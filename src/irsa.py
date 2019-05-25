"""
IRSA implementation.
"""
__author__ = "Mikhail Vilgelm"
__email__ = "mikhail.vilgelm@tum.de"

import json
import time

import numpy as np
import scipy.stats as st
import weakref

from tqdm import trange


def mean_confidence_interval(data, confidence=0.95):
    """
    Compute confidence interval on data
    :param data:
    :param confidence:
    :return:
    """

    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t._ppf((1+confidence)/2., n-1)
    return h


class IrsaResults:
    """
    Storing simulation results
    """

    def __init__(self):
        self.throughput_normalized = []
        self.packet_loss = []
        self.avg_decoding_time = 0


class BaseStation:
    """
    Base station, implementing virtual assignment of resources (random selection),
    and decoding (so far ideal decoding only implemented)
    """

    def __init__(self, degree_distr, num_resources, max_iter, enable_ideal_decoding=True):
        """

        :param degree_distr: degree distrbution to use for ues
        :param num_resources: number of resources (slots per frame)
        :param max_iter: maximum number of decoding iterations
        :param enable_ideal_decoding: if False, decoding is done in a realistic way
        """
        self.degree_distr = degree_distr
        self.num_resources = num_resources

        # TODO make a parameter
        self.enable_ideal_decoding = enable_ideal_decoding
        self.max_iter = max_iter

    def assign_resources(self, ues):
        """
        (Virtual) allocation of resources to UEs.
        :param ues: pool of active UEs
        :return: resources with ues allocated to them
        """

        # create resources
        resources = [IrsaResource(idx) for idx in range(self.num_resources)]

        for ue in ues:

            # random choice of the number of replicas
            num_replicas = np.random.choice(range(len(self.degree_distr)), p=self.degree_distr)

            # random resource choice
            resources_ue = np.random.choice(resources, num_replicas, replace=False)

            # avoiding cycling references, otherwise seems to memory leak
            ue.pointers = [weakref.ref(r) for r in resources_ue]
            for r in resources_ue:
                r.ues.add(ue)

        return resources

    def process_resources(self, resources, max_iter):
        """
        Takes frame slots (resources), and decodes the successful UEs
        :param resources: resources with UEs in them
        :param max_iter: maximum number of iteration for decoding
        :return: set of decoded UEs, execution time
        """

        if self.enable_ideal_decoding:
            return self.decoding_ideal(resources, max_iter)
        else:
            return self.decoding_realistic(resources, max_iter)

    def decoding_ideal(self, resources, max_iter):
        """
        Ideal decoding, assuming perfect interference cancellation
        :param resources:
        :param max_iter:
        :return:
        """
        start = time.time()

        # get clean slots to start with
        clean_slots = set([r for r in resources if len(r.ues) == 1])
        idle_count = len([r for r in resources if len(r.ues) == 0])

        decoded_ues = set()
        iter_count = 0

        while iter_count < max_iter:

            clean_slots_next_iter = set()
            while len(clean_slots) > 0:
                # get a random slot
                r = clean_slots.pop()

                ue = r.ues.pop()
                decoded_ues.add(ue)

                # cancel decoded guy everywhere
                for rx_p in ue.pointers:
                    rx = rx_p()
                    if rx != r:
                        size = rx.cancel_ue(ue)
                        if size == 1:
                            clean_slots_next_iter.add(rx)
                        elif size == 0:
                            if rx in clean_slots_next_iter:
                                clean_slots_next_iter.remove(rx)
                            else:
                                clean_slots.remove(rx)

                ue.pointers = []

            if len(clean_slots_next_iter) == 0:
                # cannot decode anything more...
                break
            else:
                clean_slots = clean_slots_next_iter

            iter_count += 1

        end = time.time()

        return decoded_ues, end - start

    def decoding_realistic(self, resources, max_iter):
        """
        TODO implement
        :param resources:
        :param max_iter:
        :return:
        """
        raise NotImplementedError


class IrsaResource:
    """
    Represents a resource
    """

    def __init__(self, idx):
        self.idx = idx
        self.ues = set()

    def process(self, mpr):
        """
        FIXME legacy, for extention to MPR
        :param mpr:
        :return:
        """
        if len(self.ues) == 0:
            return False

        coll_size = len(self.ues)

        if coll_size in mpr.keys():
            succ_probability = mpr[coll_size]

            if np.random.uniform() < succ_probability:
                return True
        else:
            return False

    def cancel_ue(self, ue):
        """
        Cancel interference from a given UE
        :param ue: intf to cancel
        :return: amount of remaining UEs
        """

        try:
            self.ues.remove(ue)
        except:
            raise

        return len(self.ues)


class IrsaUE:
    """
    Represents a User Equipment
    """

    def __init__(self, **kwargs):
        # stores pointers to replicas
        self.pointers = []

        # FIXME legacy code -- for extension
        self.tx_attempts = 0
        self.active = False

        self.success = 0
        self.fail = 0

        for key, val in kwargs.items():
            if key == "idx":
                self.idx = val
            if key == "tx_limit":
                self.tx_limit = val

    def deactivate(self, success):
        """
        Legacy function, for extensions
        :param success:
        :return:
        """
        self.active = False
        self.tx_attempts = 0
        if success:
            self.success += 1
        else:
            self.fail += 1


class Params:
    """
    Parameters class, for convenience
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def irsa_run(**kwargs):
    """
    Single simulation run
    :param kwargs: simulation parameters
    :return:
    """

    # **** Load variables ****
    pars = Params(**kwargs)

    # **** Create the instances ****

    results = IrsaResults()

    ues = []
    for idx in range(pars.num_ues):
        ues.append(IrsaUE(idx=idx, **kwargs))

    bs = BaseStation(pars.degree_distr, pars.num_resources, max_iter=pars.max_iter)

    # *** initialize *****

    # *** Start the simulation ***
    sim_range = trange(pars.sim_duration, desc="Thr", mininterval=1)
    for t in sim_range:

        active_ues = set()

        # activate UEs
        for ue in ues:
            if np.random.uniform() < pars.act_prob:
                active_ues.add(ue)

        if len(active_ues) > 0:

            success_count = 0

            resources = bs.assign_resources(active_ues)

            # processing the resources
            decoded_ues, duration = bs.process_resources(resources)

            success_count += len(decoded_ues)

            packet_loss = (len(active_ues) - success_count) / len(active_ues)

            # Update learning alg
            throughput_normalized = success_count / len(resources)

            # Store the results
            results.throughput_normalized.append(throughput_normalized)
            results.packet_loss.append(packet_loss)
            results.avg_decoding_time = results.avg_decoding_time + (duration-results.avg_decoding_time) / (t+1)

        else:

            results.throughput_normalized.append(0)
            results.packet_loss.append(0)

        if t % int(pars.sim_duration/10) == 0:
            sim_range.set_description(f"Thr: {np.mean(results.throughput_normalized)}")

    if not (pars.save_to is None or pars.save_to == ""):
        with open(pars.save_to, "w") as f:
            json.dump(results.__dict__, f)

    return results


if __name__ == "__main__":
    pass
