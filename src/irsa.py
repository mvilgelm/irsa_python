import json
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import weakref

from tqdm import trange

plt.style.use('ggplot')


def mean_confidence_interval(data, confidence=0.95):
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

    def __init__(self, degree_distr, num_resources):
        self.degree_distr = degree_distr
        self.num_resources = num_resources

    def get_degree_distr(self):
        return self.degree_distr

    def get_num_resources(self):
        return self.num_resources

    def assign_ues_to_resources(self, ues):

        num_resources = self.num_resources

        resources = [IrsaResource(idx) for idx in range(num_resources)]

        degree_distr = self.degree_distr

        for ue in ues:

            num_replicas = np.random.choice(range(len(degree_distr)), p=degree_distr)

            # choose resources
            resources_ue = np.random.choice(resources, num_replicas, replace=False)
            ue.pointers = [weakref.ref(r) for r in resources_ue]
            for r in resources_ue:
                r.ues.add(ue)

        return resources

    def decode(self, resources, max_iter):
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

        return decoded_ues, end-start


class IrsaResource:

    def __init__(self, idx):
        self.idx = idx
        self.ues = set()

    def process(self, mpr):
        """
        FIXME
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
        try:
            self.ues.remove(ue)
        except:
            raise

        return len(self.ues)


class IrsaUE:

    def __init__(self, **kwargs):
        # stores pointers to replicas
        self.pointers = []

        # legacy -- for extension
        self.tx_attempts = 0
        self.active = False

        self.success = 0
        self.fail = 0

        for key, val in kwargs.items():
            if key == "idx":
                self.idx = val
            if key == "tx_limit":
                self.tx_limit = val

    def activate(self):
        self.active = True

    def deactivate(self, success):
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

    # **** Load variables ****
    pars = Params(**kwargs)

    # **** Create the instances ****

    results = IrsaResults()

    ues = []
    for idx in range(pars.num_ues):
        ues.append(IrsaUE(idx=idx, **kwargs))

    bs = BaseStation(pars.degree_distr, pars.num_resources)

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
            collision_count = 0

            resources = bs.assign_ues_to_resources(active_ues)

            # processing the resources
            decoded_ues, duration = bs.decode(resources, pars.max_iter)

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


def irsa_run_poisson(**kwargs):

    # **** Load variables ****
    pars = Params(**kwargs)

    # **** Create the instances ****

    results = IrsaResults()

    bs = BaseStation(pars.degree_distr, pars.num_resources)

    # *** initialize *****

    # *** Start the simulation ***
    sim_range = trange(pars.sim_duration, desc="Thr", mininterval=1)
    for t in sim_range:

        # create active ues
        num_ues = np.random.poisson(pars.load * bs.num_resources)

        active_ues = []
        for idx in range(num_ues):
            active_ues.append(IrsaUE(idx=idx, **kwargs))

        if len(active_ues) > 0:

            success_count = 0
            collision_count = 0

            resources = bs.assign_ues_to_resources(active_ues)

            # processing the resources
            decoded_ues, duration = bs.decode(resources, pars.max_iter)

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
