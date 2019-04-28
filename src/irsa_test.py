import unittest
import os

import matplotlib.pyplot as plt
import numpy as np

import src.irsa as irsa


class IrsaTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # directory to store the test plots
        try:
            os.mkdir("tests")
        except FileExistsError:
            # do nothing all good
            pass
        plt.style.use('classic')

    def test_varying_frame_size(self):
        """
        We use visual benchmark by plotting the values...
        If more reliable benchmark values available, use assertAlmostEqual
        :return:
        """

        params = {"save_to": "",
                  "act_prob": 1,
                  "sim_duration": 100,
                  "traffic_type": "bernoulli",
                  "num_resources": 200,
                  "max_iter": 20,
                  "degree_distr": [0, 0, 0.5, 0.28, 0, 0, 0, 0, 0.22]}

        load_range = [0.1 * x for x in range(1, 11)]

        plt.figure()

        # benchmark values taken from the IRSA paper
        values = {50: [0.1, 0.2, 0.3, 0.4, 0.5, 0.59, 0.65, 0.6, 0.37, 0.19],
                  200: [0.1, 0.2, 0.3, 0.4, 0.5, 0.59, 0.69, 0.76, 0.47, 0.19],
                  1000: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.79, 0.57, 0.19]}

        for m in [50, 200, 1000]:

            thr = []
            thr_c = []
            params["num_resources"] = m
            params["num_ues"] = m

            for load_idx, load in enumerate(load_range):
                params["act_prob"] = load
                res = irsa.irsa_run(**params)
                thr.append(np.mean(res.throughput_normalized))
                thr_c.append(irsa.mean_confidence_interval(res.throughput_normalized))

                # No value check here, since it is different arrival distribution
                # self.assertAlmostEqual(values[m][load_idx], t, delta=tc)

            plt.errorbar(load_range, thr, yerr=thr_c, label=r"$m=%d$" % m)
            plt.plot(load_range, values[m], label=r"IRSA, $m=%d$" % m)

        plt.ylabel("Normalized throughput")
        plt.xlabel("Offered Load")
        plt.legend(loc=0)
        plt.grid(True)
        plt.savefig("tests/varying_frame_size.pdf")

    def test_packet_loss(self):
        """
        We use visual benchmark by plotting the values...
        If more reliable benchmark values available, use assertAlmostEqual
        :return:
        """

        params = {"save_to": "",
                  "act_prob": 1,
                  "sim_duration": 100,
                  "traffic_type": "bernoulli",
                  "num_resources": 200,
                  "max_iter": 20,
                  "num_ues": 200}

        load_range = [0.1 * x for x in range(1, 11)]

        plt.figure()

        degree_distrs = [[0, 1],         # slotted aloha
                        [0, 0, 1],      # 2-regular CRDSA
                        [0, 0, 0, 0, 1], # 4-regular CRDSA
                        [0, 0, 0.5, 0.28, 0, 0, 0, 0, 0.22],
                        [0, 0, 0.25, 0.6, 0, 0, 0, 0, 0.15]]

        degree_distr_labels = ["s-aloha",
                               "2-CRDSA",
                               "4-CRDSA",
                               r"$\Lambda_3$",
                               r"$\Lambda_4$"]

        for label, degree_distr in zip(degree_distr_labels, degree_distrs):

            params["degree_distr"] = degree_distr
            pktl = []
            pktl_c = []

            for load_idx, load in enumerate(load_range):
                params["act_prob"] = load
                res = irsa.irsa_run(**params)
                pktl.append(np.mean(res.packet_loss))
                pktl_c.append(irsa.mean_confidence_interval(res.packet_loss))

                # FIXME add a value check
                # self.assertAlmostEqual(values[m][load_idx], t, delta=tc)

                plt.plot(load_range, pktl, label=label)

        plt.yscale("log")
        plt.ylabel("Packet loss")
        plt.xlabel("Offered Load")
        plt.ylim((1e-4, 1e0))
        plt.legend(loc=0)
        plt.grid(True)
        plt.savefig("tests/packet_loss.pdf")


class IrsaTestPoisson(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # directory to store the test plots
        try:
            os.mkdir("tests")
        except FileExistsError:
            # do nothing all good
            pass
        plt.style.use('classic')

    def test_varying_frame_size_poisson(self):
        """
        We use visual benchmark by plotting the values...
        If more reliable benchmark values available, use assertAlmostEqual
        :return:
        """

        params = {"save_to": "",
                  "sim_duration": 100,
                  "max_iter": 20,
                  "degree_distr": [0, 0, 0.5, 0.28, 0, 0, 0, 0, 0.22]}

        load_range = [0.1 * x for x in range(1, 11)]

        plt.figure()

        # benchmark values taken from the IRSA paper (just visual reading from the plot)
        values = {50: [0.1, 0.2, 0.3, 0.4, 0.5, 0.59, 0.65, 0.6, 0.37, 0.19],
                  200: [0.1, 0.2, 0.3, 0.4, 0.5, 0.59, 0.69, 0.76, 0.47, 0.19],
                  1000: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.79, 0.57, 0.19]}

        # determined arbitrary

        for m in [50, 200, 1000]:

            thr = []
            thr_c = []
            params["num_resources"] = m

            for load_idx, load in enumerate(load_range):
                params["load"] = load
                res = irsa.irsa_run_poisson(**params)
                t = np.mean(res.throughput_normalized)
                tc = irsa.mean_confidence_interval(res.throughput_normalized)
                thr.append(t)
                thr_c.append(tc)

                # FIXME it will be certainly valuable to check whether confidence interval is not too high
                self.assertAlmostEqual(values[m][load_idx], t, delta=tc)

            plt.errorbar(load_range, thr, linestyle="--", yerr=thr_c, label=r"$m=%d$" % m)
            plt.plot(load_range, values[m], linestyle="", marker="s", label=r"IRSA, $m=%d$" % m)

        plt.ylabel("Normalized throughput")
        plt.xlabel("Offered Load")
        plt.legend(loc=0)
        plt.grid(True)
        plt.savefig("tests/varying_frame_size_poisson.pdf")

    def test_packet_loss_poisson(self):
        """
        We use visual benchmark by plotting the values...
        If more reliable benchmark values available, use assertAlmostEqual
        :return:
        """

        params = {"save_to": "",
                  "sim_duration": 100,
                  "num_resources": 200,
                  "max_iter": 20}

        load_range = [0.1 * x for x in range(1, 11)]

        plt.figure()

        degree_distrs = [[0, 1],         # slotted aloha
                        [0, 0, 1],      # 2-regular CRDSA
                        [0, 0, 0, 0, 1], # 4-regular CRDSA
                        [0, 0, 0.5, 0.28, 0, 0, 0, 0, 0.22],
                        [0, 0, 0.25, 0.6, 0, 0, 0, 0, 0.15]]

        degree_distr_labels = ["s-aloha",
                               "2-CRDSA",
                               "4-CRDSA",
                               r"$\Lambda_3$",
                               r"$\Lambda_4$"]

        for label, degree_distr in zip(degree_distr_labels, degree_distrs):

            params["degree_distr"] = degree_distr
            pktl = []
            pktl_c = []

            for load_idx, load in enumerate(load_range):
                params["load"] = load
                res = irsa.irsa_run_poisson(**params)
                pktl.append(np.mean(res.packet_loss))
                pktl_c.append(irsa.mean_confidence_interval(res.packet_loss))

                # FIXME it will be certainly valuable to check whether confidence interval is not too high
                # self.assertAlmostEqual(values[m][load_idx], t, delta=tc)

            plt.plot(load_range, pktl, label=label)

        plt.ylabel("Packet loss")
        plt.xlabel("Offered Load")
        plt.yscale("log")
        plt.ylim((1e-4, 1e0))
        plt.legend(loc=0)
        plt.grid(True)
        plt.savefig("tests/packet_loss_poisson.pdf")


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(IrsaTestPoisson))
    suite.addTest(unittest.makeSuite(IrsaTest))

    runner = unittest.TextTestRunner()
    runner.run(suite)
    # unittest.main()
