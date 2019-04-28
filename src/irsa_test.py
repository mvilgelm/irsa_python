"""
Tests for IRSA implementation. Since it is hard to find a real benchmark, the tests are basically
just visual inspections, reproducing some plots from the original IRSA article.
"""
__author__ = "Mikhail Vilgelm"
__email__ = "mikhail.vilgelm@tum.de"

import unittest
import os
import itertools
import json

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
import src.irsa as irsa


class IrsaTestFixed(unittest.TestCase):
    COLORS = itertools.cycle(["r", "b", "g", "m", "k"])
    MARKERS = itertools.cycle(["s", "o", "^", "v", "<"])

    @classmethod
    def setUpClass(cls):
        # directory to store the test plots
        try:
            os.mkdir("../tests")
        except FileExistsError:
            # do nothing all good
            pass
        plt.style.use('classic')

    def test_varying_frame_size_fixed(self):
        """
        We use visual benchmark by plotting the values...
        If more reliable benchmark values available, use assertAlmostEqual
        :return:
        """

        params = {"save_to": "",
                  "sim_duration": 100,
                  "max_iter": 20,
                  "traffic_type": "bernoulli",
                  "degree_distr": [0, 0, 0.5, 0.28, 0, 0, 0, 0, 0.22]}

        load_range = [0.1 * x for x in range(1, 11)]

        plt.figure()

        # benchmark values taken from the IRSA paper (just visual reading from the plot)
        values = {50: [0.1, 0.2, 0.3, 0.4, 0.5, 0.59, 0.65, 0.6, 0.37, 0.19],
                  200: [0.1, 0.2, 0.3, 0.4, 0.5, 0.59, 0.69, 0.76, 0.47, 0.19],
                  1000: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.79, 0.57, 0.19]}

        # determined arbitrary
        tolerance = 0.01

        results_to_store = {}

        for m in [50, 200, 1000]:

            color = next(IrsaTestFixed.COLORS)
            marker = next(IrsaTestFixed.MARKERS)

            thr = []
            thr_c = []
            params["num_resources"] = m

            for load_idx, load in enumerate(load_range):
                params["num_ues"] = int(m*load)
                params["act_prob"] = 1
                res = irsa.irsa_run(**params)
                t = np.mean(res.throughput_normalized)
                tc = irsa.mean_confidence_interval(res.throughput_normalized)
                thr.append(t)
                thr_c.append(tc)

                # FIXME it will be certainly valuable to check whether confidence interval is not too high
                self.assertAlmostEqual(values[m][load_idx], t, delta=tc+tolerance)

            results_to_store[str(m)] = thr
            results_to_store[str(m) + "_c"] = thr_c

            plt.errorbar(load_range, thr, linestyle="--", color=color, yerr=thr_c, label=r"$m=%d$" % m)
            plt.plot(load_range, values[m], linestyle="", color=color, markeredgecolor=color,
                     marker=marker, label=r"IRSA, $m=%d$" % m, markerfacecolor="None")

        with open("../tests/varying_frame_size.test", "w") as f:
            json.dump(results_to_store, f)

        plt.ylabel("Normalized throughput")
        plt.xlabel("Offered Load")
        plt.legend(loc=0)
        plt.grid(True)
        plt.savefig("../tests/varying_frame_size_fixed.pdf")

    def test_packet_loss_fixed(self):
        """
        We use visual benchmark by plotting the values...
        If more reliable benchmark values available, use assertAlmostEqual
        :return:
        """

        params = {"save_to": "",
                  "sim_duration": 1000,  # long simulations needed to capture packet loss
                  "num_resources": 200,
                  "traffic_type": "bernoulli",
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

        results_to_store = {}

        for label, degree_distr in zip(degree_distr_labels, degree_distrs):

            color=next(IrsaTestFixed.COLORS)
            marker=next(IrsaTestFixed.MARKERS)

            params["degree_distr"] = degree_distr
            pktl = []

            for load_idx, load in enumerate(load_range):
                params["num_ues"] = int(params["num_resources"]*load)
                params["act_prob"] = 1
                res = irsa.irsa_run(**params)
                mean_pktl = np.mean(res.packet_loss)
                pktl.append(mean_pktl)

                # FIXME it will be certainly valuable to check whether confidence interval is not too high
                # self.assertAlmostEqual(values[m][load_idx], t, delta=tc)
            results_to_store[label] = pktl
            plt.plot(load_range, pktl, "-"+color+marker, markeredgecolor=color,
                     markerfacecolor="None", label=label)

        with open("../tests/pkt_loss.test", "w") as f:
            json.dump(results_to_store, f)

        plt.ylabel("Packet loss")
        plt.xlabel("Offered Load")
        plt.yscale("log")
        plt.ylim((1e-4, 1e0))
        plt.legend(loc=0)
        plt.grid(True)
        plt.savefig("../tests/packet_loss_fixed.pdf")


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(IrsaTestFixed))

    runner = unittest.TextTestRunner()
    runner.run(suite)
    # unittest.main()
