"""
Example run of IRSA protocol
"""
import itertools
import matplotlib
import json
import numpy as np

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.style.use('classic')

from src.irsa import irsa_run


def example_run():

    params = {"save_to": None,                          # specify where to save simulation results
              "sim_duration": 100,                      # simulation duration (in IRSA frames)
              "max_iter": 20,                           # maximum number of iterations of SIC
              "traffic_type": "bernoulli",              # traffic type
              "degree_distr":                           # degree distribution for the amount of replicas
                  [0, 0, 0.5, 0.28, 0, 0, 0, 0, 0.22],
              "num_ues": 100,                           # number of users
              "act_prob": 0.3,                          # probability of any UE to be active in any frame
              "num_resources": 100                      # number of slots in an IRSA frame
              }

    results = irsa_run(**params)

    mean_throughput = np.mean(results.throughput_normalized)
    mean_pkt_loss = np.mean(results.packet_loss)

    print(f"Exemplary run, throughput: {mean_throughput}, packet loss: {mean_pkt_loss}")


if __name__ == "__main__":
    example_run()
