import numpy as np
import matplotlib.pyplot as plt

ipus = np.array([1, 2, 4, 8, 16])
tflops = np.array([1.23, 2.02, 3.74, 7.28, 11.93]) # SDK 2.1.0
tflops2 = np.array([1.32, 2.26, 4.00, 7.90, 13.98]) # SDK 2.2.0
storebw = np.array([0.61, 1.01, 1.87, 3.64, 5.97])
loadbw = np.array([4.29, 7.07, 13.08, 25.49, 41.76])
totalbw = np.array([4.90, 8.08, 14.95, 29.13, 47.72])
ideal_tflops = ipus*tflops[0]
ideal_tflops2 = ipus*tflops2[0]
ideal_storebw = ipus*storebw[0]
ideal_loadbw = ipus*loadbw[0]
ideal_totalbw = ipus*totalbw[0]


def plot_throughput():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(ipus, tflops, "bo-", label="throughput 2.1.0")
    ax.plot(ipus, tflops2, "go-", label="throughput 2.2.0")
    ax.plot(ipus, ideal_tflops, "b--", label="ideal speedup 2.1.0")
    ax.plot(ipus, ideal_tflops2, "g--", label="ideal speedup 2.2.0")
    ax.legend()
    ax.set_xlabel("No. IPUs")
    ax.set_ylabel("TFLOPS")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(ipus)
    ax.set_yticks(ideal_tflops)
    ax.set_xticklabels([str(i) for i in ipus])
    ax.set_yticklabels([str(i) for i in ideal_tflops])
    ax.grid()
    plt.savefig("flops.png")
    return None


def plot_bandwidths():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(ipus, loadbw, "bo-", label="load bandwidth")
    ax.plot(ipus, ideal_loadbw, "b--", label="ideal load BW speedup")
    ax.plot(ipus, storebw, "ro-", label="store bandwidth")
    ax.plot(ipus, ideal_storebw, "r--", label="ideal store BW speedup")
    ax.plot(ipus, totalbw, "go-", label="total bandwidth")
    ax.plot(ipus, ideal_totalbw, "g--", label="ideal total BW speedup")
    ax.legend()
    ax.set_xlabel("No. IPUs")
    ax.set_ylabel("TB/s")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(ipus)
    ax.set_yticks(ideal_totalbw)
    ax.set_xticklabels([str(i) for i in ipus])
    ax.set_yticklabels([str(i) for i in ideal_totalbw])
    ax.grid()
    plt.savefig("bw.png")
    return None


if __name__ == "__main__":
    plot_throughput()
    # plot_bandwidths()