import numpy as np
import matplotlib.pyplot as plt

def plot_throughput():
    ipus = np.array([1, 2, 4, 8, 16])
    tflops = np.array([1.32, 2.26, 4.00, 7.91, 14.0])

    # Linear speedup
    linear_tflops = ipus*tflops[0]

    # Expected ideal speedup (given chip-to-chip communication)
    t1 = 1.95e-4 # us
    sides = np.array([320, 403, 508, 640, 806])
    float_size = 4
    num_messages = 4 # send 2 + receive 2
    elems_per_message = sides**2
    dataGB = num_messages*float_size*elems_per_message*1e-9
    delta = dataGB/320 # s
    expected_tflops = tflops[0]*ipus*t1/(t1+delta)
    expected_tflops[0] = tflops[0]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(ipus, tflops, "bo-", label="Measured throughput")
    ax.plot(ipus, linear_tflops, "r--", label="Linear speedup")
    ax.plot(ipus, expected_tflops, "g--", label="Ideal upper bound")
    ax.legend()
    ax.set_xlabel("No. IPUs")
    ax.set_ylabel("TFLOPS")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(ipus)
    ax.set_yticks(linear_tflops)
    ax.set_xticklabels([str(i) for i in ipus])
    ax.set_yticklabels([str(i) for i in linear_tflops])
    ax.grid()
    plt.savefig("flops.png")

    return None

if __name__ == "__main__":
    plot_throughput()