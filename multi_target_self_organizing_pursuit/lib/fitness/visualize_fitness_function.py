"""
visualize_fitness_function.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: FRI APR 23 2021.
"""
import numpy as np
import matplotlib.pyplot as plt


def visualize_f_repel():

    min_distance_with_predator = 2
    min_distance_with_prey = 1

    scale = 1

    nnd = np.arange(0, 4.1, 0.1)
    f_repel_prey = np.zeros(nnd.shape)
    f_repel_predator = np.zeros(nnd.shape)
    base_line = np.ones(nnd.shape)

    for idx in range(len(nnd)):

        if nnd[idx] <= min_distance_with_prey:
            f_repel_prey[idx] = \
                np.exp(-scale * (nnd[idx] - min_distance_with_prey))
        else:
            f_repel_prey[idx] = 1

        if nnd[idx] <= min_distance_with_predator:
            f_repel_predator[idx] = \
                np.exp(-scale * (nnd[idx] - min_distance_with_predator))
        else:
            f_repel_predator[idx] = 1

    plt.subplot(2, 1, 1)
    plt.plot(nnd, base_line, "r--", label="Horizontal line: value 1")
    plt.plot(nnd, f_repel_prey,
             label="f_repel_prey = exp(-(nnd_prey - D_min_prey))")
    plt.xticks(ticks=[0, 1, 2, 3, 4],
               labels=[0, "D_min_prey", 2, 3, 4])
    plt.xlabel("nnd_prey")
    plt.ylabel("Fitness value")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(nnd, base_line, "r--", label="Horizontal line: value 1")
    plt.plot(nnd, f_repel_predator,
             label="f_repel_predator = exp(-(nnd_predator - D_min_predator))")
    plt.xticks(ticks=[0, 1, 2, 3, 4],
               labels=[0, 1, "D_min_predator", 3, 4])
    plt.xlabel("nnd_predator")
    plt.ylabel("Fitness value")
    plt.legend()
    plt.grid()

    plt.show()
    pass


def visualize_binary_entropy():
    p = np.arange(0.01, 0.99, 0.01)

    binary_entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)

    plt.plot(p, binary_entropy)
    plt.xlabel("p")
    plt.ylabel("Binary entropy H(p)")
    plt.grid()
    plt.show()


def main():
    # visualize_f_repel()
    visualize_binary_entropy()
    pass


if __name__ == "__main__":
    main()
