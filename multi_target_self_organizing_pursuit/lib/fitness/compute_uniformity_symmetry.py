"""
compute_uniformity_symmetry.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rewrite the codes in
https://github.com/LijunSun90/pursuitCCPSOR/blob/master/lib/uniformity_symmetry_calculate.m

Author: Lijun SUN.
Date: Thu Sep 10 2020.
"""
import numpy as np


def compute_uniformity_symmetry(position_predators, position_prey):

    # Ranges of group position coordinates.
    # x_min, y_min = np.min(position_predators, axis=0)

    positions = np.vstack((position_prey, position_predators))

    x_max, y_max = np.max(positions, axis=0)

    # Edges.
    x_edges = [-1 - 0.5, position_prey[0] - 0.5,
               position_prey[0] + 0.5, x_max + 1]
    y_edges = [-1 - 0.5, position_prey[1] - 0.5,
               position_prey[1] + 0.5, y_max + 1]

    # Bivariate histogram bin counts.
    N, _, _ = \
        np.histogram2d(position_predators[:, 0], position_predators[:, 1],
                       bins=(x_edges, y_edges))

    # Uniformity.

    # 2 x 2.
    N_simple = np.zeros((2, 2))
    N_simple[0, :] = N[0, [0, 2]] + 0.5 * N[0, 1]
    N_simple[1, :] = N[2, [0, 2]] + 0.5 * N[2, 1]
    N_simple[:, 0] += 0.5 * N[1, 0]
    N_simple[:, 1] += 0.5 * N[1, 2]
    uniformity = np.std(N_simple)

    # Check whether it is really uniform.
    if uniformity == 0:
        N_simple -= N_simple[0, 0]
        if N_simple.sum() != 0:
            uniformity_diagonal = np.std([N[0, 0], N[2, 2], N[2, 0], N[0, 2]])
            uniformity_axis = np.std([N[0, 1], N[2, 1], N[1, 0], N[1, 2]])
            uniformity = uniformity_diagonal + uniformity_axis

    return uniformity


def test():
    pass


if __name__ == "__main__":
    test()
