"""
fitness_single_prey_pursuit.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rewrite the codes in
https://github.com/LijunSun90/pursuitCCPSOR/blob/master/lib/fitness_encircle.m

Author: Lijun SUN.
Date: Thu Sep 10 2020.
~~~~~~~~~~~~~~~~~~~~~~
Modified on: SAT 8 MAY 2021.
Improve the fitness function in the formula which
1. allows the fitness calculation for the single-prey pursuit with arbitrary
number of predators;
2. introduces the coordination conventional model.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from lib.fitness.compute_uniformity_symmetry import compute_uniformity_symmetry


class FitnessSinglePreyPursuit:
    def __init__(self, fov_scope, under_debug=False):

        self.fov_scope = fov_scope
        self.under_debug = under_debug

        fov_radius = int((self.fov_scope - 1) / 2)

        # 2D numpy array, such as np.array([fov_radius] * 2)
        self.local_own_position = np.array([fov_radius] * 2)

        # Parameters.

        # For example, when fov_scope = 3, it is
        # array([[0, 3, 6],
        #        [1, 4, 7],
        #        [2, 5, 8]])
        self.priority_matrix = \
            np.arange(0, self.fov_scope**2).reshape((self.fov_scope,
                                                     self.fov_scope)).T

        # N, E, S, W.
        self.axial_neighbors_mask = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

    def compute(self, current_position=None,
                cluster_center=None,
                cluster_other_predators=None,
                local_env_vectors=None,
                local_env_matrix=None):
        """
        :param current_position: 1d numpy array of shape (2,).
            the position that is being evaluated.
        :param cluster_center: 1d numpy array of shape (2,).
        :param cluster_other_predators: 2d numpy array of shape
            (n_predators - 1, 2).
        :param local_env_vectors: a dict,
            {"local_preys": 2d numpy array of shape (n_local_preys, 2),
             "local_predators": 2d numpy array of shape (n_local_predators, 2),
             "local_obstacles": 2d numpy array of shape (n_local_obstacles, 2).}
        :param local_env_matrix: 3d numpy array,
            where the 1st, 2nd, 3rd channel is
            the preys, predators, and obstacles, respectively.
        :return: a float.
        """

        # ##################################################
        if self.is_collide(current_position, local_env_matrix):
            fitness = np.inf
            return fitness
        else:
            exist_axial_neighboring_preys = \
                self.exist_axial_neighbors(current_position,
                                           local_env_matrix[:, :, 0])
            exist_axial_neighboring_predators = \
                self.exist_axial_neighbors(current_position,
                                           local_env_matrix[:, :, 1])

            if exist_axial_neighboring_predators:
                if not exist_axial_neighboring_preys:
                    fitness = np.inf
                    return fitness
                else:
                    fitness = self.lexicographic_convention(current_position,
                                                            local_env_vectors,
                                                            local_env_matrix)
                    return fitness

        # ##################################################
        fitness = self.fitness_single_prey_pursuit(current_position,
                                                   cluster_center,
                                                   cluster_other_predators)

        return fitness

    @staticmethod
    def is_collide(current_position, local_env_matrix):
        pixel_values_in_new_position = \
            local_env_matrix[current_position[0], current_position[1]].sum()

        collide = False
        if pixel_values_in_new_position != 0:
            collide = True

        return collide

    def exist_axial_neighbors(self, current_position, local_env_matrix):
        valid_axial_neighbors = \
            self.get_valid_axial_neighbors(current_position,
                                           local_env_matrix)

        exist_or_not = (valid_axial_neighbors.shape[0] > 0)

        return exist_or_not

    def get_valid_axial_neighbors(self, current_position, local_env_matrix):
        axial_positions = self.axial_neighbors_mask + current_position

        valid_index = []
        for idx, position in enumerate(axial_positions):
            if (position >= 0).all() and (position < self.fov_scope).all():
                if local_env_matrix[position[0], position[1]] > 0:
                    valid_index.append(idx)

        axial_positions = axial_positions[valid_index]

        return axial_positions

    def is_captured(self, prey, local_env_matrix):

        capture_positions = prey + self.axial_neighbors_mask

        valid_index = []
        for idx, position in enumerate(capture_positions):
            if (position >= 0).all() and (position < self.fov_scope).all():
                valid_index.append(idx)

        capture_positions = capture_positions[valid_index, :]

        occupied_capture_positions = \
            local_env_matrix[capture_positions[:, 0],
                             capture_positions[:, 1], :].sum(axis=1)

        # Valid only if collision is not allowed in the space.
        # Otherwise, more than one agents can occupy the same position.
        yes_no = True if (occupied_capture_positions > 0).all() and len(capture_positions) == 4 else False

        return yes_no

    def sort_entities(self, entities):
        entities_priorities = self.priority_matrix[entities[:, 0],
                                                   entities[:, 1]]
        idx = np.argsort(entities_priorities)
        entities = entities[idx]

        return entities

    def get_open_axial_neighbors(self, current_position, local_env_matrix):
        neighbors = self.axial_neighbors_mask + current_position

        valid_index = []
        for idx, position in enumerate(neighbors):
            if (position >= 0).all() and (position < self.fov_scope).all():
                if not self.is_collide(position, local_env_matrix):
                    valid_index.append(idx)

        open_neighbors = neighbors[valid_index]

        return open_neighbors

    def lexicographic_convention(self, current_position,
                                 local_env_vectors, local_env_matrix):

        # Pre-processing.
        # Add the current real predator in the local environmental info.
        local_env_matrix[self.local_own_position[0],
                         self.local_own_position[1], 1] = 1

        # Initialization.
        free_capturing_position_to_predators = dict()

        # ##################################################
        # All local free capturing positions in the priority order.

        # All local preys.
        local_preys = local_env_vectors["local_preys"]

        # All local free capturing positions.
        local_free_capturing_positions = []
        for prey in local_preys:

            if self.is_captured(prey, local_env_matrix):
                continue

            # All free capturing positions for the current prey.
            free_capturing_positions = \
                self.get_open_axial_neighbors(prey,
                                              local_env_matrix)
            for position in free_capturing_positions.tolist():
                if position in local_free_capturing_positions:
                    continue

                # Update.
                local_free_capturing_positions.append(position)

        # Sort all free capturing positions.
        local_free_capturing_positions = \
            np.asarray(local_free_capturing_positions)

        local_free_capturing_positions = \
            self.sort_entities(local_free_capturing_positions)

        # ##################################################
        # The free capturing position with the highest priority is assigned
        # to the free conventional predator with the highest priority.

        # Loop local free capturing positions in the priority order.
        free_capturing_positions_neighbors = dict()
        for position in local_free_capturing_positions:
            free_conventional_predators = []

            # All one-step away free predators.
            axial_predators = \
                self.get_valid_axial_neighbors(position,
                                               local_env_matrix[:, :, 1])

            # Remove locked predators.
            for idx, predator in enumerate(axial_predators):
                is_locked_predator = \
                    self.exist_axial_neighbors(predator,
                                               local_env_matrix[:, :, 0])

                if predator.tolist() in free_conventional_predators \
                        or is_locked_predator:
                    continue

                # Update.
                free_conventional_predators.append(predator.tolist())

            if len(free_conventional_predators) == 0:
                continue

            # Sort one-step away free predators.
            free_conventional_predators = \
                np.asarray(free_conventional_predators)

            free_conventional_predators = \
                self.sort_entities(free_conventional_predators)

            free_capturing_positions_neighbors[tuple(position)] = free_conventional_predators

            # Assignment.
            for predator in free_conventional_predators.tolist():
                # The predator has already be assigned to a free capturing
                # position with a higher priority.
                if predator in free_capturing_position_to_predators.values():
                    continue

                free_capturing_position_to_predators[tuple(position)] = \
                    predator

                break

        # ##################################################
        # Get the fitness value.

        assigned_predator = None

        key = tuple(current_position)
        if key in free_capturing_position_to_predators.keys():
            assigned_predator = \
                free_capturing_position_to_predators[key]

        if assigned_predator == self.local_own_position.tolist():
            fitness = -1
            # The assigned capturing  position has other neighboring pursuers that has a higher priority.
            # With the uncertain capturing position exist in the local view,
            # the pursuer cannot move to that assigned capturing position to avoid possible collisions.
            neighboring_pursuers = free_capturing_positions_neighbors[tuple(current_position)]
            uncertain_capturing_positions_are_allocated = False
            for key in free_capturing_position_to_predators.keys():
                if 0 in key:
                    uncertain_capturing_positions_are_allocated = True
                    break
            if len(neighboring_pursuers) > 1 and uncertain_capturing_positions_are_allocated:
                fitness = np.inf
        else:
            fitness = np.inf

        return fitness

    @staticmethod
    def fitness_single_prey_pursuit(current_position=None,
                                    cluster_center=None,
                                    cluster_other_predators=None):

        # ##################################################
        # Part - closure: the group center is located inside the convex hull.

        position_predators = \
            np.vstack((current_position, cluster_other_predators))

        # When the number of partners is less than three,
        # the fitness_closure does not take effect.
        fitness_closure = 0

        if position_predators.shape[0] >= 3:
            # 1d numpy array.
            try:
                hull = ConvexHull(position_predators)
                # 2d numpy array.
                hull_vertices = position_predators[hull.vertices, :]
            except QhullError:
                # When all points are in line.
                hull_vertices = position_predators.copy()

            polygon = Polygon(hull_vertices)
            point_prey = Point(cluster_center)
            is_in = polygon.contains(point_prey)

            fitness_closure = 1 - is_in

        # ##################################################
        # Part - group expanse.

        fitness_expanse = np.mean(
            np.linalg.norm(position_predators - cluster_center, ord=2, axis=1))

        # ##################################################
        # Part - uniformity.

        fitness_uniformity = compute_uniformity_symmetry(position_predators,
                                                         cluster_center)
        # Output.
        fitness = fitness_closure + fitness_expanse + fitness_uniformity

        return fitness


def test():
    local_preys_matrix = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]])
    local_predators_matrix = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]])
    local_obstacles_matrix = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]])
    local_env_matrix = \
        np.stack((local_preys_matrix,
                  local_predators_matrix,
                  local_obstacles_matrix), axis=2)

    # Vector representation.
    # 2d numpy array of shape (x, 2) where 0 <= x.
    local_preys = np.asarray(np.where(local_preys_matrix == 1)).T
    local_predators = np.asarray(np.where(local_predators_matrix == 1)).T
    local_obstacles = np.asarray(np.where(local_obstacles_matrix == 1)).T
    local_env_vectors = {
        "local_preys": local_preys,
        "local_predators": local_predators,
        "local_obstacles": local_obstacles}

    # Cluster.
    cluster_center = np.array([3, 5])
    cluster_other_predators = np.array([4, 4])

    local_own_position = np.array([3, 3])
    axial_neighbors_mask = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
    axial_neighbors = axial_neighbors_mask + local_own_position

    fitness_computer = FitnessSinglePreyPursuit().compute

    for current_position in axial_neighbors:
        fitness = \
            fitness_computer(local_own_position=local_own_position,
                             current_position=current_position,
                             cluster_center=cluster_center,
                             cluster_other_predators=cluster_other_predators,
                             local_env_vectors=local_env_vectors,
                             local_env_matrix=local_env_matrix)
        print("Fitness for", current_position,
              "is", fitness)

    # Fitness for [2 3] is 23251407.699364424
    # Fitness for [3 4] is 1.819479216882342
    # Fitness for [4 3] is 26911661.737208813
    # Fitness for [3 2] is 2.8194792168823417
    pass


if __name__ == "__main__":
    test()
