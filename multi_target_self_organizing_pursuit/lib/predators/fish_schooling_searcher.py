"""
fish_schooling_searcher.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

References:
[1] Camazine, Scott, Jean-Louis Deneubourg, Nigel R. Franks, James Sneyd,
Guy Theraula, and Eric Bonabeau. Self-organization in biological systems.
Princeton university press, 2003.
[2] Andreas Huth, Christian Wissel, The simulation of the movement of fish
schools, Journal of Theoretical Biology, Volume 156, Issue 3, 1992,
Pages 365-385, ISSN 0022-5193, https://doi.org/10.1016/S0022-5193(05)80681-2.

Author: Lijun SUN.
Date: WED JUN 1 2021.
"""
import copy
import numpy as np
from sklearn.metrics import pairwise_distances

from lib.agents.basic_matrix_agent import BasicMatrixAgent


class FishSchoolingSearcher(BasicMatrixAgent):
    def __init__(self, fov_scope, under_debug=False):

        super().__init__()

        # 11
        self.fov_scope = fov_scope
        self.under_debug = under_debug

        # 5
        self.fov_radius = int((self.fov_scope - 1) / 2)

        # 2D numpy array, such as np.array([fov_radius] * 2)
        self.own_position = np.array([self.fov_radius] * 2)

        # ##################################################
        # Parameters.

        # Only the predators in the valid local scope can be identified as
        # free or not.
        self.min_secure_distance = 2
        # self.max_schooling_distance = self.fov_radius - 1
        self.max_secure_scope_min_idx = 1
        self.max_secure_scope_max_idx = self.fov_scope - 2
        self.repel_area_mask = None
        self.repel_area = None

        # ##################################################
        # Sharing variables.
        self.local_env_matrix = None

        self.last_local_env_matrix = None
        self.last_own_position = np.zeros((0, 2), dtype=int)
        self.last_own_moving_direction = \
            np.asarray(self.action_direction[np.random.choice(4, 1)[0] + 1])
        
    def get_action(self, local_env_matrix, local_env_vectors):

        # ##################################################
        # Sharing variables.
        if self.local_env_matrix is not None:
            self.last_local_env_matrix = self.local_env_matrix.copy()

        self.local_env_matrix = local_env_matrix

        # ##################################################
        # Classify the neighbors into:
        # free predators;
        # obstacles: obstacles + locked predator.
        local_predators = local_env_vectors["local_predators"]

        # Identify locked predators.
        free_predators_idx = []
        for idx, predator in enumerate(local_predators):
            # Do not consider predators outside the valid local scope since
            # it may be impossible to identify whether they are free or not.
            if (predator == 0).any() and (predator == self.fov_scope).any():
                continue

            if self.is_free_predator(predator):
                free_predators_idx.append(idx)

        free_predators = local_predators[free_predators_idx]

        # ##################################################
        # Rule 1.

        # Single-agent search.
        if free_predators.shape[0] == 0:

            next_position = self.own_position + self.last_own_moving_direction
            if not self.is_collide(next_position, self.local_env_matrix):
                next_action = \
                    self.direction_action[tuple(self.last_own_moving_direction)]
            else:
                next_action = \
                    self.random_walk_in_single_agent_system_without_memory(
                        self.own_position, self.local_env_matrix)

            self.last_own_moving_direction = \
                np.asarray(self.action_direction[next_action])
            self.last_own_position = \
                self.own_position - self.last_own_moving_direction

            print("Searcher: Single-agent search.")
            return next_action

        # ##################################################
        # Rule 2.

        # Pairwise distances with free predators.
        current_position = self.own_position.reshape(1, -1)
        distance_with_free_predators = pairwise_distances(current_position,
                                                          free_predators,
                                                          metric='manhattan')[0]

        # Repel by repulsive predators.
        nearest_distance_with_free_predators = \
            np.min(distance_with_free_predators)

        if nearest_distance_with_free_predators <= self.min_secure_distance:

            repulsive_free_predators_idx = np.where(
                distance_with_free_predators <= self.min_secure_distance)
            repulsive_free_predators = \
                free_predators[repulsive_free_predators_idx]

            directions_closer_to_repulsive_predators = []
            for predator in repulsive_free_predators:
                offsets = self.get_offsets(self.own_position, predator)
                for idx, offset in enumerate(offsets):
                    if offset == 0:
                        continue

                    direction = [0, 0]
                    direction[idx] = np.sign(offset)
                    directions_closer_to_repulsive_predators.append(direction)

            candidate_next_direction = []
            for next_direction in self.axial_neighbors_mask.tolist():
                if next_direction in directions_closer_to_repulsive_predators:
                    continue

                next_position = self.own_position + next_direction
                if not self.is_collide(next_position, self.local_env_matrix):
                    candidate_next_direction.append(next_direction)

            n_candidate_next_direction = len(candidate_next_direction)
            if n_candidate_next_direction == 0:
                next_action = 0
            else:
                idx = np.random.choice(n_candidate_next_direction, 1)[0]
                next_direction = candidate_next_direction[idx]
                next_action = self.direction_action[tuple(next_direction)]

            self.last_own_moving_direction = \
                np.asarray(self.action_direction[next_action])
            self.last_own_position = \
                self.own_position - self.last_own_moving_direction

            print("Searcher: Repel by repulsive predators.")
            return next_action

        # ##################################################
        # Rule 3.

        # Attract by attractive predators.

        if (free_predators <= self.max_secure_scope_min_idx).any() or \
                (free_predators >= self.max_secure_scope_max_idx).any():

            attractive_free_predators = []
            for predator in free_predators:
                if (predator <= self.max_secure_scope_min_idx).any() or \
                        (predator >= self.max_secure_scope_max_idx).any():

                    attractive_free_predators.append(predator)

            attractive_free_predators = np.asarray(attractive_free_predators)

            directions_away_from_attractive_predators = []
            for predator in attractive_free_predators:
                offsets = self.get_offsets(predator, self.own_position)
                for idx, offset in enumerate(offsets):
                    if offset == 0:
                        continue

                    direction = [0, 0]
                    direction[idx] = np.sign(offset)
                    directions_away_from_attractive_predators.append(direction)

            candidate_next_direction = []
            for next_direction in self.axial_neighbors_mask.tolist():
                if next_direction in \
                        directions_away_from_attractive_predators:
                    continue

                next_position = self.own_position + next_direction
                if not self.is_collide(next_position, self.local_env_matrix):
                    candidate_next_direction.append(next_direction)

            n_candidate_next_direction = len(candidate_next_direction)
            if n_candidate_next_direction == 0:
                next_action = 0
            else:
                idx = np.random.choice(n_candidate_next_direction, 1)[0]
                next_direction = candidate_next_direction[idx]
                next_action = self.direction_action[tuple(next_direction)]

            self.last_own_moving_direction = \
                np.asarray(self.action_direction[next_action])
            self.last_own_position = \
                self.own_position - self.last_own_moving_direction

            print("Searcher: Attract by attractive predators.")
            return next_action

        # ##################################################
        # Rule 4.

        # Otherwise, move towards the neighborhood center.
        # directions_to_neighborhood_center = []
        # for predator in free_predators:
        #     direction = self.get_offsets(self.own_position, predator).tolist()
        #     directions_to_neighborhood_center.append(direction)
        #
        # direction_to_neighborhood_center = \
        #     np.sum(directions_to_neighborhood_center, axis=0)
        # idx = np.argmax(np.abs(direction_to_neighborhood_center))
        # next_direction = [0, 0]
        # next_direction[idx] = np.sign(direction_to_neighborhood_center[idx])
        #
        # next_position = self.own_position + next_direction
        # if not self.is_collide(next_position, self.local_env_matrix):
        #     next_action = self.direction_action[tuple(next_direction)]
        # else:
        #     next_action = \
        #         self.random_walk_in_single_agent_system_without_memory(
        #             self.own_position, self.local_env_matrix)
        #
        # print("Neighborhood center, next_action", next_action)

        # Align with the moving directions of neighbors.
        sorted_moving_direction_list = self.get_neighbors_moving_directions()

        next_action = None
        for moving_direction in sorted_moving_direction_list:
            next_position = self.own_position + moving_direction
            if not self.is_collide(next_position, self.local_env_matrix):
                next_action = self.direction_action[tuple(moving_direction)]
                break

        if next_action is None:
            next_position = self.own_position + self.last_own_moving_direction
            if not self.is_collide(next_position, self.local_env_matrix):
                next_action = \
                    self.direction_action[tuple(self.last_own_moving_direction)]
            else:
                next_action = \
                    self.random_walk_in_single_agent_system_without_memory(
                        self.own_position, self.local_env_matrix)

        self.last_own_moving_direction = \
            np.asarray(self.action_direction[next_action])
        self.last_own_position = \
            self.own_position - self.last_own_moving_direction

        print("Searcher: Align with neighbors.")
        return next_action

    def is_free_predator(self, predator):
        axial_neighbors = predator + self.axial_neighbors_mask

        valid_index = []
        for idx, position in enumerate(axial_neighbors):
            if (position >= 0).all() and (position < self.fov_scope).all():
                valid_index.append(idx)

        axial_neighbors = axial_neighbors[valid_index, :]

        axial_neighboring_preys = \
            self.local_env_matrix[axial_neighbors[:, 0],
                                  axial_neighbors[:, 1], 0].sum()

        # Valid only if collision is not allowed in the space.
        # Otherwise, more than one agents can occupy the same position.
        yes_no = True
        if axial_neighboring_preys > 0:
            yes_no = False

        return yes_no

    def get_neighbors_moving_directions(self):
        # No previous information.
        if self.last_local_env_matrix is None:
            return [[0, 0]]

        # ##################################################
        last_predators_matrix = self.last_local_env_matrix[:, :, 1]
        padded_last_predators_matrix = \
            np.pad(last_predators_matrix,
                   pad_width=((1, 1), (1, 1)),
                   mode="constant",
                   constant_values=(0, 0))

        predators_positions = \
            np.asarray(np.where(padded_last_predators_matrix > 0)).T
        predators_positions -= self.last_own_moving_direction

        padded_predators_matrix_in_last_frame = \
            np.zeros((self.fov_scope + 2, self.fov_scope + 2), dtype=int)
        padded_predators_matrix_in_last_frame[predators_positions[:, 0],
                                              predators_positions[:, 1]] = 1

        # ##################################################
        current_predators_matrix = self.local_env_matrix[:, :, 1]
        padded_current_predators_matrix = \
            np.pad(current_predators_matrix,
                   pad_width=((1, 1), (1, 1)),
                   mode="constant",
                   constant_values=(0, 0))

        padded_current_predators_positions = \
            np.asarray(np.where(padded_last_predators_matrix > 0)).T

        # ##################################################
        # Assignment problem:
        # assign the predators in the last frame
        # to the predators in the current frame.
        moving_direction_matrix = np.zeros((self.fov_scope, self.fov_scope),
                                           dtype=int)

        n_valid_predator_neighbor_list = []
        for current_predator in padded_current_predators_positions:
            _, n_neighbors = \
                self.get_valid_neighbors(current_predator,
                                         padded_predators_matrix_in_last_frame)
            n_valid_predator_neighbor_list.append(n_neighbors)

        # First assign the predator with least predator neighbors in last frame.
        sorted_idx = np.argsort(n_valid_predator_neighbor_list)
        padded_current_predators_positions = \
            padded_current_predators_positions[sorted_idx]

        moving_direction_list = []
        for current_predator in padded_current_predators_positions:
            valid_neighbors, n_valid_neighbors = \
                self.get_valid_neighbors(current_predator,
                                         padded_predators_matrix_in_last_frame)

            if n_valid_neighbors == 0:
                moving_direction = [0, 0]
            else:
                idx = np.random.choice(n_valid_neighbors, 1)[0]
                predator_in_last_frame = valid_neighbors[idx]
                moving_direction = self.get_offsets(predator_in_last_frame,
                                                    current_predator).tolist()

                # Remove the already assigned predator in the last frame.
                padded_predators_matrix_in_last_frame[
                    predator_in_last_frame[0], predator_in_last_frame[0]] = 0

            # Update.
            moving_direction_list.append(moving_direction)

        # Summary.
        num_of_each_direction = []
        for direction in moving_direction_list:
            n = moving_direction_list.count(direction)
            num_of_each_direction.append(n)

        max_to_min_idx = np.argsort(num_of_each_direction)[::-1]
        sorted_moving_direction_list = \
            np.asarray(moving_direction_list)[max_to_min_idx].tolist()

        return sorted_moving_direction_list

    def get_valid_neighbors(self, position, local_env_matrix):
        axial_neighbors = position + self.axial_neighbors_mask
        valid_idx = []
        for idx, neighbor in enumerate(axial_neighbors):
            if local_env_matrix[neighbor[0], neighbor[1]] > 0:
                valid_idx.append(idx)

        valid_neighbors = axial_neighbors[valid_idx]
        n_valid_neighbors = valid_neighbors.shape[0]

        return valid_neighbors, n_valid_neighbors


def test():
    pass


if __name__ == "__test__":
    test()
