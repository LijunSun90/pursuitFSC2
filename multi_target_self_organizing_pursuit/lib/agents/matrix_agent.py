"""
matrix_agent.py
~~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: SAT APR 25 2021.
"""
from abc import ABCMeta, abstractmethod

import copy
import numpy as np
from sklearn.metrics import pairwise_distances


# Base class.
class MatrixAgent(metaclass=ABCMeta):
    def __init__(self, env, idx_agent, under_debug=False):
        self.env = env
        self.idx_agent = idx_agent
        self.under_debug = under_debug

        # Get environment parameters.

        self.fov_scope = self.env.fov_scope
        self.fov_radius = self.env.fov_radius

        # 2d numpy array of shape (4, 2).
        self.axial_neighbors_mask = self.env.axial_neighbors_mask

        # 2d numpy array of shape (8, 2).
        self.two_steps_away_neighbors_mask = \
            self.env.two_steps_away_neighbors_mask

        # Own parameters.
        self.is_prey = False

        self.local_env_matrix = None
        self.local_env_vectors = None

        self.global_own_position = None
        # Relative position in one agent's local view.
        # If fov_scope = 7, own_position = [3, 3].
        self.local_own_position = np.array([self.fov_radius] * 2)
        self.offsets_from_local_positions_to_global_positions = None

        self.memory = dict()

        self.set_is_prey_or_not()
        self.reset()

    @abstractmethod
    def set_is_prey_or_not(self, true_false=False):
        self.is_prey = true_false

    def reset(self):
        # 0. Initialize variables.
        self.memory = dict()

        # 1. Perceive.
        # self.global_own_position, self.local_env_matrix = \
            # self.env.perceive(idx_agent=self.idx_agent, is_prey=self.is_prey,
            #                   remove_current_agent=True)
        self.global_own_position, self.local_env_matrix = \
            self.env.perceive(idx_agent=self.idx_agent, is_evader=self.is_prey,
                              remove_current_agent=True)
        self.offsets_from_local_positions_to_global_positions = \
            self.global_own_position - self.local_own_position

        # 2. Update local env vector representation.
        self.update_local_env_vectors()

        # 3. Update memory.
        self.memory["last_local_position"] = np.zeros((0, 2), dtype=int)

    def get_global_position(self):
        return self.global_own_position.copy()

    def get_local_position(self):
        return self.local_own_position.copy()

    def update_local_env_vectors(self):
        """
        local_env_matrix -> local_env_vectors.
        """
        # Parse.
        local_preys_matrix = self.local_env_matrix[:, :, 0].copy()
        local_predators_matrix = self.local_env_matrix[:, :, 1].copy()
        local_obstacles_matrix = self.local_env_matrix[:, :, 2].copy()

        # Vector representation.
        # 2d numpy array of shape (x, 2) where 0 <= x.
        local_preys = np.asarray(np.where(local_preys_matrix == 1)).T
        local_predators = np.asarray(np.where(local_predators_matrix == 1)).T
        local_obstacles = np.asarray(np.where(local_obstacles_matrix == 1)).T

        # Remove local captured preys.
        for prey in local_preys:
            self.is_captured(prey)

        self.local_env_vectors = {
            "local_preys": local_preys,
            "local_predators": local_predators,
            "local_obstacles": local_obstacles}

    def is_captured(self, prey_position):

        capture_positions = prey_position + self.axial_neighbors_mask

        valid_index = []
        for idx, position in enumerate(capture_positions):
            if (position >= 0).all() and (position < self.fov_scope).all():
                valid_index.append(idx)

        capture_positions = capture_positions[valid_index, :]

        occupied_capture_positions = \
            self.local_env_matrix[capture_positions[:, 0],
                                  capture_positions[:, 1], :].sum(axis=1)

        # Valid only if collision is not allowed in the space.
        # Otherwise, more than one agents can occupy the same position.
        yes_no = True if (occupied_capture_positions > 0).all() and len(capture_positions) == 4 else False

        return yes_no

    def get_action(self):

        # 1. Perceive.
        # self.global_own_position, self.local_env_matrix = \
        #     self.env.perceive(idx_agent=self.idx_agent, is_prey=self.is_prey,
        #                       remove_current_agent=True)
        self.global_own_position, self.local_env_matrix = \
            self.env.perceive(idx_agent=self.idx_agent, is_evader=self.is_prey,
                              remove_current_agent=True)
        self.offsets_from_local_positions_to_global_positions = \
            self.global_own_position - self.local_own_position

        # 2. Update local env vector representation.
        self.update_local_env_vectors()

        # 3. Policy. Customized part.
        next_action = self.policy()

        # 4. Update memory.
        self.memory["last_local_position"] = self.local_own_position - \
            np.asarray(self.env.action_direction[next_action])

        return next_action

    @abstractmethod
    def policy(self):
        next_action = None

        return next_action

    def get_open_axial_neighbors(self, local_position_concerned):
        """
        :param local_position_concerned: 1d numpy with the shape (2,).
        :return: 2d numpy array with the shape (x, 2), where 0 <= x <= 4,
                 depending on how many axial neighbors are still open,
                 i.e., not be occupied.
        """
        neighbors = self.axial_neighbors_mask + local_position_concerned

        open_idx = []
        for idx, neighbor in enumerate(neighbors):
            if not self.is_collide(neighbor):
                open_idx.append(idx)

        open_neighbors = neighbors[open_idx, :]

        return open_neighbors.copy()

    def is_collide(self, new_position):
        """
        Check the whether ``new_position`` collide with others.

        :param new_position: 1d numpy array with the shape (2,).
        :return: boolean, indicates  valid or not.
        """
        pixel_values_in_new_position = \
            self.local_env_matrix[new_position[0], new_position[1], :].sum()

        collide = False
        if pixel_values_in_new_position != 0:
            collide = True

        return collide

    def random_walk_in_single_agent_system_without_memory(self):
        """
        Open space:
            There is no other agents in the neighborhood,
            but only the the agent itself and obstacles such as boundaries.

        Policy:
            Move to a random open one-step away position.
        """

        # 2d numpy array with the shape (x, 2), where 0 <= x <= 4.
        open_neighbors = self.get_open_axial_neighbors(self.local_own_position)

        if open_neighbors.shape[0] == 0:
            next_position = self.local_own_position.copy()
        else:
            n_candidates = len(open_neighbors)
            # Random select one candidate.
            idx = np.random.choice(n_candidates, 1)[0]
            next_position = open_neighbors[idx]

        # Get the action.
        # 1d numpy array with the shape(2,), np.array([delta_x, delta_y]).
        # direction = self.env.get_offsets(self.local_own_position, next_position)
        direction = next_position - self.local_own_position
        next_action = self.env.direction_action[tuple(direction)]

        return next_action

    def random_walk_in_single_agent_system_with_memory(self):
        """
        Open space:
            There is no other agents in the neighborhood,
            but only the the agent itself and obstacles such as boundaries.

        Policy:
            Move to a random open one-step away position that is not
            where I come from in the last time step.
        """

        # 2d numpy array with the shape (x, 2), where 0 <= x <= 4.
        open_neighbors = self.get_open_axial_neighbors(self.local_own_position)

        if open_neighbors.shape[0] == 0:
            next_position = self.local_own_position.copy()
        else:
            candidates = []
            last_local_position = self.memory["last_local_position"]
            for neighbor in open_neighbors:
                # Avoid returning to the position in the last time step.
                if neighbor.tolist() != last_local_position.tolist():
                    candidates.append(neighbor)

            n_candidates = len(candidates)
            if n_candidates == 0:
                # Only one open neighbor, walk from it in the last time step.
                next_position = last_local_position.copy()
            else:
                # Random select one candidate.
                idx = np.random.choice(n_candidates, 1)[0]
                next_position = candidates[idx]

        # Get the action.
        # 1d numpy array with the shape(2,), np.array([delta_x, delta_y]).
        # direction = self.env.get_offsets(self.local_own_position, next_position)
        direction = next_position - self.local_own_position
        next_action = self.env.direction_action[tuple(direction)]

        return next_action

    @staticmethod
    def random_walk_in_multiple_agent_system_with_memory():
        """
        MAS space:
            There are other agents in the neighborhood,
            which cannot be
             ignored even if the current agent only wants
            a random walk.

        Policy:
            Move to a random open one-step away position that is:
            1. not the last position where I came from;
            2. not a risky position that may collide with other agents.
        """
        next_action = 0

        return next_action

    def repel_by_neighborhood(self):
        """
        Repulsive potential is determined by:
            1. local closet obstacles,
            2. local closet predators.
        """

        # Parameters.
        epsilon = 1e-5
        max_repulsive_force = 1 / epsilon

        # Neighborhood info.
        # 2D numpy array of shape (n, 2) where n >= 0.
        local_predators = self.local_env_vectors["local_predators"]
        # 2D numpy array of shape (n, 2) where n >= 0.
        local_obstacles = self.local_env_vectors["local_obstacles"]

        # Open one-step away positions.
        # 2D numpy array with the shape (x, 2), where 0 <= x <= 4.
        open_neighbor_positions = \
            self.get_open_axial_neighbors(self.local_own_position)

        # No open neighboring positions.
        n_positions = open_neighbor_positions.shape[0]
        if n_positions == 0:
            next_position = self.local_own_position.copy()
        else:
            repulsive_force_by_local_closet_obstacle = np.zeros((n_positions,))
            repulsive_force_by_local_closet_predators = np.zeros((n_positions,))

            for idx, position in enumerate(open_neighbor_positions):
                current_position = position.reshape(1, -1)

                # 1. Repulsive by from local closet obstacles.
                if local_obstacles.shape[0] > 0:
                    distance_with_local_obstacles = \
                        pairwise_distances(current_position,
                                           local_obstacles,
                                           metric='manhattan')
                    # > 0.
                    nearest_distance_with_local_obstacles = \
                        np.min(distance_with_local_obstacles)

                    repulsive_force_by_local_closet_obstacle[idx] = \
                        1 / max(nearest_distance_with_local_obstacles, epsilon)

                # 2. Repulsive by from local closet predators.
                if local_predators.shape[0] > 0:
                    distance_with_local_predators = \
                        pairwise_distances(current_position,
                                           local_predators,
                                           metric='manhattan')
                    # > 0.
                    nearest_distance_with_local_predators = \
                        np.min(distance_with_local_predators)

                    # Force = inf, when d == 0 or d == 1.
                    repulsive_force_by_local_closet_predators[idx] = 1 /\
                        max(nearest_distance_with_local_predators - 1,
                            epsilon)

            repulsive_force = repulsive_force_by_local_closet_obstacle + \
                repulsive_force_by_local_closet_predators

            if (repulsive_force == max_repulsive_force).all():
                next_position = self.local_own_position.copy()
            else:
                # Move to the position with the least repulsive force.
                idx_least_repulsive_force = np.argmin(repulsive_force)
                next_position = \
                    open_neighbor_positions[idx_least_repulsive_force, :]

        # Get the action.
        # 1d numpy array with the shape(2,), np.array([delta_x, delta_y]).
        # direction = self.env.get_offsets(self.local_own_position,
        #                                  next_position)
        direction = next_position - self.local_own_position
        next_action = self.env.direction_action[tuple(direction)]

        return next_action


def test():
    pass


if __name__ == "__main__":
    test()
