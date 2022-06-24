"""
basic_matrix_agent.py
~~~~~~~~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: WED JUN 2 2021.
"""
import copy
import numpy as np


class BasicMatrixAgent:
    # Encoding.
    # Clockwise.
    # 0     1  2  3  4  5   6   7   8
    # still N  E  S  W  NE  SE  SW  NW
    action_direction = {0: (0, 0),
                        1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1),
                        5: (-1, 1), 6: (1, 1), 7: (1, -1), 8: (-1, -1)}
    direction_action = \
        dict([(value, key) for key, value in action_direction.items()])

    # N, E, S, W.
    axial_neighbors_mask = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

    def __init__(self):

        pass

    def get_offsets(self, from_position, to_position):
        """
        :param from_position: 1d numpy array with the shape(2,).
        :param to_position: 1d numpy array with the shape(2,).
        :return: 1d numpy array with the shape(2,),
                 np.array([delta_x, delta_y]).
        """
        from_x, from_y = from_position
        to_x, to_y = to_position

        delta_x = self.get_an_offset(from_x, to_x)
        delta_y = self.get_an_offset(from_y, to_y)

        delta = np.array([delta_x, delta_y])

        return delta.copy()

    @staticmethod
    def get_an_offset(from_x, to_x):
        """
        :param from_x: int, 0 <= from_x < self.world_width or self.world_height
        :param to_x: int, 0 <= to_x < self.world_width or self.world_height
        :return:
        """
        delta_x = to_x - from_x
        return delta_x.copy()

    def get_open_axial_neighbors(self, local_position_concerned,
                                 local_env_matrix):
        """
        :param local_position_concerned: 1d numpy with the shape (2,).
        :param local_env_matrix: 3d numpy array with the shape (m, n, 3).
        :return: 2d numpy array with the shape (x, 2), where 0 <= x <= 4,
                 depending on how many axial neighbors are still open,
                 i.e., not be occupied.
        """
        neighbors = self.axial_neighbors_mask + local_position_concerned

        open_idx = []
        for idx, neighbor in enumerate(neighbors):
            if not self.is_collide(neighbor, local_env_matrix):
                open_idx.append(idx)

        open_neighbors = neighbors[open_idx, :]

        return open_neighbors.copy()

    @staticmethod
    def is_collide(current_position, local_env_matrix):
        pixel_values_in_new_position = \
            local_env_matrix[current_position[0], current_position[1]].sum()

        collide = False
        if pixel_values_in_new_position != 0:
            collide = True

        return collide

    def random_walk_in_single_agent_system_without_memory(self,
                                                          local_own_position,
                                                          local_env_matrix):
        """
        Open space:
            There is no other agents in the neighborhood,
            but only the the agent itself and obstacles such as boundaries.

        Policy:
            Move to a random open one-step away position.
        """

        # 2d numpy array with the shape (x, 2), where 0 <= x <= 4.
        open_neighbors = self.get_open_axial_neighbors(local_own_position,
                                                       local_env_matrix)

        if open_neighbors.shape[0] == 0:
            next_position = local_own_position.copy()
        else:
            n_candidates = len(open_neighbors)
            # Random select one candidate.
            idx = np.random.choice(n_candidates, 1)[0]
            next_position = open_neighbors[idx]

        # Get the action.
        # 1d numpy array with the shape(2,), np.array([delta_x, delta_y]).
        direction = self.get_offsets(local_own_position, next_position)
        next_action = self.direction_action[tuple(direction)]

        return next_action

    def random_walk_in_multiple_agent_system_without_memory(self,
                                                            local_own_position,
                                                            local_env_matrix,
                                                            agents):
        """
        :param local_own_position
        :param local_env_matrix
        :param agents: a list

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
        # 2d numpy array with the shape (x, 2), where 0 <= x <= 4.
        open_neighbors = self.get_open_axial_neighbors(local_own_position,
                                                       local_env_matrix)

        valid_idx = []
        for idx, neighbor in enumerate(open_neighbors):
            dangerous = False
            neighboring_positions = self.axial_neighbors_mask + neighbor
            for position in neighboring_positions:
                if position.tolist() in agents:
                    dangerous = True

            if not dangerous:
                valid_idx.append(idx)

        open_neighbors = open_neighbors[valid_idx]

        if open_neighbors.shape[0] == 0:
            next_position = local_own_position.copy()
        else:
            n_candidates = len(open_neighbors)
            # Random select one candidate.
            idx = np.random.choice(n_candidates, 1)[0]
            next_position = open_neighbors[idx]

        # Get the action.
        # 1d numpy array with the shape(2,), np.array([delta_x, delta_y]).
        direction = self.get_offsets(local_own_position, next_position)
        next_action = self.direction_action[tuple(direction)]

        return next_action


def test():
    pass


if __name__ == "__main__":
    test()
