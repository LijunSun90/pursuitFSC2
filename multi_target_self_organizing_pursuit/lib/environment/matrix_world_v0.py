"""
matrix_world.py
~~~~~~~~~~~~~~~~~~~

AUTHOR: LIJUN SUN.
DATE: MON 27 APR 2020.
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Modified: TUE 17 NOV 2020.
1. Modify the fov scope as an odd number.
2. Delete the communication code.
Modified: THU 7 JAN 2021.
1. Add the control parameter to control the display of the frame title.
Modified: SAT APR 25 2021.
1. Add and modify some comments.
2. Change the part of variable name from "prey" to "preys".
"""
import os
import copy
import numpy as np
import shutil

import matplotlib.pyplot as plt


class MatrixWorld:
    # Encoding.
    # Clockwise.
    # 0     1  2  3  4  5   6   7   8
    # still N  E  S  W  NE  SE  SW  NW
    action_direction = {0: (0, 0),
                        1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1),
                        5: (-1, 1), 6: (1, 1), 7: (1, -1), 8: (-1, -1)}
    direction_action = \
        dict([(value, key) for key, value in action_direction.items()])
    actions_orthogonal = list(range(5))
    actions_diagonal = list(range(9))

    @classmethod
    def create_padded_env_matrix_from_vectors(cls, world_rows, world_columns,
                                              fov_scope, preys, predators,
                                              obstacles):
        """
        :param world_rows: int.
        :param world_columns: int.
        :param fov_scope: int. An odd number.
        :param preys: 2d numpy array of shape (x, 2) where 0 <= x.
        :param predators: 2d numpy array of shape (x, 2) where 0 <= x.
        :param obstacles: 2d numpy array of shape (x, 2) where 0 <= x.
        :return: 3d numpy array of shape (world_rows + fov_scope - 1,
                                          world_columns + fov_scope - 1, 4)
            channel 0: the preys matrix,
            channel 1: the predators matrix,
            channel 2: is the obstacles matrix.
            channel 3: unknown map.
            In a channel, the pixel value is 1 in an agent's location, else 0.
        """
        # Parameters.
        fov_radius = int(0.5 * (fov_scope - 1))
        fov_offsets_in_padded = np.array([fov_radius] * 2)
        fov_pad_width = np.array([fov_radius] * 2)
        # [lower_bound, upper_bound).
        fov_mask_in_padded = \
            np.array([[-fov_radius] * 2, [fov_radius + 1] * 2]) + \
            fov_offsets_in_padded

        # Create matrix.
        padded_env_matrix = np.zeros((world_rows + fov_scope - 1,
                                      world_columns + fov_scope - 1, 4),
                                     dtype=int)

        for channel in [0, 1, 2, 3]:
            if channel == 2:
                # Obstacles matrix are padded with 1, borders are obstacles.
                padded_value = 1
            else:
                padded_value = 0

            if channel == 3:
                # Unknown map matrix is initially all 1s.
                env_matrix_channel = np.ones((world_rows, world_columns),
                                             dtype=int)
            else:
                env_matrix_channel = np.zeros((world_rows, world_columns),
                                              dtype=int)

            padded_env_matrix[:, :, channel] = \
                np.pad(env_matrix_channel,
                       pad_width=((fov_pad_width[0], fov_pad_width[1]),
                                  (fov_pad_width[0], fov_pad_width[1])),
                       mode="constant",
                       constant_values=(padded_value, padded_value))

        # Write data.
        positions_in_padded = preys + fov_offsets_in_padded
        padded_env_matrix[positions_in_padded[:, 0],
                          positions_in_padded[:, 1], 0] = 1

        positions_in_padded = predators + fov_offsets_in_padded
        padded_env_matrix[positions_in_padded[:, 0],
                          positions_in_padded[:, 1], 1] = 1

        positions_in_padded = obstacles + fov_offsets_in_padded
        padded_env_matrix[positions_in_padded[:, 0],
                          positions_in_padded[:, 1], 2] = 1

        padded_env_matrix[:, :, 3] = \
            cls.update_env_matrix_unknown_map(fov_mask_in_padded,
                                              padded_env_matrix[:, :, 3],
                                              predators)

        return copy.deepcopy(padded_env_matrix)

    @classmethod
    def update_env_matrix_unknown_map(cls, fov_mask_in_padded,
                                      padded_env_matrix_unknown_map, predators):
        """
        :param fov_mask_in_padded: 2d numpy array of shape (2, 2),
            which is [[row_min, column_min], [row_max, column_max]].
        :param padded_env_matrix_unknown_map: 2d numpy array of shape
            (world_rows + fov_scope - 1, world_columns + fov_scope - 1).
        :param predators: 2d numpy array of shape (x, 2) or
            1d numpy array of shape (2,).
        :return: 2d numpy array of the same shape of
            `padded_env_matrix_unknown_map`.

        Mark the local perceptible scope of a predator as known region.
        """
        # 1d to 2d array.
        if len(predators.shape) == 1:
            predators = predators.reshape((1, -1))

        for predator in predators:
            fov_idx = predator + fov_mask_in_padded
            padded_env_matrix_unknown_map[fov_idx[0, 0]: fov_idx[1, 0],
                                          fov_idx[0, 1]: fov_idx[1, 1]] = 0

        return copy.deepcopy(padded_env_matrix_unknown_map)

    @classmethod
    def get_inf_norm_distance(cls, from_position, to_position):
        """
        :param from_position: 1d numpy array with the shape(2,).
        :param to_position: 1d numpy array with the shape(2,).
        :return: int, the inf-norm.
        """
        delta = cls.get_offsets(from_position, to_position)

        distance = np.linalg.norm(delta, ord=np.inf).astype(int)

        return distance.copy()

    @classmethod
    def get_distance(cls, from_position, to_position):
        """
        :param from_position: 1d numpy array with the shape(2,).
        :param to_position: 1d numpy array with the shape(2,).
        :return: int, the 1-norm.

        Manhattan distance or City distance.
        """
        delta = cls.get_offsets(from_position, to_position)

        distance = np.linalg.norm(delta, ord=1).astype(int)

        return distance.copy()

    @classmethod
    def get_offsets(cls, from_position, to_position):
        """
        :param from_position: 1d numpy array with the shape(2,).
        :param to_position: 1d numpy array with the shape(2,).
        :return: 1d numpy array with the shape(2,),
                 np.array([delta_x, delta_y]).
        """
        from_x, from_y = from_position
        to_x, to_y = to_position

        delta_x = cls.get_an_offset(from_x, to_x)
        delta_y = cls.get_an_offset(from_y, to_y)

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

    @classmethod
    def render_a_matrix(cls, env_matrix, is_display=True,
                        is_fixed_size=False,
                        grid_on=True, tick_labels_on=False):
        # 1. Render.

        ax_image = plt.imshow(env_matrix, origin="upper")

        # 2. Set figure parameters.

        # add the colorbar using the figure's method,
        # telling which mappable we're talking about and
        # which axes object it should be near
        ax_image.figure.colorbar(ax_image, ax=ax_image.axes)

        ax_image.figure.set_frameon(True)
        if is_fixed_size:
            ax_image.figure.set_figwidth(10)
            ax_image.figure.set_figheight(10)

        # 3. Set axes parameters.

        x_ticks = np.arange(-0.5, env_matrix.shape[1], 1)
        y_ticks = np.arange(-0.5, env_matrix.shape[0], 1)

        ax_image.axes.set_xticks(x_ticks)
        ax_image.axes.set_yticks(y_ticks)
        if not tick_labels_on:
            ax_image.axes.set_xticklabels([])
            ax_image.axes.set_yticklabels([])
        else:
            ax_image.axes.set_xticklabels(x_ticks, rotation=90)

        ax_image.axes.margins(0, 0)

        ax_image.axes.tick_params(which='both', direction='in',
                                  left=False, bottom=False,
                                  right=False, top=False)
        ax_image.axes.grid(grid_on)

        # 4. Control the display.

        if is_display:
            # plt.pause(interval)
            plt.show()

        # 5.
        plt.close()

    def __init__(self,
                 world_rows, world_columns,
                 n_preys=1, n_predators=4,
                 fov_scope=11,
                 obstacle_density=0,
                 save_path="./data/frames/"
                 ):
        """
        :param world_rows: int, corresponds to the 1st axis.
        :param world_columns: int, corresponds to the 2nd axis.
        :param n_preys: int, >= 0.
        :param n_predators: int, >= 0.
        :param fov_scope: int, >=1, an odd integer.
            The scope of the field of view of agents.
            The agent locates in the center of its own local field of view.
        :param obstacle_density: float.
        :param save_path: string.
        """

        # Set parameters.

        self.world_rows = world_rows
        self.world_columns = world_columns
        self.world_scope = np.array([self.world_rows, self.world_columns])

        self.n_preys = n_preys
        self.n_predators = n_predators

        # FOV parameters.

        self.fov_scope = fov_scope
        self.fov_radius = int(0.5 * (self.fov_scope - 1))

        self.fov_offsets_in_padded = np.array([self.fov_radius] * 2)
        self.fov_pad_width = np.array([self.fov_radius] * 2)

        # [lower_bound, upper_bound).
        self.fov_mask_in_padded = \
            np.array([[-self.fov_radius] * 2, [self.fov_radius + 1] * 2]) +\
            self.fov_offsets_in_padded

        self.fov_global_scope_in_padded = \
            np.array([[0, 0], [self.world_rows, self.world_columns]]) +\
            self.fov_offsets_in_padded

        # Neighbors.

        # N, E, S, W.
        self.axial_neighbors_mask = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        self.two_steps_away_neighbors_mask = \
            np.array([[-2, 0], [0, 2], [2, 0], [0, -2],
                      [-1, 1], [1, 1], [1, -1], [-1, -1]])

        self.obstacle_density = obstacle_density

        self.save_path = save_path

        # Processed variables.

        self.n_cells = self.world_rows * self.world_columns
        self.n_obstacles = round(self.n_cells * self.obstacle_density)

        # Get coordinates of the whole world.
        # 0, 1, ..., (world_rows - 1); 0, 1, ..., (world_columns - 1).
        # For example,
        # array([[[0, 0, 0],
        #         [1, 1, 1],
        #         [2, 2, 2]],
        #        [[0, 1, 2],
        #         [0, 1, 2],
        #         [0, 1, 2]]])
        self.meshgrid_x, self.meshgrid_y = \
            np.mgrid[0:self.world_rows, 0:self.world_columns]

        # Example of meshgrid[0].flatten() is
        # array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        # Example of meshgrid[1].flatten() is
        # array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.xs, self.ys = self.meshgrid_x.flatten(), self.meshgrid_y.flatten()

        # Rendering parameters.
        self.title_template = 'Step = %d'

        self.x_ticks = np.arange(-0.5, self.world_columns, 1)
        self.y_ticks = np.arange(-0.5, self.world_rows, 1)

        # Saving parameters.
        self.frame_prefix = "MatrixWorld"
        self.frame_no = 0

        # Indicate whether the game is over.
        self.done = False

        # 2-D numpy array,
        # where each row is a 2-D point in the global coordinate system.
        # Shape: (n_preys, 2)
        self.preys = None
        # Shape: (n_predators, 2)
        self.predators = None
        # Shape: (None, 2)
        self.obstacles = None

        self.env_matrix = None
        self.padded_env_matrix = None

    def set_frame_prefix(self, frame_prefix):
        """
        :param frame_prefix: str

        Modify the parameter self.frame_prefix.
        """
        self.frame_prefix = frame_prefix

    def reset(self, set_seed=False, seed=0):
        if set_seed:
            np.random.seed(seed)

        # 0. Reset parameters.
        self.frame_no = 0

        # 1. Create obstacles.
        empty_cells_index = np.arange(self.n_cells).tolist()

        self.obstacles, empty_cells_index = \
            self.random_select(empty_cells_index.copy(), self.n_obstacles)

        # 2. Create preys.
        self.preys, empty_cells_index = \
            self.random_select(empty_cells_index.copy(), self.n_preys)

        # 3. Create predators.
        self.predators, empty_cells_index = \
            self.random_select(empty_cells_index.copy(), self.n_predators)

        # Matrix representation.
        # shape: (world_rows, world_columns, 4)
        # channel 0: the preys matrix,
        # channel 1: the predators matrix,
        # channel 2: is the obstacles matrix.
        # channel 3: unknown map.
        # In each channel, the pixel value is 1 in an agent's location, else 0.
        self.padded_env_matrix = self.create_padded_env_matrix_from_vectors(
            self.world_rows, self.world_columns, self.fov_scope,
            self.preys, self.predators, self.obstacles)

        self.env_matrix = self.padded_env_matrix[
                          self.fov_global_scope_in_padded[0, 0]:
                          self.fov_global_scope_in_padded[1, 0],
                          self.fov_global_scope_in_padded[0, 1]:
                          self.fov_global_scope_in_padded[1, 1], :]

        # Restore the random.
        # if set_seed:
        #     np.random.seed()

    def random_select(self, empty_cells_index, n_select):
        """
        Random select ``n_select`` cells out of the total cells
        ``empty_cells_index``.

        :param empty_cells_index: a list of integers where each integer
                                  corresponds to some kind of index.
        :param n_select: int, >=0.
        :return: (entities, empty_cells_index), where ``entities`` is a
                 (n_select, 2) numpy array, and ``empty_cells_index`` is a
                 (n - n_select, 2) numpy array.
        """
        # Indexes.
        idx_entities = np.random.choice(empty_cells_index, n_select,
                                        replace=False)

        # Maintain the left empty cells.
        for idx in idx_entities:
            empty_cells_index.remove(idx)

        # Coordinates.
        # Indexes in 2D space.
        #       |       |     |
        # 0 -idx:0--idx:1--idx:2-
        # 1 -idx:3--idx:4--idx:5-
        # 2 -idx:6--idx:7--idx:8-
        #       |       |     |
        #       0       1     2
        xs_entities, ys_entities = self.xs[idx_entities], self.ys[idx_entities]

        # Get the entities positions.
        entities = np.vstack((xs_entities, ys_entities)).T

        return copy.deepcopy(entities), copy.deepcopy(empty_cells_index)

    def update_a_prey(self, idx_prey, new_position):
        """
        :param idx_prey: int, >=0.
        :param new_position: 1d numpy array of shape (2,).
        :return: None.
        """
        # 1. Update vector.
        old_position = self.get_a_prey(idx_prey)
        self.preys[idx_prey, :] = new_position

        # 2. Update the channel.
        # self.env_matrix[[old_position[0], new_position[0]],
        #                 [old_position[1], new_position[1]], 0] += [-1, 1]

        old_position_in_padded = old_position + self.fov_offsets_in_padded
        new_position_in_padded = new_position + self.fov_offsets_in_padded

        self.padded_env_matrix[old_position_in_padded[0],
                               old_position_in_padded[1], 0] -= 1
        self.padded_env_matrix[new_position_in_padded[0],
                               new_position_in_padded[1], 0] += 1

    def update_a_predator(self, idx_predator, new_position):
        """
        :param idx_predator: int, >=0.
        :param new_position: 1d numpy array of shape (2,).
        :return: None.
        """
        # 1. Update vector.
        old_position = self.get_a_predator(idx_predator)
        self.predators[idx_predator, :] = new_position

        # 2. Update unknown map.
        fov_idx = self.fov_mask_in_padded + new_position
        self.padded_env_matrix[fov_idx[0, 0]: fov_idx[1, 0],
                               fov_idx[0, 1]: fov_idx[1, 1], 3] = 0

        # 3. Update predator channel.
        # self.env_matrix[[old_position[0], new_position[0]],
        #                 [old_position[1], new_position[1]], 1] += [-1, 1]

        old_position_in_padded = old_position + self.fov_offsets_in_padded
        new_position_in_padded = new_position + self.fov_offsets_in_padded

        self.padded_env_matrix[old_position_in_padded[0],
                               old_position_in_padded[1], 1] -= 1
        self.padded_env_matrix[new_position_in_padded[0],
                               new_position_in_padded[1], 1] += 1

    def perceive(self, idx_agent, is_prey=False, remove_current_agent=True):
        """
        :param idx_agent: int, >= 0.
        :param is_prey: boolean.
        :param remove_current_agent:
        :return: a tuple, (own_position, local_env_matrix),
            "own_position" is 1d numpy array of shape (2,).
            "local_matrix" is 3d numpy array of shape
                (self.fov_scope, self.fov_scope, 3)
                with each channel being
                (local_prey, local_predators, local_obstacles).
        """
        if is_prey:
            # 1d numpy array with the shape (2,).
            center_position = self.get_a_prey(idx_agent)
        else:
            # 1d numpy array with the shape (2,).
            center_position = self.get_a_predator(idx_agent)

        fov_idx = self.fov_mask_in_padded + center_position

        local_preys = \
            self.padded_env_matrix[fov_idx[0, 0]: fov_idx[1, 0],
                                   fov_idx[0, 1]: fov_idx[1, 1], 0].copy()
        local_predators = \
            self.padded_env_matrix[fov_idx[0, 0]: fov_idx[1, 0],
                                   fov_idx[0, 1]: fov_idx[1, 1], 1].copy()
        local_obstacles = \
            self.padded_env_matrix[fov_idx[0, 0]: fov_idx[1, 0],
                                   fov_idx[0, 1]: fov_idx[1, 1], 2].copy()

        if remove_current_agent:
            if is_prey:
                local_preys[self.fov_offsets_in_padded[0],
                            self.fov_offsets_in_padded[1]] -= 1
            else:
                local_predators[self.fov_offsets_in_padded[0],
                                self.fov_offsets_in_padded[1]] -= 1

        local_env_matrix = \
            np.stack((local_preys, local_predators, local_obstacles), axis=2)

        return center_position.copy(), local_env_matrix.copy()

    def perceive_globally(self, idx_agent, is_prey=False):
        """
        :param idx_agent: int, >= 0.
        :param is_prey: boolean.
        :return: a dict.
        """
        if is_prey:
            # 1d numpy array with the shape (2,).
            center_position = self.get_a_prey(idx_agent)
        else:
            # 1d numpy array with the shape (2,).
            center_position = self.get_a_predator(idx_agent)

        env_vectors = dict()
        env_vectors["own_position"] = center_position
        env_vectors["all_preys"] = self.get_all_preys()
        env_vectors["all_predators"] = self.get_all_predators()
        env_vectors["found_obstacles"] = self.get_all_obstacles()

        return copy.deepcopy(env_vectors)

    def get_a_prey(self, idx_prey=0):
        """
        :param idx_prey:
        :return: 1d numpy array with the shape (2,).
        """
        return self.preys[idx_prey, :].copy()

    def get_a_predator(self, idx_predator):
        """
        :param idx_predator:
        :return: 1d numpy array with the shape (2,).
        """
        return self.predators[idx_predator, :].copy()

    def get_all_preys(self):
        return self.preys.copy()

    def get_all_predators(self):
        return self.predators.copy()

    def get_all_obstacles(self):
        return self.obstacles.copy()

    def act(self, idx_agent, action, is_prey=False):
        """
        :param idx_agent: index of a predator or a prey.
        :param action: int, 0 ~ 5 or 0 ~ 9 depending on ``self.move_diagonal``.
        :param is_prey: if False, move the predator;
                        if True, move the prey.
        :return: a tuple, (executable, collide) where both are boolean,
            indicate whether the action is executable or not, and
            indicate whether there is a collision.
            Change the position of the ``idx_agent`` if it is valid.
        """
        if not is_prey:
            # Shape: (2, )
            from_position = self.get_a_predator(idx_agent)
        else:
            from_position = self.get_a_prey(idx_agent)

        to_position = self.move_to(from_position, action)

        # Check validation.
        # Include the execution status of keeping still.
        if to_position.tolist() == from_position.tolist():
            collide = False
            return collide

        collide = self.is_collide(to_position)

        # Change the position.
        if is_prey:
            self.update_a_prey(idx_agent, to_position)
        else:
            self.update_a_predator(idx_agent, to_position)

        return collide

    def move_to(self, from_position, action):
        """
        :param from_position: 1d numpy array with the shape: (2, ).
        :param action: int, 0 ~ 5 or 0 ~ 9 depending on ``self.move_diagonal``.
        :return: 1d numpy array with the shape: (2, ).

        The position if the ``action`` is performed, regardless of its
        validation.
        """
        direction = self.action_direction[action]
        to_position = from_position + direction

        return to_position.copy()

    def is_collide(self, new_position):
        """
        Check the whether ``new_position`` collide with others in the global
        scope.

        ``new_position`` is valid
        if it additionally does not locate out the grid world boundaries.
        If it move out of the boundaries, it can also been seen that the agent
        collides with the boundaries, and so also a kind of collision.

        :param new_position: 1d numpy array with the shape (2,).
        :return: boolean, indicates  valid or not.
        """
        if (new_position < [0, 0]).any() or \
                (new_position >= [self.world_rows, self.world_columns]).any():
            collide = True
            return collide

        new_position = new_position + self.fov_offsets_in_padded
        pixel_values_in_new_position = \
            self.padded_env_matrix[new_position[0], new_position[1], :-1].sum()

        collide = False
        if pixel_values_in_new_position != 0:
            collide = True

        return collide

    def is_all_captured(self):
        all_preys = self.get_all_preys()

        n_captured = 0
        # yes_no = True
        for prey in all_preys:
            capture_positions = \
                self.axial_neighbors_mask + prey + self.fov_offsets_in_padded

            occupied_capture_positions = \
                self.padded_env_matrix[capture_positions[:, 0],
                                       capture_positions[:, 1], :-1].sum(axis=1)

            # Valid only if collision is not allowed in the space.
            # Otherwise, more than one agents can occupy the same position.
            n_captured += 1 if (occupied_capture_positions > 0).all() else 0
            # if occupied_capture_positions != 4:
            #     yes_no = False
            #     break

        capture_rate = n_captured / len(all_preys)

        # return yes_no
        return capture_rate == 1, capture_rate

    def get_scope_mask_in_env_matrix(self, circle_center, radius):
        """
        :param circle_center: 1d numpy array of shape (2,).
        :param radius: int, > 0.
        :return: 2d numpy array of shape (2, 2),
            which is [[row_min, column_min], [row_max, column_max]]
        """
        scope_mask = np.array([[-radius] * 2, [radius] * 2]) + circle_center

        index_min = np.maximum([0, 0], scope_mask[0, :])
        index_max = np.minimum([self.world_rows, self.world_columns],
                               scope_mask[1, :])

        scope_mask = np.vstack((index_min, index_max))

        return copy.deepcopy(scope_mask)

    def render(self, is_display=True, interval=0.001,
               is_save=False, is_fixed_size=False,
               grid_on=True, tick_labels_on=False,
               show_predator_idx=False,
               show_prey_idx=False,
               show_frame_title=True,
               use_input_env_matrix=False,
               env_matrix=None):

        # 0. Prepare the directory.

        # if self.frame_no == 0 and is_save:
        #     self.create_directory()

        # White: 255, 255, 255.
        # Black: 0, 0, 0.
        # Yellow, 255, 255, 0.
        # Silver: 192, 192, 192

        # Background: white.
        # Prey: red. Predator: blue. Obstacle: black. Unknown regions: yellow.

        #               R    G    B
        # Background: 255, 255, 255
        # Prey:       255,   0,   0
        # Predator:     0,   0, 255
        # Obstacle:     0,   0,   0
        # Unknown:    255, 255,   0
        # Pursuer fov:255, 255,   0

        # 1. Prepare data.

        if not use_input_env_matrix:
            env_matrix = self.env_matrix.copy()

        # White world.
        rgb_env = np.ones((env_matrix.shape[0], env_matrix.shape[1], 3))

        # Fov scope of pursuers: green, 255, 255, 0.
        # 0 -> 1, 0 -> 1.
        # rgb_env[:, :, [0, 2]] = \
        #     np.logical_and(rgb_env[:, :, [0, 2]],
        #                    env_matrix[:, :, 3].reshape(env_matrix.shape[0], env_matrix.shape[1], 1))

        # for i_row in range(env_matrix.shape[0]):
        #     for j_column in range(env_matrix.shape[1]):
        #         if env_matrix[i_row, j_column, 0] == 1:
        #             rgb_env[i_row, j_column, :] = [1, 0, 0]
        #         if env_matrix[i_row, j_column, 1] == 1:
        #             rgb_env[i_row, j_column, :] = [0, 0, 1]
        #         if env_matrix[i_row, j_column, 2] == 1:
        #             rgb_env[i_row, j_column, :] = [0, 0, 0]

        # Prey: red, 255, 0, 0.
        # 1 -> 0.
        rgb_env[:, :, [1, 2]] = \
            np.logical_xor(rgb_env[:, :, [1, 2]],
                           env_matrix[:, :, 0].reshape(env_matrix.shape[0],
                                                       env_matrix.shape[1],
                                                       1))
        # Predator: blue, 0, 0, 255.
        # 1 -> 0.
        rgb_env[:, :, [0, 1]] = \
            np.logical_xor(rgb_env[:, :, [0, 1]],
                           env_matrix[:, :, 1].reshape(env_matrix.shape[0],
                                                       env_matrix.shape[1],
                                                       1))

        # Obstacle: black, 0, 0, 0.
        # 1 -> 0.
        rgb_env = \
            np.logical_xor(rgb_env,
                           env_matrix[:, :, 2].reshape(env_matrix.shape[0],
                                                       env_matrix.shape[1],
                                                       1))
        # Unknown map: green, 255, 255, 0.
        # 1 -> 0, 0 -> 0.
        # rgb_env[:, :, 2] = \
        #     np.logical_and(rgb_env[:, :, 2],
        #                    1 - env_matrix[:, :, 3])

        # Fov scope of pursuers: green, 255, 255, 0.
        # 0 -> 1, 0 -> 1.
        # rgb_env[:, :, 2] = \
        #     np.logical_and(rgb_env[:, :, 2],
        #                    env_matrix[:, :, 3])

        rgb_env = rgb_env * 255

        # 2. Render.

        ax_image = plt.imshow(rgb_env, origin="upper")

        if show_predator_idx:
            for idx, predator in enumerate(self.get_all_predators()):
                text = plt.text(predator[1], predator[0], str(idx))

        if show_prey_idx:
            for idx, prey in enumerate(self.get_all_preys()):
                text = plt.text(prey[1], prey[0], str(idx))

        # 3. Set figure parameters.

        ax_image.figure.set_frameon(True)
        if is_fixed_size:
            ax_image.figure.set_figwidth(10)
            ax_image.figure.set_figheight(10)

        # 4. Set axes parameters.

        ax_image.axes.set_xticks(self.x_ticks)
        ax_image.axes.set_yticks(self.y_ticks)
        if not tick_labels_on:
            ax_image.axes.set_xticklabels([])
            ax_image.axes.set_yticklabels([])
        else:
            ax_image.axes.set_xticklabels(self.x_ticks, rotation=90)

        ax_image.axes.margins(0, 0)

        ax_image.axes.tick_params(which='both', direction='in',
                                  left=False, bottom=False,
                                  right=False, top=False)
        ax_image.axes.grid(grid_on)

        # 5. Set title.
        if show_frame_title:
            plt.title(self.title_template % self.frame_no)

        # 6. Control the display.

        if is_display:
            plt.pause(interval)
            # plt.show()
            # pass

        if is_save:
            plt.savefig(self.save_path + self.frame_prefix + "{0:0=4d}".
                        format(self.frame_no), bbox_inches='tight')

            # plt.imsave(self.save_path + self.frame_prefix + "{0:0=4d}".
            #            format(self.frame_no) + ".png",
            #            arr=rgb_env, format="png")

        # 7. Update.

        self.frame_no += 1

        # 8.
        plt.close()

    def create_directory(self):
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.makedirs(self.save_path)


def test_perceive(env, idx_predator):
    # 0, 2, 6
    local_perception = env.perceive(idx_agent=idx_predator)
    print("Local perception of predator", idx_predator, ":")
    print("Own position :", local_perception[0])
    print("Prey :\n", local_perception[1][:, :, 0])
    print("Predator :\n", local_perception[1][:, :, 1])
    print("Obstacle :\n", local_perception[1][:, :, 2])


def test_act(env):
    env.act(idx_agent=12, action=2)


def test():
    print("Testing ...")

    world_rows = 40
    world_columns = 40

    n_preys = 4
    # n_predators = 4 * (n_preys + 1)
    n_predators = 4 * n_preys

    env = MatrixWorld(world_rows, world_columns,
                      n_preys=n_preys, n_predators=n_predators)

    env.reset(set_seed=True, seed=3)

    print("Step 0...")
    env.render(is_display=True, interval=0.5,
               is_save=True, is_fixed_size=False,
               grid_on=True, tick_labels_on=False,
               show_predator_idx=True,
               show_prey_idx=True,
               show_frame_title=False)

    # test_perceive(env, idx_predator=12)

    # print("Step 1...")
    # test_act(env)
    #
    # env.render(is_display=True, interval=0.5,
    #            is_save=True, is_fixed_size=False,
    #            grid_on=True, tick_labels_on=True,
    #            show_predator_idx=True,
    #            show_prey_idx=True)
    #
    # test_perceive(env, idx_predator=12)


if __name__ == "__main__":
    test()
