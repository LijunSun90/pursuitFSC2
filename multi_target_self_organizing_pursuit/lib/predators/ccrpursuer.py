"""
ccrpursuer.py
~~~~~~~~~~~~~

CCRPursuer: parallel cooperative co-evolutionary pursuer.

Improve the PCCPSO-R in the paper:

[1] Lijun Sun, Chao Lyu, Yuhui Shi and Chin-Teng Lin,
"Multiple-preys pursuit based on biquadratic assignment problem,"
2021 IEEE Congress on Evolutionary Computation (CEC), 2021, pp. ?-?, doi: ?.

AUTHOR: Lijun SUN.
Date: MON 10 MAY, 2021.
"""
import copy
import numpy as np

from lib.fitness.fitness_single_prey_pursuit import FitnessSinglePreyPursuit


class CCRPursuer:

    # Encoding.
    # Clockwise.
    # 0     1  2  3  4  5   6   7   8
    # still N  E  S  W  NE  SE  SW  NW
    action_direction = {0: (0, 0),
                        1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1),
                        5: (-1, 1), 6: (1, 1), 7: (1, -1), 8: (-1, -1)}
    direction_action = \
        dict([(value, key) for key, value in action_direction.items()])

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

    def __init__(self, fov_scope, under_debug=False):

        self.fov_scope = fov_scope
        self.under_debug = under_debug

        self.fov_radius = int((self.fov_scope - 1) / 2)

        # 2D numpy array, such as np.array([fov_radius] * 2)
        self.own_position = np.array([self.fov_radius] * 2)

        # N, E, S, W.
        self.axial_neighbors_mask = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        self.axial_neighbors = self.axial_neighbors_mask + self.own_position

        self.fitness_computer = FitnessSinglePreyPursuit(self.fov_scope,
                                                         self.under_debug)

        # ##################################################
        # Algorithm parameters.

        # PSO parameters.
        self.w = 1
        self.c_1 = 2
        self.c_2 = 2

        # General parameters.
        # Include 1 current predator and 4 virtual robots which locates
        # at the 4 one-step away axial neighboring positions.
        self.population_size = 5

        # ##################################################
        # Runtime variables initialization.

        # Virtual robots are only deployed in the 4 one-step away
        # axial neighboring positions since only these positions can allow
        # the safe parallel decision making in the cooperative coevolutionary
        # framework.
        self.virtual_robots = self.axial_neighbors

        self.population = np.zeros((self.population_size, 2), dtype=int)
        self.population[0, :] = self.own_position.copy()
        self.population[1:, :] = self.virtual_robots

        # Global best one in the population.
        self.position_pg = np.zeros(2)

        self.fitness = np.zeros(self.population_size)
        # Fitness value of the global best individual in the subpopulation.
        self.fitness_pg = np.inf

        # Sharing variables between functions.
        self.local_env_matrix = None
        self.local_env_vectors = None

        self.cluster_center = None
        self.cluster_other_members = None

        self.last_local_position = None

    def get_action(self, local_env_matrix, local_env_vectors,
                   cluster_center, cluster_other_members,
                   last_local_position):

        # ##################################################
        # Sharing variables between functions.

        self.local_env_matrix = local_env_matrix
        self.local_env_vectors = local_env_vectors
        self.cluster_center = cluster_center
        self.cluster_other_members = cluster_other_members
        self.last_local_position = last_local_position

        # ##################################################
        # CCRPursuer

        # 0. Check whether the real predator is in the capturing position.
        exist_axial_neighboring_preys = \
            self.fitness_computer.exist_axial_neighbors(
                self.own_position, self.local_env_matrix[:, :, 0])

        if exist_axial_neighboring_preys:
            next_position = self.own_position
        else:
            # The first individual in the population is always the real
            # predator's position.
            self.population[0, :] = self.own_position

            # 1. Update the current fitness values due to the past changes.
            # Re-evaluate the fitness value of these robots in the
            # subpopulation.
            self.evaluate_subpopulation()

            # 2. Update the real robot in the subpopulation.
            self.update_real_robot()

            next_position = self.population[0, :].copy()

        ##################################################
        # Get the action.
        # 1d numpy array with the shape(2,), np.array([delta_x, delta_y]).
        direction = self.get_offsets(self.own_position, next_position)
        next_action = self.direction_action[tuple(direction)]

        return next_action

    def evaluate_subpopulation(self):
        """
        Rewrite the code in
        https://github.com/LijunSun90/pursuitCCPSOR/blob/master/lib/fitness_subpopulation_restart.m

        Modify the parameters:
        self.fitness, self.fitness_pi, self.fitness_pg,
                      self.position_pi, self.position_pg
        """

        # 1. Evaluate the fitness.
        for idx in range(0, self.population_size):

            current_individual = self.population[idx, :]

            self.fitness[idx] = self.evaluate(current_individual)
        # 2.
        # Initialize global best individual (particle) and get the fitness_pg
        self.fitness_pg = np.min(self.fitness)
        # Random select one if there are more than one minimum values.
        all_idx_min = np.where(self.fitness == self.fitness_pg)[0]
        idx_min = np.random.choice(all_idx_min)
        self.position_pg = self.population[idx_min, :].copy()

    def evaluate(self, current_position):
        fitness = self.fitness_computer.compute(
                current_position=current_position,
                cluster_center=self.cluster_center,
                cluster_other_predators=self.cluster_other_members,
                local_env_vectors=self.local_env_vectors,
                local_env_matrix=self.local_env_matrix)

        return fitness

    def update_real_robot(self):
        """
        Modify the parameters:
        self.population, self.position_pi, self.position_pg,
        self.fitness, self.fitness_pi, self.fitness_pg
        """

        # Generate the new generation.
        velocity_new = self.position_pg - self.population[0, :]
        position_new = self.population[0, :] + velocity_new

        # Fitness evaluation.
        fitness_p_new = self.evaluate(position_new)
        # Generation update.
        if fitness_p_new < self.fitness[0]:

            self.fitness[0] = fitness_p_new
            self.population[0, :] = position_new.copy()

            # Update the global best.
            if fitness_p_new <= self.fitness_pg:
                self.fitness_pg = fitness_p_new
                self.position_pg = position_new.copy()


def test():
    pass


if __name__ == "__main__":
    test()
