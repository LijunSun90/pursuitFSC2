"""
random_prey.py
~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: SAT APR 25 2021.
~~~~~~~~~~~~~~~~~~~~~~
"""
import copy
import numpy as np

from lib.agents.matrix_agent import MatrixAgent

# For testing.
from lib.environment.matrix_world import MatrixWorld


class RandomPrey(MatrixAgent):
    def __init__(self, env, idx_prey, under_debug=False):
        super(RandomPrey, self).__init__(env, idx_prey, under_debug)

    def set_is_prey_or_not(self, true_false=True):
        self.is_prey = true_false

    def policy(self):

        # Here, since we don't evolve the preys' policies,
        # in simulation, they observe and make decisions one-by-one,
        # which act as if they have considered each other
        # in the preys' parallel decision making.
        # So, with the above assumptions, an open space no memory random walk
        # is safe here.
        next_action = self.random_walk_in_single_agent_system_without_memory()

        return next_action


def test():
    world_rows = 40
    world_columns = 40

    n_prey = 4
    n_predators = 4 * (n_prey + 1)

    env = MatrixWorld(world_rows, world_columns,
                      n_preys=n_prey, n_predators=n_predators)
    env.reset(set_seed=True, seed=0)

    # 0, 16, 5, 2
    idx_prey = 0
    prey = RandomPrey(env, idx_prey)

    print("is_prey:", prey.is_prey)

    print("Step 0...")
    env.render(is_display=True, interval=0.5,
               is_save=True, is_fixed_size=False,
               grid_on=True, tick_labels_on=True,
               show_predator_idx=True,
               show_prey_idx=True)

    print("Step 1...")
    action = prey.get_action()
    env.act(idx_agent=0, action=action, is_prey=prey.is_prey)
    print("Action:", action)

    env.render(is_display=True, interval=0.5,
               is_save=True, is_fixed_size=False,
               grid_on=True, tick_labels_on=True,
               show_predator_idx=True,
               show_prey_idx=True)


if __name__ == "__main__":
    test()
