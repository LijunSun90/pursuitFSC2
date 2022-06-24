"""
do_nothing_predator.py
~~~~~~~~~~~~~~~~~~~~~~

AUTHOR: Lijun SUN.
Date: TUE JAN 5 2021.
"""
from lib.agents.matrix_agent import MatrixAgent

# For testing.
from lib.environment.matrix_world import MatrixWorld


class DoNothingPrey(MatrixAgent):
    def __init__(self, env, idx_prey, under_debug=False):
        super(DoNothingPrey, self).__init__(env, idx_prey, under_debug)

    def set_is_prey_or_not(self, true_false=True):
        self.is_prey = true_false

    def policy(self):
        # Do nothing.
        next_action = 0

        return next_action


def test():
    world_rows = 40
    world_columns = 40

    n_prey = 4
    n_predators = 4 * (n_prey + 1)

    env = MatrixWorld(world_rows, world_columns,
                      n_preys=n_prey, n_predators=n_predators)
    env.reset(set_seed=True, seed=0)

    prey = DoNothingPrey(env=env, idx_prey=0, under_debug=False)

    print("DoNothingPrey next action:", prey.get_action())


if __name__ == "__main__":
    test()
