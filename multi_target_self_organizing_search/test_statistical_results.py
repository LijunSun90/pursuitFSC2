import os
import numpy as np


def main():
    game_setup_list_1 = []

    # Single searcher.
    # game_setup = {"x_size": 20, "y_size": 20, "n_pursuers": 1, "n_targets": 5}
    # game_setup_list_1.append(game_setup)
    # game_setup = {"x_size": 40, "y_size": 40, "n_pursuers": 1, "n_targets": 5}
    # game_setup_list_1.append(game_setup)
    # game_setup = {"x_size": 60, "y_size": 60, "n_pursuers": 1, "n_targets": 5}
    # game_setup_list_1.append(game_setup)
    # game_setup = {"x_size": 80, "y_size": 80, "n_pursuers": 1, "n_targets": 5}
    # game_setup_list_1.append(game_setup)

    # Swarm performance with the same swarm size.
    game_setup = {"x_size": 20, "y_size": 20, "n_pursuers": 8, "n_targets": 50}
    game_setup_list_1.append(game_setup)
    game_setup = {"x_size": 40, "y_size": 40, "n_pursuers": 8, "n_targets": 50}
    game_setup_list_1.append(game_setup)
    game_setup = {"x_size": 60, "y_size": 60, "n_pursuers": 8, "n_targets": 50}
    game_setup_list_1.append(game_setup)
    game_setup = {"x_size": 80, "y_size": 80, "n_pursuers": 8, "n_targets": 50}
    game_setup_list_1.append(game_setup)
    # game_setup = {"x_size": 100, "y_size": 100, "n_pursuers": 8, "n_targets": 50}
    # game_setup_list_1.append(game_setup)

    algorithm_list_1 = [
                        # "test_sos_random_walk.py",
                        # "test_sos_systematic_searcher.py",
                        # "test_sos_apexdqn.py",
                        "test_sos_maddpg.py",
                        # "test_sos_actor_critic.py",
                        ]

    for setup in game_setup_list_1:
        for algorithm in algorithm_list_1:
            command = "".join(["python ", algorithm,
                               " --x_size=", str(setup["x_size"]),
                               " --y_size=", str(setup["y_size"]),
                               " --n_pursuers=", str(setup["n_pursuers"]),
                               " --n_targets=", str(setup["n_targets"])])
            print("Command:", command)
            os.system(command)

    pass


if __name__ == "__main__":
    main()
    print("DONE!")
