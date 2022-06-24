import os
import numpy as np


def main():
    game_setup_list_1 = []
    # Single-target pursuit.
    game_setup = {"x_size": 6, "y_size": 6, "n_pursuers": 4, "n_targets": 1}
    game_setup_list_1.append(game_setup)

    for setup in game_setup_list_1:
        command = "".join(["python run_sop.py ",
                           " --x_size=", str(setup["x_size"]),
                           " --y_size=", str(setup["y_size"]),
                           " --n_pursuers=", str(setup["n_pursuers"]),
                           " --n_targets=", str(setup["n_targets"]),
                           " --n_episodes=", str(100)])
        print("Command:", command)
        os.system(command)

    pass


if __name__ == "__main__":
    main()
    print("DONE!")
