import os
import os.path as osp
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def main():
    folders = [
               "./results/sos_random_walk/",
               "./results/sos_systematic_searcher/",
               # "./results/sos_MADDPG/policy_s8/checkpoint60000/",
               "./results/sos_Apex-DQN/APEX_pursuit_2af28_00000_0_2022-04-30_20-13-23/",
               # "./results/sos_Actor-Critic/sos_actor_critic_s5/",
               # "./results/sos_Actor-Critic-P/sos_actor_critic_p_s1/",
               "./results/sos_Actor-Critic/sos_actor_critic_p_s3/"
               ]
    algorithms = [
                  'Random_walk',
                  "Systematic",
                  # "MADDPG",
                  "Apex-DQN",
                  "Actor-Critic",
                  # "Actor-Critic-P"
                  ]

    game_setup_list_1 = []

    # Single searcher.
    game_setup = {"x_size": 20, "y_size": 20, "n_pursuers": 1, "n_targets": 5}
    game_setup_list_1.append(game_setup)
    game_setup = {"x_size": 40, "y_size": 40, "n_pursuers": 1, "n_targets": 5}
    game_setup_list_1.append(game_setup)
    game_setup = {"x_size": 60, "y_size": 60, "n_pursuers": 1, "n_targets": 5}
    game_setup_list_1.append(game_setup)
    game_setup = {"x_size": 80, "y_size": 80, "n_pursuers": 1, "n_targets": 5}
    game_setup_list_1.append(game_setup)

    # Swarm performance with the same swarm size.
    # game_setup = {"x_size": 20, "y_size": 20, "n_pursuers": 8, "n_targets": 50}
    # game_setup_list_1.append(game_setup)
    # game_setup = {"x_size": 40, "y_size": 40, "n_pursuers": 8, "n_targets": 50}
    # game_setup_list_1.append(game_setup)
    # game_setup = {"x_size": 60, "y_size": 60, "n_pursuers": 8, "n_targets": 50}
    # game_setup_list_1.append(game_setup)
    # game_setup = {"x_size": 80, "y_size": 80, "n_pursuers": 8, "n_targets": 50}
    # game_setup_list_1.append(game_setup)
    # game_setup = {"x_size": 100, "y_size": 100, "n_pursuers": 8, "n_targets": 50}
    # game_setup_list_1.append(game_setup)

    makers = ["<:r", ">:g", "x:b", "o:m", '^:k']
    # 20, 16
    font_size = 12
    # 7
    font_size_diff = 2

    fig, ax1 = plt.subplots()

    ax1 = plt.subplot(311)
    ax1.set_ylabel('Episode reward', fontsize=font_size)
    ax1.tick_params('x', labelbottom=False)
    plt.yticks(fontsize=font_size-font_size_diff)

    ax2 = plt.subplot(312, sharex=ax1)
    ax2.set_ylabel('Search rate', fontsize=font_size)
    ax2.tick_params('x', labelbottom=False)
    plt.yticks(fontsize=font_size-font_size_diff)

    ax3 = plt.subplot(313, sharex=ax1)
    ax3.set_ylabel('Episode length', fontsize=font_size)
    plt.xticks(ticks=range(len(algorithms)), labels=algorithms, fontsize=font_size, rotation=15)
    plt.yticks(fontsize=font_size-font_size_diff)

    for idx_setup, setup in enumerate(game_setup_list_1):
        label = "x".join([str(setup["x_size"]), str(setup["y_size"])])
        file_name = "_".join(['progress_test_parallel',
                             str(setup["x_size"]), str(setup["y_size"]),
                             str(setup["n_pursuers"]), str(setup["n_targets"]), '.txt'])

        episode_reward_avg_list = []
        episode_reward_std_list = []
        capture_rate_avg_list = []
        capture_rate_std_list = []
        episode_length_avg_list = []
        episode_length_std_list = []
        
        for idx_algorithm, folder in enumerate(folders):
            file_path = osp.join(folder, file_name)
            data = pd.read_table(file_path)
            n_pursuers = setup["n_pursuers"]

            episode_reward_avg = data["AverageEpRet"].values[0]
            episode_reward_std = data["StdEpRet"].values[0]
            capture_rate_avg = data["AverageCaptureRate"].values[0]
            capture_rate_std = data["StdCaptureRate"].values[0]
            episode_length_avg = data["AverageEpLen"].values[0]
            episode_length_std = data["StdEpLen"].values[0]

            episode_reward_avg_list.append(episode_reward_avg)
            episode_reward_std_list.append(episode_reward_std)
            capture_rate_avg_list.append(capture_rate_avg)
            capture_rate_std_list.append(capture_rate_std)
            episode_length_avg_list.append(episode_length_avg)
            episode_length_std_list.append(episode_length_std)

        ax1.errorbar(range(len(algorithms)), episode_reward_avg_list, episode_reward_std_list,
                     fmt=makers[idx_setup], label=label, alpha=0.9)
        ax2.errorbar(range(len(algorithms)), capture_rate_avg_list, capture_rate_std_list,
                     fmt=makers[idx_setup], label=label, alpha=0.9)
        ax3.errorbar(range(len(algorithms)), episode_length_avg_list, episode_length_std_list,
                     fmt=makers[idx_setup], label=label, alpha=0.9)

    plt.legend(loc='best', fontsize=font_size-font_size_diff)
    # plt.savefig("result_single_searcher.png", bbox_inches='tight', dpi=100)
    fig.tight_layout()
    plt.show()
    pass


if __name__ == "__main__":
    main()
    print('DONE!')
