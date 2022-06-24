import os
import os.path as osp
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def main():
    folders = [
        "./data/",
    ]

    game_setup_list_1 = []

    # Swarm performance with the different pursuer swarm size and target swarm size.
    # for n_targets in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    for n_targets in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        game_setup = {"x_size": 40, "y_size": 40, "n_pursuers": 4 * n_targets, "n_targets": n_targets}
        # game_setup = {"x_size": 80, "y_size": 80, "n_pursuers": 4 * n_targets, "n_targets": n_targets}
        game_setup_list_1.append(game_setup)

    n_episodes = 0
    n_episodes_with_collisions = 0
    for idx_setup, setup in enumerate(game_setup_list_1):
        label = "x".join([str(setup["x_size"]), str(setup["y_size"])])
        file_name = "_".join(['sop',
                              str(setup["x_size"]), str(setup["y_size"]),
                              str(setup["n_pursuers"]), str(setup["n_targets"]), '.txt'])

        folder = folders[0]
        file_path = osp.join(folder, file_name)
        data = pd.read_table(file_path)

        collisions = data["Collisions"].values
        n_episodes += len(collisions)
        n_episodes_with_collisions += np.count_nonzero(collisions)

    print("# episode with collisions vs. # episode without collisions vs. total episodes vs. collision episode ratio:",
          n_episodes_with_collisions, n_episodes - n_episodes_with_collisions, n_episodes,
          n_episodes_with_collisions / n_episodes)

    # makers = ["<:r", ">:g", "x:b", "o:m", "^:c", "v:y", "s:k"] * 5
    makers = ["<:r", ">:g", "x:b", "o:m", "^:c"] * 6
    # 20, 16
    font_size = 18
    # 7
    font_size_diff = 2

    fig, ax1 = plt.subplots()
    fig.tight_layout()

    ax1 = plt.subplot(311)
    ax1.set_ylabel('Capture rate', fontsize=font_size)
    ax1.tick_params('x', labelbottom=False)
    plt.yticks(fontsize=font_size - font_size_diff)

    ax2 = plt.subplot(312, sharex=ax1)
    ax2.set_ylabel('Episode length', fontsize=font_size)
    ax2.tick_params('x', labelbottom=False)
    plt.yticks(fontsize=font_size - font_size_diff)

    ax3 = plt.subplot(313, sharex=ax1)
    ax3.set_ylabel('Episode collisions', fontsize=font_size)
    plt.yticks(fontsize=font_size - font_size_diff)

    plt.xlabel('No. of targets / No. of pursuers', fontsize=font_size)
    plt.xticks(ticks=range(len(game_setup_list_1)),
               labels=[str(setup["n_targets"]) + "/" + str(setup["n_pursuers"]) for setup in game_setup_list_1],
               fontsize=font_size, rotation=0)
    plt.yticks(fontsize=font_size - font_size_diff)

    capture_rate_avg_list = []
    capture_rate_std_list = []
    episode_length_avg_list = []
    episode_length_std_list = []
    episode_collisions_avg_list = []
    episode_collisions_std_list = []
    episode_time_avg_list = []
    episode_time_std_list = []

    for idx_setup, setup in enumerate(game_setup_list_1):
        label = "x".join([str(setup["x_size"]), str(setup["y_size"])])
        file_name = "_".join(['sop',
                              str(setup["x_size"]), str(setup["y_size"]),
                              str(setup["n_pursuers"]), str(setup["n_targets"]), 'statistical_result.txt'])

        folder = folders[0]
        file_path = osp.join(folder, file_name)
        data = pd.read_table(file_path)

        capture_rate_avg = data["CaptureRateAvg"].values[0]
        capture_rate_std = data["CaptureRateStd"].values[0]
        episode_length_avg = data["EpLenAvg"].values[0]
        episode_length_std = data["EpLenStd"].values[0]
        episode_collisions_avg = data["CollisionsAvg"].values[0]
        episode_collisions_std = data["CollisionsStd"].values[0]
        episode_time_avg = data["Time(s)Avg"].values[0]
        episode_time_std = data["Time(s)Std"].values[0]

        capture_rate_avg_list.append(capture_rate_avg)
        capture_rate_std_list.append(capture_rate_std)
        episode_length_avg_list.append(episode_length_avg)
        episode_length_std_list.append(episode_length_std)
        episode_collisions_avg_list.append(episode_collisions_avg)
        episode_collisions_std_list.append(episode_collisions_std)
        episode_time_avg_list.append(episode_time_avg)
        episode_time_std_list.append(episode_time_std)

    ax1.errorbar(range(len(game_setup_list_1)), capture_rate_avg_list, capture_rate_std_list,
                 fmt='.--', alpha=0.9)
    ax2.errorbar(range(len(game_setup_list_1)), episode_length_avg_list, episode_length_std_list,
                 fmt='.--', alpha=0.9)
    ax3.errorbar(range(len(game_setup_list_1)), episode_collisions_avg_list, episode_collisions_std_list,
                 fmt='.--', alpha=0.9)

    for idx, value in enumerate(capture_rate_avg_list):
        ax1.text(x=idx, y=value, s="{:.2f}".format(value), fontsize=font_size - font_size_diff)
    idx = 0
    for avg, std in zip(capture_rate_avg_list, capture_rate_std_list):
        offset = 0
        if std < 1e-3:
            idx += 1
            continue
        elif std < 1e-1:
            # offset = 0.15
            offset = 0
        ax1.text(x=idx, y=avg - std - offset, s="±{:.3f}".format(std), fontsize=font_size - font_size_diff)
        idx += 1

    for idx, value in enumerate(episode_length_avg_list):
        ax2.text(x=idx, y=value+10, s=value, fontsize=font_size - font_size_diff)
    idx = 0
    for avg, std in zip(episode_length_avg_list, episode_length_std_list):
        if std < 1e-3:
            idx += 1
            continue
        ax2.text(x=idx, y=avg - std, s="±{:.3f}".format(std), fontsize=font_size - font_size_diff)
        idx += 1

    for idx, value in enumerate(episode_collisions_avg_list):
        ax3.text(x=idx, y=value, s=value, fontsize=font_size - font_size_diff)
    idx = 0
    for avg, std in zip(episode_collisions_avg_list, episode_collisions_std_list):
        offset = 0
        if std < 1e-3:
            idx += 1
            continue
        elif std < 5:
            # offset = 5
            offset = 0
        ax3.text(x=idx, y=avg - std - offset, s="±{:.3f}".format(std), fontsize=font_size - font_size_diff)
        idx += 1

    # plt.legend(loc='best', fontsize=font_size - font_size_diff)
    # plt.savefig("result_single_searcher.png", bbox_inches='tight', dpi=100)
    # fig.tight_layout()
    plt.show()
    pass


if __name__ == "__main__":
    main()
    print('DONE!')
