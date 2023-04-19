import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_log_folder_FSC2", type=str,
                        default="./data/")

    parser.add_argument("--data_log_folder_ActorCritic", type=str,
                        default="./data/log_actor_critic/sop_actor_critic/")

    parser.add_argument("--data_log_folder_MAPPO", type=str,
                        default="./data/log_mappo/sop_mappo/")

    parser.add_argument("--data_log_folder_IPPO", type=str,
                        default="./data/log_mappo/sop_ippo/")

    return parser.parse_args()


def plot_errorbar(ax, data_mean, data_std, label, marker=".--", color='b'):

    x = range(len(data_mean))

    ax.errorbar(x=x, y=data_mean, yerr=data_std,
                fmt=marker, color=color, alpha=0.9, label=label)

    ax.fill_between(x, data_mean - data_std, data_mean + data_std,
                    color=color, alpha=0.15)

    # Text.

    font_size = 10
    y_axis_offset = 0.3 * (np.max(data_mean) - np.min(data_mean))

    for idx, (avg, std) in enumerate(zip(data_mean, data_std)):

        if avg < 1:

            y_axis_offset = 0.15 if avg > 0.5 else 1

        ax.text(x=idx, y=avg, s="{:.3f}".format(avg), fontsize=font_size)

        if std > 1e-3:

            ax.text(x=idx, y=avg + y_axis_offset, s="±{:.3f}".format(std), fontsize=font_size)

    pass


def plot_errorbar_simplified(ax, data_mean, data_std, label, marker, color):

    x = range(len(data_mean))

    ax.plot(x, data_mean, alpha=0.9, label=label, marker=marker, color=color)

    ax.fill_between(x, data_mean - data_std, data_mean + data_std,
                    color=color, alpha=0.15)

    pass


def get_game_setup_list():

    game_setup_list = []

    # Swarm performance with the different pursuer swarm size and target swarm size.
    # for n_targets in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    for n_targets in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        game_setup = {"x_size": 40, "y_size": 40, "n_pursuers": 4 * n_targets, "n_targets": n_targets}
        # game_setup = {"x_size": 80, "y_size": 80, "n_pursuers": 4 * n_targets, "n_targets": n_targets}
        game_setup_list.append(game_setup)

    return game_setup_list


def create_figure_axes(game_setup_list):

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
    plt.xticks(ticks=range(len(game_setup_list)),
               labels=[str(setup["n_targets"]) + "/" + str(setup["n_pursuers"]) for setup in game_setup_list],
               fontsize=font_size, rotation=0)
    plt.yticks(fontsize=font_size - font_size_diff)

    ax2.set_ylim([0, 510])
    ax3.set_ylim([-0.2, 10])

    return ax1, ax2, ax3


def create_figure_ax_collision(game_setup_list):

    # 20, 16
    font_size = 18

    # 7
    font_size_diff = 2

    fig, ax = plt.subplots()
    fig.tight_layout()

    ax.set_ylabel('Episode collisions', fontsize=font_size)

    plt.xlabel('No. of targets / No. of pursuers', fontsize=font_size)

    plt.xticks(ticks=range(len(game_setup_list)),
               labels=[str(setup["n_targets"]) + "/" + str(setup["n_pursuers"]) for setup in game_setup_list],
               fontsize=font_size, rotation=0)
    plt.yticks(fontsize=font_size - font_size_diff)

    # Set font size of scientific notation.

    ax.yaxis.get_offset_text().set_fontsize(font_size - font_size_diff)
    # ax.yaxis.offsetText.set_fontsize(24)

    ax.set_ylim([0, 1500])

    return ax


def get_data(game_setup_list, data_log_folder):

    label = data_log_folder.split("/")[-2].split("_")[-1]

    if label == "data":

        label = "FSC2"

    elif label == "critic":

        label = "Actor-Critic"

    else:
        # MAPPO, IPPO.
        label = label.upper()

    capture_rate_avg_list = []
    capture_rate_std_list = []
    episode_length_avg_list = []
    episode_length_std_list = []
    episode_collisions_avg_list = []
    episode_collisions_std_list = []

    for idx_setup, setup in enumerate(game_setup_list):

        file_name = "_".join(['sop',
                              str(setup["x_size"]), str(setup["y_size"]),
                              str(setup["n_pursuers"]), str(setup["n_targets"]), 'statistical_result.txt'])

        file_path = os.path.join(data_log_folder, file_name)

        data = pd.read_table(file_path)

        capture_rate_avg = data["CaptureRateAvg"].values[0]
        capture_rate_std = data["CaptureRateStd"].values[0]
        episode_length_avg = data["EpLenAvg"].values[0]
        episode_length_std = data["EpLenStd"].values[0]
        episode_collisions_avg = data["CollisionsAvg"].values[0]
        episode_collisions_std = data["CollisionsStd"].values[0]

        capture_rate_avg_list.append(capture_rate_avg)
        capture_rate_std_list.append(capture_rate_std)
        episode_length_avg_list.append(episode_length_avg)
        episode_length_std_list.append(episode_length_std)
        episode_collisions_avg_list.append(episode_collisions_avg)
        episode_collisions_std_list.append(episode_collisions_std)

    capture_rate_avg_list = np.array(capture_rate_avg_list)
    capture_rate_std_list = np.array(capture_rate_std_list)
    episode_length_avg_list = np.array(episode_length_avg_list)
    episode_length_std_list = np.array(episode_length_std_list)
    episode_collisions_avg_list = np.array(episode_collisions_avg_list)
    episode_collisions_std_list = np.array(episode_collisions_std_list)

    return capture_rate_avg_list, capture_rate_std_list, \
        episode_length_avg_list, episode_length_std_list, \
        episode_collisions_avg_list, episode_collisions_std_list, label


def plot_data(ax1, ax2, ax3, game_setup_list, data_log_folder, marker, color, simplified=False):

    capture_rate_avg_list, capture_rate_std_list, \
        episode_length_avg_list, episode_length_std_list, \
        episode_collisions_avg_list, episode_collisions_std_list, label = \
        get_data(game_setup_list, data_log_folder)

    if simplified:
        plot_errorbar_simplified(ax1, capture_rate_avg_list, capture_rate_std_list, label, marker, color)
        plot_errorbar_simplified(ax2, episode_length_avg_list, episode_length_std_list, label, marker, color)
        plot_errorbar_simplified(ax3, episode_collisions_avg_list, episode_collisions_std_list, label, marker, color)
    else:
        plot_errorbar(ax1, capture_rate_avg_list, capture_rate_std_list, label, marker, color)
        plot_errorbar(ax2, episode_length_avg_list, episode_length_std_list, label, marker, color)
        plot_errorbar(ax3, episode_collisions_avg_list, episode_collisions_std_list, label, marker, color)

    plt.legend()

    pass


def plot_data_collision(ax, game_setup_list, data_log_folder, marker, color):

    capture_rate_avg_list, capture_rate_std_list, \
        episode_length_avg_list, episode_length_std_list, \
        episode_collisions_avg_list, episode_collisions_std_list, label = \
        get_data(game_setup_list, data_log_folder)

    plot_errorbar_simplified(ax, episode_collisions_avg_list, episode_collisions_std_list, label, marker, color)

    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    plt.legend()

    pass


def main():

    all_args = parse_args()

    game_setup_list = get_game_setup_list()

    #

    ax1, ax2, ax3 = create_figure_axes(game_setup_list)

    plot_data(ax1, ax2, ax3, game_setup_list, all_args.data_log_folder_FSC2, marker=".--", color='b', simplified=False)

    plot_data(ax1, ax2, ax3, game_setup_list, all_args.data_log_folder_ActorCritic, marker="x", color='g',
              simplified=True)

    plot_data(ax1, ax2, ax3, game_setup_list, all_args.data_log_folder_MAPPO, marker="s", color='r', simplified=True)

    plot_data(ax1, ax2, ax3, game_setup_list, all_args.data_log_folder_IPPO, marker="*", color='C9', simplified=True)

    #

    ax = create_figure_ax_collision(game_setup_list)

    plot_data_collision(ax, game_setup_list, all_args.data_log_folder_ActorCritic, marker="x", color='g')

    plot_data_collision(ax, game_setup_list, all_args.data_log_folder_MAPPO, marker="s", color='r')

    plot_data_collision(ax, game_setup_list, all_args.data_log_folder_IPPO, marker="*", color='C9')

    plt.show()

    pass


if __name__ == "__main__":
    main()
    print("COMPLETE!")

