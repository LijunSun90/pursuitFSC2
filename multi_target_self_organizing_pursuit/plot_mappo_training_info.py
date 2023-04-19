import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():

    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_log_folder", type=str, default="data/log_mappo/sop_mappo")
    # parser.add_argument("--data_filename", type=str, default="experiment20230408131414.txt")
    # parser.add_argument("--data_log_folder", type=str, default="data/log_mappo/sop_mappo_global_state")
    # parser.add_argument("--data_filename", type=str, default="experiment20230411152837.txt")
    parser.add_argument("--data_log_folder", type=str, default="data/log_mappo/sop_ippo")
    parser.add_argument("--data_filename", type=str, default="experiment20230416150216.txt")

    return parser.parse_args()


def get_log_data(all_args):

    data_log_folder = os.path.join(all_args.data_log_folder, all_args.data_filename)

    data = pd.read_table(data_log_folder)

    return data


def compute_moving_average(data, window_size=20):
    """
    :param data: (n_data,).
    :param window_size: int.
    :return: (n_data,).

    Example behavior:

    data: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    moving_average: [1 1 1 3 3 3 3 3 3 3]

    data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    moving_average: [ 0  1  2  6  9 12 15 18 21 24]
    """

    data = np.array(data).squeeze()
    n_data = len(data)

    window_size = min(max(1, window_size), n_data)

    cumulative_sum = np.cumsum(data)
    intermediate_value = cumulative_sum[window_size:] - cumulative_sum[:-window_size]
    moving_average = np.hstack((data[:window_size], intermediate_value / window_size))

    return moving_average


def main():
    
    all_args = parse_args()

    data_log_folder = all_args.data_log_folder

    transparent = 0.7

    data = get_log_data(all_args)
    print("data:\n", data)

    epochs = data['epoch']

    plt.figure()
    plt.plot(epochs, data['episode_length'])
    y_moving_average = compute_moving_average(data['episode_length'])
    plt.plot(epochs, y_moving_average, 'r-.', label="Moving average")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Episode length")
    output_figure_path = os.path.join(data_log_folder, "episode_length.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.figure()
    plt.plot(epochs, data['capture_rate'])
    y_moving_average = compute_moving_average(data['capture_rate'])
    plt.plot(epochs, y_moving_average, 'r-.', label="Moving average")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Episode capture rate")
    output_figure_path = os.path.join(data_log_folder, "capture_rate.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.figure()
    plt.plot(epochs, data['episode_return'])
    y_moving_average = compute_moving_average(data['episode_return'])
    plt.plot(epochs, y_moving_average, 'r-.', label="Moving average")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Episode return")
    output_figure_path = os.path.join(data_log_folder, "episode_return.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    truncate_start_epoch = 50
    truncate_end_epoch = 6370
    plt.figure()
    plt.plot(epochs.iloc[truncate_start_epoch:truncate_end_epoch],
             data['episode_return'].iloc[truncate_start_epoch:truncate_end_epoch])
    plt.xlabel("Epoch")
    plt.ylabel("Episode return")
    output_figure_path = os.path.join(data_log_folder, "episode_return_truncated.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.figure()
    plt.plot(epochs, data['episode_n_multiagent_collision_events'], label='Collide agent', alpha=transparent)
    plt.plot(epochs, data['episode_n_collision_with_obstacles'], label='Collide obstacle', alpha=transparent)
    plt.plot(epochs, data['episode_n_collision_with_boundaries'], label='Collide boundaries', alpha=transparent)
    plt.ylim([0, 100])
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Episode collisions")
    output_figure_path = os.path.join(data_log_folder, "episode_collisions.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.figure()
    plt.plot(epochs, data['loss_value'], label='loss_value')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    output_figure_path = os.path.join(data_log_folder, "loss_value.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.figure()
    plt.plot(epochs, data['loss_policy'], label='loss_policy', alpha=transparent)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    output_figure_path = os.path.join(data_log_folder, "loss_policy.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.figure()
    plt.plot(epochs, data['epoch_time_s'])
    plt.xlabel("Epoch")
    plt.ylabel("Epoch time")
    output_figure_path = os.path.join(data_log_folder, "epoch_time.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.show()

    pass


if __name__ == "__main__":
    main()
    print("COMPLETE!")

