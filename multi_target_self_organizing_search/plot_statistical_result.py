import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import os.path as osp
import numpy as np


def main():
    data_path = "./results/"
    # Modify this line according to the name of data path.
    methods = ["MADDPG", "Apex-DQN", "Actor-Critic"]

    # x_label = "Training iteration"
    x_label = "Episode"
    y_labels = ["Average episode reward", "Episode length", "Capture rate", "Collisions", "Collisions with obstacles",
                "Time (s)"]

    all_dataset = get_all_data(data_path, methods)

    for idx_label, y_label in enumerate(y_labels):
        # plt.figure()
        dataset = all_dataset[[x_label, y_label, "Experiment", "Method"]]
        plot_data(data_path, dataset, x_label, y_label)

    plt.show()


def plot_data(data_path, data, x_label, y_label):
    # print(data)
    sns.set_theme(style="darkgrid")
    # "sd" means to draw the standard deviation of the data.
    # Method 1. multiple curves.
    # plt.figure()
    # sns.lineplot(x=x_column_name, y=y_column_name, hue=group_variable, data=data, ci='sd')
    # Method 2.
    # plt.figure()
    # sns.lineplot(x=x_column_name, y=y_column_name, ci='sd', data=data)
    # Method 3. one mean curve with std.
    # sns.relplot(x=x_label, y=y_label, kind="line", ci="sd", data=data, hue="Method")
    face_grid = sns.relplot(data=data, x=x_label, y=y_label, hue="Method", ci="sd", kind="line", alpha=0.7)
    old_legend = face_grid.legend
    old_legend.remove()
    # if y_label is "Average episode reward" or "Capture rate":
    #     sns.move_legend(face_grid, loc='lower right')
    # elif y_label is "Episode length" or "Collisions" or "Collisions with obstacles" or "Time (s)":
    #     sns.move_legend(face_grid, loc='upper right')

    font_size = 16
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel(x_label, fontsize=font_size)
    if y_label == "Capture rate":
        plt.ylabel("Search rate", fontsize=font_size)
    else:
        plt.ylabel(y_label, fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.tight_layout(pad=0.5)

    output_file_name = osp.join(data_path, "result_" + y_label.replace(' ', '_') + ".png")
    plt.tight_layout()
    plt.savefig(output_file_name)
    print("Write to file:", output_file_name)


def get_all_data(data_path, methods):
    all_dataset = []
    for method in methods:
        sub_data_path = osp.join(data_path, 'sos_' + method + '/')
        print('Searching', sub_data_path)
        all_dataset.append(get_data(sub_data_path, method))

    all_dataset = pd.concat(all_dataset, ignore_index=True)
    # print(all_dataset)

    return all_dataset


def get_data(data_path, method):
    # x_label = "Training iteration"
    x_label = "Episode"
    y_labels = ["Average episode reward", "Episode length", "Capture rate", "Collisions", "Collisions with obstacles",
                "Time (s)"]

    n_pursuers = 8
    dataset = []
    exp_idx = 0
    for folder, sub_folders, files in os.walk(data_path):
        if 'progress.txt' in files:
            print("Found", osp.join(folder, 'progress.txt'))
            exp_data = pd.read_table(osp.join(folder, 'progress.txt'))

            if method == "Actor-Critic" or method == "Actor-Critic-P":
                x_column_name = "Epoch"
                y_column_names = ["AverageEpRet", "EpLen", "AverageCaptureRate", "AverageCollisions",
                                  "AverageCollideObstacles", "Time"]
            elif method == "Apex-DQN":
                x_column_name = "episodes_total"
                y_column_names = ["episode_reward_mean", "episode_len_mean",
                                  "custom_metrics/capture_rate_mean", "custom_metrics/collisions_mean",
                                  "custom_metrics/collisions_with_obstacles_mean", "time_total_s"]
            elif method == "MADDPG":
                x_column_name = "Episode"
                y_column_names = ["EpRet", "EpLen", "CaptureRate", "Collisions", "CollisionsWithObstacles", "Time(s)"]
            else:
                print("ERROR")
                return

            column_names = [x_column_name] + y_column_names

            if method == "Actor-Critic" or method == "Actor-Critic-P":
                n_processes = exp_data[['Processes']].iloc[0].values[0]

            exp_data = exp_data[column_names]

            exp_data.rename(columns={x_column_name: x_label}, inplace=True)
            for idx, old_column_name in enumerate(y_column_names):
                exp_data.rename(columns={old_column_name: y_labels[idx]}, inplace=True)

            if method == "Actor-Critic" or method == "Actor-Critic-P":
                exp_data[['Episode']] *= n_processes

            if method == "Apex-DQN":
                exp_data[['Average episode reward']] /= n_pursuers
                exp_data[['Episode length']] /= n_pursuers

            if method == "MADDPG":
                # exp_data[['Training iteration']] /= 6
                exp_data[['Time (s)']] = exp_data[['Time (s)']].cumsum()
                # exp_data = exp_data.iloc[0:100]
                exp_data = exp_data.iloc[0::6]

            exp_data.insert(len(exp_data.columns), "Experiment", exp_idx)
            exp_data.insert(len(exp_data.columns), "Method", method)
            dataset.append(exp_data)
            exp_idx += 1

    dataset = pd.concat(dataset, ignore_index=True)
    # print(dataset)

    return dataset


if __name__ == "__main__":
    main()
    print('SUCCESS!')
