import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import os.path as osp
import numpy as np


def main():
    data_path = "./results/sos_Apex-DQN/APEX_pursuit_b945a_00000_0_2022-05-07_16-55-47/"
    dataset, group_variable = get_data(data_path)

    x_column_name = "episodes_total"
    y_column_names = ["episode_reward_mean", "episode_len_mean",
                      "custom_metrics/capture_rate_mean", "custom_metrics/collisions_mean",
                      "custom_metrics/collisions_with_obstacles_mean", "time_total_s"]

    x_label = "Episode"
    y_labels = ["Average total reward", "Episode length", "Capture rate", "Collisions", "Collisions with obstacles",
                "Time (s)"]

    for idx, y_column_name in enumerate(y_column_names):
        plot_data(dataset, x_column_name, y_column_name, x_label, y_labels[idx], group_variable=group_variable)
        output_name = osp.join(data_path, 'result_' + y_labels[idx] + '.png')
        plt.savefig(output_name)
        print('Write to file:', output_name)
        pass

    plt.show()
    pass


def plot_data(data, x_column_name, y_column_name, x_label, y_label, group_variable):
    print('Plotting', y_column_name)
    data = pd.concat(data, ignore_index=True)
    sns.set_theme(style="darkgrid")
    # "sd" means to draw the standard deviation of the data.
    # Method 1.
    plt.figure()
    sns.lineplot(x=x_column_name, y=y_column_name, hue=group_variable, data=data, ci='sd',
                 legend='full', palette='tab10', alpha=0.7)
    # Method 2.
    # plt.figure()
    # sns.lineplot(x=x_column_name, y=y_column_name, ci='sd', data=data)
    # Method 3.
    # sns.relplot(x=x_column_name, y=y_column_name, kind="line", ci="sd", data=data)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout(pad=0.5)


def get_data(data_path):
    n_pursuers = 8
    dataset = []
    group_variable = "Experiment"
    # group_variable = "Batch size"
    exp_idx = 0
    for folder, sub_folders, files in os.walk(data_path):
        if 'progress.txt' in files:
            exp_data = pd.read_table(osp.join(folder, 'progress.txt'))

            exp_data[['episode_reward_mean']] /= n_pursuers
            exp_data[['episode_len_mean']] /= n_pursuers

            # batch_size = folder.split('_')[-1]
            # exp_data.insert(len(exp_data.columns), group_variable, batch_size)
            exp_data.insert(len(exp_data.columns), group_variable, exp_idx)
            dataset.append(exp_data)
            print('Experiment ID:', exp_idx, 'Folder:', folder)
            exp_idx += 1

    return dataset, group_variable


if __name__ == "__main__":
    main()
