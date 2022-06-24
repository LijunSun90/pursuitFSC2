import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import os.path as osp
import numpy as np


def main():
    data_path = "./results/sos_MADDPG/learning_curves_s9/"
    dataset, group_variable = get_data(data_path)

    x_column_name = "Episode"
    y_column_names = ["EpRet", "EpLen", "CaptureRate", "Collisions", "CollisionsWithObstacles", "PLoss", "QLoss", "Time(s)"]

    x_label = "Episode"
    y_labels = ["Average total reward", "Episode length", "Capture rate", "Collisions", "Collisions with obstacles",
                "Policy loss", "Value loss", "Time (s)"]

    for idx, y_column_name in enumerate(y_column_names):
        plot_data(dataset, x_column_name, y_column_name, x_label, y_labels[idx], group_variable=group_variable)
        output_name = osp.join(data_path, 'result_smooth_50_' + y_column_name + '.png')
        plt.savefig(output_name)
        print('Write to file:', output_name)
        pass

    # plt.show()


def plot_data(data, x_column_name, y_column_name, x_label, y_label, group_variable):
    print('Plotting', y_column_name)

    smooth_width = 50
    # Smooth data with moving window average.
    # That is,
    #   smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
    # where the 'smooth' param is width of that window (2k + 1).
    y = np.ones(smooth_width)
    for datum in data:
        x = np.asarray(datum[y_column_name])
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        datum[y_column_name] = smoothed_x

    data = pd.concat(data, ignore_index=True)
    # print(data)
    sns.set_theme(style="darkgrid")
    # "sd" means to draw the standard deviation of the data.
    # Method 1. multiple curves.
    plt.figure()
    sns.lineplot(x=x_column_name, y=y_column_name, hue=group_variable, data=data, ci='sd',
                 palette='tab10', legend='full', alpha=0.5)
    # Method 2.
    # plt.figure()
    # sns.lineplot(x=x_column_name, y=y_column_name, ci='sd', data=data)
    # Method 3. one mean curve with std.
    # sns.relplot(x=x_column_name, y=y_column_name, kind="line", ci="sd", data=data)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout(pad=0.5)


def get_data(data_path):
    dataset = []
    group_variable = "Experiment"
    # group_variable = "Batch size"
    exp_idx = 0
    for folder, sub_folders, files in os.walk(data_path):
        if 'progress.txt' in files:
            exp_data = pd.read_table(osp.join(folder, 'progress.txt'))
            # exp_data = pd.read_csv(osp.join(folder, 'progress.csv'))
            exp_data[['Time(s)']] = exp_data[['Time(s)']].cumsum()

            # batch_size = folder.split('_')[-1]
            # exp_data.insert(len(exp_data.columns), group_variable, batch_size)
            exp_data.insert(len(exp_data.columns), group_variable, exp_idx)

            dataset.append(exp_data)
            print('Experiment ID:', exp_idx, 'Folder:', folder)
            exp_idx += 1

    return dataset, group_variable


if __name__ == "__main__":
    main()
    print('SUCCESS!')
