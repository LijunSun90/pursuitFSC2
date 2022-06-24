import numpy as np
import random
import pandas as pd
import argparse
import os
import os.path as osp
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tqdm

from actor_critic.common.models import MLPActorCritic


def parse_args(data_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, nargs='*', default=data_path)

    parser.add_argument('--use_seed', '-s', default=True)
    parser.add_argument('--n_runs', '-nr', type=int, default=100)
    parser.add_argument('--n_steps', '-ns', type=int, default=1000)

    return parser.parse_args()


def main():
    data_path = "./results/sos_Actor-Critic/"
    arg_list = parse_args(data_path)
    all_dataset = get_all_data(arg_list)

    plot_data(data_path, all_dataset, 'action_encoding', 'action_distribution')
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
    sns.relplot(data=data, x=x_label, y=y_label, hue="Model", ci="sd", kind="line")

    plt.plot(range(5), [1 / 5] * 5, 'r--', label="Uniform random")

    plt.xlabel("Action")
    plt.ylabel("Probability")
    plt.xticks(ticks=range(5), labels=['0: Left', '1: Right', '2: Up', '3: Down', '4: Stay'])
    plt.tight_layout(pad=0.5)

    output_file_name = osp.join(data_path, "result_action_distribution_models.png")
    plt.savefig(output_file_name)
    print("Write to file:", output_file_name)


def get_all_data(arg_list):
    all_dataset = []
    folder = arg_list.data_path
    sub_folders = []
    for seed in range(10):
        sub_folders.append("sos_actor_critic_p_s" + str(seed))

    for sub_folder in sub_folders:
        # for folder, sub_folders, files in os.walk(arg_list.data_path):
        # if 'pyt_save' in sub_folders:
        # model_path = osp.join(folder, 'pyt_save/')
        model_path = osp.join(folder, sub_folder, 'pyt_save/')
        model_name = convert_model_format(model_path)
        # training_seed = folder[-1]
        training_seed = sub_folder[-1]
        data = get_data(model_name, training_seed, arg_list)
        all_dataset.append(data)

    # uniform_random_data = pd.DataFrame({'action_encoding': range(5),
    #                                     'action_distribution': [1 / 5] * 5,
    #                                     'testing_seed': np.nan,
    #                                     'Model': ["Uniform random"] * 5})
    # all_dataset.append(uniform_random_data)

    all_dataset = pd.concat(all_dataset, ignore_index=True)
    return all_dataset


def get_data(model_name, training_seed, arg_list):

    obs_dim = 11 * 11 * 3
    act_dim = 5
    ac_model = MLPActorCritic(obs_dim, act_dim, hidden_sizes=[400, 300])
    ac_model.load_state_dict(torch.load(model_name))
    print("Loaded model:", model_name)

    observation = np.zeros((obs_dim,))
    action_encoding = list(range(act_dim))
    seed_list = range(arg_list.n_runs)

    action_distribution_list = []
    for i_run in range(arg_list.n_runs):
        seed = seed_list[i_run]
        if arg_list.use_seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        actions = []
        for i_step in range(arg_list.n_steps):
            a, v, logp = ac_model.step(torch.as_tensor(observation, dtype=torch.float32))
            actions.append(a.item())

        action_distribution = np.zeros(act_dim)
        for idx, action in enumerate(action_encoding):
            action_distribution[idx] = actions.count(action)

        action_distribution /= arg_list.n_steps
        action_distribution = pd.DataFrame({'action_encoding': action_encoding,
                                            'action_distribution': action_distribution,
                                            'testing_seed': [seed] * act_dim,
                                            'Model': [training_seed] * act_dim})
        action_distribution_list.append(action_distribution)

    data = pd.concat(action_distribution_list, ignore_index=True)
    return data


def convert_model_format(model_path):
    # # <class 'spinup.utils.core.MLPActorCritic'>
    saved_models_indexes = [int(x.split('.')[0][5:]) for x in os.listdir(model_path) if len(x) > 8 and 'model' in x]
    last_saved_models_idx = '%d' % max(saved_models_indexes) if len(saved_models_indexes) > 0 else ''
    model_name = osp.join(model_path, 'model' + last_saved_models_idx + '.pt')
    model = torch.load(model_name)
    print("Loaded model:", model_name)

    # Get parameters.
    output_model_name = osp.join(model_path, 'model' + last_saved_models_idx + '.pth')
    torch.save(model.state_dict(), output_model_name)
    print("Save model:", output_model_name)

    return output_model_name


if __name__ == "__main__":
    start_time = time.time()
    main()
    duration = time.time() - start_time
    print("Time {:.3f} s = {:.3f} min.".format(duration, duration / 60))
    print('SUCCESS!')
