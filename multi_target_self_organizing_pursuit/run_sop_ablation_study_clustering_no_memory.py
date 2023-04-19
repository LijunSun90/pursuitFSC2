"""
Author: Lijun Sun.
Date: Thu 5 May, 2022.
"""
import os
import time
import numpy as np
import argparse
import torch
from datetime import datetime

from lib.environment.matrix_world import MatrixWorld

from lib.preys.do_nothing_prey import DoNothingPrey
from lib.preys.random_prey import RandomPrey

from lib.predators.self_organized_predator_ablation_study_clustering_no_memory \
    import SelfOrganizedPredatorClusteringNoMemory as SelfOrganizedPredator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_seed', '-s', default=True)

    parser.add_argument('--n_episodes', '-n', type=int, default=100)
    parser.add_argument('--x_size', '-xs', type=int, default=40)
    parser.add_argument('--y_size', '-ys', type=int, default=40)
    parser.add_argument('--n_pursuers', '-np', type=int, default=4*4)
    parser.add_argument('--n_targets', '-nt', type=int, default=4)
    parser.add_argument('--episode_len', '-l', type=int, default=500)

    parser.add_argument('--render', '-r', default=False)
    parser.add_argument('--save_frames', '-sf', default=False)

    data_log_folder = "data/ablation_study/clustering_no_memory"
    os.makedirs(data_log_folder, exist_ok=True)
    parser.add_argument('--data_log_folder', type=str, default=data_log_folder)

    return parser.parse_args()


def run_policy(arg_list):
    # ##################################################
    # Initialization.
    logger_output_fname = "_".join([arg_list.data_log_folder + '/sop',
                                    str(arg_list.x_size), str(arg_list.y_size),
                                    str(arg_list.n_pursuers), str(arg_list.n_targets), '.txt'])
    logger_output_fname_2 = "_".join([arg_list.data_log_folder + '/sop',
                                      str(arg_list.x_size), str(arg_list.y_size),
                                      str(arg_list.n_pursuers), str(arg_list.n_targets), 'statistical_result.txt'])
    log_rows = [["Episode", "EpLen", "CaptureRate", "Collisions", "Time(s)"]]

    env = create_environment(arg_list.x_size, arg_list.y_size, arg_list.n_targets, arg_list.n_pursuers)

    # Random seed.
    seed_list = np.arange(arg_list.n_episodes + 1)
    seed = seed_list[0]
    if arg_list.use_seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        # env.reset(set_seed=True, seed=seed)
        env.reset(seed=seed)

    # env.reset()
    preys = initialize_prey_swarm(env)
    predators = initialize_predator_swarm(env)

    i_episode, ep_len, ep_collisions = 0, 0, 0
    ep_len_list, ep_collisions_list, ep_capture_rate_list, ep_time_list = [], [], [], []
    start_time = time.time()

    ##################################################
    # Run.
    while i_episode < arg_list.n_episodes:
        if ep_len == 0 and (arg_list.render or arg_list.save_frames):
            render_frame(env, arg_list.render, arg_list.save_frames)

        # 1. The preys swarm observe, decide, and move.
        for idx_prey in range(arg_list.n_targets):
            action_prey = preys[idx_prey].get_action()
            # env.act(idx_prey, action_prey, is_prey=True)
            env.act(idx_prey, action_prey, is_evader=True)

        # 2. The predators observe and make decisions in parallel.
        # Determine the priorities.
        # priorities = np.random.permutation(n_predators)
        priorities = np.arange(arg_list.n_pursuers)
        action_predators = [None] * arg_list.n_pursuers
        for idx_predator in priorities:
            action_predator = predators[idx_predator].get_action()
            action_predators[idx_predator] = action_predator

        # 3. The predators move in parallel.
        for idx_predator in priorities:
            action_predator = action_predators[idx_predator]
            # collide = env.act(idx_predator, action_predator, is_prey=False)
            collide = env.act(idx_predator, action_predator, is_evader=False)
            ep_collisions += 1 if collide else 0
            if collide:
                pursuer = predators[idx_predator]
                print("Collision! ep_len, idx_predator, role, position:",
                      ep_len, idx_predator, pursuer.role, pursuer.global_own_position)

        ep_len += 1

        if arg_list.render or arg_list.save_frames:
            render_frame(env, arg_list.render, arg_list.save_frames)

        timeout = (ep_len == arg_list.episode_len)
        is_terminal, capture_rate = env.is_all_captured()
        terminal = is_terminal or timeout
        if terminal:
            ep_time = time.time() - start_time

            print('Seed %3d \t Episode %3d \t EpLen %d \t CaptureRate %.3f \t Collisions %d Time %.3f' %
                  (seed, i_episode, ep_len, capture_rate, ep_collisions, ep_time))

            ep_len_list.append(ep_len)
            ep_capture_rate_list.append(capture_rate)
            ep_collisions_list.append(ep_collisions)
            ep_time_list.append(ep_time)
            log_rows.append([str(i_episode), str(ep_len), str(capture_rate), str(ep_collisions), str(ep_time)])

            i_episode += 1
            if arg_list.use_seed:
                seed = seed_list[i_episode]
                torch.manual_seed(seed)
                np.random.seed(seed)
                # env.reset(set_seed=True, seed=seed)
                env.reset(seed=seed)

            preys = initialize_prey_swarm(env)
            predators = initialize_predator_swarm(env)

            ep_len, ep_collisions = 0, 0
            start_time = time.time()

    # Statistical result.
    with open(logger_output_fname, 'w') as fp:
        fp.write("\n".join(["\t".join(row) for row in log_rows]))

    print("Write to file:", logger_output_fname)

    log_rows_2 = [["NEpisodes",
                   "EpLenAvg", "EpLenStd", "EpLenMax", "EpLenMin",
                   "CaptureRateAvg", "CaptureRateStd", "CaptureRateMax", "CaptureRateMin",
                   "CollisionsAvg", "CollisionsStd", "CollisionsMax", "CollisionsMin",
                   "Time(s)Avg", "Time(s)Std", "Time(s)Max", "Time(s)Min"]]
    one_row = [str(len(ep_len_list))]
    one_row += get_statistical_result(ep_len_list)
    one_row += get_statistical_result(ep_capture_rate_list)
    one_row += get_statistical_result(ep_collisions_list)
    one_row += get_statistical_result(ep_time_list)
    log_rows_2.append(one_row)
    with open(logger_output_fname_2, 'w') as fp:
        fp.write("\n".join(["\t".join(row) for row in log_rows_2]))

    print("Write to file:", logger_output_fname_2)
    pass


def get_statistical_result(data):
    return [str(np.mean(data)), str(np.std(data)), str(np.max(data)), str(np.min(data))]


def create_environment(world_x_size, world_y_size, n_preys, n_predators):

    # Configurations.
    world_rows = world_x_size
    world_columns = world_y_size
    fov_scope = 11

    # env = MatrixWorld(world_rows, world_columns,
    #                   n_preys=n_preys, n_predators=n_predators,
    #                   fov_scope=fov_scope)
    env = MatrixWorld(world_rows, world_columns,
                      n_evaders=n_preys, n_pursuers=n_predators,
                      fov_scope=fov_scope)

    return env


def render_frame(env, is_display, is_save):

    # Modify `is_display`.
    env.render(is_display=is_display,
               is_save=is_save, is_fixed_size=False,
               grid_on=True, tick_labels_on=True,
               show_predator_idx=True,
               show_prey_idx=True)


def initialize_prey_swarm(env):
    """
    :param env:
    :return: A list of prey,
             which are some kind of prey class instance.
    """
    # n_preys = env.n_preys
    n_preys = env.n_evaders

    # [8]
    # np.arange(0, n_prey, 1).tolist()
    debug_idx = []

    preys = []
    for idx in range(n_preys):
        under_debug = False
        if idx in debug_idx:
            under_debug = True

        # prey = DoNothingPrey(env, idx, under_debug=under_debug)
        prey = RandomPrey(env, idx, under_debug=under_debug)
        preys.append(prey)

    return preys


def initialize_predator_swarm(env):
    """
    :param env:
    :return: A list of predators,
             which are some kind of predator class instance.
    """

    # n_predators = env.n_predators
    n_predators = env.n_pursuers

    # [8]
    # np.arange(0, n_predators, 1).tolist()
    debug_idx = []

    predators = []
    for idx in range(n_predators):
        under_debug = False
        if idx in debug_idx:
            under_debug = True

        predator = SelfOrganizedPredator(env, idx, under_debug=under_debug)
        predators.append(predator)

    return predators


if __name__ == "__main__":
    arg_list = parse_args()
    run_policy(arg_list)
