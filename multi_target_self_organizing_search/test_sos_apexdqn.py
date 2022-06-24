import os
import os.path as osp
import time
from datetime import datetime
from pathlib import Path
import pickle5 as pickle
import argparse
import PIL
import numpy as np
import random

import ray
import ray.rllib.agents.dqn.apex as apex
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from pursuit_game import pursuit_vsos as pursuit
from pursuit_game.supersuit import flatten_v0

from apex_dqn.custom_model import MLPModel
from actor_critic.common.logx import EpochLogger


# Modify the data_path and checkpoint filename.
data_path = "./results/sos_Apex-DQN/APEX_pursuit_2af28_00000_0_2022-04-30_20-13-23/"
checkpoint_folder = "checkpoint_003067"
checkpoint_file = "checkpoint-3067"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('f_path', type=str, nargs='*', default=data_path)
    parser.add_argument('checkpoint_folder', type=str, nargs='*', default=checkpoint_folder)
    parser.add_argument('checkpoint_file', type=str, nargs='*', default=checkpoint_file)

    parser.add_argument('--use_seed', '-s', default=True)

    parser.add_argument('--n_episodes', '-n', type=int, default=100)
    parser.add_argument('--x_size', '-xs', type=int, default=40)
    parser.add_argument('--y_size', '-ys', type=int, default=40)
    parser.add_argument('--n_pursuers', '-np', type=int, default=1)
    parser.add_argument('--n_targets', '-nt', type=int, default=5)
    parser.add_argument('--episode_len', '-l', type=int, default=500)

    parser.add_argument('--render', '-r', default=False)
    parser.add_argument('--save_frames', '-sf', default=False)

    return parser.parse_args()


def run_policy(env, env_config, checkpoint_folder, checkpoint_file, arg_list,  n_pursuers, max_ep_len=None,
               num_episodes=100, use_seed=True, render=True, save_frames=False):

    # ##############################################################################
    # STEPS to render.
    # register environment.
    # register model.
    # checkpoint file -> agent restore.
    # parameters file -> config dict.
    # run

    checkpoint_path = osp.join(data_path, checkpoint_folder, checkpoint_file)

    checkpoint_path = os.path.expanduser(checkpoint_path)
    params_path = Path(checkpoint_path).parent.parent/"params.pkl"

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # logger_output_fname = 'progress_test_' + checkpoint_path.split('/')[-1] + '_' + timestamp + '.txt'
    logger_output_fname = "_".join(['progress_test_parallel',
                                    str(arg_list.x_size), str(arg_list.y_size),
                                    str(arg_list.n_pursuers), str(arg_list.n_targets),
                                    # timestamp,
                                    '.txt'])
    logger = EpochLogger(output_dir=data_path, output_fname=logger_output_fname)

    # ##############################################################################

    with open(params_path, "rb") as f:
        config = pickle.load(f)

        config['num_envs_per_worker'] = 1
        config['num_workers'] = 1
        config['num_gpus'] = 0
        config['env_config'] = env_config
        # num_workers not needed since we are not training
        # del config['num_workers']
        # del config['num_gpus']
        print('config:', config)

    ray.init(num_cpus=4)

    Agent = apex.ApexTrainer(config=config, env='pursuit')
    Agent.restore(checkpoint_path)

    # ##############################################################################
    # Random seed.
    seed_list = range(num_episodes)
    seed = seed_list[0]
    if use_seed:
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)

    env.reset()
    ep_ret, ep_len, ep_collisions, ep_collisions_with_obstacles, i_episode = 0, 0, 0, 0, 0
    agents_ep_ret = [0 for _ in range(n_pursuers)]
    frame_list = []

    while i_episode < num_episodes:
        if render:
            if not save_frames:
                env.render()
                time.sleep(1e-3)
            else:
                frame_list.append(PIL.Image.fromarray(env.render(mode='rgb_array')))

        actions = []
        dones = []
        # agent_id = 0
        for agent_id in range(n_pursuers):
        # for agent in env.agent_iter():
            o, r, d, _ = env.last_vsos(agent_id)
            o = np.reshape(o, -1)
            agents_ep_ret[agent_id] += r

            action, _, _ = Agent.get_policy("policy_0").compute_single_action(o)

            actions.append(action)
            dones.append(d)

        for agent_id in range(n_pursuers):
            d, action = dones[agent_id], actions[agent_id]
            if d:
                env.step(None)
            else:
                env.step(action)

            # agent_id += 1

            # if agent_id % len(env.possible_agents) == 0:
            #     break

        ep_len += 1
        ep_collisions += env.env.env.env.env.n_collision_events_per_multiagent_step

        timeout = (ep_len == max_ep_len)
        terminal = env.env.env.env.env.is_terminal or timeout
        if terminal:
            ep_ret = np.mean(agents_ep_ret)
            capture_rate = sum(env.env.env.env.env.evaders_gone) / len(env.env.env.env.env.evaders_gone)
            ep_collisions_with_obstacles = env.env.env.env.env.n_collision_with_obstacles
            logger.store(EpRet=ep_ret, EpLen=ep_len, CaptureRate=capture_rate,
                         Collisions=ep_collisions, CollideObstacles=ep_collisions_with_obstacles)

            if use_seed:
                print('Seed %3d \t Episode %3d \t EpRet %.3f \t EpLen %d \t CaptureRate %d \t Collisions %d \t CollideObstacles %d' %
                      (seed, i_episode, ep_ret, ep_len, capture_rate, ep_collisions, ep_collisions_with_obstacles))
            else:
                print('Episode %3d \t EpRet %.3f \t EpLen %d \t CaptureRate %d \t Collisions %d \t CollideObstacles %d' %
                      (i_episode, ep_ret, ep_len, capture_rate, ep_collisions, ep_collisions_with_obstacles))

            i_episode += 1
            if use_seed and i_episode < num_episodes:
                seed = seed_list[i_episode]
                np.random.seed(seed)
                random.seed(seed)
                env.seed(seed)

            if save_frames:
                now = datetime.now()
                time_format = "%Y%m%d%H%M%S"
                timestamp = now.strftime(time_format)
                filename = data_path + "result_pursuit_sos_" + \
                    checkpoint_path.split('/')[-1] + "_" + \
                    str("{:.3f}".format(ep_ret)) + "_" + \
                    str("{:.3f}".format(capture_rate)) + "_" + \
                    str("{:d}".format(ep_collisions)) + "_" + \
                    str("{:d}".format(ep_collisions_with_obstacles)) + "_" + \
                    timestamp + '.gif'
                images_to_gif(frame_list, filename)

            frame_list = []

            env.reset()
            ep_ret, ep_len, ep_collisions, ep_collisions_with_obstacles = 0, 0, 0, 0
            agents_ep_ret = [0 for _ in range(n_pursuers)]

    logger.log_tabular('UseSeed', use_seed)
    logger.log_tabular("NEpisodes", num_episodes)
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', with_min_and_max=True)
    logger.log_tabular("CaptureRate", with_min_and_max=True)
    logger.log_tabular("Collisions", with_min_and_max=True)
    logger.log_tabular("CollideObstacles", with_min_and_max=True)
    logger.dump_tabular()


# define how to make the environment. This way takes an optional environment config.
def env_creator(env_config):
    env = pursuit.env(max_cycles=500,
                      x_size=env_config["x_size"], y_size=env_config["y_size"],
                      n_evaders=env_config["n_targets"], n_pursuers=env_config["n_pursuers"],
                      obs_range=11,
                      surround=False, n_catch=1,
                      freeze_evaders=True)

    if "seed" in env_config.keys():
        seed = env_config["seed"]
        env.seed(seed)

    env = flatten_v0(env)
    return env


def images_to_gif(image_list, output_name):
    output_dir = osp.join(data_path, 'pngs')
    os.system("mkdir " + output_dir)
    for idx, im in enumerate(image_list):
        im.save(output_dir + "/target" + str(idx) + ".png")

    os.system("/usr/local/bin/ffmpeg -i " + output_dir + "/target%d.png " + output_name)
    os.system("rm -r " + output_dir)
    print('Write to file:', output_name)


if __name__ == "__main__":
    arg_list = parse_args()

    register_env('pursuit', lambda env_config: PettingZooEnv(env_creator(env_config)))
    ModelCatalog.register_custom_model("MLPModel", MLPModel)

    env_config = {"x_size": arg_list.x_size, "y_size": arg_list.y_size,
                  "n_targets": arg_list.n_targets, "n_pursuers": arg_list.n_pursuers}
    env = PettingZooEnv(env_creator(env_config))
    env = env_creator(env_config)

    run_policy(env,
               env_config,
               checkpoint_folder=arg_list.checkpoint_folder,
               checkpoint_file=arg_list.checkpoint_file,
               arg_list=arg_list,
               n_pursuers=arg_list.n_pursuers,
               max_ep_len=arg_list.episode_len,
               num_episodes=arg_list.n_episodes,
               use_seed=arg_list.use_seed,
               render=arg_list.render,
               save_frames=arg_list.save_frames)

    print("SUCCESS!")
