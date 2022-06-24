import argparse
import numpy as np
import random
import time
import pickle
import gym
import os
import os.path as osp

import tensorflow as tf
import tensorflow.contrib.layers as layers

import PIL
from datetime import datetime

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg.sisl.vector_env_sos import EnvSOS

from pursuit_game import pursuit_vsos_maddpg as pursuit

from actor_critic.common.logx import EpochLogger


# Modify the data_path.
data_path = "/Users/lijunsun/Workspace/selforganizing_search_pursuit/results/sos_MADDPG/policy_s8/checkpoint60000/"


def parse_args(seed):
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument('--use-seed', '-s', default=True)
    parser.add_argument("--seed", type=int, default=seed, help="random seed")
    # Environment
    parser.add_argument("--scenario", type=str, default="pursuit", help="name of the scenario script")

    parser.add_argument("--num-episodes", type=int, default=1, help="number of episodes")
    parser.add_argument('--x_size', '-xs', type=int, default=20)
    parser.add_argument('--y_size', '-ys', type=int, default=20)
    parser.add_argument('--n_pursuers', '-np', type=int, default=8)
    parser.add_argument('--n_targets', '-nt', type=int, default=50)
    parser.add_argument("--max-episode-len", type=int, default=500, help="maximum episode length")

    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="maddpg_sos_experiment", help="name of the experiment")
    # save_dir must be an absolute path, and it can be restored from a relative path.
    # Otherwise, if a relative path is given here, there are problems in restoring from relative path.
    # Also, cannot use `~` here to represent the home dir.
    # parser.add_argument("--save-dir", type=str, default="/tmp/maddpg-sos/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1, help="save model once every time this many episodes are completed")
    parser.add_argument("--save-dir", type=str, default=data_path, help="directory in which training state and model should be saved")
    # parser.add_argument("--save-dir", type=str, default="/home/lijsun/Data/workspace/selforganizing_search_pursuit/results/sos_maddpg/policy_s" + str(seed) + "/", help="directory in which training state and model should be saved")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--plots-dir", type=str, default="./results/sos_maddpg/testing_curves_s" + str(seed) + "/", help="directory where plot data is saved")
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--save-frames", default=False)
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=400, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=300, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, n_targets, n_pursuers, x_size=40, y_size=40):
    name_dict = {
        "pursuit": pursuit,
    }
    scenario = name_dict[scenario_name]
    env = EnvSOS(scenario.env, n_targets=n_targets, n_pursuers=n_pursuers, x_size=x_size, y_size=y_size)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arg_list):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    action_spaces = [env.action_space_dict[p] for p in env.agent_ids]
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, action_spaces, i, arg_list,
            local_q_func=(arg_list.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.num_agents):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, action_spaces, i, arg_list,
            local_q_func=(arg_list.good_policy=='ddpg')))
    return trainers


def images_to_gif(image_list, output_name):
    output_dir = osp.join(data_path, 'pngs')
    os.system("mkdir " + output_dir)
    for idx, im in enumerate(image_list):
        im.save(output_dir + "/target" + str(idx) + ".png")

    os.system("/usr/local/bin/ffmpeg -i " + output_dir + "/target%d.png " + output_name)
    os.system("rm -r " + output_dir)
    print("Write to file:", output_name)


def train(arg_list, logger, trainers=None):
    # Create environment
    env = make_env(arg_list.scenario, arg_list.n_targets, arg_list.n_pursuers, arg_list.x_size, arg_list.y_size)

    # Random seed.
    if arg_list.use_seed:
        seed = arg_list.seed
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        env.env.sisl_env.env.seed(seed)

    with U.single_threaded_session():
        # Create agent trainers
        obs_shape_n = [env.observation_space_dict[i].shape for i in env.agent_ids]
        num_adversaries = min(env.num_agents, arg_list.num_adversaries)
        if trainers is None:
            trainers = get_trainers(env, num_adversaries, obs_shape_n, arg_list)
        print('Using good policy {} and adv policy {}'.format(arg_list.good_policy, arg_list.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arg_list.load_dir == "":
            arg_list.load_dir = arg_list.save_dir
        if arg_list.display or arg_list.restore:
            print('Loading previous state...')
            U.load_state(arg_list.load_dir)

        saver = tf.train.Saver()
        obs_n = env.reset()

        episode_rewards = [0.0]  # mean of all agents' episode rewards in the same environment.
        agent_rewards = [[0.0] for _ in range(env.num_agents)]  # individual agent reward
        episode_step = 0  # Counter of the multi-agent step.
        train_step = 0
        episode_q_losses = []  #
        episode_p_losses = []  #
        episode_have_loss_or_not = []
        frame_list = []

        t_start = time.time()

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arg_list.max_episode_len) or done
            # collect experience
            # for i, agent in enumerate(trainers):
            #     if not done:
            #         # obs_n: [[array_shape=(364=11*11*3+1,)]_0, ..., [array_shape=(364,)]_7]]
            #         agent.experience(obs_n[i][0], action_n[i][0], rew_n[i], new_obs_n[i][0], done_n[i], terminal)

            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                # episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            # increment global step counter
            train_step += 1

            # for displaying learned policies
            if arg_list.display:
                # for displaying learned policies
                if arg_list.save_frames:
                        frame_list.append(PIL.Image.fromarray(env.env.sisl_env.render(mode='rgb_array')))
                else:
                    time.sleep(0.1)
                    env.env.sisl_env.render()  # EnvSOS.

            # update all trainers, if not in display or benchmark mode
            loss = None
            # for agent in trainers:
            #     agent.preupdate()
            # for agent in trainers:
            #     loss = agent.update(trainers, train_step)

            episode_have_loss_or_not.append(loss is not None)
            episode_q_losses.append(loss[0] if loss is not None else np.nan)
            episode_p_losses.append(loss[1] if loss is not None else np.nan)

            if terminal:
                episode_rewards[-1] = np.mean([agent_rewards[i][-1] for i in range(env.num_agents)])
                capture_rate = sum(env.env.sisl_env.env.evaders_gone) / len(env.env.sisl_env.env.evaders_gone)
                collisions = env.env.sisl_env.env.n_collision_events_per_multiagent_step
                collisions_with_obstacles = env.env.sisl_env.env.n_collision_with_obstacles
                episode_time = round(time.time() - t_start, 3)

            # save model, display training output
            if terminal and ((len(episode_rewards) % arg_list.save_rate == 0) or
                             (len(episode_rewards) > arg_list.num_episodes)):
                # U.save_state(arg_list.save_dir, saver=saver)
                if num_adversaries == 0:
                    print("Seed: {}, Steps: {}, episodes: {}, mean episode reward: {}, episode length: {}, "
                          "capture rate: {}, collisions: {}, collisions_with_obstacles: {}, time: {}".format(
                        arg_list.seed,
                        train_step,
                        len(episode_rewards),
                        np.round(np.mean(episode_rewards[-arg_list.save_rate:]), 3),
                        episode_step,
                        round(capture_rate, 3),
                        collisions,
                        collisions_with_obstacles,
                        episode_time))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arg_list.save_rate:]),
                        [np.mean(rew[-arg_list.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))

                t_start = time.time()

                logger.store(EpRet=episode_rewards[-1], EpLen=episode_step, CaptureRate=capture_rate,
                             Collisions=collisions, CollideObstacles=collisions_with_obstacles)

            if len(episode_rewards) > arg_list.num_episodes:
                print('...Finished total of {} episodes.'.format(len(episode_rewards) - 1))
                break

            if done or terminal:
                if arg_list.save_frames:
                    data_path = arg_list.save_dir
                    now = datetime.now()
                    time_format = "%Y%m%d%H%M%S"
                    timestamp = now.strftime(time_format)
                    filename = data_path + "result_pursuit_sos_" + \
                        str("{:.3f}".format(episode_rewards[-1])) + \
                        str("{:.3f}".format(episode_step)) + \
                        str("{:.3f}".format(capture_rate)) + \
                        str("{:.3f}".format(collisions)) + "_" + timestamp + ".gif"
                    images_to_gif(frame_list, filename)

                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)

    if not arg_list.save_frames:
        env.env.sisl_env.close()  # EnvSOS

    return logger, trainers


if __name__ == '__main__':
    arg_list = parse_args(seed=-1)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # logger_output_fname = 'progress_test_' + timestamp + '.txt'
    logger_output_fname = "_".join(['progress_test_parallel',
                                    str(arg_list.x_size), str(arg_list.y_size),
                                    str(arg_list.n_pursuers), str(arg_list.n_targets), '.txt'])
    logger = EpochLogger(output_dir=data_path, output_fname=logger_output_fname)
    print('Write to file:', osp.join(data_path, logger_output_fname))

    num_episodes = 100
    use_seed = True
    seed_list = range(num_episodes)
    trainers = None
    for seed in seed_list:
        arg_list = parse_args(seed)
        logger, trainers = train(arg_list, logger, trainers)

    logger.log_tabular('UseSeed', use_seed)
    logger.log_tabular("NEpisodes", num_episodes)
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', with_min_and_max=True)
    logger.log_tabular("CaptureRate", with_min_and_max=True)
    logger.log_tabular("Collisions", with_min_and_max=True)
    logger.log_tabular("CollideObstacles", with_min_and_max=True)
    logger.dump_tabular()
    print('SUCCESS!')
