import argparse
import numpy as np
import random
import time
import pickle
import gym
import os

import tensorflow as tf
import tensorflow.contrib.layers as layers

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg.sisl.vector_env_sos import EnvSOS

from pursuit_game import pursuit_vsos_maddpg as pursuit


def parse_args():
    seed = 0
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument('--use-seed', '-s', default=True)
    parser.add_argument("--seed", type=int, default=seed, help="random seed")
    # Environment
    parser.add_argument("--n_pursuers", type=int, default=8)
    parser.add_argument("--n_targets", type=int, default=50)
    parser.add_argument("--scenario", type=str, default="pursuit", help="name of the scenario script")

    parser.add_argument("--max-episode-len", type=int, default=500, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="maddpg_sos_experiment", help="name of the experiment")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    # save_dir must be an absolute path, and it can be restored from a relative path.
    # Otherwise, if a relative path is given here, there are problems in restoring from relative path.
    # Also, cannot use `~` here to represent the home dir.
    parser.add_argument("--save-dir", type=str, default="/tmp/sos_maddpg/policy/", help="directory in which training state and model should be saved")
    # parser.add_argument("--save-dir", type=str, default="/Users/lijunsun/Workspace/selforganizing_search_pursuit/results/sos_maddpg/policy_s" + str(seed) + "/", help="directory in which training state and model should be saved")
    # parser.add_argument("--save-dir", type=str, default="/home/lijsun/Data/workspace/selforganizing_search_pursuit/results/sos_maddpg/policy_s" + str(seed) + "/", help="directory in which training state and model should be saved")
    # parser.add_argument("--save-dir", type=str, default="/home/hucao/Data/ljsworkspace/selforganizing_search_pursuit/results/sos_maddpg/policy_s" + str(seed) + "/", help="directory in which training state and model should be saved")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    # parser.add_argument("--plots-dir", type=str, default="/tmp/sos_maddpg/learning_curves_s" + str(seed) + "/", help="directory where plot data is saved")
    # parser.add_argument("--plots-dir", type=str, default="./results/sos_maddpg/learning_curves_s" + str(seed) + "/", help="directory where plot data is saved")
    parser.add_argument("--plots-dir", type=str, default="./results/sos_maddpg/learning_curves_s" + str(seed) + "/", help="directory where plot data is saved")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=400, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=300, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, n_targets, n_pursuers):
    name_dict = {
        "pursuit": pursuit,
    }
    scenario = name_dict[scenario_name]
    env = EnvSOS(scenario.env, n_targets=n_targets, n_pursuers=n_pursuers)
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


def train(arg_list):
    # Create environment
    env = make_env(arg_list.scenario, arg_list.n_targets, arg_list.n_pursuers)

    # Random seed.
    if arg_list.use_seed:
        seed = arg_list.seed + 10000
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        env.env.sisl_env.env.seed(seed)

    with U.single_threaded_session():
        # Create agent trainers
        obs_shape_n = [env.observation_space_dict[i].shape for i in env.agent_ids]
        num_adversaries = min(env.num_agents, arg_list.num_adversaries)
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
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        episode_lengths = [0]  # No. of multi-agent steps.
        final_ep_lengths = []  # episode lengths for training curve.
        episode_step = 0  # Counter of the multi-agent step.
        final_ep_step = []  # episode step for training curve.
        train_step = 0
        episode_capture_rates = []  # No. of capture rate.
        final_ep_capture_rates = []  # episode capture rate for training curve.
        episode_collisions = []  # n_collision_events_per_multiagent_step
        episode_collisions_with_obstacles = []
        final_ep_collisions = []  # final episode collisions for training curve.
        final_ep_collisions_with_obstacles = []
        episode_q_losses = []  #
        final_ep_q_losses = []
        episode_p_losses = []  #
        final_ep_p_losses = []
        episode_have_loss_or_not = []
        final_have_loss_or_not = []
        episode_times = []
        final_ep_times = []

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
            for i, agent in enumerate(trainers):
                if not done:
                    # obs_n: [[array_shape=(364=11*11*3+1,)]_0, ..., [array_shape=(364,)]_7]]
                    agent.experience(obs_n[i][0], action_n[i][0], rew_n[i], new_obs_n[i][0], done_n[i], terminal)

            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                # episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            # increment global step counter
            train_step += 1

            # for displaying learned policies
            if arg_list.display:
                time.sleep(0.1)
                env.env.sisl_env.render()  # EnvSOS.
                # continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            episode_have_loss_or_not.append(loss is not None)
            episode_q_losses.append(loss[0] if loss is not None else np.nan)
            episode_p_losses.append(loss[1] if loss is not None else np.nan)

            if terminal:
                episode_rewards[-1] = np.mean([agent_rewards[i][-1] for i in range(env.num_agents)])
                capture_rate = sum(env.env.sisl_env.env.evaders_gone) / len(env.env.sisl_env.env.evaders_gone)
                collisions = env.env.sisl_env.env.n_collision_events_per_multiagent_step
                collisions_with_obstacles = env.env.sisl_env.env.n_collision_with_obstacles
                episode_time = round(time.time() - t_start, 3)
                # Keep track of performance metrics.
                episode_lengths.append(episode_step)
                episode_capture_rates.append(capture_rate)
                episode_collisions.append(collisions)
                episode_collisions_with_obstacles.append(collisions_with_obstacles)
                episode_times.append(episode_time)

            # save model, display training output
            if terminal and ((len(episode_rewards) % arg_list.save_rate == 0) or
                             (len(episode_rewards) > arg_list.num_episodes)):
                save_dir = os.path.join(arg_list.save_dir, 'checkpoint' + str(len(episode_rewards)) + '/')
                U.save_state(save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, episode length: {}, "
                          "capture rate: {}, collisions: {}, collisions_with_obstacles: {}, time: {}".format(
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

                # Keep track of final episode reward
                # final_ep_rewards.append(np.mean(episode_rewards[-arg_list.save_rate:]))
                final_ep_rewards = episode_rewards
                for rew in agent_rewards:
                    # final_ep_ag_rewards.append(np.mean(rew[-arg_list.save_rate:]))
                    final_ep_ag_rewards = rew
                # final_ep_lengths.append(np.mean(episode_lengths[-arg_list.save_rate:]))
                final_ep_lengths = episode_lengths
                # final_ep_capture_rates.append(np.mean(episode_capture_rates[-arg_list.save_rate:]))
                final_ep_capture_rates = episode_capture_rates
                # final_ep_collisions.append(np.mean(episode_collisions[-arg_list.save_rate:]))
                final_ep_collisions = episode_collisions
                # final_ep_collisions_with_obstacles.append(np.mean(episode_collisions_with_obstacles[-arg_list.save_rate:]))
                final_ep_collisions_with_obstacles = episode_collisions_with_obstacles
                # final_ep_q_losses.append(np.mean(np.array(episode_q_losses[-arg_list.save_rate:])[np.where(episode_have_loss_or_not[-arg_list.save_rate:])[0]]))
                final_ep_q_losses = episode_q_losses
                # final_ep_p_losses.append(np.mean(np.array(episode_p_losses[-arg_list.save_rate:])[np.where(episode_have_loss_or_not[-arg_list.save_rate:])[0]]))
                final_ep_p_losses = episode_p_losses
                # final_have_loss_or_not.append(any(episode_have_loss_or_not[-arg_list.save_rate:]))
                final_have_loss_or_not = episode_have_loss_or_not
                # final_ep_times.append(np.mean(episode_times[-arg_list.save_rate:]))
                final_ep_times = episode_times

            # saves final episode reward for plotting training curve later
            # if (len(episode_rewards) > arg_list.num_episodes) or (len(episode_rewards) % arg_list.save_rate == 0):
                os.makedirs(arg_list.plots_dir, exist_ok=True)
                rew_file_name = arg_list.plots_dir + arg_list.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arg_list.plots_dir + arg_list.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                length_file_name = arg_list.plots_dir + arg_list.exp_name + '_lengths.pkl'
                with open(length_file_name, 'wb') as fp:
                    pickle.dump(final_ep_lengths, fp)
                capture_file_name = arg_list.plots_dir + arg_list.exp_name + '_capture_rates.pkl'
                with open(capture_file_name, 'wb') as fp:
                    pickle.dump(final_ep_capture_rates, fp)
                collision_file_name = arg_list.plots_dir + arg_list.exp_name + '_collisions.pkl'
                with open(collision_file_name, 'wb') as fp:
                    pickle.dump(final_ep_collisions, fp)
                collisions_with_obstacles_file_name = arg_list.plots_dir + arg_list.exp_name + '_collisions_with_obstacles.pkl'
                with open(collisions_with_obstacles_file_name, 'wb') as fp:
                    pickle.dump(final_ep_collisions_with_obstacles, fp)
                q_loss_file_name = arg_list.plots_dir + arg_list.exp_name + '_q_loss.pkl'
                with open(q_loss_file_name, 'wb') as fp:
                    pickle.dump(final_ep_q_losses, fp)
                p_loss_file_name = arg_list.plots_dir + arg_list.exp_name + '_p_loss.pkl'
                with open(p_loss_file_name, 'wb') as fp:
                    pickle.dump(final_ep_p_losses, fp)
                have_loss_or_not_file_name = arg_list.plots_dir + arg_list.exp_name + '_have_loss_or_not.pkl'
                with open(have_loss_or_not_file_name, 'wb') as fp:
                    pickle.dump(final_have_loss_or_not, fp)
                time_file_name = arg_list.plots_dir + arg_list.exp_name + '_time.pkl'
                with open(time_file_name, 'wb') as fp:
                    pickle.dump(final_ep_times, fp)

            if len(episode_rewards) > arg_list.num_episodes:
                print('...Finished total of {} episodes.'.format(len(episode_rewards) - 1))
                break

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)

                t_start = time.time()

    env.env.sisl_env.close()  # EnvSOS


if __name__ == '__main__':
    arg_list = parse_args()
    train(arg_list)
    print('SUCCESS!')
