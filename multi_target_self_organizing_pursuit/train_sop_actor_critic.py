import os

import argparse
import os.path as osp
import time
import copy

import numpy as np
import random

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind

import torch

from lib.environment.matrix_world import MatrixWorld as Pursuit
from lib.preys.random_prey import RandomPrey
from lib.actor_critic.models import ModelMLP, ModelPolicy, ModelActorCritic
from lib.actor_critic.trainer_agent_actor_critic import ActorCriticAgents


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", "-s", type=int, default=0)

    # Test.

    parser.add_argument("--render", type=bool, default=False)

    # Epoch is different from episode.
    # An epoch can collect experiences of more than one or less than one episodes.
    # 5000, 1000, 100, 10000, 20000, 1e5

    parser.add_argument("--n_epochs", type=int, default=int(1e5))

    # steps_per_epoch is different from max_episode_length.

    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--max_episode_length", type=int, default=500)

    # Training.

    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:1')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    parser.add_argument("--device", type=torch.device, default=device)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--lr_policy", type=float, default=1e-4)
    parser.add_argument("--lr_value", type=float, default=1e-4)
    parser.add_argument("--train_value_iters", type=int, default=10, help="default: 80")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=5,
                        help="time duration between contiunous twice log printing.")

    # Env.

    parser.add_argument("--world_size", type=int, default=40)
    parser.add_argument("--n_pursuers", type=int, default=4*4)
    parser.add_argument("--n_evaders", type=int, default=4)
    parser.add_argument("--exp_name", type=str, default='sop_actor_critic')

    data_log_path = 'data/log_actor_critic/sop_actor_critic_train_value_iters_10/'
    os.makedirs(data_log_path, exist_ok=True)
    parser.add_argument("--data_log_path", type=str, default=data_log_path)

    parser.add_argument("--save_model_name_actor", type=str, default='sop_model_actor.pth')
    parser.add_argument("--save_model_name_critic", type=str, default='sop_model_critic.pth')

    parser.add_argument("--resume_model", type=bool, default=False)
    parser.add_argument("--resume_model_name_actor", type=str, default='sop_model_actor.pth')
    parser.add_argument("--resume_model_name_critic", type=str, default='sop_model_critic.pth')

    parser.add_argument("--output_fname", type=str, default='progress')

    parser.add_argument("--frame_folder", type=str, default=osp.join(data_log_path, "frames"))
    parser.add_argument("--video_name", type=str,
                        default=os.path.join(data_log_path, "pursuit.gif"))

    return parser.parse_args()


def initialize_model_policy(dim_input, dim_output, data_log_path, device):

    model_policy = ModelPolicy(dim_input=dim_input,
                               dim_output=dim_output,
                               hidden_sizes=(400, 300))

    model_log_path = osp.join(data_log_path)

    model_policy.load_state_dict(torch.load(osp.join(model_log_path, "Epoch2000x26sop_model_actor.pth"),
                                            map_location=device))

    return model_policy


def initialize_model_actor_critic(args, dim_input, dim_output, device):

    model_policy = ModelPolicy(dim_input=dim_input,
                               dim_output=dim_output,
                               hidden_sizes=(400, 300))

    model_value = ModelMLP(dim_input=dim_input, dim_output=1, hidden_sizes=(400, 300))

    if args.resume_model:

        model_policy.load_state_dict(torch.load(osp.join(args.data_log_path, args.resume_model_name_actor),
                                                map_location=args.device))
        model_value.load_state_dict(torch.load(osp.join(args.data_log_path, args.resume_model_name_critic),
                                               map_location=args.device))

    model_actor_critic = ModelActorCritic(model_policy=model_policy,
                                          model_value=model_value).to(device)

    return model_actor_critic


def save_model(model_policy, model_value, model_log_path_actor, model_log_path_critic):

    torch.save(model_policy.state_dict(), model_log_path_actor)
    torch.save(model_value.state_dict(), model_log_path_critic)

    print("Write to:", model_log_path_actor)
    print("Write to:", model_log_path_critic)


def initialize_evader_swarm(env):
    """
    :param env:
    :return: A list of evader,
             which are some kind of evader class instance.
    """
    n_evaders = env.n_evaders

    # [8]
    # np.arange(0, n_prey, 1).tolist()
    debug_idx = []

    evaders = []
    for idx in range(n_evaders):
        under_debug = False
        if idx in debug_idx:
            under_debug = True

        # prey = DoNothingPrey(env, idx, under_debug=under_debug)
        evader = RandomPrey(env, idx, under_debug=under_debug)
        evaders.append(evader)

    return evaders


def train(args):

    # 1. Instantiate environment.

    env = Pursuit(world_rows=args.world_size, world_columns=args.world_size,
                  n_evaders=args.n_evaders, n_pursuers=args.n_pursuers,
                  fov_scope=11,
                  occupying_based_capture=False,
                  save_path=args.frame_folder)

    dim_observation = env.fov_scope * env.fov_scope * 3
    dim_action = env.n_actions

    # 2. Set seed.

    seed = 0 + 10000

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env.reset(seed=seed)

    # Evader.

    evaders = initialize_evader_swarm(env)

    # 3. ModelPolicy.

    model_actor_critic = initialize_model_actor_critic(args,
                                                       dim_input=dim_observation,
                                                       dim_output=dim_action,
                                                       device=args.device)

    pursuers = ActorCriticAgents(model_actor_critic=model_actor_critic,
                                 lr_policy=args.lr_policy, lr_value=args.lr_value,
                                 train_value_iters=args.train_value_iters,
                                 dim_observation=dim_observation,
                                 dim_observation_unflatten=(env.fov_scope, env.fov_scope, 3),
                                 buffer_size=args.steps_per_epoch,
                                 n_agents=args.n_pursuers,
                                 n_evaders=args.n_evaders,
                                 gamma=args.gamma, lam=args.lam, device=args.device,
                                 world_size=args.world_size)

    # 4. Collect experience (data) and optimize (update) model.

    # The following three lines are together.

    env.reset()

    _, observations, env_vectors, game_done, _ = env.last(is_evader=False)

    observations = torch.as_tensor(observations, dtype=torch.double, device=args.device)

    # 5. Reset episode performance.

    start_time_per_epoch = time.time()

    episode_length_per_epoch = []
    episode_capture_rate_per_epoch = []
    episode_return_per_epoch = []
    episode_n_collision_with_boundaries_per_epoch = []
    episode_n_collision_with_obstacles_per_epoch = []
    episode_n_multiagent_collision_events_per_epoch = []

    episode_length = 0
    episode_capture_rate = 0
    episode_return = 0
    episode_n_collision_with_boundaries = 0
    episode_n_collision_with_obstacles = 0
    episode_n_multiagent_collision_events = 0

    # 6. Log info.

    data_log_path = osp.join(args.data_log_path, args.output_fname + ".txt")

    with open(data_log_path, 'w') as output_file:

        row_head = ['epoch', 'episode_length_avg', 'episode_length_std',
                    'episode_capture_rate_avg', 'episode_capture_rate_std',
                    'episode_return_avg', 'episode_return_std',
                    'episode_n_collision_with_boundaries_avg', 'episode_n_collision_with_boundaries_std',
                    'episode_n_collision_with_obstacles_avg', 'episode_n_collision_with_obstacles_std',
                    'episode_n_multiagent_collision_events_avg', 'episode_n_multiagent_collision_events_std',
                    'loss_value',
                    'loss_policy',
                    'time']

        output_file.write("\t".join(map(str, row_head)) + "\n")
        output_file.flush()

    for i_epoch in range(args.n_epochs):

        # print('-'*70)
        # print("Epoch", i_epoch)

        for t in range(args.steps_per_epoch):

            # print("Step", t)

            # Test.

            if args.render:

                env.render(is_display=True, is_save=True,
                           is_fixed_size=False, grid_on=True, tick_labels_on=False,
                           show_pursuer_idx=False, show_evader_idx=False)

            # Evader.

            # The evader swarm observe, decide, and move.
            for idx_evader in range(args.n_evaders):
                action_evader = evaders[idx_evader].get_action()
                env.act(idx_evader, action_evader, is_evader=True)

            # 7. Agents observe, make decisions.

            _, observations, _, _, _ = env.last(is_evader=False)

            observations = torch.as_tensor(observations, dtype=torch.double, device=args.device)

            # (batch_size, set_size, ...)

            actions, values = \
                pursuers.model_actor_critic.step(observations.reshape(1, *observations.shape))

            # 8. Env update.

            if not game_done:

                env.step_swarm(actions[0], is_evader=False)

            # 9. Observe after the env update.

            rewards, next_observations, next_env_vectors, game_done, (capture_rate,
                                                                      n_collision_with_boundaries,
                                                                      n_collision_with_obstacles,
                                                                      n_multiagent_collision_events) = \
                env.last(is_evader=False)

            next_observations = torch.as_tensor(next_observations, dtype=torch.double, device=args.device)

            # 10. Update episode process record.

            episode_length += 1
            episode_return += torch.mean(torch.tensor(rewards, dtype=torch.double)).item()
            episode_n_collision_with_boundaries += n_collision_with_boundaries
            episode_n_collision_with_obstacles += n_collision_with_obstacles
            episode_n_multiagent_collision_events += n_multiagent_collision_events

            # 11. Store agents' experience in buffer.

            pursuers.data_buffer.store_swarm(swarm_observations=observations,
                                             swarm_actions=actions,
                                             swarm_rewards=torch.tensor(rewards),
                                             swarm_values=values)

            # 12. Update the observation memory.

            # observations = next_observations

            # 13. Identify the game status.

            episode_timeout = (episode_length == args.max_episode_length)
            episode_terminal = game_done or episode_timeout

            # Specific number of experiences in an epoch have already been collected,
            # terminate this epoch (and prepare to start the next one).

            epoch_ended = (t == (args.steps_per_epoch - 1))

            if episode_terminal or epoch_ended:

                # Test.

                if args.render:

                    env.render(is_display=True, is_save=True,
                               is_fixed_size=False, grid_on=True, tick_labels_on=False,
                               show_pursuer_idx=False, show_evader_idx=False)

                # 14. Finish the experience collecting process.

                # If trajectory didn't reach episode_terminal state, bootstrap value target.

                if episode_timeout or epoch_ended:

                    _, observations, _, _, _ = env.last(is_evader=False)

                    observations = torch.as_tensor(observations, dtype=torch.double, device=args.device)

                    # _, values, _, _ = pursuers.model_actor_critic.step(observations.reshape(1, *observations.shape))
                    _, values = pursuers.model_actor_critic.step(observations.reshape(1, *observations.shape))

                else:

                    values = torch.zeros((1, args.n_pursuers), dtype=torch.double, device=args.device)

                pursuers.data_buffer.finish_path(values)

                # 15. Log info only when a complete episode is finished.

                if episode_terminal and ((i_epoch % args.log_interval == 0) or (i_epoch == args.n_epochs - 1)):
                # if episode_terminal:

                    # Log info.

                    episode_length_per_epoch.append(episode_length)
                    episode_capture_rate_per_epoch.append(capture_rate)
                    episode_return_per_epoch.append(episode_return)
                    episode_n_collision_with_boundaries_per_epoch.append(episode_n_collision_with_boundaries)
                    episode_n_collision_with_obstacles_per_epoch.append(episode_n_collision_with_obstacles)
                    episode_n_multiagent_collision_events_per_epoch.append(episode_n_multiagent_collision_events)

                    pass

                # 16. Reset episode parameters.

                env.reset()

                _, observations, env_vectors, game_done, _ = env.last(is_evader=False)

                observations = torch.as_tensor(observations, dtype=torch.double, device=args.device)

                episode_length = 0
                episode_return = 0
                episode_n_collision_with_boundaries = 0
                episode_n_collision_with_obstacles = 0
                episode_n_multiagent_collision_events = 0

                # Test.

                # In test mode, it is reasonable only count the episode, not the epoch,
                # since epoch is purposed to collect enough experiences for learning,
                # and there is no learning in the test mode.

                # break

        # 17. Update model and log info.

        loss_value, loss_policy = pursuers.update_model()

        # 18. Save model.

        # model_log_path = osp.join(args.data_log_path, args.model_name_pursuer)
        model_name_prefix = osp.join(args.data_log_path, "Epoch2000x" + str(i_epoch // 2000))
        model_log_path_actor = model_name_prefix + args.save_model_name_actor
        model_log_path_critic = model_name_prefix + args.save_model_name_critic

        if (i_epoch % args.save_interval == 0) or (i_epoch == args.n_epochs - 1):
            save_model(pursuers.model_actor_critic.model_policy,
                       pursuers.model_actor_critic.model_value,
                       model_log_path_actor, model_log_path_critic)

        if (i_epoch % args.log_interval == 0) or (i_epoch == args.n_epochs - 1):
            row = [i_epoch,
                   np.mean(episode_length_per_epoch), np.std(episode_length_per_epoch),
                   np.mean(episode_capture_rate_per_epoch), np.std(episode_capture_rate_per_epoch),
                   np.mean(episode_return_per_epoch), np.std(episode_return_per_epoch),
                   np.mean(episode_n_collision_with_boundaries_per_epoch),
                   np.std(episode_n_collision_with_boundaries_per_epoch),
                   np.mean(episode_n_collision_with_obstacles_per_epoch),
                   np.std(episode_n_collision_with_obstacles_per_epoch),
                   np.mean(episode_n_multiagent_collision_events_per_epoch),
                   np.std(episode_n_multiagent_collision_events_per_epoch),
                   loss_value,
                   loss_policy,
                   time.time() - start_time_per_epoch]

            with open(data_log_path, 'a') as output_file:

                output_file.write("\t".join(map(str, row)) + "\n")
                output_file.flush()

            print("Epoch: {:4d}, "
                  "loss_policy: {:7f}, "
                  "loss_value: {:7f}, "
                  "episode_length_per_epoch: {:4f}, "
                  "episode_capture_rate_per_epoch: {:4f}, "
                  "episode_n_multiagent_collision_events_per_epoch: {:4f},"
                  "episode_n_collision_with_boundaries_per_epoch: {:4f}, "
                  "episode_n_collision_with_obstacles_per_epoch: {:4f}, "
                  "episode_return_per_epoch: {:7f}, "
                  "epoch time: {:7f}".format(
                    i_epoch,
                    loss_policy,
                    loss_value,
                    np.mean(episode_length_per_epoch),
                    np.mean(episode_capture_rate_per_epoch),
                    np.mean(episode_n_multiagent_collision_events_per_epoch),
                    np.mean(episode_n_collision_with_boundaries_per_epoch),
                    np.mean(episode_n_collision_with_obstacles_per_epoch),
                    np.mean(episode_return_per_epoch),
                    time.time() - start_time_per_epoch))

        # 19. Reset epoch performance metrics.

        start_time_per_epoch = time.time()

        episode_length_per_epoch = []
        episode_capture_rate_per_epoch = []
        episode_return_per_epoch = []
        episode_n_collision_with_boundaries_per_epoch = []
        episode_n_collision_with_obstacles_per_epoch = []
        episode_n_multiagent_collision_events_per_epoch = []

    pass


def test(args):

    # 1. Instantiate environment.

    env = Pursuit(world_rows=args.world_size, world_columns=args.world_size,
                  n_evaders=args.n_evaders, n_pursuers=args.n_pursuers,
                  fov_scope=11,
                  occupying_based_capture=False,
                  save_path=osp.join(args.data_log_path, "frames"))

    dim_observation = env.fov_scope * env.fov_scope * 3
    dim_action = env.n_actions

    # Set seed.

    seed = 0 + 20000

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env.reset(seed=seed)

    # Evader.

    evaders = initialize_evader_swarm(env)

    # Policy.

    policy = initialize_model_policy(dim_input=dim_observation,
                                     dim_output=dim_action,
                                     data_log_path=args.data_log_path,
                                     device=args.device)

    # The following three lines are together.

    env.reset()

    _, observations, env_vectors, game_done, _ = env.last(is_evader=False)

    observations = torch.as_tensor(observations, dtype=torch.double, device=args.device)

    # 5. Reset episode performance.

    start_time_epochs = time.time()
    start_time_per_epoch = time.time()

    episode_length_epochs = []
    episode_capture_rate_epochs = []
    episode_return_epochs = []
    episode_n_collision_with_boundaries_epochs = []
    episode_n_collision_with_obstacles_epochs = []
    episode_n_multiagent_collision_events_epochs = []

    episode_length_per_epoch = []
    episode_capture_rate_per_epoch = []
    episode_return_per_epoch = []
    episode_n_collision_with_boundaries_per_epoch = []
    episode_n_collision_with_obstacles_per_epoch = []
    episode_n_multiagent_collision_events_per_epoch = []

    episode_length = 0
    episode_return = 0
    episode_n_collision_with_boundaries = 0
    episode_n_collision_with_obstacles = 0
    episode_n_multiagent_collision_events = 0

    # 6. Log info.

    data_log_path = osp.join(args.data_log_path, "progress_test.txt")

    with open(data_log_path, 'w') as output_file:
        row_head = ['epoch', 'episode_length_avg', 'episode_length_std',
                    'episode_capture_rate_avg', 'episode_capture_rate_std',
                    'episode_return_avg', 'episode_return_std',
                    'episode_n_collision_with_boundaries_avg',
                    'episode_n_collision_with_boundaries_std',
                    'episode_n_collision_with_obstacles_avg',
                    'episode_n_collision_with_obstacles_std',
                    'episode_n_multiagent_collision_events_avg',
                    'episode_n_multiagent_collision_events_std',
                    'time']

        output_file.write("\t".join(map(str, row_head)) + "\n")
        output_file.flush()

    for i_epoch in range(args.n_epochs):

        # print('-'*80)
        # print("Epoch", i_epoch)

        for t in range(args.steps_per_epoch):

            # print("Step", t)

            # Test.

            if args.render:

                env.render(is_display=True, is_save=True,
                           is_fixed_size=False, grid_on=True, tick_labels_on=False,
                           show_pursuer_idx=False, show_evader_idx=False)

            # Evader.

            # The evader swarm observe, decide, and move.
            for idx_evader in range(args.n_evaders):
                action_evader = evaders[idx_evader].get_action()
                env.act(idx_evader, action_evader, is_evader=True)

            # 7. Agents observe, make decisions.

            _, observations, _, _, _ = env.last(is_evader=False)

            observations = torch.as_tensor(observations, dtype=torch.double, device=args.device)

            # (batch_size, set_size, dim_)

            action_distribution_categories, _ = policy(observations.reshape(1, *observations.shape))
            # action_distribution_categories, _, _ = policy(observations.reshape(1, observations.shape[0], -1))

            actions = action_distribution_categories.sample()

            # 8. Env update.

            if not game_done:

                env.step_swarm(actions[0], is_evader=False)

            # 9. Observe after the env update.

            rewards, next_observations, next_env_vectors, game_done, (capture_rate,
                                                                      n_collision_with_boundaries,
                                                                      n_collision_with_obstacles,
                                                                      n_multiagent_collision_events) = \
                env.last(is_evader=False)

            next_observations = torch.as_tensor(next_observations, dtype=torch.double)

            # 10. Update episode process record.

            episode_length += 1
            episode_return += torch.mean(torch.tensor(rewards, dtype=torch.double)).item()
            episode_n_collision_with_boundaries += n_collision_with_boundaries
            episode_n_collision_with_obstacles += n_collision_with_obstacles
            episode_n_multiagent_collision_events += n_multiagent_collision_events

            # 12. Update the observation memory.

            observations = next_observations

            # 13. Identify the game status.

            episode_timeout = (episode_length == args.max_episode_length)
            episode_terminal = game_done or episode_timeout

            # Specific number of experiences in an epoch have already been collected,
            # terminate this epoch (and prepare to start the next one).

            epoch_ended = (t == (args.steps_per_epoch - 1))

            if episode_terminal or epoch_ended:

                # Test.

                if args.render:

                    env.render(is_display=True, is_save=True,
                               is_fixed_size=False, grid_on=True, tick_labels_on=False,
                               show_pursuer_idx=False, show_evader_idx=False)

                    os.system("ffmpeg -i " + args.frame_folder + "/MatrixWorld%4d.png " + args.video_name)

                # 15. Log info only when a complete episode is finished.

                if episode_terminal:

                    # Log info.

                    episode_length_per_epoch.append(episode_length)
                    episode_capture_rate_per_epoch.append(capture_rate)
                    episode_return_per_epoch.append(episode_return)
                    episode_n_collision_with_boundaries_per_epoch.append(episode_n_collision_with_boundaries)
                    episode_n_collision_with_obstacles_per_epoch.append(episode_n_collision_with_obstacles)
                    episode_n_multiagent_collision_events_per_epoch.append(
                        episode_n_multiagent_collision_events)

                    pass

                # 16. Reset episode parameters.

                env.reset()

                _, observations, env_vectors, game_done, _ = env.last(is_evader=False)

                observations = torch.as_tensor(observations, dtype=torch.double, device=args.device)

                episode_length = 0
                episode_return = 0
                episode_n_collision_with_boundaries = 0
                episode_n_collision_with_obstacles = 0
                episode_n_multiagent_collision_events = 0

                # Test.

                # In test mode, it is reasonable only count the episode, not the epoch,
                # since epoch is purposed to collect enough experiences for learning,
                # and there is no learning in the test mode.

                break

        row = [i_epoch,
               np.mean(episode_length_per_epoch),
               np.std(episode_length_per_epoch),
               np.mean(episode_capture_rate_per_epoch),
               np.std(episode_capture_rate_per_epoch),
               np.mean(episode_return_per_epoch),
               np.std(episode_return_per_epoch),
               np.mean(episode_n_collision_with_boundaries_per_epoch),
               np.std(episode_n_collision_with_boundaries_per_epoch),
               np.mean(episode_n_collision_with_obstacles_per_epoch),
               np.std(episode_n_collision_with_obstacles_per_epoch),
               np.mean(episode_n_multiagent_collision_events_per_epoch),
               np.std(episode_n_multiagent_collision_events_per_epoch),
               time.time() - start_time_per_epoch]

        with open(data_log_path, 'a') as output_file:

            output_file.write("\t".join(map(str, row)) + "\n")
            output_file.flush()

        print("Epoch: {:4d}, "
              "episode_length_per_epoch: {:4f}, "
              "episode_capture_rate_per_epoch: {:4f}, "
              "episode_n_multiagent_collision_events_per_epoch: {:4f},"
              "episode_n_collision_with_boundaries_per_epoch: {:4f}, "
              "episode_n_collision_with_obstacles_per_epoch: {:4f}, "
              "episode_return_per_epoch: {:7f}".format(
                i_epoch,
                np.mean(episode_length_per_epoch),
                np.mean(episode_capture_rate_per_epoch),
                np.mean(episode_n_multiagent_collision_events_per_epoch),
                np.mean(episode_n_collision_with_boundaries_per_epoch),
                np.mean(episode_n_collision_with_obstacles_per_epoch),
                np.mean(episode_return_per_epoch)))

        episode_length_epochs.append(np.mean(episode_length_per_epoch))
        episode_capture_rate_epochs.append(np.mean(episode_capture_rate_per_epoch))
        episode_return_epochs.append(np.mean(episode_return_per_epoch))
        episode_n_collision_with_boundaries_epochs.append(np.mean(episode_n_collision_with_boundaries_per_epoch))
        episode_n_collision_with_obstacles_epochs.append(np.mean(episode_n_collision_with_obstacles_per_epoch))
        episode_n_multiagent_collision_events_epochs.append(np.mean(episode_n_multiagent_collision_events_per_epoch))

        # 19. Reset epoch performance metrics.

        start_time_per_epoch = time.time()

        episode_length_per_epoch = []
        episode_capture_rate_per_epoch = []
        episode_return_per_epoch = []
        episode_n_collision_with_boundaries_per_epoch = []
        episode_n_collision_with_obstacles_per_epoch = []
        episode_n_multiagent_collision_events_per_epoch = []

    # Statistical result of epochs.

    row = [-1,
           np.mean(episode_length_epochs),
           np.std(episode_length_epochs),
           np.mean(episode_return_epochs),
           np.std(episode_return_epochs),
           np.mean(episode_n_collision_with_boundaries_epochs),
           np.std(episode_n_collision_with_boundaries_epochs),
           np.mean(episode_n_collision_with_obstacles_epochs),
           np.std(episode_n_collision_with_obstacles_epochs),
           np.mean(episode_n_multiagent_collision_events_epochs),
           np.std(episode_n_multiagent_collision_events_epochs),
           time.time() - start_time_epochs]

    with open(data_log_path, 'a') as output_file:

        output_file.write("\t".join(map(str, row)) + "\n")
        output_file.flush()

    print("Summary -  , "
          "episode_length_epochs: {:4f}, "
          "episode_n_multiagent_collision_events_epochs: {:4f},"
          "episode_n_collision_with_boundaries_epochs: {:4f}, "
          "episode_n_collision_with_obstacles_epochs: {:4f}, "
          "episode_return_epochs: {:7f}".format(
            np.mean(episode_length_epochs),
            np.mean(episode_n_multiagent_collision_events_epochs),
            np.mean(episode_n_collision_with_boundaries_epochs),
            np.mean(episode_n_collision_with_obstacles_epochs),
            np.mean(episode_return_epochs)))

    pass


def statistical_test(args):

    # Get data.

    data_file_1 = osp.join(args.data_log_path, "test_performance_obstacle_avoidance",
                           "progress_train_without_mask_test_without_mask.txt")
    data_file_2 = osp.join(args.data_log_path, "test_performance_obstacle_avoidance",
                           "progress_train_with_mask_test_with_mask.txt")

    data_1 = pd.read_table(data_file_1)
    data_2 = pd.read_table(data_file_2)

    # row_head = ['epoch', 'episode_length_avg', 'episode_length_std',
    #                     'episode_return_avg', 'episode_return_std',
    #                     'episode_n_collision_with_boundaries_avg',
    #                     'episode_n_collision_with_boundaries_std',
    #                     'episode_n_collision_with_obstacles_avg',
    #                     'episode_n_collision_with_obstacles_std',
    #                     'episode_n_multiagent_collision_events_avg',
    #                     'episode_n_multiagent_collision_events_std',
    #                     'time']

    # Episode return.

    metric_1 = data_1['episode_return_avg'][:-1]
    metric_2 = data_2['episode_return_avg'][:-1]

    result_pvalue = ttest_ind(metric_1, metric_2).pvalue

    print('episode_return_avg:', result_pvalue)

    # Episode length.

    metric_1 = data_1['episode_length_avg'][:-1]
    metric_2 = data_2['episode_length_avg'][:-1]

    result_pvalue = ttest_ind(metric_1, metric_2).pvalue

    print('episode_length_avg:', result_pvalue)

    # Obstacle collisions.

    metric_1 = data_1['episode_n_collision_with_obstacles_avg'][:-1]
    metric_2 = data_2['episode_n_collision_with_obstacles_avg'][:-1]

    result_pvalue = ttest_ind(metric_1, metric_2).pvalue

    print('episode_n_collision_with_obstacles_avg:', result_pvalue)

    # Multiagent collisions.

    metric_1 = data_1['episode_n_multiagent_collision_events_avg'][:-1]
    metric_2 = data_2['episode_n_multiagent_collision_events_avg'][:-1]

    result_pvalue = ttest_ind(metric_1, metric_2).pvalue

    print('episode_n_multiagent_collision_events_avg:', result_pvalue)

    pass


def get_log_data(data_log_path):

    # data_log_path = osp.join(data_log_path, 'progress_test.txt')
    data_log_path = osp.join(data_log_path, 'progress.txt')

    # row_head = ['epoch', 'episode_length_avg', 'episode_length_std',
    #                     'episode_return_avg', 'episode_return_std',
    #                     'episode_n_collision_with_boundaries_avg', 'episode_n_collision_with_boundaries_std',
    #                     'episode_n_collision_with_obstacles_avg', 'episode_n_collision_with_obstacles_std',
    #                     'episode_n_multiagent_collision_events_avg', 'episode_n_multiagent_collision_events_std',
    #                     'loss_value',
    #                     'loss_policy',
    #                     'time']

    data = pd.read_table(data_log_path)

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


def visualize(args):

    data_log_path = args.data_log_path

    transparent = 0.7

    data = get_log_data(data_log_path)
    print("data:\n", data)

    epochs = data['epoch']

    plt.figure()
    plt.plot(epochs, data['episode_length_avg'])
    y_moving_average = compute_moving_average(data['episode_length_avg'])
    plt.plot(epochs, y_moving_average, 'r-.', label="Moving average")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Episode length")
    output_figure_path = osp.join(data_log_path, "episode_length_avg.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.figure()
    plt.plot(epochs, data['episode_capture_rate_avg'])
    y_moving_average = compute_moving_average(data['episode_capture_rate_avg'])
    plt.plot(epochs, y_moving_average, 'r-.', label="Moving average")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Episode capture rate")
    output_figure_path = osp.join(data_log_path, "episode_capture_rate_avg.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.figure()
    plt.plot(epochs, data['episode_return_avg'])
    y_moving_average = compute_moving_average(data['episode_return_avg'])
    plt.plot(epochs, y_moving_average, 'r-.', label="Moving average")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Episode return")
    output_figure_path = osp.join(data_log_path, "episode_return_avg.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    truncate_start_epoch = 50
    truncate_end_epoch = 8000
    plt.figure()
    plt.plot(epochs.iloc[truncate_start_epoch:truncate_end_epoch],
             data['episode_return_avg'].iloc[truncate_start_epoch:truncate_end_epoch])
    plt.xlabel("Epoch")
    plt.ylabel("Episode return")
    output_figure_path = osp.join(data_log_path, "episode_return_avg_truncated.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.figure()
    plt.plot(epochs, data['episode_n_multiagent_collision_events_avg'], label='Collide agent', alpha=transparent)
    plt.plot(epochs, data['episode_n_collision_with_obstacles_avg'], label='Collide obstacle', alpha=transparent)
    plt.plot(epochs, data['episode_n_collision_with_boundaries_avg'], label='Collide boundaries', alpha=transparent)
    plt.ylim([0, 100])
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Episode collisions")
    output_figure_path = osp.join(data_log_path, "episode_collisions.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.figure()
    plt.plot(epochs, data['time'])
    plt.xlabel("Epoch")
    plt.ylabel("Epoch time")
    output_figure_path = osp.join(data_log_path, "epoch_time.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.figure()
    plt.plot(epochs, data['loss_value'], label='loss_value')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    output_figure_path = osp.join(data_log_path, "loss_value.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.figure()
    plt.plot(epochs, data['loss_policy'], label='loss_policy', alpha=transparent)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    output_figure_path = osp.join(data_log_path, "loss_policy.png")
    plt.savefig(output_figure_path, bbox_inches='tight')
    print("Save to file:", output_figure_path)

    plt.show()

    pass


if __name__ == "__main__":

    args = parse_args()

    # train(args)
    # test(args)
    # statistical_test(argsgs)
    visualize(args)

    print('COMPLETE! SUCCESS!')
