import os
import argparse
import time
import numpy as np
import random
import torch

from lib.performance_logger import PerformanceLogger
from lib.environment.matrix_world import MatrixWorld as Pursuit
from lib.preys.random_prey import RandomPrey
from lib.mappo.mappo_policy import MAPPOPolicy
from lib.mappo.trainer_agent_mappo import MAPPOAgents
from lib.mappo.shared_buffer import SharedReplayBuffer


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", "-s", type=int, default=0)

    # Test.

    parser.add_argument("--render", type=bool, default=False)

    # Epoch is different from episode.
    # An epoch can collect experiences of more than one or less than one episodes.
    # 5000, 1000, 100

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

    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")

    parser.add_argument("--use_valuenorm", action='store_false', default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_popart", action='store_true', default=False,
                        help="by default False, use PopArt to normalize rewards.")

    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=1e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')

    # ppo parameters

    parser.add_argument("--ppo_epoch", type=int, default=10,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True,
                        help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True,
                        help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters

    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')

    # save parameters

    parser.add_argument("--save_interval", type=int, default=100,
                        help="time duration between contiunous twice models saving.")

    # log parameters

    parser.add_argument("--log_interval", type=int, default=5,
                        help="time duration between contiunous twice log printing.")

    # Env.

    parser.add_argument("--world_size", type=int, default=40)
    parser.add_argument("--n_pursuers", type=int, default=4 * 4)
    parser.add_argument("--n_evaders", type=int, default=4)

    data_log_path = 'data/log_mappo/sop_mappo/'
    os.makedirs(data_log_path, exist_ok=True)
    parser.add_argument("--data_log_path", type=str, default=data_log_path)

    parser.add_argument("--save_model_name_actor", type=str, default='sop_mappo_model_actor.pth')
    parser.add_argument("--save_model_name_critic", type=str, default='sop_mappo_model_critic.pth')
    parser.add_argument("--save_model_name_value_normalizer", type=str, default='sop_mappo_model_value_normalizer.pth')

    parser.add_argument("--resume_model", type=bool, default=False)
    parser.add_argument("--resume_model_name_actor", type=str, default='sop_mappo_model_actor.pth')
    parser.add_argument("--resume_model_name_critic", type=str, default='sop_mappo_model_critic.pth')
    parser.add_argument("--resume_model_name_value_normalizer", type=str,
                        default='sop_mappo_model_value_normalizer.pth')

    return parser.parse_args()


def initialize_evader_swarm(env):
    """
    :param env:
    :return: A list of evader,
             which are some kind of evader class instance.
    """
    n_evaders = env.n_evaders

    evaders = []
    for idx in range(n_evaders):
        evader = RandomPrey(env, idx)
        evaders.append(evader)

    return evaders


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class RunExperiment:

    def __init__(self, args):

        self.args = args

        self.logger = PerformanceLogger(self.args.data_log_path)

        # Share parameters.

        self.evaders = None
        self.pursuers = None
        self.buffer = None

        # step_1_create_environment.

        self.env = None
        self.dim_observation = None
        self.dim_action = None
        self.dim_centralized_observation = None

        # step_4_experiment_over_generation

        self.game_done = False
        self.idx_generation = 0
        self.epoch_counter = 0

        pass

    def run(self):

        self.step_1_create_environment()
        self.step_2_set_seed()
        self.step_3_create_agents()
        self.step_4_experiment_over_epoch()

    def step_1_create_environment(self):

        self.env = Pursuit(world_rows=self.args.world_size, world_columns=self.args.world_size,
                           n_evaders=self.args.n_evaders, n_pursuers=self.args.n_pursuers,
                           fov_scope=11,
                           max_env_cycles=self.args.max_episode_length,
                           save_path=os.path.join(self.args.data_log_path, "frames"))

        self.dim_observation = (self.env.fov_scope ** 2) * 3
        self.dim_action = self.env.n_actions
        if self.args.use_centralized_V:
            self.dim_centralized_observation = self.dim_observation * self.args.n_pursuers
        else:
            self.dim_centralized_observation = self.dim_observation

    def step_2_set_seed(self):
        """
        First, create an environment instance.
        Second, set random seed, including that for the environment.
        Third, do all the other things.
        """

        seed = self.args.seed + 10000

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.reset(seed=seed)

    def step_3_create_agents(self):

        # Evader.

        self.evaders = initialize_evader_swarm(self.env)

        # 3. ModelPolicy.

        model_actor_critic = MAPPOPolicy(self.args,
                                         self.dim_observation,
                                         self.dim_centralized_observation,
                                         self.dim_action,
                                         self.args.device)

        if self.args.resume_model:

            model_actor_critic.actor.load_state_dict(torch.load(os.path.join(self.args.data_log_path,
                                                                             self.args.resume_model_name_actor),
                                                                map_location=self.args.device))

            model_actor_critic.critic.load_state_dict(torch.load(os.path.join(self.args.data_log_path,
                                                                              self.args.resume_model_name_critic),
                                                                 map_location=self.args.device))

        self.pursuers = MAPPOAgents(self.args, policy=model_actor_critic, device=self.args.device)

        if self.args.resume_model:

            self.pursuers.value_normalizer.load_state_dict(
                torch.load(os.path.join(self.args.data_log_path,self.args.resume_model_name_value_normalizer),
                           map_location=self.args.device))

        self.buffer = SharedReplayBuffer(self.args,
                                         self.args.n_pursuers,
                                         self.dim_observation,
                                         self.dim_centralized_observation,
                                         self.dim_action)

    def step_4_experiment_over_epoch(self):

        self.step_4_0_warm_up()

        for i_epoch in range(self.args.n_epochs):

            start_epoch_time = time.time()

            if self.args.use_linear_lr_decay:

                self.pursuers.policy.lr_decay(i_epoch, self.args.n_epochs)

            self.step_4_1_experiment_of_an_epoch()

            self.step_4_2_update_model()

            self.step_4_3_log_info_of_epoch(time.time() - start_epoch_time)

            # Update.

            self.epoch_counter += 1

            pass

        pass

    def step_4_0_warm_up(self):

        # reset env
        # obs = self.env.reset()
        self.env.reset()

        # Pursuer.

        _, observations, _, self.game_done, _ = self.env.last(is_evader=False)
        observations = observations.reshape(observations.shape[0], -1)

        # replay buffer
        if self.args.use_centralized_V:
            share_obs = observations.reshape(-1)
            share_obs = np.expand_dims(share_obs, 0).repeat(self.args.n_pursuers, axis=0)
        else:
            share_obs = observations

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = observations.copy()

    def step_4_1_experiment_of_an_epoch(self):

        episode_length = 0

        for t in range(self.args.steps_per_epoch):

            # Sample actions

            values, actions, action_log_probs, actions_env = self.collect(t)

            # Env update.

            if not self.game_done:

                self.env.step_swarm(actions_env, is_evader=False)

            # Evader.

            # The evader swarm observe, decide, and move.
            for idx_evader in range(self.args.n_evaders):
                action_evader = self.evaders[idx_evader].get_action()
                self.env.act(idx_evader, action_evader, is_evader=True)

            # Observe reward and next obs after the env update.

            rewards, next_observations, _, self.game_done, (capture_rate,
                                                            n_collision_with_boundaries,
                                                            n_collision_with_obstacles,
                                                            n_multiagent_collision_events) = \
                self.env.last(is_evader=False)
            next_observations = next_observations.reshape(next_observations.shape[0], -1)

            data = next_observations, rewards, self.game_done, values, actions, action_log_probs

            # insert data into buffer

            self.insert(data)

            # Update episode process record.

            episode_length += 1

            if (self.epoch_counter % self.args.log_interval == 0) or (self.epoch_counter == self.args.n_epochs - 1):
                self.logger.update_episode_performance(key="episode_return", value=np.mean(rewards))
                self.logger.update_episode_performance(key="episode_n_collision_with_boundaries",
                                                       value=n_collision_with_boundaries)
                self.logger.update_episode_performance(key="episode_n_collision_with_obstacles",
                                                       value=n_collision_with_obstacles)
                self.logger.update_episode_performance(key="episode_n_multiagent_collision_events",
                                                       value=n_multiagent_collision_events)

            # 7. Identify the game status.

            episode_timeout = (episode_length == self.args.max_episode_length)
            episode_terminal = self.game_done or episode_timeout

            # Specific number of experiences in an epoch have already been collected,
            # terminate this epoch (and prepare to start the next one).

            epoch_ended = (t == (self.args.steps_per_epoch - 1))

            if episode_terminal or epoch_ended:

                if (self.epoch_counter % self.args.log_interval == 0) or (self.epoch_counter == self.args.n_epochs - 1):
                    if episode_terminal:

                        self.logger.update_episode_performance(key="episode_length", value=episode_length)
                        self.logger.update_episode_performance(key="capture_rate", value=capture_rate)
                        self.logger.end_episode_performance()

                    else:

                        self.logger.reset_episode_performance()

                # Reset.

                self.env.reset()

                episode_length = 0

                pass
        pass

    @torch.no_grad()
    def collect(self, step):

        self.pursuers.prep_rollout()

        value, action, action_log_prob = self.pursuers.policy.get_actions(self.buffer.share_obs[step],
                                                                          self.buffer.obs[step])
        # [self.envs, agents, dim]
        values = _t2n(value)
        actions = _t2n(action)
        action_log_probs = _t2n(action_log_prob)

        # actions_env = np.squeeze(np.eye(self.env.action_dimension)[actions], 2)
        actions_env = action

        return values, actions, action_log_probs, actions_env

    def insert(self, data):

        obs, rewards, game_done, values, actions, action_log_probs = data

        masks = np.ones(self.args.n_pursuers, dtype=np.float32) * (game_done is False)

        if self.args.use_centralized_V:
            share_obs = obs.reshape(-1)
            share_obs = np.expand_dims(share_obs, 0).repeat(self.args.n_pursuers, axis=0)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, actions, action_log_probs, values, rewards, masks)

    def step_4_2_update_model(self):

        # compute return and update network
        self.compute()
        train_infos = self.train()

        if (self.epoch_counter % self.args.log_interval == 0) or (self.epoch_counter == self.args.n_epochs - 1):

            self.logger.update_epoch_performance(key="loss_value", value=train_infos["value_loss"])
            self.logger.update_epoch_performance(key="loss_policy", value=train_infos["policy_loss"])
            self.logger.update_epoch_performance(key="distribution_entropy", value=train_infos["dist_entropy"])
            self.logger.update_epoch_performance(key="actor_gradient_norm", value=train_infos["actor_grad_norm"].item())
            self.logger.update_epoch_performance(key="critic_gradient_norm",
                                                 value=train_infos["critic_grad_norm"].item())
            self.logger.update_epoch_performance(key="mappo_ratio", value=train_infos["ratio"].item())

        if (self.epoch_counter % self.args.save_interval == 0) or (self.epoch_counter == self.args.n_epochs - 1):

            self.save()

    def step_4_3_log_info_of_epoch(self, epoch_time):

        if (self.epoch_counter % self.args.log_interval == 0) or (self.epoch_counter == self.args.n_epochs - 1):
            self.logger.update_epoch_performance(key="epoch_time_s", value=epoch_time)
            self.logger.log_dump_epoch_performance(self.epoch_counter)

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.pursuers.prep_rollout()
        next_values = self.pursuers.policy.get_values(self.buffer.share_obs[-1])
        next_values = _t2n(next_values)
        self.buffer.compute_returns(next_values, self.pursuers.value_normalizer)

    def train(self):
        """Train policies with data in buffer. """
        self.pursuers.prep_training()
        train_infos = self.pursuers.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""

        model_name_prefix = os.path.join(self.args.data_log_path, "Epoch2000x" + str(self.epoch_counter // 2000))
        model_log_path_actor = model_name_prefix + self.args.save_model_name_actor
        model_log_path_critic = model_name_prefix + self.args.save_model_name_critic
        model_log_path_value_normalizer = model_name_prefix + self.args.save_model_name_value_normalizer

        policy_actor = self.pursuers.policy.actor
        torch.save(policy_actor.state_dict(), model_log_path_actor)

        policy_critic = self.pursuers.policy.critic
        torch.save(policy_critic.state_dict(), model_log_path_critic)

        if self.pursuers._use_valuenorm:
            policy_vnorm = self.pursuers.value_normalizer
            torch.save(policy_vnorm.state_dict(), model_log_path_value_normalizer)


if __name__ == "__main__":

    args = parse_args()

    run_an_experiment = RunExperiment(args)

    run_an_experiment.run()

    print("COMPLETE!")
