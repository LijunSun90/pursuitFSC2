import scipy.signal
import os.path as osp

import torch


class ExperienceBuffer:

    def __init__(self, dim_observation, dim_observation_unflatten, buffer_size, n_agents,
                 gamma=0.99, lam=0.95, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        self.device = device

        # self.buffer_observation = torch.zeros((buffer_size, n_agents, dim_observation), dtype=torch.double,
        #                                       device=device)

        self.buffer_observation = \
            torch.zeros((buffer_size, n_agents, *dim_observation_unflatten), dtype=torch.double, device=device)

        self.buffer_observation_target_pursuer_obstacle = \
            torch.zeros((buffer_size, n_agents, *dim_observation_unflatten), dtype=torch.double, device=device)

        self.buffer_action = torch.zeros((buffer_size, n_agents), dtype=torch.long, device=device)

        # For compute return.

        self.buffer_reward = torch.zeros((buffer_size, n_agents), dtype=torch.double, device=device)

        # For compute advantage.

        self.buffer_value = torch.zeros((buffer_size, n_agents), dtype=torch.double, device=device)

        self.buffer_advantage = torch.zeros((buffer_size, n_agents), dtype=torch.double, device=device)

        self.buffer_return = torch.zeros((buffer_size, n_agents), dtype=torch.double, device=device)

        # For computing advantage.

        self.gamma, self.lam = gamma, lam

        self.ptr, self.path_start_idx, self.max_buffer_size = 0, 0, buffer_size

    def store_swarm(self, swarm_observations, swarm_actions, swarm_rewards, swarm_values):
        """
        :param swarm_observations: (n_agents, dim_observation).
        :param swarm_actions: (n_agents,).
        :param swarm_rewards: (n_agents,).
        :param swarm_values: (n_agents,).
        """

        # Buffer has to have room so you can store.
        assert self.ptr < self.max_buffer_size

        self.buffer_observation[self.ptr] = swarm_observations
        self.buffer_action[self.ptr] = swarm_actions
        self.buffer_reward[self.ptr] = swarm_rewards
        self.buffer_value[self.ptr] = swarm_values

        self.ptr += 1

    def finish_path(self, last_swarm_value):
        """
        :param last_swarm_value: (n_agents,).
        """

        path_slice = slice(self.path_start_idx, self.ptr)

        rewards = torch.vstack((self.buffer_reward[path_slice, :], last_swarm_value))
        values = torch.vstack((self.buffer_value[path_slice, :], last_swarm_value))

        # The next two lines implement GAE-Lambda advantage calculation.

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.buffer_advantage[path_slice] = \
            torch.from_numpy(self.discount_cumulative_sum(deltas.tolist(), self.gamma * self.lam, axis=0).copy())

        # The next line computes rewards-to-go, to be targets for the value function.

        self.buffer_return[path_slice] = \
            torch.from_numpy(self.discount_cumulative_sum(rewards.tolist(), self.gamma, axis=0)[:-1].copy())

        self.path_start_idx = self.ptr

    def get(self):

        # Buffer has to be full before you can get. Why?
        # Ensure one epoch has enough experiences to train the model.
        # One epoch can have the experiences of more than one or less than one episode.

        assert self.ptr == self.max_buffer_size

        # The next two lines implement the advantage normalization trick.

        advantage_mean = torch.mean(self.buffer_advantage)
        advantage_std = torch.std(self.buffer_advantage, unbiased=False)

        if advantage_std != 0:
            self.buffer_advantage = (self.buffer_advantage - advantage_mean) / advantage_std

        data = dict(obs=self.buffer_observation,
                    act=self.buffer_action,
                    ret=self.buffer_return,
                    adv=self.buffer_advantage
                    )

        self.ptr, self.path_start_idx = 0, 0

        return data

    def discount_cumulative_sum(self, x, discount, axis=0):
        """
        Magic from rllab for computing discounted cumulative sums of vectors.
        Input:
            Vector x =
            [x0,
             x1,
             x2]
        Output:
            [x0 + discount * x1 + discount^2 * x2,
             x1 + discount * x2,
             x2]
        """
        # scipy.signal.lfilter(b, a, x, axis=-1, zi=None)
        # scipy.signal.lfilter([1], [1, -0.9], [1, 2, 3])
        # array([1.  , 2.9 , 5.61])

        # return scipy.signal.lfilter([1], [1, float(-discount)], x.flip(dims=(0,)), axis=axis)[::-1]

        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=axis)[::-1]


def test_2d_discount_cumulative_sum():
    # array([1.  , 2.9 , 5.61])
    scipy.signal.lfilter([1], [1, -0.9], [1, 2, 3], axis=0)

    # array([[1.  , 1.  ],
    #        [2.9 , 2.9 ],
    #        [5.61, 5.61]])
    scipy.signal.lfilter([1], [1, -0.9], [[1, 1], [2, 2], [3, 3]], axis=0)
    pass

