"""
rl_searcher.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: MON NOV 22 2021.
"""
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from lib.agents.basic_matrix_agent import BasicMatrixAgent


class RLPursuer(BasicMatrixAgent):
    def __init__(self, fov_scope, under_debug=False):

        super().__init__()

        # 11
        self.fov_scope = fov_scope
        self.under_debug = under_debug

        # 5
        self.fov_radius = int((self.fov_scope - 1) / 2)

        # 2D numpy array, such as np.array([fov_radius] * 2)
        self.own_position = np.array([self.fov_radius] * 2)

        self.local_env_matrix = None

        # Load the trained model.
        model_path = './lib/predators/data/models/model_stp_rl_pursuer.pth'

        obs_dim = self.fov_scope * self.fov_scope * 3
        act_dim = 5
        self.model = MLPActorCritic(obs_dim, act_dim, hidden_sizes=[400, 300])
        self.model.load_state_dict(torch.load(model_path))

    def is_captured(self, prey_position):

        capture_positions = prey_position + self.axial_neighbors_mask

        valid_index = []
        for idx, position in enumerate(capture_positions):
            if (position >= 0).all() and (position < self.fov_scope).all():
                valid_index.append(idx)

        capture_positions = capture_positions[valid_index, :]

        occupied_capture_positions = \
            self.local_env_matrix[capture_positions[:, 0],
                                  capture_positions[:, 1], :].sum(axis=1)

        # Valid only if collision is not allowed in the space.
        # Otherwise, more than one agents can occupy the same position.
        yes_no = True if (occupied_capture_positions > 0).all() else False

        return yes_no, capture_positions

    def preprocess_observations(self, local_env_matrix, local_env_vectors, global_own_position,
                                captured_targets_map, locked_pursuers_map):
        # local_env_matrix: channel 0-2: preys, predators, obstacles.
        # self.local_env_matrix = local_env_matrix.copy()
        #
        # local_preys = local_env_vectors['local_preys']
        #
        # for prey in local_preys:
        #     is_captured_prey, capture_positions = self.is_captured(prey)
        #     if is_captured_prey:
        #         print('ljs - Treat captured targets and locked pursuers as obstacles.')
        #         # Move captured targets and locked predators to the obstacle channel.
        #         local_env_matrix[prey[0], prey[1], [0, 2]] = [0, 1]
        #         local_env_matrix[capture_positions[:, 0], capture_positions[:, 1], 1] = 0
        #         local_env_matrix[capture_positions[:, 0], capture_positions[:, 1], 2] = 1

        # Take captured targets and locked pursuers as obstacles.
        # local_global_offset = global_own_position - self.own_position
        #
        # local_captured_targets_map = \
        #     captured_targets_map[local_global_offset[0]: local_global_offset[0] + self.fov_scope,
        #                          local_global_offset[1]: local_global_offset[1] + self.fov_scope]
        # local_locked_pursuers_map = \
        #     locked_pursuers_map[local_global_offset[0]: local_global_offset[0] + self.fov_scope,
        #                         local_global_offset[1]: local_global_offset[1] + self.fov_scope]
        #
        # local_captured_targets_x, local_captured_targets_y = np.nonzero(local_captured_targets_map)
        # local_locked_pursuers_x, local_locked_pursuers_y = np.nonzero(local_locked_pursuers_map)
        #
        # for target_x, target_y in zip(local_captured_targets_x, local_captured_targets_y):
        #     local_env_matrix[target_x, target_y, [0, 2]] = [0, 1]
        #
        # for pursuer_x, pursuer_y in zip(local_locked_pursuers_x, local_locked_pursuers_y):
        #     local_env_matrix[pursuer_x, pursuer_y, [1, 2]] = [0, 1]

        # Policy model input:
        # Shape: (H, W, 3).
        # Channel 0-2: obstacles, pursuers, evaders.
        # 0: border walls: 1.0; centered obstacle: -1.0.
        # 1: pursuers, value is the number of pursuers.
        # 2: evaders, value is the number of evaders.
        observation = local_env_matrix[:, :, [2, 1, 0]]

        # Add the current agent own position in its observation.
        observation[self.own_position[0], self.own_position[1], 1] += 1

        return observation

    def get_action(self, local_env_matrix, local_env_vectors, global_own_position,
                   captured_targets_map, locked_pursuers_map):
        # Channel 0-2: obstacles, pursuers, evaders.
        observation = self.preprocess_observations(local_env_matrix, local_env_vectors,
                                                   global_own_position, captured_targets_map, locked_pursuers_map)

        observation = np.reshape(observation, -1)

        with torch.no_grad():
            observation = torch.as_tensor(observation, dtype=torch.float32)
            action = self.model.act(observation)

        # PettingZoo action encoding:
        # 0: left, 1: right, 2: up, 3: down, 4: stay.
        # [x, y]
        # 0: [-1, 0]
        # 1: [1, 0]
        # 2: [0, 1]
        # 3: [0, -1]
        # 4: [0, 0]
        # Coordinate:
        # ---------> x
        # |
        # |
        # v y

        # Matrix action encoding:
        # 0: stay, 1: up, 2: right, 3: down, 4: left.
        # [x, y]
        # 0: [0, 0]
        # 1: [-1, 0]
        # 2: [0, 1]
        # 3: [1, 0]
        # 4: [0, -1]
        # Coordinate:
        # ---------> y
        # |
        # |
        # V x

        if action == 0:
            next_action = 4
        elif action == 1:
            next_action = 2
        elif action == 2:
            next_action = 3
        elif action == 3:
            next_action = 1
        else:
            next_action = 0

        # Check the validation of the action.
        direction = self.action_direction[next_action]
        next_position = self.own_position + direction
        if self.is_collide(next_position, local_env_matrix):
            next_action = 0

        return next_action


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(64, 64), activation=nn.ReLU):
        super().__init__()

        # Build policy depending on action space.
        self.pi = MLPCategoricalActor(obs_dim, action_dim, hidden_sizes, activation)

        # Build value function.
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)

        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)

        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]

    return nn.Sequential(*layers)


def test():
    pass


if __name__ == '__main__':
    test()
