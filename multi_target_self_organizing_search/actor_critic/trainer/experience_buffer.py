import numpy as np
import torch

from actor_critic.common.utils import combined_shape, discount_cumulative_sum
from actor_critic.common.mpi_tools import mpi_statistics_scalar


class ExperienceBuffer:
    """
    A buffer for storing trajectories experienced by an agent interacting with the
    environment, and using Generalized Advantage Estimation (GAE-Lambda) for
    calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, n_agents, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(n_agents, size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(n_agents, size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros((n_agents, size), dtype=np.float32)
        self.rew_buf = np.zeros((n_agents, size), dtype=np.float32)
        self.ret_buf = np.zeros((n_agents, size), dtype=np.float32)
        self.val_buf = np.zeros((n_agents, size), dtype=np.float32)
        self.logp_buf = np.zeros((n_agents, size), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.n_agents = n_agents

    def store(self, obs, act, rew, val, logp, agent_id):
        # Buffer has to have room so you can store.
        assert self.ptr < self.max_size
        self.obs_buf[agent_id][self.ptr] = obs
        self.act_buf[agent_id][self.ptr] = act
        self.rew_buf[agent_id][self.ptr] = rew
        self.val_buf[agent_id][self.ptr] = val
        self.logp_buf[agent_id][self.ptr] = logp

        if agent_id == (self.n_agents - 1):
            self.ptr += 1

    def finish_path(self, last_val):
        """
        Call this at the end of a trajectory, or when one gets cut off by an epoch ending.
        This looks back in the buffer to where the trajectory started, and uses rewards
        and values estimates from the whole trajectory to compute advantage estimates with
        GAE-Lambda, as well as to compute the rewards-to-for for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended because the agent reached
        a terminal state (died), and otherwise should be V(s_T), the value function estimated
        for the last state. This allows us to bootstrap the reward-to-go calculation to
        account for timesteps beyond the arbitrary horizon.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = [np.append(self.rew_buf[agent_id][path_slice], last_val[agent_id]) for agent_id in range(self.n_agents)]
        vals = [np.append(self.val_buf[agent_id][path_slice], last_val[agent_id]) for agent_id in range(self.n_agents)]

        # The next two lines implement GAE-Lambda advantage calculation.
        for agent_id in range(self.n_agents):
            deltas = rews[agent_id][:-1] + self.gamma * vals[agent_id][1:] - vals[agent_id][:-1]

            self.adv_buf[agent_id][path_slice] = discount_cumulative_sum(deltas, self.gamma * self.lam)

            # The next line computes rewards-to-go, to be targets for the value function.
            self.ret_buf[agent_id][path_slice] = discount_cumulative_sum(rews[agent_id], self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from the buffer,
        with advantages appropriately normalized (shifted to have mean zero and
        std one). Also, resets some pointers in the buffer.
        """
        # Buffer has to be full before you can get.
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # The next two lines implement the advantage normalization trick.
        for agent_id in range(self.n_agents):
            adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf[agent_id])
            self.adv_buf[agent_id] = (self.adv_buf[agent_id] - adv_mean) / adv_std

        data = dict(obs=np.concatenate(self.obs_buf, axis=0),
                    act=np.concatenate(self.act_buf, axis=0),
                    ret=np.concatenate(self.ret_buf, axis=0),
                    adv=np.concatenate(self.adv_buf, axis=0),
                    logp=np.concatenate(self.logp_buf, axis=0))

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

