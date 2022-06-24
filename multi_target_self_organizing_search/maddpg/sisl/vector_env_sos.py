import numpy as np
import gym

class SislWrapper:
    def __init__(self, sisl_env):
        self.n = sisl_env.num_agents
        self.observation_space = [sisl_env.observation_space_dict[p] for p in sisl_env.agent_ids]
        self.action_space = [sisl_env.action_space_dict[p] for p in sisl_env.agent_ids]
        self.base_action_space = self.action_space[0]
        self.sisl_env = sisl_env

    def from_dict(self, d):
        return [d[i] for i in range(self.n)]

    def step(self, action):
        action_space = self.base_action_space
        if isinstance(action_space,gym.spaces.Box):
            action = np.asarray(action)
            action = np.clip(action,action_space.low,action_space.high)
        else:
            action = np.argmax(action,axis=1)

        observation_dict, reward, done_dict, info_dict = self.sisl_env.step(action)
        observation_dict = self.from_dict(observation_dict)
        reward_dict = self.from_dict(reward)#]*self.n
        done_dict = self.from_dict(done_dict)
        info_dict = [{}]*self.n
        assert len(observation_dict[0].shape) == 1
        #observations = [obs.flatten() for obs in observation_dict]

        return observation_dict, reward_dict, done_dict, info_dict

    def reset(self):
        return self.from_dict(self.sisl_env.reset())


class EnvSOS:
    def __init__(self, env_constructor, n_targets, n_pursuers, x_size, y_size):
        self.env = SislWrapper(env_constructor(max_cycles=500,
                                               x_size=x_size, y_size=y_size,
                                               n_evaders=n_targets, n_pursuers=n_pursuers,
                                               obs_range=11,
                                               surround=False, n_catch=1,
                                               freeze_evaders=True))
        self.observation_space_dict = self.env.observation_space
        self.action_space_dict = self.env.action_space
        self.num_agents = self.env.n
        self.agent_ids = list(range(self.num_agents))
        self.dones = np.zeros((self.num_agents,), dtype=np.bool)

    def step(self, actions):
        actions = np.asarray(actions)
        actions = np.squeeze(actions)

        if not self.dones.all():
            obs_n, reward_n, done_n, info_n = self.env.step(actions)
            for i in range(self.num_agents):
                if self.dones[i]:
                    reward_n[i] = 0
        else:
            obs_n = [np.zeros(self.observation_space_dict[0].shape)]*self.num_agents
            reward_n = [0]*self.num_agents
            done_n = [True]*self.num_agents
            info_n = [{}]*self.num_agents

        cur_dones = np.array(done_n, dtype=bool)
        self.dones = self.dones | cur_dones

        obs_n = trans_list([obs_n])

        return obs_n, reward_n, self.dones, info_n

    def reset(self):
        self.dones[:] = False
        return trans_list([self.env.reset()])


def trans_list(l):
    return [[l[i][j] for i in range(len(l))] for j in range(len(l[0]))]

