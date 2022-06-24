import numpy as np
import gym
from multiprocessing import Process, Queue
import signal
import os

class SislWrapper:
    def __init__(self,sisl_env):
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

class VectorEnv:
    def __init__(self,env_constructor,num_envs):
        self.envs = [SislWrapper(env_constructor()) for _ in range(num_envs)]
        self.env = self.envs[0]
        self.observation_space_dict = self.env.observation_space
        self.action_space_dict = self.env.action_space
        self.num_agents = self.env.n
        self.agent_ids = list(range(self.num_agents))
        self.dones = np.zeros((num_envs,self.num_agents),dtype=np.bool)
        self.num_envs = num_envs

    def step(self,actions):
        actions = np.asarray(actions)
        if len(actions.shape) < 3:
            actions = np.expand_dims(actions,2)
        actions = np.transpose(actions,axes=(1,0,2))

        obs_ns = []
        reward_ns = []
        dones_ns = []
        infos_ns = []
        for act,env,was_done in zip(actions,self.envs,self.dones):
            if not was_done.all():
                obs,rew,done,info = env.step(act)
                for i in range(self.num_agents):
                    if was_done[i]:
                        rew[i] = 0
            else:
                obs = [np.zeros(self.observation_space_dict[0].shape)]*self.num_agents
                rew = [0]*self.num_agents
                done = [True]*self.num_agents
                info = [{}]*self.num_agents
            obs_ns.append(obs)
            reward_ns.append(rew)
            infos_ns.append(info)
            dones_ns.append(done)
        cur_dones = np.array(dones_ns,dtype=np.bool)
        #new_dones = cur_dones & ~self.dones
        self.dones = self.dones | cur_dones
        obs_ns = trans_list(obs_ns)
        reward_ns = trans_list(reward_ns)
        infos_ns = trans_list(infos_ns)

        return obs_ns,reward_ns,np.all(self.dones,axis=1),infos_ns

    def reset(self):
        self.dones[:] = False
        return trans_list([env.reset() for env in self.envs])

def run_multiproc_env(env_constructor,num_envs,in_queue,out_queue):
    vec_env = VectorEnv(env_constructor,num_envs)
    while True:
        (val, action) = in_queue.get()
        if val == 'reset':
            obs = vec_env.reset()
            out_queue.put((False,obs))
        elif val == 'step':
            data = vec_env.step(action)
            out_queue.put((True,data))
        elif val == 'term':
            return
        else:
            assert False, "val must be reset, step or term"

def sig_handle(signal_object, argvar):
    raise KeyboardInterrupt()

def init_parallel_env():
    signal.signal(signal.SIGINT, sig_handle)
    signal.signal(signal.SIGTERM, sig_handle)

def trans_list(l):
    return [[l[i][j] for i in range(len(l))] for j in range(len(l[0]))]

class ParallelVectorEnv:
    def __init__(self, env_constructor, num_envs, num_cpus):
        assert num_envs % num_cpus == 0
        self.envs_per_cpu = num_envs//num_cpus
        self.num_cpus = num_cpus
        self.in_queues = [Queue(4) for _ in range(num_cpus)]
        self.out_queues = [Queue(4) for _ in range(num_cpus)]

        self.procs = []
        for i in range(num_cpus):
            p = Process(target=run_multiproc_env, args=(env_constructor,self.envs_per_cpu,self.in_queues[i],self.out_queues[i]))
            p.start()
            self.procs.append(p)

        self.env = SislWrapper(env_constructor())
        self.observation_space_dict = self.env.observation_space
        self.action_space_dict = self.env.action_space
        self.num_agents = self.env.n
        self.agent_ids = list(range(self.num_agents))
        self.num_envs = num_envs

    def reset(self):
        for q in self.in_queues:
            q.put(('reset',None))
        arglist = [q.get()[1] for q in self.out_queues]
        new_l = trans_list(arglist)
        fin_list = [sum(l,[]) for l in new_l]
        return fin_list

    def step(self, actions):
        actions = np.transpose(np.asarray(actions),axes=(1,0,2))
        bs = self.envs_per_cpu
        sectioned_action = [actions[i*bs:(i+1)*bs] for i in range(self.num_cpus)]
        for act,q in zip(sectioned_action,self.in_queues):
            act = np.transpose(act,axes=((1,0,2)))
            q.put(('step',act))
        results = [q.get()[1] for q in self.out_queues]
        trans_results = trans_list(results)
        fin_results = [None]*4
        for i in range(len(trans_results)):
            if i != 2:
                trans = trans_list(trans_results[i])
                fin_results[i] = [sum(l,[]) for l in trans]
        fin_results[2] = np.concatenate(trans_results[2],axis=0)

        return fin_results

    def close(self):
        for q in self.in_queues:
            q.put(('term',None))
        return [p.join() for p in self.procs]
