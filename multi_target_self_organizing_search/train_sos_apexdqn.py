import numpy as np
import random
import sys
import argparse

from ray import tune
# PettingZoo API is not directly compatible with rllib, but it can be converted into an rllib MultiAgentEnv.
from ray.tune.registry import register_env
# import rllib pettingzoo interface
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog

from pursuit_game.supersuit import flatten_v0
from pursuit_game import pursuit_vsos as pursuit

from apex_dqn.custom_model import MLPModel
from apex_dqn.custom_callbacks import CustomPursuitCallbacks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=11)
    parser.add_argument("--n_pursuers", type=int, default=8)
    parser.add_argument("--n_targets", type=int, default=50)

    return parser.parse_args()


def train(arg_list):
    # Set seed for the search algorithm/schedulers:
    seed = arg_list.seed + 10000
    np.random.seed(seed)
    random.seed(arg_list.seed)

    # register that way to make the environment under an rllib name
    register_env('pursuit', lambda env_config: PettingZooEnv(env_creator(env_config)))
    # now you can use `pursuit_vsop` as an environment

    policies = {"policy_0": gen_policy(0, arg_list)}
    policy_ids = list(policies.keys())

    tune.run(
        "APEX",
        name="ADQN",
        stop={"episodes_total": 60000},
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="results/sos_apexdqn/",
        # Redirect stdout and stderror to files.
        log_to_file=True,
        config={
            # Environment specific.
            "env": "pursuit",

            # General.
            "log_level": "INFO",
            "num_gpus": 1,
            "num_workers": 8,
            "num_envs_per_worker": 8,
            # "num_gpus": 0,
            # "num_workers": 1,
            # "num_envs_per_worker": 1,
            "seed": seed,  # Set seed for worker.
            # "env_config": {"seed": seed},  # Set seed for env creator. Comment or not do not influence result.
            "env_config": {"n_targets": arg_list.n_targets, "n_pursuers": arg_list.n_pursuers},
            "gamma": .99,

            "callbacks": CustomPursuitCallbacks,

            # Method specific.
            "rollout_fragment_length": 32,
            "train_batch_size": 1024,
            "target_network_update_freq": 50000,
            "timesteps_per_iteration": 25000,
            "learning_starts": 80000,
            "compress_observations": False,

            "dueling": True,
            "double_q": True,
            "num_atoms": 1,
            "noisy": False,
            "n_step": 3,
            "lr": 1e-5,
            "adam_epsilon": 1.5e-4,   # default: 1e-8
            "buffer_size": int(1e5),
            "exploration_config": {
                "final_epsilon": 0.01,
                "epsilon_timesteps": 200000,
            },
            "prioritized_replay": True,
            "prioritized_replay_alpha": 0.5,
            "prioritized_replay_beta": 0.4,
            "final_prioritized_replay_beta": 1.0,
            "prioritized_replay_beta_annealing_timesteps": 2000000,

            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (lambda agent_id: policy_ids[0]),
            },
        },
    )


# define how to make the environment. This way takes an optional environment config.
def env_creator(env_config):
    env = pursuit.env(max_cycles=500,
                      x_size=40, y_size=40,
                      n_evaders=env_config["n_targets"], n_pursuers=env_config["n_pursuers"],
                      obs_range=11,
                      surround=False, n_catch=1,
                      freeze_evaders=True)

    if "seed" in env_config.keys():
        seed = env_config["seed"]
        env.seed(seed)

    env = flatten_v0(env)
    return env


def gen_policy(i, arg_list):
    seed = arg_list.seed
    env_config = {"n_targets": arg_list.n_targets, "n_pursuers": arg_list.n_pursuers}

    # ModelCatalog.register_custom_model("MLPModel", MLPModel)
    ModelCatalog.register_custom_model("MLPModel", MLPModel)

    config = {
        "model": {
            "custom_model": "MLPModel",
        },
        "gamma": 0.99,
    }

    test_env = PettingZooEnv(env_creator(env_config))
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    return None, obs_space, act_space, config


if __name__ == "__main__":
    arg_list = parse_args()
    train(arg_list)
    print("DONE!")
