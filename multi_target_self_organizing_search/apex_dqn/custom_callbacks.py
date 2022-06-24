"""
Author: Lijun Sun.
Date: 2022.
"""
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict


class CustomPursuitCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, \
            "ERROR: `on_episode_start()` callback should be called right " \
            "after env reset!"

        # capture_rates.
        # episode.hist_data["capture_rates"] = []

        # n_collision_events_per_multiagent_step
        episode.user_data["n_collision_events_per_multiagent_step"] = []
        # episode.hist_data["n_collision_events_per_multiagent_step"] = []

        # n_agents_collide_with_others_per_multiagent_step
        episode.user_data["n_agents_collide_with_others_per_multiagent_step"] = []
        # episode.hist_data["n_agents_collide_with_others_per_multiagent_step"] = []

        # episode.hist_data["collisions"] = []
        # episode.hist_data["collisions_per_agent"] = []

        # collisions_with_obstacles
        # episode.hist_data["collisions_with_obstacles"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Make sure this episode is ongoing.
        assert episode.length > 0, \
            "ERROR: `on_episode_step()` callback should not be called right " \
            "after env reset!"

        # <ray.rllib.env.multi_agent_env.MultiAgentEnvWrapper object at 0x7fc16117d9d0>
        # <PettingZooEnv instance>
        pettingzoo_instances = base_env.get_unwrapped()[0]
        # aec_observation_lambda<pursuit_v3>
        aec_observation_lambda_instance = pettingzoo_instances.env
        pettingzoo_env = aec_observation_lambda_instance.env
        # <class 'pettingzoo.utils.wrappers.order_enforcing.OrderEnforcingWrapper'>
        order_enforcing_wrapped_env = pettingzoo_env.env
        # <class 'pettingzoo.sisl.pursuit.pursuit.raw_env'>
        # raw_env = order_enforcing_wrapped_env.__getattr__("unwrapped")
        # if no use "flatten_v0()".
        raw_env = order_enforcing_wrapped_env.env
        # <class 'pettingzoo.sisl.pursuit.pursuit_base.Pursuit'>
        env = raw_env.env

        # n_collision_events_per_multiagent_step
        n_collision_events_per_multiagent_step = env.n_collision_events_per_multiagent_step
        episode.user_data["n_collision_events_per_multiagent_step"].append(n_collision_events_per_multiagent_step)

        # n_agents_collide_with_others_per_multiagent_step
        n_agents_collide_with_others_per_multiagent_step = env.n_agents_collide_with_others_per_multiagent_step
        episode.user_data["n_agents_collide_with_others_per_multiagent_step"].append(n_agents_collide_with_others_per_multiagent_step)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        # <ray.rllib.env.multi_agent_env.MultiAgentEnvWrapper object at 0x7fc16117d9d0>
        # <PettingZooEnv instance>
        pettingzoo_instances = base_env.get_unwrapped()[0]
        # aec_observation_lambda<pursuit_v3>
        aec_observation_lambda_instance = pettingzoo_instances.env
        pettingzoo_env = aec_observation_lambda_instance.env
        # <class 'pettingzoo.utils.wrappers.order_enforcing.OrderEnforcingWrapper'>
        order_enforcing_wrapped_env = pettingzoo_env.env
        # <class 'pettingzoo.sisl.pursuit.pursuit.raw_env'>
        # raw_env = order_enforcing_wrapped_env.__getattr__("unwrapped")
        # if no use "flatten_v0()".
        raw_env = order_enforcing_wrapped_env.env
        # <class 'pettingzoo.sisl.pursuit.pursuit_base.Pursuit'>
        env = raw_env.env

        # capture_rate.
        capture_rate = sum(env.evaders_gone) / len(env.evaders_gone)

        episode.custom_metrics["capture_rate"] = capture_rate
        # episode.hist_data["capture_rates"].append(capture_rate)

        # collisions_with_obstacles
        collisions_with_obstacles = env.n_collision_with_obstacles
        episode.custom_metrics["collisions_with_obstacles"] = collisions_with_obstacles
        # episode.hist_data["collisions_with_obstacles"].append(collisions_with_obstacles)

        # n_collision_events_per_multiagent_step
        # episode.hist_data["n_collision_events_per_multiagent_step"] = episode.user_data["n_collision_events_per_multiagent_step"]

        # collisions
        collisions = sum(episode.user_data["n_collision_events_per_multiagent_step"])
        episode.custom_metrics["collisions"] = collisions
        # episode.hist_data["collisions"].append(episode.custom_metrics["collisions"])

        # n_agents_collide_with_others_per_multiagent_step
        # episode.hist_data["n_agents_collide_with_others_per_multiagent_step"] = \
        #     episode.user_data["n_agents_collide_with_others_per_multiagent_step"]

        # collisions_per_agent
        collisions_per_agent = sum(episode.user_data["n_agents_collide_with_others_per_multiagent_step"])
        episode.custom_metrics["collisions_per_agent"] = collisions_per_agent
        # episode.hist_data["collisions_per_agent"].append(collisions_per_agent)

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

