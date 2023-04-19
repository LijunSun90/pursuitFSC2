import os
import torch

from lib.mappo.models import ModelPolicy


class RLPursuerMAPPO:

    def __init__(self, env, data_log_folder, device=torch.device("cpu")):

        self.env = env
        self.data_log_folder = data_log_folder
        self.device = device

        self.policy_model = self.load_policy_model()

        self.role = "RL"
        self.swarm_global_positions = None

    def load_policy_model(self):

        model_filename = os.path.join(self.data_log_folder, "Epoch2000x14sop_mappo_model_actor.pth")
        print("Load model from:", model_filename)

        policy_model = ModelPolicy(dim_input=11 * 11 * 3, dim_output=5, device=self.device)

        policy_model.load_state_dict(torch.load(model_filename, map_location=self.device))

        return policy_model

    def get_action(self):

        _, current_observation, env_vectors, _, _ = self.env.last(is_evader=False)
        current_observation = current_observation.reshape(current_observation.shape[0], -1)
        current_observation = torch.tensor(current_observation, dtype=torch.double)
        action, _ = self.policy_model(current_observation)
        action = action.numpy()

        self.swarm_global_positions = env_vectors["all_pursuers"]

        return action

