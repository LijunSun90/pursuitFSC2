import torch
from .models import ModelMLP, ModelPolicy
from .utils.util import update_linear_schedule


class MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.
    """
    def __init__(self, args, obs_shape, cent_obs_shape, action_dim, device):

        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_shape = obs_shape
        self.share_obs_shape = cent_obs_shape
        self.action_dim = action_dim

        self.device = device

        self.actor = ModelPolicy(self.obs_shape, self.action_dim, device=self.device).to(self.device)

        self.critic = ModelMLP(cent_obs_shape, 1).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):

        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)

        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs):

        obs = torch.as_tensor(obs, dtype=torch.double, device=self.device)
        cent_obs = torch.as_tensor(cent_obs, dtype=torch.double, device=self.device)

        actions, action_log_probs = self.actor(obs)

        values = self.critic(cent_obs)

        return values, actions, action_log_probs

    def get_values(self, cent_obs):

        cent_obs = torch.as_tensor(cent_obs, dtype=torch.double, device=self.device)

        values = self.critic(cent_obs)

        return values

    def evaluate_actions(self, cent_obs, obs, action):

        obs = torch.as_tensor(obs, dtype=torch.double, device=self.device)
        cent_obs = torch.as_tensor(cent_obs, dtype=torch.double, device=self.device)

        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, action)

        values = self.critic(cent_obs)

        return values, action_log_probs, dist_entropy

    def act(self, obs):

        obs = torch.as_tensor(obs, dtype=torch.double, device=self.device)

        actions, _ = self.actor(obs)

        return actions
