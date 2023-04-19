import time
import torch
import torch.nn as nn

from .experience_buffer import ExperienceBuffer


class ActorCriticAgents:

    def __init__(self, model_actor_critic, lr_policy, lr_value, train_value_iters,
                 dim_observation, dim_observation_unflatten,
                 buffer_size, n_agents, n_evaders, gamma, lam, device, world_size):

        self.train_value_iters = train_value_iters
        self.n_agents = n_agents
        self.n_evaders = n_evaders
        self.world_size = world_size

        self.device = device

        # ModelPolicy.

        self.model_actor_critic = model_actor_critic

        self.model_value = model_actor_critic.model_value

        self.model_policy = model_actor_critic.model_policy

        # Optimizer.

        self.optimizer_value = torch.optim.Adam(self.model_value.parameters(), lr=lr_value)

        self.optimizer_policy = torch.optim.Adam(self.model_policy.parameters(), lr=lr_policy)

        # Weight to balance different losses.

        self.alpha = 0.5

        # Data buffer.

        self.data_buffer = ExperienceBuffer(dim_observation=dim_observation,
                                            dim_observation_unflatten=dim_observation_unflatten,
                                            buffer_size=buffer_size,
                                            n_agents=n_agents,
                                            gamma=gamma, lam=lam, device=device)

        # Share parameters.

        self.data = None

    def update_model(self):

        self.data = self.data_buffer.get()

        # Policy STP model update.

        self.optimizer_policy.zero_grad()

        loss_policy_stp = self.compute_loss_policy_stp()

        loss_policy_stp.backward()

        self.optimizer_policy.step()

        # Value model update.

        loss_value = 0
        for _ in range(self.train_value_iters):
            self.optimizer_value.zero_grad()
            loss_value = self.compute_loss_value()
            loss_value.backward()
            self.optimizer_value.step()

        # return loss_value.item() if torch.is_tensor(loss_value) else loss_value, \
        #     loss_policy_agent_coordination.item(), \
        #     loss_policy_target.item() if torch.is_tensor(loss_policy_target) else loss_policy_target, \
        #     loss_policy_collision.item()

        return loss_value.item(), loss_policy_stp.item()

    def compute_loss_policy_stp(self):
        # Get data.

        batch_set_observation, batch_set_action, batch_set_advantage = \
            self.data['obs'], self.data['act'], self.data['adv']

        # ModelPolicy loss.

        # action_distribution_category, sigmoid_actions, action_probabilities, action_log_probability

        # _, _, _, batch_set_log_probabilities = self.model_policy(batch_set_observation, batch_set_action)

        _, batch_set_log_probabilities = self.model_policy(batch_set_observation, batch_set_action)

        loss_policy_stp = - (batch_set_log_probabilities * batch_set_advantage).mean()

        return loss_policy_stp

    def compute_loss_value(self):
        # Get data.

        batch_set_observation, batch_set_return = self.data['obs'], self.data['ret']

        # Value loss.

        batch_set_value = self.model_value(batch_set_observation.reshape(*(batch_set_observation.shape[:2]), -1))

        loss_value = ((batch_set_value - batch_set_return) ** 2).mean()

        return loss_value

