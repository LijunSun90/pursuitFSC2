import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
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


def main():
    # model_path = './data/models/model1637589954.pth'
    model_path = './data/models/model.pt'

    # <class MLPActorCritic'>
    model = torch.load(model_path)

    # Get parameters.
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # obs_dim = 11 * 11 * 3
    # act_dim = 5
    # ac_model = MLPActorCritic(obs_dim, act_dim, hidden_sizes=[400, 300])
    # ac_model.load_state_dict(torch.load(model_path))

    print("DONE!")


if __name__ == '__main__':
    main()

