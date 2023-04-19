import time
import itertools
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F


class ModelMLP(nn.Module):

    def __init__(self, dim_input, dim_output, hidden_sizes=(400, 300), with_layer_normalization=True):
        super(ModelMLP, self).__init__()

        self.with_layer_normalization = with_layer_normalization

        self.fc1 = nn.Linear(dim_input, hidden_sizes[0], dtype=torch.double)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], dtype=torch.double)
        self.fc3 = nn.Linear(hidden_sizes[1], dim_output, dtype=torch.double)

        self.layer_normalize_1 = nn.LayerNorm(hidden_sizes[0], dtype=torch.double)
        self.layer_normalize_2 = nn.LayerNorm(hidden_sizes[1], dtype=torch.double)
        self.layer_normalize_3 = nn.LayerNorm(dim_output, dtype=torch.double)

        self.relu = nn.ReLU()

    def forward(self, x):

        # If no layer normalization, even with BCE loss clamp, loss will be nan after some time in some experiments.

        if self.with_layer_normalization:

            x = self.relu(self.layer_normalize_1(self.fc1(x)))
            x = self.relu(self.layer_normalize_2(self.fc2(x)))

            logits = self.layer_normalize_3(self.fc3(x))

        else:

            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))

            logits = self.fc3(x)

        return logits.squeeze(dim=-1)


class ModelActorCritic(nn.Module):

    def __init__(self, model_policy, model_value):
        super(ModelActorCritic, self).__init__()

        self.model_policy = model_policy
        self.model_value = model_value

    def step(self, observation):
        """
        This function is for collect experience.

        :param observation: (batch_size, n_rows, n_columns, n_channels=3).
        :return: a tuple of two elements for collecting experience.
        """

        with torch.no_grad():

            # (batch_size, dim_action)

            action_distribution_category, _ = self.model_policy(observation)

            action = action_distribution_category.sample()

            value = self.model_value(observation.reshape(*(observation.shape[:2]), -1)).squeeze(dim=-1)

        return action, value


class ModelPolicy(nn.Module):

    def __init__(self, dim_input, dim_output, hidden_sizes=(400, 300), with_layer_normalization=True):

        super(ModelPolicy, self).__init__()

        self.model = ModelMLP(dim_input=dim_input,
                              dim_output=dim_output,
                              hidden_sizes=hidden_sizes,
                              with_layer_normalization=with_layer_normalization)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, observation, action=None):
        """
        :param observation: (batch_size, dim_observation).
        :return:
        """

        # logits = self.model(observation)
        logits = self.model(observation.reshape(*(observation.shape[:2]), -1))

        action_distribution_category = Categorical(logits=logits)

        softmax_actions = self.softmax(logits)

        # sigmoid_actions = self.sigmoid(logits)

        action_log_probability = self.log_probability_from_distribution(action_distribution_category, action)

        return action_distribution_category, action_log_probability

    @staticmethod
    def log_probability_from_distribution(distribution, action):

        return distribution.log_prob(action) if action is not None else None


def test_model_mlp():

    # (400, 300): total_parameters: 3883.
    # (16, 16): total_parameters: 543

    model = ModelMLP(dim_input=11 * 11 * 3, dim_output=5, hidden_sizes=(16, 16))

    total_parameters = 0
    for name, parameter in model.named_parameters():
        current_n_parameters = sum(parameter.shape)
        total_parameters += current_n_parameters
        print(name, current_n_parameters)

    print("total_parameters:", total_parameters)


def main():

    test_model_mlp()

    pass


if __name__ == "__main__":
    main()
    print('COMPLETE!')
    print('SUCCESS!')
