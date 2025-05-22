import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

'''
Code from: https://github.com/denisyarats/pytorch_sac
'''

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning.
    Critic approximates the value function Q(s,a),
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)



if __name__ == "__main__":
    print("---- This module is not meant to be run directly. ----")

    # Testing data
    obs_dim = 4
    action_dim = 2
    hidden_dim = 256
    hidden_depth = 2

    critic = DoubleQCritic(obs_dim, action_dim, hidden_dim, hidden_depth)
    print(critic)

    # dummy observation and actions
    batch_size = 4
    obs = torch.randn(batch_size, obs_dim)
    action = torch.randn(batch_size, action_dim)

    q1, q2 = critic(obs, action)

    print(f"{q1.shape=} {q1}\n{q2.shape=} {q2=}")