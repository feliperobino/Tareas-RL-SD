import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        action_mean = self.actor_mean(x)

        action_logstd = self.actor_logstd.expand_as(action_mean)

        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        action = torch.clamp(action, -1.0, 1.0)
        log_prob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)
        value = self.critic(x)

        return action, log_prob, entropy, value