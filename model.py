import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):

    def __init__(self, obs_dim=9, act_dim=4):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.actor = nn.Linear(128, act_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)

        action = torch.tanh(self.actor(x))
        value = self.critic(x)

        return action, value