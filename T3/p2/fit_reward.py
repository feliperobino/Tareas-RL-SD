import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RewardNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, output_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        return x

    def get_difference_logit(self, p_A, p_B):
        return self.forward(p_A) - self.forward(p_B)

