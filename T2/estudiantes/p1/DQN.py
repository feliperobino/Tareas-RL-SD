import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, done, next_state, reward):
        self.memory.append((state, action, done, next_state, reward))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, dones, next_states, rewards = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.memory)



class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=128):
        super().__init__()
        self.num_inputs = num_inputs
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2), num_outputs),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return self.net(x)


class DQN:
    def __init__(
        self,
        gamma=0.99,
        tau=0.005,
        hidden_size=128,
        num_inputs=128,
        num_outputs=5,
        replay_memory_size=50000,
        batch_size=64,
        lr=1e-3,
        device=None,
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(num_inputs, num_outputs, hidden_size).to(self.device)
        self.target_net = QNetwork(num_inputs, num_outputs, hidden_size).to(self.device)
        self.num_actions = num_outputs
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(replay_memory_size)

    # Epsilon-greedy policy
    def calc_action(self, state, epsilon):
        self.policy_net.eval()
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return int(torch.argmax(q_values, dim=1).item())

    # Soft update for target network
    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.policy_net.train()
        states, actions, dones, next_states, rewards = self.memory.sample(self.batch_size)
        states = states.to(self.device).squeeze()
        actions = actions.to(self.device)
        dones = dones.to(self.device)
        next_states = next_states.to(self.device).squeeze()
        rewards = rewards.to(self.device)

        # Current Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        self.soft_update()
        return loss.item()

def epsilon_schedule(episode, max_episodes, eps_start=1.0, eps_end=0.00):
    frac = max(0.0, 1.0 - episode / max_episodes)
    return eps_end + (eps_start - eps_end) * frac

