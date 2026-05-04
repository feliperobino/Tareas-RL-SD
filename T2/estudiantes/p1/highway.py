import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
import torch

from DQN import DQN, ReplayMemory, epsilon_schedule

config_dict = {
    "observation": {
        "type": "LidarObservation",
        "cells": 64,
        "maximum_range": 64,
        "normalise": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 5,
    "vehicles_count": 40,
    "duration": 100,  # [s]
    "initial_spacing": 2,
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,

    # Reward info
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
    "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "normalize_reward": True,
}


train_env = gym.make("highway-v0", config=config_dict)
obs_dim = int(np.prod(train_env.observation_space.shape))
num_actions = train_env.action_space.n
max_steps = config_dict["duration"] * config_dict["policy_frequency"]

DQN_agent = DQN(
    gamma=0.99,
    tau=0.005,
    hidden_size=128,
    num_inputs=obs_dim,
    num_outputs=num_actions,
    replay_memory_size=50000,
    batch_size=64,
    lr=1e-3,
    device=None,
)


num_episodes = 1000
eval_episodes = 20

reward_history = []
loss_history = []
collision_history = []

for episode in range(num_episodes):
    print(f"Episode {episode + 1}/{num_episodes}")
    state, info = train_env.reset()
    episode_reward = 0.0
    episode_losses = []
    crashed = False
    epsilon = epsilon_schedule(episode, num_episodes, eps_start=1.0, eps_end=0.05)

    for _ in range(max_steps):
        action = DQN_agent.calc_action(state, epsilon)
        next_state, reward, done, truncated, info = train_env.step(action)

        crashed = crashed or bool(info.get("crashed", False))
        DQN_agent.memory.push(state, action, done or truncated, next_state, reward)

        loss = DQN_agent.learn()
        if loss is not None:
            episode_losses.append(loss)

        episode_reward += reward
        state = next_state

        if done or truncated:
            break

    reward_history.append(episode_reward)
    collision_history.append(1 if crashed else 0)
    loss_history.append(float(np.mean(episode_losses)) if episode_losses else np.nan)
    print(f"  Reward: {episode_reward:.3f}, Loss: {loss_history[-1]:.3f}, Epsilon: {epsilon:.3f}, Collision: {crashed}")


torch.save(DQN_agent.policy_net.state_dict(), "dqn_highway.pth")

eval_env = gym.make("highway-v0", config=config_dict, render_mode="human")
eval_rewards = []
eval_collisions = 0

for _ in range(eval_episodes):
    state, info = eval_env.reset()
    episode_reward = 0.0
    crashed = False

    for _ in range(max_steps):
        action = DQN_agent.calc_action(state, 0.0)
        next_state, reward, done, truncated, info = eval_env.step(action)

        crashed = crashed or bool(info.get("crashed", False))
        episode_reward += reward
        state = next_state

        if done or truncated:
            break

    eval_rewards.append(episode_reward)
    eval_collisions += int(crashed)


print(f"Training episodes: {num_episodes}")
print(f"Average training reward: {np.mean(reward_history):.3f}")
print(f"Collision rate during training: {100.0 * np.mean(collision_history):.2f}%")
print(f"Average evaluation reward: {np.mean(eval_rewards):.3f}")
print(f"Evaluation collision rate: {100.0 * eval_collisions / eval_episodes:.2f}%")

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axes[0].plot(reward_history, label="Reward acumulado")
axes[0].set_ylabel("Reward")
axes[0].legend()

axes[1].plot(loss_history, label="Loss de entrenamiento", color="tab:orange")
axes[1].set_ylabel("Loss")
axes[1].legend()

collision_rate = np.cumsum(collision_history) / (np.arange(len(collision_history)) + 1)
axes[2].plot(collision_rate, label="Tasa acumulada de colisión", color="tab:red")
axes[2].set_xlabel("Episodio")
axes[2].set_ylabel("Colisión")
axes[2].legend()

plt.tight_layout()
plt.show()

eval_env.close()
train_env.close()