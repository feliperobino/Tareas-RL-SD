import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer



class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 action_low: np.ndarray, action_high: np.ndarray,
                 hidden: int = 128):
        super().__init__()

        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

    def _squash_and_scale(self, raw: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(raw)
        return self.action_low + (squashed + 1.0) * 0.5 * (self.action_high - self.action_low)

    def _unsquash(self, action: torch.Tensor) -> torch.Tensor:
        squashed = (action - self.action_low) / (self.action_high - self.action_low) * 2.0 - 1.0
        squashed = squashed.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.atanh(squashed)

    def _log_abs_det_jacobian(self, raw: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(raw)
        scale = 0.5 * (self.action_high - self.action_low)
        return torch.log(scale * (1.0 - squashed.pow(2)) + 1e-6).sum(-1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        mean = self.actor_mean(x)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            raw = dist.sample()
            action = self._squash_and_scale(raw)
        else:
            raw = self._unsquash(action)

        raw_log_prob = dist.log_prob(raw).sum(-1)
        log_prob = raw_log_prob - self._log_abs_det_jacobian(raw)

        entropy = dist.entropy().sum(-1)
        value = self.critic(x).squeeze(-1)

        return action, log_prob, entropy, value

    def get_action_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        return self._squash_and_scale(self.actor_mean(x))




class PPO:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 act_low: np.ndarray,
                 act_high: np.ndarray,
                 cfg,
                 ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_low = torch.from_numpy(act_low).float().to(device)
        self.act_high = torch.from_numpy(act_high).float().to(device)
        self.cfg = cfg
        self.agent = Agent(self.obs_dim, self.act_dim, self.act_low, self.act_high).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.cfg.learning_rate, eps=1e-5)
        self.global_step = 0
        self.global_episodes = 0

    def compute_gae(self, rewards, values, dones, next_value, next_done):
        T = len(rewards)
        advs = torch.zeros(T)
        last = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                nnt, nval = 1.0 - next_done, next_value
            else:
                nnt, nval = 1.0 - dones[t + 1], values[t + 1]
            delta   = rewards[t] + self.cfg.gamma * nval * nnt - values[t]
            last    = delta + self.cfg.gamma * self.cfg.gae_lambda * nnt * last
            advs[t] = last
        return advs, advs + values


    def collect_rollouts(self, env):
        # Tensors for the updates
        rollout_observations = torch.zeros(self.cfg.num_steps, self.obs_dim)
        rollout_actions = torch.zeros(self.cfg.num_steps, self.act_dim)
        rollout_log_probs = torch.zeros(self.cfg.num_steps)
        rollout_rewards = torch.zeros(self.cfg.num_steps)
        rollout_dones = torch.zeros(self.cfg.num_steps)
        rollout_values = torch.zeros(self.cfg.num_steps)

        # Metrics
        episode_returns = []
        episode_costs = []

        # Initialize env
        obs, _ = env.reset()
        obs = torch.from_numpy(obs).float()

        current_episode_return = 0.0
        current_episode_cost = 0.0


        # Collect experience
        for step in range(self.cfg.num_steps):
            self.global_step += 1
            rollout_observations[step] = obs

            # Sample an action
            with torch.no_grad():
                action, log_prob, _, value = self.agent.get_action_and_value(obs.to(device))
                action = torch.clip(action, self.act_low, self.act_high)

            rollout_actions[step] = action.cpu()
            rollout_log_probs[step] = log_prob.cpu()
            rollout_values[step] = value.cpu()

            # Step the environment
            next_obs_np, reward, cost, terminated, truncated, info = env.step(action.cpu().numpy())

            # Check if the episode was finished
            episode_ended = bool(terminated) or bool(truncated)

            # Get metrics
            current_episode_return += float(reward)
            current_episode_cost += float(cost)

            # Store scaled reward for the critic, and the done flag for GAE
            rollout_rewards[step] = torch.tensor(float(reward))
            rollout_dones[step] = torch.tensor(float(episode_ended))

            # If the episode ended, log its return and reset
            if episode_ended:
                self.global_episodes +=1
                self.global_step = 0
                episode_returns.append(current_episode_return)
                episode_costs.append(current_episode_cost)
                print('Episode: {} | Returns: {} | Costs: {}'.format(self.global_episodes, current_episode_return, current_episode_cost))

                current_episode_return = 0.0
                current_episode_cost = 0
                next_obs_np, _ = env.reset()

            obs = torch.from_numpy(next_obs_np).float()
            current_done = torch.tensor(float(episode_ended))

        # Bootstrap to get the final value
        with torch.no_grad():
            bootstrap_value = self.agent.get_value(obs.to(device)).cpu()


        return {'observations': rollout_observations,
                'actions': rollout_actions,
                'log_probs': rollout_log_probs,
                'rewards': rollout_rewards,
                'dones': rollout_dones,
                'values': rollout_values,
                'bootstrap_value': bootstrap_value,
                'last_done': current_done}, (episode_returns, episode_costs)


    def update(self, rollouts):
        rollout_observations = rollouts['observations']
        rollout_actions = rollouts['actions']
        rollout_log_probs = rollouts['log_probs']
        rollout_rewards = rollouts['rewards']
        rollout_dones = rollouts['dones']
        rollout_values = rollouts['values']
        bootstrap_value = rollouts['bootstrap_value']
        last_done = rollouts['last_done']


        advantages, returns = self.compute_gae(
            rewards=rollout_rewards,
            values=rollout_values,
            dones=rollout_dones,
            next_value=bootstrap_value,
            next_done=last_done,
        )

        # Flatten and move to GPU
        b_observations = rollout_observations.to(device)
        b_actions = rollout_actions.to(device)
        b_log_probs = rollout_log_probs.to(device)
        b_values = rollout_values.to(device)
        b_advantages = advantages.to(device)
        b_returns = returns.to(device)

        # PPO Updates
        batch_indices = np.arange(self.cfg.batch_size)

        policy_losses, value_losses, entropies = [], [], []
        approx_kls, clip_fractions = [], []

        for epoch in range(self.cfg.update_epochs):
            np.random.shuffle(batch_indices)

            for mb_start in range(0, self.cfg.batch_size, self.cfg.mini_batch_size):
                mb_end = mb_start + self.cfg.mini_batch_size
                mb_indices = batch_indices[mb_start:mb_end]

                # Forward pass on this mini-batch
                _, new_log_prob, entropy, new_value = self.agent.get_action_and_value(
                    b_observations[mb_indices], b_actions[mb_indices]
                )

                # Loss calculation
                log_ratio = new_log_prob - b_log_probs[mb_indices]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.cfg.clip_coef).float().mean().item()
                    approx_kls.append(approx_kl)
                    clip_fractions.append(clip_frac)

                # Normalise advantages
                mb_advantages = b_advantages[mb_indices]
                if self.cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / \
                                    (mb_advantages.std() + 1e-8)

                # Policy loss
                surrogate_unclipped = -mb_advantages * ratio
                surrogate_clipped = -mb_advantages * ratio.clamp(1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)
                policy_loss = torch.max(surrogate_unclipped, surrogate_clipped).mean()

                # Value loss
                new_value = new_value.view(-1)
                if self.cfg.clip_vloss:
                    value_loss_unclipped = (new_value - b_returns[mb_indices]) ** 2
                    value_clipped = (b_values[mb_indices] +
                                     (new_value - b_values[mb_indices]).clamp(-self.cfg.clip_coef, self.cfg.clip_coef))
                    value_loss_clipped = (value_clipped - b_returns[mb_indices]) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((new_value - b_returns[mb_indices]) ** 2).mean()

                entropy_loss = entropy.mean()
                total_loss = policy_loss - self.cfg.ent_coef * entropy_loss + self.cfg.vf_coef * value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())

        policy_loss = np.array(policy_losses).mean()
        value_loss = np.array(value_losses).mean()
        entropy = np.array(entropies).mean()
        print('Policy loss: {} | Value loss: {} | entropy: {}'.format(policy_loss, value_loss, entropy))

        return policy_loss, value_loss, entropy


    def train(self, env):
        self.agent.train()
        total_updates  = self.cfg.total_timesteps // self.cfg.batch_size

        global_episode_returns = []
        global_episode_costs = []
        policy_losses = []
        value_losses = []
        entropies = []

        # Iterate between
        for update in range(1, total_updates + 1):
            print('Update: {} | {} '.format(update, total_updates))
            # Collect rollouts
            rollouts, (episode_returns, episode_costs) = self.collect_rollouts(env)

            # Update
            policy_loss, value_loss, entropy = self.update(rollouts)

            # Metrics
            global_episode_returns += episode_returns
            global_episode_costs += episode_costs
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)


        return global_episode_returns, global_episode_costs, policy_losses, value_losses, entropies



    def evaluate(self, env, n_episodes: int = 10):
        self.agent.eval()
        for ep in range(1, n_episodes + 1):
            obs, _ = env.reset()
            obs = torch.from_numpy(obs).float()

            ep_return = 0.0
            ep_cost = 0.0
            done = False
            steps = 0
            while not done:
                with torch.no_grad():
                    action = self.agent.get_action_deterministic(obs.to(device))
                    action = torch.clip(action, self.act_low, self.act_high)

                next_obs_np, reward, cost, terminated, truncated, info = env.step(action.cpu().numpy())
                ep_return += float(reward)
                ep_cost += float(cost)
                steps += 1
                done = bool(terminated) or bool(truncated)

                obs = torch.from_numpy(next_obs_np).float()

            print('Eval Episode {} | Returns: {} | Costs: {}'.format(ep, ep_return, ep_cost))



@dataclass
class PPOConfig:
    seed = 1
    total_timesteps = 300_000

    learning_rate = 3e-4
    anneal_rl = True

    num_steps = 10_000
    num_minibatches = 32
    update_epochs = 10

    gamma = 0.99
    gae_lambda = 0.95

    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    norm_adv = True
    target_kl = None

    batch_size = num_steps
    mini_batch_size = batch_size // num_minibatches



if __name__ == "__main__":
    import safety_gymnasium

    env_id = "SafetyCarGoal2-v0"
    env = safety_gymnasium.make(env_id)
    ppo = PPO(cfg=PPOConfig, obs_dim=env.observation_space.shape[0],
              act_dim=env.action_space.shape[0],
              act_low=env.action_space.low,
              act_high=env.action_space.high)

    episode_returns, episode_costs, policy_losses, value_losses, entropies = ppo.train(safety_gymnasium.make(env_id))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(episode_returns, label='returns')
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(episode_costs, label='costs')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(policy_losses, label='Policy Loss')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(value_losses, label='Value Loss')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(entropies, label='Entropy')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


    ppo.evaluate(safety_gymnasium.make(env_id, render_mode='human'))