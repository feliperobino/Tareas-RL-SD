# adaptado de cleanrl\ppo_continuous_action.py
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from dwpt_rl import DWPTContinuousShareEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    seed: int = 1
    cuda: bool = True

    total_timesteps: int = 500_000

    learning_rate: float = 3e-4

    num_steps: int = 1024

    gamma: float = 0.99
    gae_lambda: float = 0.95

    num_minibatches: int = 8
    update_epochs: int = 10

    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    max_grad_norm: float = 0.5

    anneal_lr: bool = True

    save_model: bool = True

    # runtime filled
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class DWPTEnvAdapter(gym.Wrapper):
    """Adapter to make DWPTContinuousShareEnv compatible with gymnasium API."""
    def reset(self, seed=None, options=None):
        # Call parent reset ignoring options (DWPTContinuousShareEnv doesn't support it)
        obs, info = self.env.reset(seed=seed)
        # Convert obs to numpy array if dict
        if isinstance(obs, dict):
            obs = np.array(list(obs.values()), dtype=np.float32)
        else:
            obs = np.array(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Convert obs to numpy array if dict
        if isinstance(obs, dict):
            obs = np.array(list(obs.values()), dtype=np.float32)
        else:
            obs = np.array(obs, dtype=np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), info


def make_env(system_knowledge='full'):
    env = DWPTContinuousShareEnv(system_knowledge=system_knowledge)
    env = DWPTEnvAdapter(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeObservation(env)
    return env


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
        logprob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)
        return action, logprob, entropy, self.critic(x)
    
    def get_deterministic_action(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        action_mean = self.actor_mean(x)
        return torch.clamp(action_mean, -1.0, 1.0)


def train_ppo(args, system_knowledge='full'):
    # runtime batch sizes will be computed after env creation
    run_name = f"ppo_dwpt__seed{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup (single env for easier debugging)
    env = make_env(system_knowledge=system_knowledge)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # runtime batch sizes
    args.batch_size = int(args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = int(args.total_timesteps // args.batch_size)

    agent = Agent(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup (single-env rollout)
    obs = torch.zeros((args.num_steps, obs_dim)).to(device)
    actions = torch.zeros((args.num_steps, action_dim)).to(device)
    logprobs = torch.zeros(args.num_steps).to(device)
    rewards = torch.zeros(args.num_steps).to(device)
    dones = torch.zeros(args.num_steps).to(device)
    values = torch.zeros(args.num_steps).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, info = env.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = 0

    # HISTORIES
    episode_returns = []
    value_losses = []
    policy_losses = []
    entropy_losses = []
    mean_rewards = []

    mean_eta = []
    mean_pout = []
    mean_phase = []
    
    # Episode counter for progress printing
    num_episodes = 0
    print(f"\n{'='*80}")
    print(f"Starting PPO training")
    print(f"  Total timesteps: {args.total_timesteps}")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Steps per rollout: {args.num_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {device}")
    print(f"{'='*80}\n")
    
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # rollout accumulators for physics metrics
        rollout_eta = []
        rollout_pout = []
        rollout_phase = []

        for step in range(0, args.num_steps):

            global_step += 1
            obs[step] = next_obs
            dones[step] = float(next_done)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                # clamp actions to [-1,1] before env.step
                action = torch.clamp(action, -1.0, 1.0)
                values[step] = value.flatten()
            actions[step] = action.squeeze()
            logprobs[step] = logprob.squeeze()

            # execute the game and log data.
            action_np = action.cpu().numpy().squeeze()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            step_info = info

            next_done = bool(terminated or truncated)
            if next_done:
                next_obs, _ = env.reset()

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)

            # collect physics metrics from info or env attributes
            if step_info and "eta" in step_info:
                rollout_eta.append(step_info.get("eta"))
            elif hasattr(env.unwrapped, "eta"):
                rollout_eta.append(getattr(env.unwrapped, "eta"))

            if step_info and "p_out" in step_info:
                rollout_pout.append(step_info.get("p_out"))
            elif hasattr(env.unwrapped, "pout"):
                rollout_pout.append(getattr(env.unwrapped, "pout"))

            if step_info and "phase_active_mag_deg" in step_info:
                rollout_phase.append(step_info.get("phase_active_mag_deg"))
            elif hasattr(env.unwrapped, "phase_active_mag_deg"):
                rollout_phase.append(getattr(env.unwrapped, "phase_active_mag_deg"))

            if step_info and "episode" in step_info:
                num_episodes += 1
                episode_returns.append(step_info["episode"]["r"])
                # Print every 10 episodes
                if num_episodes % 10 == 0:
                    avg_return = np.mean(episode_returns[-10:])
                    print(f"[Episode {num_episodes}] global_step={global_step}, "
                          f"iteration={iteration}/{args.num_iterations}, "
                          f"episodic_return={step_info['episode']['r']:.2f}, "
                          f"avg_return_last_10={avg_return:.2f}")
                writer.add_scalar("charts/episodic_return", step_info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", step_info["episode"]["l"], global_step)
                # physics logging
                if hasattr(env.unwrapped, "pout"):
                    writer.add_scalar("physics/pout", float(env.unwrapped.pout), global_step)
                if hasattr(env.unwrapped, "eta"):
                    writer.add_scalar("physics/eta", float(env.unwrapped.eta), global_step)
                if hasattr(env.unwrapped, "phase_active_mag_deg"):
                    writer.add_scalar("physics/phase", float(env.unwrapped.phase_active_mag_deg), global_step)
                if hasattr(env.unwrapped, "share"):
                    writer.add_scalar("physics/share", float(env.unwrapped.share), global_step)
                if hasattr(env.unwrapped, "f_sw"):
                    writer.add_scalar("physics/f_sw", float(env.unwrapped.f_sw), global_step)

        # after rollout collection, record mean rollout reward and physics means
        try:
            mean_rewards.append(rewards.mean().item())
        except Exception:
            mean_rewards.append(float(torch.mean(rewards).item()))

        if len(rollout_eta) > 0:
            mean_eta.append(float(np.mean(rollout_eta)))
        else:
            mean_eta.append(float('nan'))
        if len(rollout_pout) > 0:
            mean_pout.append(float(np.mean(rollout_pout)))
        else:
            mean_pout.append(float('nan'))
        if len(rollout_phase) > 0:
            mean_phase.append(float(np.mean(rollout_phase)))
        else:
            mean_phase.append(float('nan'))

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - float(next_done)
                    nextvalues = next_value.squeeze()
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # batch is already flat for single env rollouts
        b_obs = obs
        b_logprobs = logprobs
        b_actions = actions
        b_advantages = advantages
        b_returns = returns
        b_values = values

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args.minibatch_size > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # record PPO losses (use last computed batch losses)
        try:
            value_losses.append(v_loss.item())
        except Exception:
            value_losses.append(float('nan'))
        try:
            policy_losses.append(pg_loss.item())
        except Exception:
            policy_losses.append(float('nan'))
        try:
            entropy_losses.append(entropy_loss.item())
        except Exception:
            entropy_losses.append(float('nan'))

        # record rewards and a few key losses for plotting
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)

        # checkpoint every 10 iterations
        if iteration % 10 == 0:
            torch.save(agent.state_dict(), "ppo_dwpt.pt")
            elapsed = time.time() - start_time
            print(f"[Iteration {iteration}/{args.num_iterations}] "
                  f"Global step: {global_step}, "
                  f"Episodes: {num_episodes}, "
                  f"Elapsed: {elapsed:.1f}s, "
                  f"Last policy loss: {policy_losses[-1]:.4f}, "
                  f"Last value loss: {value_losses[-1]:.4f}")

    if args.save_model:
        model_path = f"runs/{run_name}/ppo_dwpt.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    env.close()
    writer.close()
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Total timesteps: {global_step}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    if len(episode_returns) > 0:
        print(f"  Avg episode return: {np.mean(episode_returns):.2f}")
        print(f"  Max episode return: {np.max(episode_returns):.2f}")
    else:
        print("  Avg episode return: n/a (no completed episodes recorded)")
        print("  Max episode return: n/a (no completed episodes recorded)")
    print(f"{'='*80}\n")

    history = {
        "episode_returns": episode_returns,
        "value_losses": value_losses,
        "policy_losses": policy_losses,
        "entropy_losses": entropy_losses,
        "mean_rewards": mean_rewards,
        "mean_eta": mean_eta,
        "mean_pout": mean_pout,
        "mean_phase": mean_phase,
    }

    return agent, history


def evaluate_policy(agent, env, device, num_episodes=5):
    agent.eval()

    trajectories = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.Tensor(obs).to(device)
        done = False

        ep_return = 0.0

        # per-episode trajectory storage
        freqs = []
        shares = []
        powers = []
        etas = []
        phases = []
        positions = []

        while not done:
            with torch.no_grad():
                action = agent.get_deterministic_action(obs)
            action_np = action.cpu().numpy().squeeze()
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated or truncated)
            obs = torch.Tensor(obs).to(device)

            ep_return += float(reward)

            # collect trajectory info from env
            # safely read attributes from wrapped envs (NormalizeObservation, RecordEpisodeStatistics)
            def _env_attr(e, name):
                if hasattr(e, name):
                    return getattr(e, name)
                if hasattr(e, 'unwrapped') and hasattr(e.unwrapped, name):
                    return getattr(e.unwrapped, name)
                return float('nan')

            freqs.append(_env_attr(env, 'f_sw'))
            shares.append(_env_attr(env, 'share'))
            powers.append(_env_attr(env, 'pout'))
            etas.append(_env_attr(env, 'eta'))
            phases.append(_env_attr(env, 'phase_active_mag_deg'))
            positions.append(_env_attr(env, 'x_pos'))

        print(f"eval episode {ep} return: {ep_return}")
        trajectories.append({
            'return': ep_return,
            'freqs': freqs,
            'shares': shares,
            'powers': powers,
            'etas': etas,
            'phases': phases,
            'positions': positions,
        })

    return trajectories