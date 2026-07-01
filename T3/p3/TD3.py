import copy
import random
import time
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform_references(references):
    x     = references[0][:, 0].reshape(1, -1)
    y     = references[0][:, 1].reshape(1, -1)
    z     = references[0][:, 2].reshape(1, -1)
    x_dot = references[1][:, 0].reshape(1, -1)
    y_dot = references[1][:, 1].reshape(1, -1)
    z_dot = references[1][:, 2].reshape(1, -1)
    X_ref = np.concatenate([x, x_dot, y, y_dot, z, z_dot], axis=0)
    X_ref = np.concatenate([X_ref, np.zeros_like(X_ref)], axis=0)
    return X_ref


def linear_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    frac = max(0.0, 1.0 - step / float(total_steps))
    return end + (start - end) * frac


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def rolling_mean(v, w=20):
    if len(v) == 0:
        return []
    return [float(np.mean(v[max(0, i - w + 1): i + 1])) for i in range(len(v))]


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1_000_000):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.act[self.ptr] = np.asarray(act, dtype=np.float32)
        self.rew[self.ptr] = float(rew)
        self.next_obs[self.ptr] = np.asarray(next_obs, dtype=np.float32)
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device)
        act = torch.as_tensor(self.act[idx], dtype=torch.float32, device=device)
        rew = torch.as_tensor(self.rew[idx], dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device)
        done = torch.as_tensor(self.done[idx], dtype=torch.float32, device=device)
        return obs, act, rew, next_obs, done




class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_low: np.ndarray, act_high: np.ndarray, hidden: int = 256):
        super().__init__()
        self.register_buffer("act_low", torch.as_tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.as_tensor(act_high, dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

        # zero-init last layer
        last_linear = self.net[5]
        init.uniform_(last_linear.weight, -0.003, 0.003)
        init.uniform_(last_linear.bias, -0.003, 0.003)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        x = 0.5 * (x + 1.0)
        return self.act_low + x * (self.act_high - self.act_low)


class CriticTwin(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = obs_dim + act_dim

        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x)



@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    batch_size: int = 128
    replay_size: int = 500_000

    start_steps: int = 10_000
    update_after: int = 2_000
    update_every: int = 1

    grad_clip_norm: Optional[float] = 5

    policy_noise_frac: float = 0.1
    noise_clip_frac: float = 0.2
    exploration_start_frac: float = 0.2
    exploration_end_frac: float = 0.02


class TD3Agent:
    def __init__(
        self,
        reward_function,
        obs_dim: int,
        act_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        cfg: TD3Config,
        device: torch.device,
        max_pos_error=3.0,
        max_angle_deg=60.0,

    ):
        self.cfg = cfg
        self.device = device
        self.act_low_np = np.asarray(act_low, dtype=np.float32)
        self.act_high_np = np.asarray(act_high, dtype=np.float32)
        self.act_low = torch.as_tensor(self.act_low_np, dtype=torch.float32, device=device)
        self.act_high = torch.as_tensor(self.act_high_np, dtype=torch.float32, device=device)

        self.actor = Actor(obs_dim, act_dim, self.act_low_np, self.act_high_np).to(device)
        self.critic = CriticTwin(obs_dim, act_dim).to(device)

        self.actor_targ = copy.deepcopy(self.actor).to(device)
        self.critic_targ = copy.deepcopy(self.critic).to(device)

        for p in self.actor_targ.parameters():
            p.requires_grad = False
        for p in self.critic_targ.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.reward_function = reward_function
        self.total_it = 0
        self.max_angle_rad = np.deg2rad(max_angle_deg)
        self.max_pos_error = max_pos_error
        self.metrics = {
            "episode_returns": [],
            "update_steps": [],
            "critic_loss": [],
            "actor_loss": [],
            "q1_mean": [],
            "q1_std": [],
            "target_mean": [],
            "target_std": [],
            "eval_steps": [],
            "eval_returns": [],
            "actions": []}

    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_scale=0.0) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(obs_t).cpu().numpy()[0]

        if np.any(np.asarray(noise_scale) > 0.0):
            a = a + noise_scale * np.random.randn(*a.shape)

        return np.clip(a, self.act_low_np, self.act_high_np).astype(np.float32)


    def update(self, rb: ReplayBuffer):
        cfg = self.cfg
        self.total_it += 1

        obs, act, rew, next_obs, done = rb.sample(cfg.batch_size, self.device)
        with torch.no_grad():
            action_range = self.act_high - self.act_low

            noise = torch.randn_like(act) * (cfg.policy_noise_frac)
            noise = noise.clamp(-cfg.noise_clip_frac, cfg.noise_clip_frac)

            next_act = torch.clamp((self.actor_targ(next_obs) - self.act_low) / action_range + noise, min=0, max=1)

            q1_t, q2_t = self.critic_targ(next_obs, next_act)
            q_t = torch.min(q1_t, q2_t)
            y = rew + cfg.gamma * (1.0 - done) * q_t

        q1, q2 = self.critic(obs, (act - self.act_low) / action_range)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
        self.critic_opt.step()

        info = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float("nan"),
            "q1_mean": float(q1.mean().item()),
            "q1_std": float(q1.std().item()),
            "target_mean": float(y.mean().item()),
            "target_std": float(y.std().item()),
        }

        if self.total_it % cfg.policy_delay == 0:
            actor_loss = -self.critic.q1_only(obs, (self.actor(obs) - self.act_low) / action_range).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if cfg.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
            self.actor_opt.step()

            with torch.no_grad():
                for p, p_t in zip(self.actor.parameters(), self.actor_targ.parameters()):
                    p_t.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)
                for p, p_t in zip(self.critic.parameters(), self.critic_targ.parameters()):
                    p_t.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)

            info["actor_loss"] = float(actor_loss.item())

        return info

    def sample_unknown(self, unknown_params):
        dicto_out = {}
        for k, v in unknown_params.items():
            mu = v[0]
            sigma = v[1]
            dicto_out[k] = np.random.normal(mu, sigma)
        return dicto_out

    def build_obs(self, state, ref, ppo_action, obs_norm, f_hover):
        error = state - ref
        obs   = np.concatenate([
            state      / obs_norm,               # 12
            error      / obs_norm,               # 12
            ppo_action / f_hover,  # 4
        ])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def is_unstable(self, state, ref):
        pos_error = np.linalg.norm(state[[0, 2, 4]] - ref[[0, 2, 4]])
        flipped   = np.any(np.abs(state[6:9]) > self.max_angle_rad)
        return pos_error > self.max_pos_error or flipped


    def train(self, env, references, known_params, unknown_params, seed: int = 0, steps: int = 50_000, eval_every: int = 1_000):
        set_seed(seed)

        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'p3'))
        os.makedirs(results_dir, exist_ok=True)

        obs_dim = int(len(env.x))*2 + 4
        act_dim = 4

        # Initialization
        f_hover = known_params['m'] * known_params['g'] / 4.0
        max_steps = references.shape[1]

        # Agent
        rb = ReplayBuffer(obs_dim, act_dim, capacity=self.cfg.replay_size)


        params = {**known_params, **self.sample_unknown(unknown_params)}
        state, _ = env.reset(x0=references[:, 0], params=params)

        t_idx = 0
        ref = references[:, t_idx]
        action = np.array([f_hover]*act_dim)
        obs = self.build_obs(state, ref, action, env.obs_norm, f_hover)
        obs = torch.from_numpy(obs)

        ep_return = 0.0
        ep_length = 0
        ep_num = 0

        # para guardar el modelo
        best_return = -np.inf
        best_actor_state = None

        window_info = []
        ep_length_history = []

        for t in range(1, steps + 1):
            noise_frac = linear_schedule(t, int(steps), self.cfg.exploration_start_frac, self.cfg.exploration_end_frac)
            noise_scale = noise_frac * (self.act_high_np - self.act_low_np)

            if t < self.cfg.start_steps:
                action = np.array([np.random.uniform(self.act_low_np[i], self.act_high_np[i]) for i in range(act_dim)])
            else:
                action = self.act(obs, noise_scale=noise_scale)

            # Step environment
            state, _ = env.step(action)
            t_idx = min(env.t, max_steps - 1)
            ref = references[:, t_idx]

            # Reward
            reward = self.reward_function(ref, state, action)

            # Termination
            traj_done = (env.t >= max_steps)
            unstable = self.is_unstable(state, ref)
            if unstable:
                reward -= 10 * (max_steps - env.t)
            done = traj_done or unstable

            ep_return += float(reward)
            ep_length += 1

            next_obs = torch.from_numpy(self.build_obs(state, ref, action, env.obs_norm, f_hover))


            # Add to the replay buffer
            rb.add(obs, action, reward, next_obs, float(done))

            obs = next_obs

            if done:
                ep_num += 1
                self.metrics["episode_returns"].append(ep_return)
                # guardamos el mejor
                if ep_return > best_return:
                    best_return = ep_return
                    best_actor_state = {
                        k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()
                    }

                print(
                    f"[train] step={t:>7d}  ep={ep_num:>4d}  "
                    f"return={ep_return:>8.2f}  len={ep_length:>4d}  "
                )
                ep_length_history.append(ep_length)
                ep_return = 0.0
                ep_length = 0


                # Resample randomized parameters for next episode
                params = {**known_params, **self.sample_unknown(unknown_params)}
                state, _ = env.reset(x0=references[:, 0], params=params)
                t_idx = 0
                ref = references[:, t_idx]
                action = np.array([f_hover] * act_dim)
                obs = self.build_obs(state, ref, action, env.obs_norm, f_hover)


            if t >= self.cfg.update_after and t % self.cfg.update_every == 0:
                info = self.update(rb)
                window_info.append(info)

            if t % eval_every == 0:

                if window_info:
                    cl = float(np.mean([i["critic_loss"] for i in window_info]))
                    al_v = [i["actor_loss"] for i in window_info if not np.isnan(i["actor_loss"])]
                    al = float(np.mean(al_v)) if al_v else float("nan")
                    qm = float(np.mean([i["q1_mean"] for i in window_info]))
                    qs = float(np.mean([i["q1_std"] for i in window_info]))
                    tm = float(np.mean([i["target_mean"] for i in window_info]))
                    ts = float(np.mean([i["target_std"] for i in window_info]))

                    self.metrics["update_steps"].append(t)
                    self.metrics["critic_loss"].append(cl)
                    self.metrics["actor_loss"].append(al)
                    self.metrics["q1_mean"].append(qm)
                    self.metrics["q1_std"].append(qs)
                    self.metrics["target_mean"].append(tm)
                    self.metrics["target_std"].append(ts)

        if best_actor_state is None:
            best_actor_state = {
                k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()
            }

        best_model_path = os.path.join(results_dir, "best_td3_actor.pt")
        torch.save(best_actor_state, best_model_path)
        print(f"[train] best model saved to {best_model_path} | best_return={best_return:.2f}")


        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        plt.plot(ep_length_history, label='Len')
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(self.metrics['episode_returns'], label='Return')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'td3_train_episode_returns.png'), dpi=200, bbox_inches='tight')
        plt.show()

        return self.metrics


    def evaluate(
            self,
            env,
            references,
            known_params,
    ):

        self.actor.eval()
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'p3'))
        os.makedirs(results_dir, exist_ok=True)

        act_dim = 4
        f_hover = known_params["m"] * known_params["g"] / 4.0
        max_steps = references.shape[1]

        # Reset without new params ("real system")
        state, _ = env.reset(x0=references[:, 0])
        t_idx = 0
        ref = references[:, t_idx]

        action = np.array([f_hover]*act_dim)
        obs = self.build_obs(
            state,
            ref,
            action,
            env.obs_norm,
            f_hover,
        )

        states = []
        refs = []
        actions = []
        rewards = []
        pos_errors = []

        ep_return = 0.0
        unstable = False

        for k in range(max_steps - 1):
            ref = references[:, t_idx]
            noise_scale = 0.0
            action = self.act(obs, noise_scale=noise_scale)

            # Step environment
            state, _ = env.step(action)

            t_idx = min(env.t, max_steps - 1)
            ref = references[:, t_idx]

            reward = self.reward_function(ref, state, action)

            traj_done = env.t >= max_steps

            if unstable:
                reward -= 10.0 * (max_steps - env.t)

            done = traj_done or unstable

            ep_return += float(reward)

            states.append(np.asarray(state))
            refs.append(np.asarray(ref))
            actions.append(np.asarray(action))
            rewards.append(float(reward))

            pos_error = np.linalg.norm(state[[0, 2, 4]] - ref[[0, 2, 4]])
            pos_errors.append(float(pos_error))

            obs = self.build_obs(
                state,
                ref,
                action,
                env.obs_norm,
                f_hover,
            )

            if done:
                break

        states = np.asarray(states)
        refs = np.asarray(refs)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        pos_errors = np.asarray(pos_errors)

        total_return = float(np.sum(rewards)) if len(rewards) > 0 else 0.0
        mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
        rmse_pos = float(np.sqrt(np.mean(pos_errors ** 2))) if len(pos_errors) > 0 else np.nan

        t = np.arange(len(states)) * env.Ts


        plt.figure(figsize=(15, 7))

        plt.subplot(3, 1, 1)
        plt.plot(t, refs[:, 0], "k--", label="x ref")
        plt.plot(t, states[:, 0], label="x TD3")
        plt.ylabel("x [m]")
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t, refs[:, 2], "k--", label="y ref")
        plt.plot(t, states[:, 2], label="y TD3")
        plt.ylabel("y [m]")
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(t, refs[:, 4], "k--", label="z ref")
        plt.plot(t, states[:, 4], label="z TD3")
        plt.ylabel("z [m]")
        plt.xlabel("t [s]")
        plt.grid(True)
        plt.legend()

        plt.suptitle(
            f"TD3 eval — return={total_return:.2f}, "
            f"RMSE pos={rmse_pos:.3f}, steps={len(states)}, unstable={unstable}"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'td3_eval_xyz.png'), dpi=200, bbox_inches='tight')

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            refs[:, 0],
            refs[:, 2],
            refs[:, 4],
            "k--",
            linewidth=2,
            label="Reference",
        )

        ax.plot(
            states[:, 0],
            states[:, 2],
            states[:, 4],
            linewidth=1.5,
            label="TD3 trajectory",
        )

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Quadrotor TD3 — Reference vs Followed Trajectory")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'td3_eval_trajectory_3d.png'), dpi=200, bbox_inches='tight')

        plt.figure(figsize=(15, 8))

        plt.subplot(2, 1, 1)
        for i in range(act_dim):
            plt.plot(t, actions[:, i], label=f"u{i + 1}")
        plt.axhline(f_hover, linestyle="--", linewidth=1.0, label="hover")
        plt.ylabel("final thrust [N]")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t, rewards, label="reward")
        plt.xlabel("t [s]")
        plt.ylabel("reward")
        plt.grid(True)
        plt.legend()

        plt.suptitle("TD3 actions and rewards")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'td3_eval_actions_rewards.png'), dpi=200, bbox_inches='tight')
        plt.show()

        print(
            f"[eval] return={total_return:.2f} | "
            f"mean_reward={mean_reward:.3f} | "
            f"rmse_pos={rmse_pos:.3f} | "
            f"steps={len(states)} | "
            f"unstable={unstable} | "
        )


        return {
            "return": total_return,
            "mean_reward": mean_reward,
            "rmse_pos": rmse_pos,
            "unstable": bool(unstable),
            "steps": int(len(states)),
            "states": states,
            "references": refs,
            "actions": actions,
            "rewards": rewards,
            "pos_errors": pos_errors,
        }

if __name__ == "__main__":
    from scripts.Quadrotor3D import Quadrotor3D
    from scripts.QuadrotorTrajectories import QuadrotorTrajectory

    Ts = 0.01

    # Known parameters
    known_params = {
        'm': 0.027,  # kg
        'L': 0.0397,  # m
        'g': 9.8,  # m/s²
    }

    # Unknown parameters
    unknown_params = {
        'Ixx': [0.6e-5, 3.0e-5],
        'Iyy': [0.6e-5, 3.0e-5],
        'Izz': [1.0e-5, 5.0e-5],
        'KF': [2.0e-10, 4.5e-10],
        'KM': [4.0e-12, 1.3e-11],
    }

    # Real system
    system = Quadrotor3D(Ts=Ts)

    # Trajectory
    trajectory_gen = QuadrotorTrajectory(sample_time=Ts)
    LENGTH = 12.0
    references_raw = trajectory_gen.helix(length=LENGTH, num_cycles=3, radius=1, rise=0.5,  center=(0., 0., 0.5))
    references = transform_references(references_raw)


    ##### Cost Function #######
    Q = np.diag([
        5., 2.,  # x,  ẋ
        5., 2.,  # y,  ẏ
        5., 2.,  # z,  ż
        1., 1., 0.1,  # φ,  θ,  ψ
        0.1, 0.1, 0.1,  # p,  q,  r
    ])
    R = np.eye(4) * 0.01
    f_hover = known_params['m'] * known_params['g'] / 4.0
    f_max = 2.5 * f_hover

    def reward_function_simple(ref, states, inputs):
        err = ref - states
        state_cost = err.reshape(1, -1) @ Q @ err.reshape(-1, 1)
        u_cost = inputs.reshape(1, -1) @ R @ inputs.reshape(-1, 1)
        return -state_cost[0, 0] - u_cost[0, 0]


    @dataclass
    class TD3Config:
        gamma: float = 0.99
        tau: float = 0.005
        policy_delay: int = 2

        actor_lr: float = 3e-4
        critic_lr: float = 3e-4

        batch_size: int = 256
        replay_size: int = 50_000

        start_steps: int =15_000
        update_after: int = 5_000
        update_every: int = 1

        grad_clip_norm: Optional[float] = 5

        policy_noise_frac = 0.05
        noise_clip_frac = 0.1
        exploration_start_frac = 0.1
        exploration_end_frac = 0.001

    # Transform uniform to gaussian
    unknown_params_gaussian = {}
    for k, v_range in unknown_params.items():
        mu_k = (v_range[1] + v_range[0]) / 2.0
        sigma = abs(mu_k - v_range[0]) / 2
        unknown_params_gaussian[k] = [mu_k, sigma]


    agent = TD3Agent(
        reward_function=reward_function_simple,
        obs_dim=len(system.x)*2 + 4,
        act_dim= 4,
        act_low=np.array([0, 0, 0, 0]),
        act_high= 2.5*np.array([f_hover, f_hover, f_hover, f_hover]),
        cfg=TD3Config(),
        device=device,
        max_pos_error=2.0,
        max_angle_deg=60.0)

    agent.train(copy.deepcopy(system), references, known_params, unknown_params_gaussian, seed=0,
                steps=250_000)

    eval_results = agent.evaluate(
        env=copy.deepcopy(system),
        references=references,
        known_params=known_params)