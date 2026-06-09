import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

matplotlib.use("Agg")



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)



def reward_function(obs):
    return 1


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: torch.device):
        self.device = device
        self.max_size = size
        self.ptr = 0
        self.len = 0

        self.obs      = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts     = np.zeros((size, act_dim), dtype=np.float32)
        self.rews     = np.zeros((size, 1),       dtype=np.float32)
        self.done     = np.zeros((size, 1),       dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr]      = obs
        self.acts[self.ptr]     = act
        self.rews[self.ptr]     = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr]     = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.len = min(self.len + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.len, size=batch_size)
        return dict(
            obs      = to_tensor(self.obs[idx],      self.device),
            acts     = to_tensor(self.acts[idx],     self.device),
            rews     = to_tensor(self.rews[idx],     self.device),
            next_obs = to_tensor(self.next_obs[idx], self.device),
            done     = to_tensor(self.done[idx],     self.device),
        )


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class CriticQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(128, 128, 128)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden_sizes, 1])

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


LOG_STD_MIN = -20
LOG_STD_MAX = 2
LOG_STD_MIN = -5
LOG_STD_MAX = 2

class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float, hidden_sizes=(128, 128, 64)):
        super().__init__()
        self.act_limit = act_limit
        self.net = mlp([obs_dim, *hidden_sizes, 2 * act_dim])

    def forward(self, obs):
        mu_logstd = self.net(obs)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z)
        act = self.act_limit * a


        # log prob with tanh correction
        logp_z = dist.log_prob(z).sum(dim=-1, keepdim=True)
        correction = torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        logp_a = logp_z - correction

        mu_act = self.act_limit * torch.tanh(mu)
        return act, logp_a, mu_act



@dataclass
class SACConfig:
    seed: int = 0

    total_steps: int  = 200_000
    start_steps: int  = 5_000
    update_after: int = 1_000
    update_every: int = 50
    batch_size: int   = 256
    replay_size: int  = 200_000

    gamma: float = 0.99
    tau:   float = 0.005

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float  = 3e-4

    hidden_sizes: Tuple[int, int] = (256, 256)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    eval_every:    int = 10_000
    eval_episodes: int = 5



class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit, cfg: SACConfig):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)

        self.actor = SquashedGaussianActor(obs_dim, act_dim, act_limit, cfg.hidden_sizes).to(self.device)
        self.q1    = CriticQ(obs_dim, act_dim, cfg.hidden_sizes).to(self.device)
        self.q2    = CriticQ(obs_dim, act_dim, cfg.hidden_sizes).to(self.device)

        self.q1_t = CriticQ(obs_dim, act_dim, cfg.hidden_sizes).to(self.device)
        self.q2_t = CriticQ(obs_dim, act_dim, cfg.hidden_sizes).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.critic_lr
        )

        # Automatic temperature tuning
        #self.target_entropy = -float(act_dim)
        self.target_entropy = -0.5 * float(act_dim)  # = -1.0, more exploration
        self.log_alpha  = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_opt  = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        self.act_limit = act_limit

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = to_tensor(obs, self.device).unsqueeze(0)
        a, _, mu_a = self.actor.sample(obs_t)
        out = mu_a if deterministic else a
        return out.squeeze(0).cpu().numpy()

    def _soft_update(self, net, targ):
        tau = self.cfg.tau
        for p, p_t in zip(net.parameters(), targ.parameters()):
            p_t.data.mul_(1 - tau)
            p_t.data.add_(tau * p.data)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs      = batch["obs"]
        act      = batch["acts"]
        rew      = batch["rews"]
        next_obs = batch["next_obs"]
        done     = batch["done"]

        # Critic Update
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(next_obs)
            q_next = torch.min(self.q1_t(next_obs, next_a), self.q2_t(next_obs, next_a))
            y = rew + self.cfg.gamma * (1 - done) * (q_next - self.alpha * next_logp)

        q1_val = self.q1(obs, act)
        q2_val = self.q2(obs, act)
        q_loss = F.mse_loss(q1_val, y) + F.mse_loss(q2_val, y)

        self.critic_opt.zero_grad()
        q_loss.backward()
        self.critic_opt.step()

        # Actor update
        a_pi, logp_pi, _ = self.actor.sample(obs)
        q_pi  = torch.min(self.q1(obs, a_pi), self.q2(obs, a_pi))
        pi_loss = (self.alpha * logp_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        pi_loss.backward()
        #nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
        self.actor_opt.step()

        # Temp adjustment (optional)
        alpha_loss = -(self.alpha * (logp_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1, self.q1_t)
        self._soft_update(self.q2, self.q2_t)

        return {
            "q_loss":     float(q_loss.item()),
            "pi_loss":    float(pi_loss.item()),
            "alpha":      float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
            "logp_pi":    float(logp_pi.mean().item()),
            "q1_mean":    float(q1_val.mean().item()),
        }



def train(env, cfg: SACConfig):
    set_seed(cfg.seed)


    obs_dim   = env.observation_space.shape[0]
    act_dim   = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    print(f"device : {cfg.device}")
    print(f"obs_dim: {obs_dim}  act_dim: {act_dim}  act_limit: {act_limit}")

    agent = SACAgent(obs_dim, act_dim, act_limit, cfg)
    rb    = ReplayBuffer(obs_dim, act_dim, cfg.replay_size, agent.device)

    metrics = {key: [] for key in [
        "episode_returns",
        "mean_rewards",
        "update_steps",
        "q_loss", "pi_loss",
        "alpha", "alpha_loss",
        "logp_pi", "q1_mean",
    ]}

    obs, _ = env.reset(seed=cfg.seed)
    ep_ret, ep_len = 0.0, 0
    ep_num = 0

    training_start = time.time()

    for t in range(1, cfg.total_steps + 1):

        if t < cfg.start_steps:
            act = env.action_space.sample()
        else:
            act = agent.act(obs, deterministic=False)

        next_obs, rew, terminated, truncated, info = env.step(act)
        done_flag = bool(terminated or truncated)
        done = float(done_flag)

        rb.add(obs, act, rew, next_obs, done)

        obs     = next_obs
        ep_ret += rew
        ep_len += 1

        if done_flag:
            ep_num += 1
            metrics["episode_returns"].append((t, ep_ret))
            metrics["mean_rewards"].append(ep_ret / max(ep_len, 1))
            recent      = metrics["episode_returns"][-10:]
            recent_avg  = np.mean([r for _, r in recent])
            sps         = int(t / (time.time() - training_start))
            print(
                f"[train] step={t:>7d}  ep={ep_num:>4d}  "
                f"return={ep_ret:>8.1f}  len={ep_len:>4d}  "
                f"avg10={recent_avg:>7.1f}  sps={sps}"
            )
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0


        if t >= cfg.update_after and t % cfg.update_every == 0:
            info_list = {k: [] for k in ["q_loss", "pi_loss", "alpha", "alpha_loss", "logp_pi", "q1_mean"]}
            for _ in range(cfg.update_every):
                batch = rb.sample(cfg.batch_size)
                info  = agent.update(batch)
                for k in info_list:
                    info_list[k].append(info[k])

            metrics["update_steps"].append(t)
            for k in info_list:
                metrics[k].append(float(np.mean(info_list[k])))


    env.close()
    return metrics, agent



