import os, copy, pickle
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
import torch.nn as nn 

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# imports locales
from p3.TD3 import Actor, CriticTwin, ReplayBuffer, rolling_mean
from fit_reward import RewardNet, FEATURES, profile_to_vec 
from dagger import DAGGER                              

from scripts.fermentation_env import WineFermentationEnv
from scripts.expert_controller import WineExpertController
from scripts.quality_functions import maceration_extraction, get_wine_quality_score

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = "results/p2/d"
os.makedirs(OUT, exist_ok=True)

N_EPISODES = 400
NOISE_START = 0.15
NOISE_END = 0.01

@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.003
    policy_delay: int = 2

    actor_lr: float = 3e-5
    critic_lr: float = 3e-4

    batch_size: int = 128
    replay_size: int = 500_000

    start_steps: int = 10_000
    update_after: int = 8_000
    update_every: int = 1

    grad_clip_norm: Optional[float] = 5

    policy_noise_frac: float = 0.05
    noise_clip_frac: float = 0.10
    exploration_start_frac: float = 0.15
    exploration_end_frac: float = 0.02
    episode_horizon_steps: Optional[int] = None
    temp_slew_rate: float = 0.5
    exploration_noise_abs: Optional[np.ndarray] = None

### TD3 WINE
class WineTD3:
    def __init__(self, obs_dim, act_dim, act_low, act_high,
                 reward_net: RewardNet, reward_mu, reward_std, dagger: DAGGER, config: TD3Config):
        
        self.config = config
        self.cfg = config
        self.dagger = dagger
        self.reward_net = reward_net
        self.reward_mu = reward_mu
        self.reward_std = reward_std
        self.device = device
        self.max_episode_steps = config.episode_horizon_steps

        lo, hi = np.asanyarray(act_low, dtype=np.float32), np.asanyarray(act_high, dtype=np.float32)
        self.act_low_np = lo
        self.act_high_np = hi
        self.act_low_t = torch.as_tensor(lo, device=device)
        self.act_high_t = torch.as_tensor(hi, device=device)
        self.config.exploration_noise_abs = np.array([0.05, 0.30], dtype=np.float32) if self.config.exploration_noise_abs is None else np.asarray(self.config.exploration_noise_abs, dtype=np.float32)
        self.act_low = self.act_low_t
        self.act_high = self.act_high_t

        self.actor = Actor(obs_dim, act_dim, lo, hi).to(device)
        self.critic = CriticTwin(obs_dim, act_dim).to(device)
        self.actor_targ = copy.deepcopy(self.actor)
        self.critic_targ = copy.deepcopy(self.critic)
        for p in list(self.actor_targ.parameters()) + list(self.critic_targ.parameters()):
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.total_it = 0

    # init from dagger
    def init_from_dagger(self, sd):
        self.actor.load_state_dict(sd)
        self.actor_targ.load_state_dict(sd)
        print("Actor initialized from DAGGER.")

    # normalización de estados
    def _norm(self, obs):
        mn, mx = self.dagger.state_lims
        return (obs - mn) / (mx - mn + 1e-8).astype(np.float32)
    
    # reward (SPARSE)
    def _reward(self, env, done):

        # damos reward 0 si no terminó
        if not done:
            return 0.0
        
        profile = maceration_extraction(env.get_trajectory())
        x = (profile_to_vec(profile)-self.reward_mu)/self.reward_std

        ## reward por perfil de vinos!
        with torch.no_grad():
            r = self.reward_net(torch.tensor(x, device=device).unsqueeze(0)).item()

        finished = env._state.S < 1.0 ### premio para guíar si es que termina fermentación! (para guíar exploración)
        completion_bonus = 2.0 if finished else 0.0

        return float(r) + completion_bonus


    def _episode_terminal(self, env_done: bool, step_idx: int) -> bool:
        if self.max_episode_steps is None:
            return env_done
        return env_done or (step_idx + 1 >= self.max_episode_steps)
    
    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_scale=0.0, prev_action=None) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(obs_t).cpu().numpy()[0]

        if np.any(np.asarray(noise_scale) > 0.0):
            noise_abs = self.config.exploration_noise_abs * (noise_scale / max(NOISE_START, 1e-8))
            a = a + noise_abs * np.random.randn(*a.shape)

        if prev_action is not None:
            temp_delta = np.clip(
                a[0] - prev_action[0],
                -self.config.temp_slew_rate,
                self.config.temp_slew_rate,
            )
            a[0] = prev_action[0] + temp_delta
        return np.clip(a, self.act_low_np, self.act_high_np)

    def update(self, rb: ReplayBuffer):
        config = self.config
        self.total_it += 1

        obs, act, rew, next_obs, done = rb.sample(config.batch_size, self.device)
        with torch.no_grad():
            action_range = self.act_high - self.act_low

            noise = torch.randn_like(act) * (config.policy_noise_frac * action_range)
            noise = noise.clamp(-config.noise_clip_frac * action_range, config.noise_clip_frac * action_range)

            next_act = torch.clamp((self.actor_targ(next_obs) - self.act_low) / action_range + noise, min=0, max=1)

            q1_t, q2_t = self.critic_targ(next_obs, next_act)
            q_t = torch.min(q1_t, q2_t)
            y = rew + config.gamma * (1.0 - done) * q_t
            y = torch.clamp(y, -20.0, 10.0)

        q1, q2 = self.critic(obs, (act - self.act_low) / action_range)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), config.grad_clip_norm)
        self.critic_opt.step()

        info = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float("nan"),
            "q1_mean": float(q1.mean().item()),
            "q1_std": float(q1.std().item()),
            "target_mean": float(y.mean().item()),
            "target_std": float(y.std().item()),
        }

        if self.total_it % config.policy_delay == 0:
            actor_loss = -self.critic.q1_only(obs, (self.actor(obs) - self.act_low) / action_range).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), config.grad_clip_norm)
            self.actor_opt.step()

            with torch.no_grad():
                for p, p_t in zip(self.actor.parameters(), self.actor_targ.parameters()):
                    p_t.data.mul_(1.0 - config.tau).add_(config.tau * p.data)
                for p, p_t in zip(self.critic.parameters(), self.critic_targ.parameters()):
                    p_t.data.mul_(1.0 - config.tau).add_(config.tau * p.data)

            info["actor_loss"] = float(actor_loss.item())

        return info
    

    def train(self, n_episodes=N_EPISODES, noise_start=NOISE_START, noise_end=NOISE_END):
        obs_dim = len(self.dagger.state_lims[0])
        rb = ReplayBuffer(obs_dim, len(self.act_low_np), self.config.replay_size)

        ep_returns, true_scores, c_losses, a_losses = [], [], [], []
        best_score = -np.inf
        best_path = f"{OUT}/best_td3_wine_policy.pt"
        warmup_episodes = max(1, int(self.config.exploration_start_frac * n_episodes))
        decay_episodes = max(warmup_episodes + 1, int((1.0 - self.config.exploration_end_frac) * n_episodes))

        for ep in range(1, n_episodes+1):
            if ep <= warmup_episodes:
                noise = noise_start
            elif ep >= decay_episodes:
                noise = noise_end
            else:
                frac = (ep - warmup_episodes) / max(1, decay_episodes - warmup_episodes)
                noise = noise_start + frac * (noise_end - noise_start)
            env = WineFermentationEnv();  obs = env.reset();  done = False;  ep_r = 0.0
            a = None
            step_idx = 0
            
            while not done:
                on = self._norm(obs);  a = self.act(on, noise, a)
                nobs, env_done, _ = env.step(a)
                done = self._episode_terminal(env_done, step_idx)
                r = self._reward(env, done);  ep_r += r
                rb.add(on, a, r, self._norm(nobs), float(done))
                obs = nobs
                if rb.size >= self.config.update_after:
                    info = self.update(rb)
                    c_losses.append(info["critic_loss"])
                    if not np.isnan(info["actor_loss"]):
                        a_losses.append(info["actor_loss"])
                step_idx += 1

            ep_returns.append(ep_r)
            ts = get_wine_quality_score(maceration_extraction(env.get_trajectory()))
            true_scores.append(ts)
            if ts > best_score:
                best_score = ts
                torch.save(self.actor.state_dict(), best_path)
            if ep % 20 == 0: print(f"ep={ep}/{n_episodes}  reward={ep_r:.3f}  score={ts:.4f}  noise={noise:.3f}")

        return ep_returns, true_scores, c_losses, a_losses
    
    
    def rollout(self):
        env = WineFermentationEnv();  obs = env.reset();  done = False
        states, actions = [], []
        a = None
        step_idx = 0
        while not done:
            a = self.act(self._norm(obs), prev_action=a)
            states.append(obs.copy());  actions.append(a.copy())
            obs, env_done, _ = env.step(a)
            done = self._episode_terminal(env_done, step_idx)
            step_idx += 1
        profile = maceration_extraction(env.get_trajectory())
        score   = get_wine_quality_score(profile)
        x = (profile_to_vec(profile)-self.reward_mu)/self.reward_std
        with torch.no_grad():
            pred = self.reward_net(torch.tensor(x, device=device).unsqueeze(0)).item()
        states, actions = _pad_to_horizon(np.array(states), np.array(actions), self.max_episode_steps)
        return states, actions, score, pred

### ROLLOUTS DE CADA CONTROLADOR
def _pad_to_horizon(states, actions, max_steps):
    if max_steps is None or len(states) >= max_steps:
        return states, actions

    pad_n = max_steps - len(states)
    last_state = states[-1:]
    last_action = actions[-1:]
    states = np.concatenate([states, np.repeat(last_state, pad_n, axis=0)], axis=0)
    actions = np.concatenate([actions, np.repeat(last_action, pad_n, axis=0)], axis=0)
    return states, actions


def _run_env(actor_fn, max_steps=None):
    env = WineFermentationEnv()
    obs = env.reset()
    done = False
    states, actions = [], []
    step_idx = 0

    while not done:
        a = actor_fn(obs)
        states.append(obs.copy())
        actions.append(a.copy())
        obs, env_done, _ = env.step(a)
        done = env_done or (max_steps is not None and step_idx + 1 >= max_steps)
        step_idx += 1

    profile = maceration_extraction(env.get_trajectory())
    states, actions = _pad_to_horizon(np.array(states), np.array(actions), max_steps)

    return states, actions, get_wine_quality_score(profile)

def expert_rollout(max_steps=None):
    controller = WineExpertController()
    controller.reset()

    return _run_env(lambda obs: controller(obs, float(obs[7])), max_steps=max_steps)

def dagger_rollout(actor, dagger, max_steps=None):
    def act(obs):
        mn, mx = dagger.state_lims
        on = ((obs-mn) / (mx-mn+1e-8)).astype(np.float32)
        with torch.no_grad():
            return actor(torch.tensor(on, device=device).unsqueeze(0)).cpu().numpy()[0]
    return _run_env(act, max_steps=max_steps)

#### PLOTS!
OBS_NAMES = ["Res. Sugar","Etanol","Yeast vivo","Yeast muerto","N asim.","CO₂ dis.","T","t [h]"]
ACT_NAMES = ["T setpoint","N añadido"]
 
def plot_training(ep_returns, true_scores, c_losses, a_losses):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Entrenamiento TD3 – Sparse Reward")
    for ax, data, title, color in zip(axes.flatten(),
        [ep_returns, true_scores, c_losses, a_losses],
        ["Retorno por época","Score verdadero","Critic Loss","Actor Loss"],
        ["steelblue","seagreen","tomato","darkorange"]):
        ax.plot(data, alpha=0.35, color=color)
        ax.plot(rolling_mean(data, 20), color=color, linewidth=2)
        ax.set_title(title);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/training_curves.png", dpi=150);  plt.show()
 
def plot_comparison(rl_runs, dag_runs, exp_runs):
    s_rl = np.stack([r[0] for r in rl_runs], axis=0)
    a_rl = np.stack([r[1] for r in rl_runs], axis=0)
    sc_rl = np.array([r[2] for r in rl_runs], dtype=np.float32)

    s_dag = np.stack([r[0] for r in dag_runs], axis=0)
    a_dag = np.stack([r[1] for r in dag_runs], axis=0)
    sc_dag = np.array([r[2] for r in dag_runs], dtype=np.float32)

    s_exp = np.stack([r[0] for r in exp_runs], axis=0)
    a_exp = np.stack([r[1] for r in exp_runs], axis=0)
    sc_exp = np.array([r[2] for r in exp_runs], dtype=np.float32)

    s_rl_mean, s_rl_std = s_rl.mean(axis=0), s_rl.std(axis=0)
    s_dag_mean, s_dag_std = s_dag.mean(axis=0), s_dag.std(axis=0)
    s_exp_mean, s_exp_std = s_exp.mean(axis=0), s_exp.std(axis=0)

    a_rl_mean, a_rl_std = a_rl.mean(axis=0), a_rl.std(axis=0)
    a_dag_mean, a_dag_std = a_dag.mean(axis=0), a_dag.std(axis=0)
    a_exp_mean, a_exp_std = a_exp.mean(axis=0), a_exp.std(axis=0)

    fig, axes = plt.subplots(3, 3, figsize=(16,11))
    fig.suptitle("Comparación: RL vs DAGGER vs Experto")
    for i, ax in enumerate(axes.flatten()):
        if i < s_rl_mean.shape[1]:
            ax.plot(s_exp_mean[:,i], "k-",  lw=1.2, label="Experto")
            ax.fill_between(np.arange(s_exp_mean.shape[0]), s_exp_mean[:,i] - s_exp_std[:,i], s_exp_mean[:,i] + s_exp_std[:,i], color="k", alpha=0.12)
            ax.plot(s_dag_mean[:,i], "b--", lw=1.2, label="DAGGER")
            ax.fill_between(np.arange(s_dag_mean.shape[0]), s_dag_mean[:,i] - s_dag_std[:,i], s_dag_mean[:,i] + s_dag_std[:,i], color="b", alpha=0.12)
            ax.plot(s_rl_mean[:,i],  "r-",  lw=1.2, label="TD3-RL")
            ax.fill_between(np.arange(s_rl_mean.shape[0]), s_rl_mean[:,i] - s_rl_std[:,i], s_rl_mean[:,i] + s_rl_std[:,i], color="r", alpha=0.12)
            ax.set_title(OBS_NAMES[i] if i < len(OBS_NAMES) else f"obs[{i}]")
            ax.legend(fontsize=7);  ax.grid(True, alpha=0.3)
    plt.tight_layout();  plt.savefig(f"{OUT}/states_comparison.png", dpi=150);  plt.show()
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Acciones: RL vs DAGGER vs Experto")
    for i, ax in enumerate(axes):
        ax.plot(a_exp_mean[:,i], "k-", lw=1.2, label="Experto")
        ax.fill_between(np.arange(a_exp_mean.shape[0]), a_exp_mean[:,i] - a_exp_std[:,i], a_exp_mean[:,i] + a_exp_std[:,i], color="k", alpha=0.12)
        ax.plot(a_dag_mean[:,i], "b--",lw=1.2, label="DAGGER")
        ax.fill_between(np.arange(a_dag_mean.shape[0]), a_dag_mean[:,i] - a_dag_std[:,i], a_dag_mean[:,i] + a_dag_std[:,i], color="b", alpha=0.12)
        ax.plot(a_rl_mean[:,i],  "r-", lw=1.2, label="TD3-RL")
        ax.fill_between(np.arange(a_rl_mean.shape[0]), a_rl_mean[:,i] - a_rl_std[:,i], a_rl_mean[:,i] + a_rl_std[:,i], color="r", alpha=0.12)
        ax.set_title(ACT_NAMES[i]);  ax.legend();  ax.grid(True, alpha=0.3)
    plt.tight_layout();  plt.savefig(f"{OUT}/actions_comparison.png", dpi=150);  plt.show()
 
    fig, ax = plt.subplots(figsize=(6,4))
    scores_mean  = [float(sc_exp.mean()), float(sc_dag.mean()), float(sc_rl.mean())]
    scores_std  = [float(sc_exp.std()), float(sc_dag.std()), float(sc_rl.std())]
    bars = ax.bar(["Experto","DAGGER","TD3-RL"], scores_mean, yerr=scores_std, color=["gray","steelblue","tomato"], edgecolor="k", alpha=0.85, capsize=5)
    for b, s in zip(bars, scores_mean): ax.text(b.get_x()+b.get_width()/2, b.get_height()+.02, f"{s:.3f}", ha="center")
    ax.set_ylabel("Score de calidad");  ax.set_title("Scores finales");  ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout();  plt.savefig(f"{OUT}/score_comparison.png", dpi=150);  plt.show()

###############################
##           MAIN            ##
###############################

if __name__ == "__main__":
    # Cargar datos y NORMALIZAR
    with open("p2/data/expert_trajectories.pkl","rb") as f: expert_trajs = pickle.load(f)

    expert_steps = []
    for traj in expert_trajs:
        expert_steps.append(len(traj["trajectory"]))
    episode_horizon_steps = int(np.max(expert_steps))

    dag_helper = DAGGER()
    dag_helper.state_lims = dag_helper.get_normalization_params(expert_trajs)
    obs_dim = len(dag_helper.state_lims[0]);  act_dim = 2
 
    # DAGGER (pretrained)
    dagger_actor = Actor(obs_dim, act_dim, dag_helper.act_lims[0], dag_helper.act_lims[1]).to(device)
    dagger_actor.load_state_dict(torch.load("results/p2/dagger/dagger_policy.pt", map_location=device))
    dagger_actor.eval()
 
    # Red de Rewards
    dataset  = pickle.load(open("p2/data/comparisons.pkl","rb"))
    all_vecs = np.array([profile_to_vec(s[k]) for s in dataset for k in ("profile_A","profile_B")])
    mu, std  = all_vecs.mean(0), all_vecs.std(0)+1e-8
 
    rnet = RewardNet(len(FEATURES), 1).to(device)
    rnet.load_state_dict(torch.load("best_reward_model.pt", map_location=device))
    rnet.eval()
 
    # TD3
    cfg = TD3Config(episode_horizon_steps=episode_horizon_steps)
    agent = WineTD3(obs_dim, act_dim, dag_helper.act_lims[0], dag_helper.act_lims[1],
                    rnet, mu, std, dag_helper, cfg)
    agent.init_from_dagger(dagger_actor.state_dict())
 
    # TRAIN
    ep_returns, true_scores, c_losses, a_losses = agent.train(n_episodes=N_EPISODES)
    agent.actor.load_state_dict(torch.load(f"{OUT}/best_td3_wine_policy.pt", map_location=device))
    agent.actor_targ.load_state_dict(agent.actor.state_dict())
    torch.save(agent.actor.state_dict(), f"{OUT}/td3_wine_policy.pt")
 
    # PLOT
    plot_training(ep_returns, true_scores, c_losses, a_losses)
 
    rl_res  = [agent.rollout()          for _ in range(5)]
    dag_res = [dagger_rollout(dagger_actor, dag_helper, max_steps=episode_horizon_steps) for _ in range(5)]
    exp_res = [expert_rollout(max_steps=episode_horizon_steps)         for _ in range(5)]
 
    for name, res in [("Experto", exp_res), ("DAGGER", dag_res), ("TD3-RL", rl_res)]:
        scores = [r[2] for r in res]
        print(f"{name:<10} score={np.mean(scores):.4f} ± {np.std(scores):.4f}")
 
    plot_comparison(rl_res, dag_res, exp_res)
   
 
