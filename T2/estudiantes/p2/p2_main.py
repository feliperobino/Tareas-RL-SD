from dwpt_rl import DWPTContinuousShareEnv, DWPTParams
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

params = DWPTParams()
env = DWPTContinuousShareEnv(params=params, seed=0, system_knowledge='full')
env.reset(variable=False)

# ============================================================================
# PERFIL DE ACOPLAMIENTO MAGNÉTICO: k1 y k2 vs posición
positions = np.linspace(0, env.p.coil_spacing_m * 2, 200)
k1_values = []
k2_values = []

for pos in positions:
    k1, k2 = env.coupling_profile(pos)
    k1_values.append(k1)
    k2_values.append(k2)

plt.figure(figsize=(10, 6))
plt.plot(positions, k1_values, 'b-', label='k1 (acoplamiento coil1-secondary)', linewidth=2)
plt.plot(positions, k2_values, 'r-', label='k2 (acoplamiento coil2-secondary)', linewidth=2)
plt.axvline(x=0, color='b', linestyle='--', alpha=0.5)
plt.axvline(x=env.p.coil_spacing_m, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Posición del vehículo (m)', fontsize=12)
plt.ylabel('Coeficiente de acoplamiento (k)', fontsize=12)
plt.title('Perfil de acoplamiento magnético', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('coupling_profile.png')
plt.close()

# ============================================================================
# POTENCIA vs FRECUENCIA para distintos valores de k1
frequencies = np.linspace(env.p.f_min, env.p.f_max, 100)
k1_test_values = [0.05, 0.10, 0.15, 0.20, 0.26]

plt.figure(figsize=(12, 6))
for k1_test in k1_test_values:
    p_out_values = []
    for f in frequencies:
        sol = env.solve_network(f_sw=f, k1=k1_test, k2=0.0, share=0.0)
        p_out_values.append(sol["P_out_target"])
    plt.plot(frequencies / 1e3, p_out_values, marker='o', markersize=3, label=f'k1={k1_test:.2f}', linewidth=2)

plt.xlabel('Frecuencia de conmutación (kHz)', fontsize=12)
plt.ylabel('Potencia de salida (W)', fontsize=12)
plt.title('Potencia transferida vs Frecuencia (share=0)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('power_vs_freq.png')
plt.close()

# ============================================================================
# EFICIENCIA vs FRECUENCIA para distintos valores de k1
plt.figure(figsize=(12, 6))
for k1_test in k1_test_values:
    eta_values = []
    for f in frequencies:
        sol = env.solve_network(f_sw=f, k1=k1_test, k2=0.0, share=0.0)
        eta_values.append(sol["eta_target"])
    plt.plot(frequencies / 1e3, eta_values, marker='o', markersize=3, label=f'k1={k1_test:.2f}', linewidth=2)

plt.xlabel('Frecuencia de conmutación (kHz)', fontsize=12)
plt.ylabel('Eficiencia (η)', fontsize=12)
plt.title('Eficiencia vs Frecuencia (share=0)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('efficiency_vs_freq.png')
plt.close()

# ============================================================================
# POTENCIA vs DESFASE: barriendo frecuencia para distintos k1
plt.figure(figsize=(12, 6))
for k1_test in k1_test_values:
    p_out_phase = []
    phase_values = []
    for f in frequencies:
        sol = env.solve_network(f_sw=f, k1=k1_test, k2=0.0, share=0.0)
        p_out_phase.append(sol["P_out_target"])
        phase_values.append(sol["phase_active_mag_deg"])
    plt.plot(phase_values, p_out_phase, marker='o', markersize=3, label=f'k1={k1_test:.2f}', linewidth=2)

plt.xlabel('Desfase de corriente (grados)', fontsize=12)
plt.ylabel('Potencia de salida (W)', fontsize=12)
plt.title('Potencia vs Desfase (variando frecuencia, share=0)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('power_vs_phase.png')
plt.close()

# ============================================================================
# POTENCIA vs SHARE para pares (k1, k2)
shares = np.linspace(0.0, 1.0, 50)
k_pairs = [(0.05, 0.20), (0.15, 0.15), (0.20, 0.05), (0.26, 0.26)]

plt.figure(figsize=(12, 6))
for k1_test, k2_test in k_pairs:
    p_out_share = []
    for share in shares:
        sol = env.solve_network(f_sw=86.5e3, k1=k1_test, k2=k2_test, share=share)
        p_out_share.append(sol["P_out_target"])
    plt.plot(shares, p_out_share, marker='o', markersize=4, label=f'k1={k1_test:.2f}, k2={k2_test:.2f}', linewidth=2)

plt.xlabel('Distribución de potencia (share)', fontsize=12)
plt.ylabel('Potencia de salida (W)', fontsize=12)
plt.title('Potencia vs Share (frecuencia fija en 86.5 kHz)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('power_vs_share.png')
plt.close()


import torch
import torch.nn as nn
import torch.optim as optim
from sac import SACConfig, SACAgent, ReplayBuffer, to_tensor, set_seed, mlp, reward_function


class PPOAgent:
    def __init__(self, obs_dim, act_dim, act_limit, lr=3e-4, gamma=0.99, lam=0.95, hidden=256, device='cpu'):
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.act_limit = act_limit
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim), nn.Tanh()
        ).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        ).to(self.device)
        
        self.log_std = nn.Parameter(torch.zeros(act_dim, device=self.device))
        self.eps = 1e-6
        
        self.actor_opt = optim.Adam(list(self.actor.parameters()) + [self.log_std], lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.trajectory = []

    def _get_dist(self, obs_t):
        mu = self.actor(obs_t)
        std = torch.exp(self.log_std).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def _log_prob_from_action(self, obs_t, action_t):
        scaled = torch.clamp(action_t / self.act_limit, -0.99999, 0.99999)
        z = 0.5 * torch.log((1 + scaled) / (1 - scaled + 1e-7))
        dist = self._get_dist(obs_t)
        logp_z = dist.log_prob(z).sum(dim=-1)
        logp_pi = logp_z - torch.log(1 - scaled.pow(2) + 1e-7).sum(dim=-1)
        return logp_pi

    @torch.no_grad()
    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self._get_dist(obs_t)
        z = dist.rsample()
        a = torch.tanh(z) * self.act_limit
        logp = self._log_prob_from_action(obs_t, a)
        val = self.critic(obs_t).squeeze(0)
        return a.squeeze(0).cpu().numpy(), float(logp.item()), float(val.item())

    def store_transition(self, obs, act, logp, rew, val, done):
        self.trajectory.append((obs, act, logp, rew, val, done))

    def _compute_gae(self, rewards, values, dones, last_value):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = last_value if t == len(rewards) - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * nonterminal - values[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, last_value=0.0, epochs=10, batch_size=128):
        obs_np = np.array([t[0] for t in self.trajectory], dtype=np.float32)
        act_np = np.array([t[1] for t in self.trajectory], dtype=np.float32)
        logp_np = np.array([t[2] for t in self.trajectory], dtype=np.float32)
        rew_np = np.array([t[3] for t in self.trajectory], dtype=np.float32)
        val_np = np.array([t[4] for t in self.trajectory], dtype=np.float32)
        done_np = np.array([t[5] for t in self.trajectory], dtype=np.float32)

        adv_np, ret_np = self._compute_gae(rew_np, val_np, done_np, float(last_value))
        adv_max = np.abs(adv_np).max() + 1e-8
        adv_np = adv_np / adv_max

        obs_b = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        act_b = torch.tensor(act_np, dtype=torch.float32, device=self.device)
        logp_old = torch.tensor(logp_np, dtype=torch.float32, device=self.device)
        adv_b = torch.tensor(adv_np, dtype=torch.float32, device=self.device)
        ret_b = torch.tensor(ret_np, dtype=torch.float32, device=self.device)
        
        n_traj = len(self.trajectory)
        actor_losses = []
        critic_losses = []
        for _ in range(epochs):
            idx = np.random.permutation(n_traj)
            for i in range(0, n_traj, batch_size):
                batch_idx = idx[i:i+batch_size]
                
                obs_batch = obs_b[batch_idx]
                act_batch = act_b[batch_idx]
                logp_old_batch = logp_old[batch_idx]
                adv_batch = adv_b[batch_idx]
                ret_batch = ret_b[batch_idx]
                
                logp_new = self._log_prob_from_action(obs_batch, act_batch)
                
                ratio = torch.exp(logp_new - logp_old_batch)
                ratio_clipped = torch.clamp(ratio, 1-0.1, 1+0.1)
                surr1 = ratio * adv_batch
                surr2 = ratio_clipped * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()
                actor_losses.append(float(actor_loss.item()))
                
                val_pred = self.critic(obs_batch).squeeze()
                critic_loss = torch.mean((val_pred - ret_batch) ** 2)
                
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()
                critic_losses.append(float(critic_loss.item()))
        
        self.trajectory = []
        return {
            'actor_loss': float(np.mean(actor_losses)) if actor_losses else np.nan,
            'critic_loss': float(np.mean(critic_losses)) if critic_losses else np.nan,
        }

    def get_value(self, obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.critic(obs_t).squeeze().item()


def train_ppo(env, episodes=500, steps_per_ep=400, device='cpu'):
    obs_dim = np.prod(env.observation_space.shape) if hasattr(env.observation_space, 'shape') else len(env.reset(obs_format='list')[0])
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    
    agent = PPOAgent(obs_dim, act_dim, act_limit, lr=1e-3, device=device)
    
    metrics = {'returns': [], 'p_out_avg': [], 'p_out_max': [], 'actor_loss': [], 'critic_loss': []}
    
    for ep in range(episodes):
        obs, _ = env.reset(variable=True)
        obs_arr = np.array(list(obs.values())) if isinstance(obs, dict) else obs
        ep_ret, ep_p_out_list = 0.0, []
        
        for _ in range(steps_per_ep):
            act, logp, val = agent.act(obs_arr)
            next_obs, terminated = env.step(act)
            next_obs_arr = np.array(list(next_obs.values())) if isinstance(next_obs, dict) else next_obs
            
            p_out_raw = next_obs['p_out'] if isinstance(next_obs, dict) else reward_function(next_obs)
            rew = float(p_out_raw / max(env.p.p_rated, 1e-6))
            
            agent.store_transition(obs_arr, act, logp, rew, val, float(terminated))
            ep_p_out_list.append(p_out_raw)
            ep_ret += rew
            obs_arr = next_obs_arr
            
            if terminated:
                break
        
        last_value = 0.0 if terminated else agent.get_value(obs_arr)
        info = agent.update(last_value=last_value, epochs=10)
        metrics['returns'].append(ep_ret)
        metrics['p_out_avg'].append(np.mean(ep_p_out_list))
        metrics['p_out_max'].append(np.max(ep_p_out_list))
        metrics['actor_loss'].append(info['actor_loss'])
        metrics['critic_loss'].append(info['critic_loss'])
        
        if (ep + 1) % 20 == 0:
            print(f"PPO ep={ep+1}/{episodes} ret={ep_ret:.1f} p_avg={np.mean(ep_p_out_list):.1f}")
    
    return metrics, agent


def train_sac(env, episodes=500, steps_per_ep=400, device='cpu'):
    obs_dim = np.prod(env.observation_space.shape) if hasattr(env.observation_space, 'shape') else len(env.reset(obs_format='list')[0])
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    
    cfg = SACConfig(device=device, total_steps=episodes*steps_per_ep, batch_size=32, start_steps=100)
    agent = SACAgent(obs_dim, act_dim, act_limit, cfg)
    rb = ReplayBuffer(obs_dim, act_dim, 50000, agent.device)
    
    metrics = {'returns': [], 'p_out_avg': [], 'p_out_max': [], 'q_loss': [], 'pi_loss': [], 'alpha': []}
    
    for ep in range(episodes):
        obs, _ = env.reset(variable=True)
        obs_arr = np.array(list(obs.values())) if isinstance(obs, dict) else obs
        ep_ret, ep_p_out_list, t_step = 0.0, [], 0
        ep_q_loss, ep_pi_loss, ep_alpha = [], [], []
        
        for _ in range(steps_per_ep):
            if ep * steps_per_ep + t_step < cfg.start_steps:
                act = env.action_space.sample()
            else:
                act = agent.act(obs_arr, deterministic=False)
            
            next_obs, terminated = env.step(act)
            next_obs_arr = np.array(list(next_obs.values())) if isinstance(next_obs, dict) else next_obs
            p_out_raw = next_obs['p_out'] if isinstance(next_obs, dict) else reward_function(next_obs)
            rew = float(p_out_raw / max(env.p.p_rated, 1e-6))
            
            rb.add(obs_arr, act, rew, next_obs_arr, float(terminated))
            ep_p_out_list.append(p_out_raw)
            ep_ret += rew
            obs_arr = next_obs_arr
            t_step += 1
            
            if ep * steps_per_ep + t_step >= cfg.update_after and (ep * steps_per_ep + t_step) % cfg.update_every == 0:
                batch = rb.sample(cfg.batch_size)
                info = agent.update(batch)
                ep_q_loss.append(info['q_loss'])
                ep_pi_loss.append(info['pi_loss'])
                ep_alpha.append(info['alpha'])
            
            if terminated:
                break
        
        metrics['returns'].append(ep_ret)
        metrics['p_out_avg'].append(np.mean(ep_p_out_list))
        metrics['p_out_max'].append(np.max(ep_p_out_list))
        metrics['q_loss'].append(float(np.mean(ep_q_loss)) if ep_q_loss else np.nan)
        metrics['pi_loss'].append(float(np.mean(ep_pi_loss)) if ep_pi_loss else np.nan)
        metrics['alpha'].append(float(np.mean(ep_alpha)) if ep_alpha else np.nan)
        
        if (ep + 1) % 20 == 0:
            print(f"SAC ep={ep+1}/{episodes} ret={ep_ret:.1f} p_avg={np.mean(ep_p_out_list):.1f}")
    
    return metrics, agent


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nTraining PPO and SAC on device: {device}\n")

env_ppo = DWPTContinuousShareEnv(params=params, seed=42, system_knowledge='full')
metrics_ppo, agent_ppo = train_ppo(env_ppo, episodes=50, steps_per_ep=200, device=device)

env_sac = DWPTContinuousShareEnv(params=params, seed=42, system_knowledge='full')
metrics_sac, agent_sac = train_sac(env_sac, episodes=50, steps_per_ep=200, device=device)


def moving_average(x, w=20):
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    if w <= 1 or arr.size < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode='valid')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(metrics_ppo['returns'], label='PPO', linewidth=2)
axes[0].plot(metrics_sac['returns'], label='SAC', linewidth=2)
axes[0].set_xlabel('Episodio')
axes[0].set_ylabel('Retorno acumulado')
axes[0].set_title('Comparación PPO vs SAC')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(metrics_ppo['p_out_avg'], label='PPO', linewidth=2)
axes[1].plot(metrics_sac['p_out_avg'], label='SAC', linewidth=2)
axes[1].set_xlabel('Episodio')
axes[1].set_ylabel('Potencia transferida promedio (W)')
axes[1].set_title('Potencia promedio por episodio')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(metrics_ppo['p_out_max'], label='PPO', linewidth=2)
axes[2].plot(metrics_sac['p_out_max'], label='SAC', linewidth=2)
axes[2].set_xlabel('Episodio')
axes[2].set_ylabel('Potencia transferida máxima (W)')
axes[2].set_title('Potencia máxima por episodio')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_comparison.png')
plt.close()

# Guardar métricas de entrenamiento
np.savez(
    'metrics_ppo.npz',
    returns=np.array(metrics_ppo['returns']),
    p_out_avg=np.array(metrics_ppo['p_out_avg']),
    p_out_max=np.array(metrics_ppo['p_out_max']),
    actor_loss=np.array(metrics_ppo['actor_loss']),
    critic_loss=np.array(metrics_ppo['critic_loss'])
)
np.savez(
    'metrics_sac.npz',
    returns=np.array(metrics_sac['returns']),
    p_out_avg=np.array(metrics_sac['p_out_avg']),
    p_out_max=np.array(metrics_sac['p_out_max']),
    q_loss=np.array(metrics_sac['q_loss']),
    pi_loss=np.array(metrics_sac['pi_loss']),
    alpha=np.array(metrics_sac['alpha'])
)

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
axes2[0, 0].plot(moving_average(metrics_ppo['returns'], 20), label='PPO MA(20)', linewidth=2)
axes2[0, 0].plot(moving_average(metrics_sac['returns'], 20), label='SAC MA(20)', linewidth=2)
axes2[0, 0].set_title('Retorno suavizado')
axes2[0, 0].grid(True, alpha=0.3)
axes2[0, 0].legend()

axes2[0, 1].plot(metrics_ppo['actor_loss'], label='PPO actor_loss', linewidth=1.5)
axes2[0, 1].plot(metrics_ppo['critic_loss'], label='PPO critic_loss', linewidth=1.5)
axes2[0, 1].set_title('PPO losses')
axes2[0, 1].grid(True, alpha=0.3)
axes2[0, 1].legend()

axes2[1, 0].plot(metrics_sac['q_loss'], label='SAC q_loss', linewidth=1.5)
axes2[1, 0].plot(metrics_sac['pi_loss'], label='SAC pi_loss', linewidth=1.5)
axes2[1, 0].set_title('SAC losses')
axes2[1, 0].grid(True, alpha=0.3)
axes2[1, 0].legend()

axes2[1, 1].plot(metrics_sac['alpha'], label='SAC alpha', linewidth=1.5)
axes2[1, 1].set_title('SAC temperatura alpha')
axes2[1, 1].grid(True, alpha=0.3)
axes2[1, 1].legend()

plt.tight_layout()
plt.savefig('training_diagnostics.png')
plt.close()

print(f"\nPPO - Retorno final: {metrics_ppo['returns'][-1]:.2f}, P_out promedio: {metrics_ppo['p_out_avg'][-1]:.2f}, P_out máx: {metrics_ppo['p_out_max'][-1]:.2f}")
print(f"SAC - Retorno final: {metrics_sac['returns'][-1]:.2f}, P_out promedio: {metrics_sac['p_out_avg'][-1]:.2f}, P_out máx: {metrics_sac['p_out_max'][-1]:.2f}")



