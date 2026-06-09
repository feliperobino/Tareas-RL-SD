import os

from dwpt_rl import DWPTContinuousShareEnv, DWPTParams, oracle_controller
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ppo_continuous_action import *
from sac import SACConfig, train as train_sac
import torch

print("\n" + "="*80)
print("Escenario a correr:")
print("  [1] full")
print("  [2] no_comm")
print("="*80)
user_choice = input("1 o 2?").strip()

if user_choice == "2":
    system_knowledge = 'no_comm'
    suffix = "_no_comm"
    print(f"Running NO_COMM\n")
else:
    system_knowledge = 'full'
    suffix = ""
    print(f"Running FULL\n")

params = DWPTParams()
env = DWPTContinuousShareEnv(params=params, seed=0, system_knowledge=system_knowledge)
env.reset(variable=False)

base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, f"results{suffix}")
os.makedirs(path, exist_ok=True)

# ============================================================================
# k1 y k2 vs posición
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
plt.savefig(os.path.join(path, 'coupling_profile.png'))
plt.close()

# ============================================================================
# P vs f para distintos valores de k1
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
plt.savefig(os.path.join(path, 'power_vs_freq.png'))
plt.close()

# ============================================================================
# eta vs f para distintos valores de k1
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
plt.savefig(os.path.join(path, 'efficiency_vs_freq.png'))
plt.close()

# ============================================================================
# P vs phi barriendo frecuencia vs k1
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
plt.savefig(os.path.join(path, 'power_vs_phase.png'))
plt.close()

# ============================================================================
# P vs share vs pares (k1, k2)
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
plt.savefig(os.path.join(path, 'power_vs_share.png'))
plt.close()

#############
# Entrenamiento PPO + SAC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=============================")
print("             PPO             ")
print("=============================")

args = Args()
ppo_agent, ppo_history = train_ppo(args, system_knowledge=system_knowledge)

print("=============================")
print("             SAC             ")
print("=============================")

sac_cfg = SACConfig(seed=args.seed, total_steps=args.total_timesteps)
sac_env = make_env(system_knowledge=system_knowledge)
sac_history, sac_agent = train_sac(sac_env, sac_cfg)

# ============================
# Comparar PPO vs SAC vs oráculo
# ============================
combos = [
    (5.0, 4.5),
    (10.0, 5.1),
    (12.0, 6.5),
]


def run_controller_on_condition(controller_kind, controller_obj, env, device):
    obs, _ = env.reset(seed=0)
    obs_dict = env.unwrapped._get_state('dict')
    done = False
    times = []
    pouts = []
    etas = []
    fs = []
    shares = []
    phase1s = []
    phase2s = []
    t = 0.0

    while not done:
        if controller_kind == 'oracle':
            action = oracle_controller(obs_dict, env.unwrapped)
        elif controller_kind == 'ppo':
            obs_arr = np.asarray(obs, dtype=np.float32)
            obs_tensor = torch.as_tensor(obs_arr, device=device)
            with torch.no_grad():
                action = controller_obj.get_deterministic_action(obs_tensor)
            action = action.cpu().numpy().squeeze()
        elif controller_kind == 'sac':
            obs_arr = np.asarray(obs, dtype=np.float32)
            action = controller_obj.act(obs_arr, deterministic=True)
        else:
            raise ValueError(f"Unknown controller kind: {controller_kind}")

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        times.append(t)
        pouts.append(env.unwrapped.pout)
        etas.append(env.unwrapped.eta)
        fs.append(env.unwrapped.f_sw)
        shares.append(env.unwrapped.share)
        phase1s.append(env.unwrapped.phase1_deg)
        phase2s.append(env.unwrapped.phase2_deg)

        t += env.unwrapped.p.dt
        obs_dict = env.unwrapped._get_state('dict')

    return {
        'time': np.array(times),
        'pout': np.array(pouts),
        'eta': np.array(etas),
        'f_sw': np.array(fs),
        'share': np.array(shares),
        'phase1_deg': np.array(phase1s),
        'phase2_deg': np.array(phase2s),
    }


def plot_three_way_comparison(results_by_label, metric_key, ylabel, title, output_file):
    styles = {
        'agent_ppo': {'label': 'agent_ppo', 'linestyle': '-'},
        'agent_sac': {'label': 'agent_sac', 'linestyle': '--'},
        'agent_oracle': {'label': 'agent_oracle', 'linestyle': ':'},
    }
    plt.figure(figsize=(10, 6))
    for label, result in results_by_label.items():
        plt.plot(
            result['time'],
            result[metric_key],
            label=styles[label]['label'],
            linestyle=styles[label]['linestyle'],
            linewidth=2,
        )
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


for speed, r_load in combos:
    params_tmp = DWPTParams()
    params_tmp.speed_mps = speed
    params_tmp.r_load_dc = r_load

    eval_env_ppo = make_env(system_knowledge=system_knowledge)
    eval_env_sac = make_env(system_knowledge=system_knowledge)
    eval_env_oracle = DWPTContinuousShareEnv(params=params_tmp, seed=0, system_knowledge=system_knowledge)

    # force same physical condition on all environments
    eval_env_ppo.unwrapped.p.speed_mps = speed
    eval_env_ppo.unwrapped.p.r_load_dc = r_load
    eval_env_sac.unwrapped.p.speed_mps = speed
    eval_env_sac.unwrapped.p.r_load_dc = r_load

    res_ppo = run_controller_on_condition('ppo', ppo_agent, eval_env_ppo, device)
    res_sac = run_controller_on_condition('sac', sac_agent, eval_env_sac, device)
    res_oracle = run_controller_on_condition('oracle', None, eval_env_oracle, device)

    label_suffix = f"speed_{speed}_rload_{r_load}{suffix}"
    results_by_label = {
        'agent_ppo': res_ppo,
        'agent_sac': res_sac,
        'agent_oracle': res_oracle,
    }

    plot_three_way_comparison(
        results_by_label,
        'pout',
        'P_out (W)',
        f'P_out vs time ({label_suffix})',
        os.path.join(path, f'compare_pout_{label_suffix}.png'),
    )
    plot_three_way_comparison(
        results_by_label,
        'eta',
        'Eta',
        f'Eta vs time ({label_suffix})',
        os.path.join(path, f'compare_eta_{label_suffix}.png'),
    )
    plot_three_way_comparison(
        results_by_label,
        'f_sw',
        'f_sw (Hz)',
        f'f_sw vs time ({label_suffix})',
        os.path.join(path, f'compare_fsw_{label_suffix}.png'),
    )
    plot_three_way_comparison(
        results_by_label,
        'share',
        'share',
        f'share vs time ({label_suffix})',
        os.path.join(path, f'compare_share_{label_suffix}.png'),
    )
    plot_three_way_comparison(
        results_by_label,
        'phase1_deg',
        'phase1 (deg)',
        f'phase1 vs time ({label_suffix})',
        os.path.join(path, f'compare_phase1_{label_suffix}.png'),
    )
    plot_three_way_comparison(
        results_by_label,
        'phase2_deg',
        'phase2 (deg)',
        f'phase2 vs time ({label_suffix})',
        os.path.join(path, f'compare_phase2_{label_suffix}.png'),
    )

    eval_env_ppo.close()
    eval_env_sac.close()
    eval_env_oracle.close()


###############################################################################
# TRAINING PLOTS
###############################################################################

plt.figure(figsize=(10, 6))
plt.plot(ppo_history["episode_returns"])
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("PPO Episode Returns")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(path, "ppo_episode_returns.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(ppo_history["mean_rewards"])
plt.xlabel("Iteration")
plt.ylabel("Mean Rollout Reward")
plt.title("PPO Mean Rollout Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(path, "ppo_mean_rewards.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(ppo_history["policy_losses"])
plt.xlabel("Update")
plt.ylabel("Policy Loss")
plt.title("PPO Policy Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(path, "ppo_policy_loss.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(ppo_history["value_losses"])
plt.xlabel("Update")
plt.ylabel("Value Loss")
plt.title("PPO Value Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(path, "ppo_value_loss.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(ppo_history["entropy_losses"])
plt.xlabel("Update")
plt.ylabel("Entropy")
plt.title("PPO Policy Entropy")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(path, "ppo_entropy.png"))
plt.close()


def save_line_plot(y_values, title, xlabel, ylabel, output_name):
    plt.figure(figsize=(10, 6))
    plt.plot(y_values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, output_name))
    plt.close()


save_line_plot(
    [ret for _, ret in sac_history["episode_returns"]],
    "SAC Episode Returns",
    "Episode",
    "Return",
    "sac_episode_returns.png",
)
save_line_plot(
    sac_history["mean_rewards"],
    "SAC Mean Rewards",
    "Episode",
    "Mean Reward",
    "sac_mean_rewards.png",
)
save_line_plot(
    sac_history["q_loss"],
    "SAC Q Loss",
    "Update",
    "Q Loss",
    "sac_q_loss.png",
)
save_line_plot(
    sac_history["pi_loss"],
    "SAC Policy Loss",
    "Update",
    "Policy Loss",
    "sac_pi_loss.png",
)
save_line_plot(
    sac_history["alpha"],
    "SAC Alpha",
    "Update",
    "Alpha",
    "sac_alpha.png",
)
save_line_plot(
    sac_history["alpha_loss"],
    "SAC Alpha Loss",
    "Update",
    "Alpha Loss",
    "sac_alpha_loss.png",
)
save_line_plot(
    sac_history["logp_pi"],
    "SAC Log Prob Policy",
    "Update",
    "Logp Pi",
    "sac_logp_pi.png",
)
save_line_plot(
    sac_history["q1_mean"],
    "SAC Q1 Mean",
    "Update",
    "Q1 Mean",
    "sac_q1_mean.png",
)