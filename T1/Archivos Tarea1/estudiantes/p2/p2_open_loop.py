from TwinRotorODE import TwinRotor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

params = {
    "Jp": 0.038,
    "Jy": 0.043,
    "mgl": 0.32,
    "Kpp": 0.204,
    "Kyp": 0.051,
    "Kpt": 0.011,
    "Kyt": 0.072,
    "Bp": 0.05,
    "By": 0.03,
    "Tm": 0.11,
    "Tt": 0.10,
    "JrOmega": 0.02,
}

Ts = 0.01
N = 2000
step_amp = 0.05
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)


def run_case(u_step_idx, step_amp_value):
    system = TwinRotor(params, Ts)
    # Equilibrio  x0 = 0, x1 = 0
    x_eq, u_eq = system.get_eq_point(x0=0, x1=0)
    system.x = x_eq
    system.t = 0

    u = np.array(u_eq, dtype=float)
    u[u_step_idx] += step_amp_value

    states = np.zeros((N, 6))
    outputs = np.zeros((N, 2))

    for k in range(N):
        x, y = system.sim(u=u.tolist())
        states[k, :] = np.array(x, dtype=float)
        outputs[k, :] = np.array(y, dtype=float)

    return np.array(x_eq, dtype=float), np.array(u_eq, dtype=float), u, states, outputs


x_eq_1, u_eq_1, u_1, states_1, outputs_1 = run_case(u_step_idx=0, step_amp_value=step_amp)
x_eq_2, u_eq_2, u_2, states_2, outputs_2 = run_case(u_step_idx=1, step_amp_value=step_amp)

t_vec = np.arange(N) * Ts

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

axes[0, 0].plot(t_vec, states_1[:, 0], label="phi")
axes[0, 0].plot(t_vec, states_1[:, 1], label="psi")
axes[0, 0].set_title("Step in pitch input (u0)")
axes[0, 0].set_ylabel("angles [rad]")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[1, 0].plot(t_vec, np.full_like(t_vec, u_1[0]), label="u0")
axes[1, 0].plot(t_vec, np.full_like(t_vec, u_1[1]), label="u1")
axes[1, 0].set_ylabel("inputs [V]")
axes[1, 0].set_xlabel("time [s]")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

axes[0, 1].plot(t_vec, states_2[:, 0], label="phi")
axes[0, 1].plot(t_vec, states_2[:, 1], label="psi")
axes[0, 1].set_title("Step in yaw input (u1)")
axes[0, 1].set_ylabel("angles [rad]")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

axes[1, 1].plot(t_vec, np.full_like(t_vec, u_2[0]), label="u0")
axes[1, 1].plot(t_vec, np.full_like(t_vec, u_2[1]), label="u1")
axes[1, 1].set_ylabel("inputs [V]")
axes[1, 1].set_xlabel("time [s]")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

fig.suptitle("Twin Rotor open-loop response from equilibrium x0=0, x1=0")
fig.tight_layout()
fig.savefig(os.path.join(results_dir, "open_loop_steps.png"), dpi=150)

phi_change_u0 = np.mean(states_1[-200:, 0] - x_eq_1[0])
psi_change_u0 = np.mean(states_1[-200:, 1] - x_eq_1[1])
phi_change_u1 = np.mean(states_2[-200:, 0] - x_eq_2[0])
psi_change_u1 = np.mean(states_2[-200:, 1] - x_eq_2[1])
