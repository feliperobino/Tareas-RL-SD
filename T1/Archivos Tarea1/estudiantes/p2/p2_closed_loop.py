from TwinRotorODE import TwinRotor, TwinRotorEnhanced
from KF import DiscreteKalmanFilter
import numpy as np
import control
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

# LQR weights
Q_lqr = np.diag([80.0, 80.0, 2.0, 2.0, 0.1, 0.1])
R_lqr = np.diag([0.4, 0.4])

# KF weights
Q_kf = np.diag([5e-5, 5e-5, 5e-3, 5e-3, 5e-4, 5e-4])
R_kf = np.diag([0.01 ** 2, 0.005 ** 2])
P0 = np.eye(6) * 5.0


def simulate_reference(reference):
    system = TwinRotor(
        params,
        Ts,
        disturbance_time=[1000, 1500],
        disturbance_strength=0.1,
    )

    x_eq, u_eq = system.get_eq_point(x0=reference[0], x1=reference[1])
    x_eq = np.array(x_eq, dtype=float).reshape(-1, 1)
    u_eq = np.array(u_eq, dtype=float).reshape(-1, 1)
    y_eq = x_eq[:2, :]

    system.x = x_eq.squeeze().tolist()

    # matrices linealizadas del eq (re linealizar en pi = 0 y pi/4)
    Ad, Bd, Cd, Dd, _ = system.get_linear_system(x_eq.squeeze().tolist(), u_eq.squeeze().tolist())

    # LQR desde control.dlqr
    K, _, E = control.dlqr(Ad, Bd, Q_lqr, R_lqr) 
    K = np.asarray(K, dtype=float)

    # kalman
    kf = DiscreteKalmanFilter.create(
        Ad,
        Bd,
        Cd,
        Q_kf,
        R_kf,
        P_init=P0,
        D=Dd,
        x_init=np.zeros((6, 1)),
    )

    states = np.zeros((N, 6))
    states_est = np.zeros((N, 6))
    measurements = np.zeros((N, 2))
    controls = np.zeros((N, 2))

    u_cmd = u_eq.copy() # comando init

    # simular
    for k in range(N):
        x, y = system.sim(u=u_cmd.squeeze().tolist())
        x = np.array(x, dtype=float).reshape(-1, 1)
        y = np.array(y, dtype=float).reshape(-1, 1)

        states[k, :] = x.squeeze()
        measurements[k, :] = y.squeeze()
        controls[k, :] = u_cmd.squeeze()

        y_dev = y - y_eq
        u_dev = u_cmd - u_eq

        x_hat_dev = kf.step(u=u_dev, y=y_dev)["x_filt"]
        x_hat = x_hat_dev + x_eq
        states_est[k, :] = x_hat.squeeze()

        u_dev_next = -K @ x_hat_dev
        u_cmd = u_eq + u_dev_next
        u_cmd = np.clip(u_cmd, -5.0, 5.0)

    # medir errores
    err = np.abs(states[:, :2] - y_eq.squeeze())
    iae_phi = np.sum(err[:, 0]) * Ts
    iae_psi = np.sum(err[:, 1]) * Ts
    iae_total = iae_phi + iae_psi

    i0, i1 = [1000, 1500]
    iae_dist_phi = np.sum(err[i0:i1, 0]) * Ts # iae phi disturbance
    iae_dist_psi = np.sum(err[i0:i1, 1]) * Ts # iae psi disturbance
    iae_dist_total = iae_dist_phi + iae_dist_psi # total iae disturbance

    metrics = {
        "iae_phi": iae_phi,
        "iae_psi": iae_psi,
        "iae_total": iae_total,
        "iae_dist_phi": iae_dist_phi,
        "iae_dist_psi": iae_dist_psi,
        "iae_dist_total": iae_dist_total,
    }

    return {
        "ref": reference,
        "x_eq": x_eq.squeeze(),
        "u_eq": u_eq.squeeze(),
        "K": K,
        "states": states,
        "states_est": states_est,
        "measurements": measurements,
        "controls": controls,
        "metrics": metrics,
    }


def main():
    references = [
        (0.0, 0.0),
        (0.0, np.pi / 4),
        (np.pi / 4, np.pi / 4),
    ]

    results = [simulate_reference(ref) for ref in references]

    t_vec = np.arange(N) * Ts
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

    # plt
    for idx, res in enumerate(results):
        r0, r1 = res["ref"]
        states = res["states"]
        states_est = res["states_est"]
        controls = res["controls"]

        ax_ang = axes[idx, 0]
        ax_u = axes[idx, 1]

        ax_ang.plot(t_vec, states[:, 0], label="phi (nonlinear)", lw=1.1)
        ax_ang.plot(t_vec, states[:, 1], label="psi (nonlinear)", lw=1.1)
        ax_ang.plot(t_vec, states_est[:, 0], "--", label="phi_hat (KF)", lw=1.0)
        ax_ang.plot(t_vec, states_est[:, 1], "--", label="psi_hat (KF)", lw=1.0)
        ax_ang.axhline(r0, color="k", ls=":", lw=1.0, label="phi_ref" if idx == 0 else None)
        ax_ang.axhline(r1, color="gray", ls=":", lw=1.0, label="psi_ref" if idx == 0 else None)
        ax_ang.axvspan(1000 * Ts, 1500 * Ts, color="red", alpha=0.08)
        ax_ang.set_ylabel("angles [rad]")
        ax_ang.set_title(f"Ref=({r0:.3f}, {r1:.3f}) | IAE={res['metrics']['iae_total']:.3f}")
        ax_ang.grid(True, alpha=0.3)
        ax_ang.legend(loc="upper right", ncol=2, fontsize=8)

        ax_u.plot(t_vec, controls[:, 0], label="u0", lw=1.1)
        ax_u.plot(t_vec, controls[:, 1], label="u1", lw=1.1)
        ax_u.axvspan(1000 * Ts, 1500 * Ts, color="red", alpha=0.08)
        ax_u.set_ylabel("input [V]")
        ax_u.grid(True, alpha=0.3)
        ax_u.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    fig.suptitle("Closed-loop de twin rotor con LQR y Kalman Filter", fontsize=16)
    fig.tight_layout()

    fig_path = os.path.join(results_dir, "closed_loop_kf_lqr.png")
    fig.savefig(fig_path, dpi=150)

    print("=== RESUMEN ===")
    for i, res in enumerate(results, start=1):
        ref = res["ref"]
        m = res["metrics"]
        print(
            f"Case {i} ref=({ref[0]:.5f}, {ref[1]:.5f}) | "
            f"IAE_phi={m['iae_phi']:.4f}, IAE_psi={m['iae_psi']:.4f}, IAE_total={m['iae_total']:.4f}, "
            f"IAE_dist={m['iae_dist_total']:.4f}"
        )

if __name__ == "__main__":
    main()
