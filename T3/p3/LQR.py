import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time
import os
from scipy.linalg import solve_discrete_are


# Default cost matrices used by reward_function_simple.
Q_lqr = np.diag([
    5., 2.,  # x, x_dot
    5., 2.,  # y, y_dot
    5., 2.,  # z, z_dot
    1., 1., 0.1,  # phi, theta, psi
    0.1, 0.1, 0.1,  # p, q, r
])
R_lqr = np.eye(4) * 0.01


def transform_references(references_raw):
    x = references_raw[0][:, 0].reshape(1, -1)
    y = references_raw[0][:, 1].reshape(1, -1)
    z = references_raw[0][:, 2].reshape(1, -1)
    x_dot = references_raw[1][:, 0].reshape(1, -1)
    y_dot = references_raw[1][:, 1].reshape(1, -1)
    z_dot = references_raw[1][:, 2].reshape(1, -1)
    X_ref = np.concatenate([x, x_dot, y, y_dot, z, z_dot], axis=0)
    X_ref = np.concatenate([X_ref, np.zeros_like(X_ref)], axis=0)
    return X_ref


class LQR:
    def __init__(self, model, n, m, Q, R, u_lims=[]):
        self.n = n
        self.m = m
        self.model = model
        self.Q = Q
        self.R = R
        self.u_lims = u_lims
        self.K, self.x_eq, self.u_eq = self.lqr_matrix()

    def lqr_matrix(self):
        # 1) Linearize the system for the given reference
        x_eq, u_eq = self.model.get_eq_point(z_ref=1)  # Input corresponding to z
        Ad, Bd, Cd = self.model.get_linear_system(x_eq=x_eq, u_eq=u_eq)
        x_eq = np.array(x_eq)
        u_eq = np.array(u_eq)

        # 2) Solve for K
        P = solve_discrete_are(Ad, Bd, self.Q, self.R)
        K = np.linalg.inv(self.R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad
        return K, x_eq, u_eq

    # Construct the LQR problem
    def __call__(self, x, ref):
        x = np.array(x)
        ref = np.array(ref)

        # 1) Center at new eq
        x = x - self.x_eq
        ref = ref - self.x_eq
        u_min = self.u_lims[0] - self.u_eq[0]
        u_max = self.u_lims[1] - self.u_eq[0]

        # 2) Produce the output
        x_error = x - ref
        u = - self.K @ x_error
        u_out = np.clip(u + self.u_eq, self.u_lims[0], self.u_lims[1])
        return u_out
    
    def recompute_K(self, Q, R): ## helper nueva para meter sintonización de valores con TD3
        Ab, Bd, _ = self.model.get_linear_system(x_eq=self.x_eq, u_eq=self.u_eq)
        P = solve_discrete_are(Ab, Bd, Q, R)

        self.K = np.linalg.inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ab
        self.Q = Q
        self.R = R


def reward_function_complex(ref, state, a_rl, prev_a_rl, f_hover):
        pos_err = state[[0, 2, 4]] - ref[[0, 2, 4]]
        vel_err = state[[1, 3, 5]] - ref[[1, 3, 5]]
        roll_pitch = state[[6, 7]]
        body_rates = state[[9, 10, 11]]

        pos_cost = np.sum((pos_err / 2.0) ** 2)
        vel_cost = np.sum((vel_err / 1) ** 2)
        angle_cost = np.sum((roll_pitch / np.deg2rad(60.0)) ** 2)
        rate_cost = np.sum((body_rates / 5.0) ** 2)

        if prev_a_rl is None:
            smooth_cost = 0.0
        else:
            smooth_cost = np.sum(((a_rl - prev_a_rl) / f_hover) ** 2)

        reward = (
                - 5.0 * pos_cost
                - 0.5 * vel_cost
                - 0 * angle_cost
                - 0 * rate_cost
                - 0.5 * smooth_cost
        )

        return float(reward)


def reward_function_simple(ref, states, inputs, Q=Q_lqr, R=R_lqr): # To be used for the residual task
    err = ref - states
    state_cost = err.reshape(1, -1) @ Q @ err.reshape(-1, 1)
    u_cost = inputs.reshape(1, -1) @ R @ inputs.reshape(-1, 1)
    return -state_cost[0, 0] - u_cost[0, 0]



if __name__ == '__main__':
    from scripts.Quadrotor3D import Quadrotor3D
    from scripts.QuadrotorTrajectories import QuadrotorTrajectory
    import copy

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'p3'))
    os.makedirs(results_dir, exist_ok=True)

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

    # System
    Ts = 0.01
    system = Quadrotor3D(Ts=Ts)


    # Trajectory
    trajectory_gen = QuadrotorTrajectory(sample_time=Ts)
    LENGTH = 12
    references_raw = trajectory_gen.helix(length=LENGTH, num_cycles=3, radius=1, rise=0.5,  center=(0., 0., 0.5))
    references = transform_references(references_raw)


    # Controller
    Q_lqr = np.diag([
        5., 2.,  # x,  ẋ
        5., 2.,  # y,  ẏ
        5., 2.,  # z,  ż
        1., 1., 0.1,  # φ,  θ,  ψ
        0.1, 0.1, 0.1,  # p,  q,  r
    ])
    R_lqr = np.eye(4) * 0.01
    rng = np.random.default_rng(seed=42)
    unk_params = {k: rng.uniform(low=v[0], high=v[1]) for k, v in unknown_params.items()}

    # LQR has a wrong fixed model
    lqr_model_params = known_params.copy()
    lqr_model_params.update(unk_params)
    lqr_internal_model = Quadrotor3D(Ts=Ts, params=lqr_model_params)
    u_min = 0
    n = len(lqr_internal_model.x)
    m = 4
    u_max = (lqr_internal_model.p['m'] * lqr_internal_model.p['g'] / 4.0) * 2.5
    lqr = LQR(lqr_internal_model, n, m, Q_lqr, R_lqr, u_lims=[u_min, u_max])

    # Simulation
    N_steps = references.shape[1]
    x_current = references[:, 0]
    system.x = list(x_current)
    f_hover = known_params['m'] * known_params['g'] / 4.0
    last_u = np.ones(4)*f_hover

    x_list = []
    x_est_list = []
    u_list = []
    y_list = []
    total_reward = 0
    total_reward_simp = 0
    init_time = time.time()
    
    rmse_inst_list = []
    pos_sq_err_hist = []

    for k in range(N_steps):
        # Extract the reference
        ref = references[:, k]

        # Get LQR action
        u_opt = lqr(x_current, ref)

        # Step the nonlinear system
        _, y = system.step(u_opt)


        total_reward += reward_function_complex(ref, x_current, u_opt,last_u, f_hover)
        total_reward_simp += reward_function_simple(ref, x_current, u_opt)


        x_current = np.array(system.x)
        pos_err = x_current[[0, 2, 4]] - ref[[0, 2, 4]]
        pos_sq_err = pos_err ** 2
        pos_sq_err_hist.append(pos_sq_err)

        rmse_inst = float(np.sqrt(np.mean(pos_sq_err)))
        rmse_inst_list.append(rmse_inst)

        rmse_cum = float(np.sqrt(np.mean(np.array(pos_sq_err_hist))))
        x_list.append(np.array(system.x).reshape(-1, 1))
        y_list.append(np.array(y).reshape(-1, 1))
        u_list.append(np.array(u_opt).reshape(-1, 1))

        last_u = u_opt

        if k % 100 == 0:
            print('Step {}|{} | RMSE_inst(pos): {:.4f} m | RMSE_cum(pos): {:.4f} m'.format(k, N_steps, rmse_inst, rmse_cum))

    print('Total time: {}'.format(time.time() - init_time))
    print('Total reward: {}'.format(total_reward))
    print('Total reward Simp: {}'.format(total_reward_simp))

    x_list = np.concatenate(x_list, axis=1)
    y_list = np.concatenate(y_list, axis=1)
    u_list = np.concatenate(u_list, axis=1)
    print(x_list.shape, references.shape)

    pos_err_all = x_list[[0, 2, 4], :] - references[[0, 2, 4], :]
    rmse_xyz = np.sqrt(np.mean(pos_err_all**2, axis=1))
    rmse_pos = np.sqrt(np.mean(pos_err_all**2))
    print('RMSE final x,y,z [m] = {}'.format(rmse_xyz))
    print('RMSE final posicion global [m] = {}'.format(rmse_pos))


    plt.figure(figsize=(15, 7))
    plt.subplot(4, 1, 1)
    plt.plot(references[0, :], 'k--')
    plt.plot(x_list[0, :], label='x')
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(references[2, :], 'k--')
    plt.plot(x_list[2, :], label='y')
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(references[4, :], 'k--')
    plt.plot(x_list[4, :], label='z')
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 4)
    for i in range(u_list.shape[0]):
        plt.plot(u_list[i, :], label='u{}'.format(i + 1))
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lqr_tracking_xyz_u.png'), dpi=200, bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(references[0, :], references[2, :], references[4, :],
            color='royalblue', linewidth=2, linestyle='--', label='Reference')
    ax.plot(x_list[0, :], x_list[2, :], x_list[4, :],
            color='tomato', linewidth=1.5, label='LQR trajectory')


    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Quadrotor LQR — Reference vs Followed Trajectory')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lqr_trajectory_3d.png'), dpi=200, bbox_inches='tight')
    plt.show()