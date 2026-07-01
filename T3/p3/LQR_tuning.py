import numpy as np
import torch
from LQR import LQR
import copy
import os
from dataclasses import dataclass
from typing import Any, Optional
import matplotlib.pyplot as plt
import time
from scipy.linalg import solve_discrete_are

from LQR import LQR, reward_function_complex, transform_references, reward_function_simple
from TD3 import TD3Agent, TD3Config, ReplayBuffer, set_seed, linear_schedule

from scripts.Quadrotor3D import Quadrotor3D
from scripts.QuadrotorTrajectories import QuadrotorTrajectory



# matrices base
Q_lqr = np.diag([
    5., 2.,  # x,  ẋ
    5., 2.,  # y,  ẏ
    5., 2.,  # z,  ż
    1., 1., 0.1,  # φ,  θ,  ψ
    0.1, 0.1, 0.1,  # p,  q,  r
])
R_lqr = np.eye(4) * 0.01


## dims de estado y acción
N_Q = 12 # q diag tiene 12 elementos
N_R = 4 # r diag tiene 4 elementos
act_dim = N_Q + N_R

# escalamiento de ganancias bounded
act_low = np.ones(act_dim)*0.1
act_high = np.ones(act_dim)*10.0

## observación del agente TD3
def build_obs_tuning(state, ref, obs):
    # agente ve error de estado + estado actual (no ve u pq no controla)
    error = state - ref
    obs = np.concatenate([
        state / obs,
        error / obs,
    ])
    return np.clip(obs, -10.0, 10.0).astype(np.float32)

obs_dim = 24 # (12 de estado + 12 de error)

# aplicar acción del TD3 al LQR
def apply_action_to_lqr(lqr, action, Q_prev, R_prev):
    # accion es vec de 16 elems: escaldos de cada valor de Q y R
    q_scales = action[:N_Q] # primeros 12 para Q
    r_scales = action[N_Q:] # otros para R

    Q_new = np.diag(np.diag(Q_prev) * q_scales)
    R_new = np.diag(np.diag(R_prev) * r_scales)

    lqr.recompute_K(Q_new, R_new)
    return Q_new, R_new

### loop de entrenamiento (igual al de TD3 pero cambio de acción y obs)
u_max = (Quadrotor3D().p['m'] * Quadrotor3D().p['g'] / 4.0) * 2.5
lqr_base = LQR(model=Quadrotor3D(), n=12, m=4, Q=Q_lqr, R=R_lqr, u_lims=[0, u_max])



def train(self, env, lqr_base, references, known_params, unknown_params, seed, steps: int=100_000, eval_every: int=1000):
    set_seed(seed)

    lqr = copy.deepcopy(lqr_base)
    prev_action = None
    prev_u_lqr = None


    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'p3'))
    os.makedirs(results_dir, exist_ok=True)

    # Initialization
    f_hover = known_params['m'] * known_params['g'] / 4.0
    max_steps = references.shape[1]

    # Agent
    rb = ReplayBuffer(obs_dim, act_dim, capacity=self.cfg.replay_size)


    params = {**known_params, **self.sample_unknown(unknown_params)}
    state, _ = env.reset(x0=references[:, 0], params=params)

    t_idx = 0
    ref = references[:, t_idx]
    
    # acción inicial 1
    action = np.ones(act_dim)

    obs = build_obs_tuning(state, ref, env.obs_norm)
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
        # state, _ = env.step(action)

        # state y obs son del LQR
        apply_action_to_lqr(lqr, action, Q_lqr, R_lqr)
        u_lqr = lqr(state, ref)
        state, _ = env.step(u_lqr)

        t_idx = min(env.t, max_steps - 1)
        ref = references[:, t_idx]

        # Reward va con acción del LQR, no del TD3 -> indirecto
        reward = reward_function_complex(ref, state, u_lqr, prev_u_lqr, f_hover)

        # Termination
        traj_done = (env.t >= max_steps)
        unstable = self.is_unstable(state, ref)
        if unstable:
            reward -= 10 * (max_steps - env.t)
        done = traj_done or unstable

        ep_return += float(reward)
        ep_length += 1

        next_obs = torch.from_numpy(build_obs_tuning(state, ref, env.obs_norm))


        # Add to the replay buffer
        rb.add(obs, action, reward, next_obs, float(done))

        prev_action = action
        prev_u_lqr = u_lqr
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

            # copia del lqr para resetear cada ep
            lqr = copy.deepcopy(lqr_base)
            prev_action = None
            prev_u_lqr = None

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
            action = np.ones(act_dim)
            # obs = self.build_obs(state, ref, action, env.obs_norm, f_hover)
            # obs también cambia
            obs = build_obs_tuning(state, ref, env.obs_norm)


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
    plt.savefig(os.path.join(results_dir, 'td3_lqr_train_episode_returns.png'), dpi=200, bbox_inches='tight')
    plt.show()

    return self.metrics

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

    ## AGENTE TD3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = TD3Agent(
        reward_function=None, # se calcula externa con el del LQR
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
        cfg=TD3Config(),
        device=device,
        max_pos_error=2.0,
        max_angle_deg=60.0,
    )
    unknown_params_gaussian = {
    k: [(v[1]+v[0])/2.0, abs((v[1]+v[0])/2.0 - v[0])/2.0]
    for k, v in unknown_params.items()
    }

    ## ENTRENAMOS!!
    best_model_path = os.path.join(results_dir, 'best_td3_actor.pt')
    train_ans = input('Entrenar? [s/N]: ').strip().lower()
    do_train = train_ans in ('s', 'si', 'sí', 'y', 'yes', '1', 'true')

    if do_train:
        train(agent, copy.deepcopy(system), lqr, references,
            known_params, unknown_params_gaussian,
            seed=0, steps=100_000, eval_every=1_000)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"No existe el modelo guardado."
        )

    agent.actor.load_state_dict(
        torch.load(best_model_path, map_location=device))
    agent.actor.eval()

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

    print('Total time LQR: {}'.format(time.time() - init_time))
    print('Total reward LQR: {}'.format(total_reward))
    print('Total reward Simp LQR: {}'.format(total_reward_simp))

    x_list = np.concatenate(x_list, axis=1)
    y_list = np.concatenate(y_list, axis=1)
    u_list = np.concatenate(u_list, axis=1)
    print(x_list.shape, references.shape)

    pos_err_all = x_list[[0, 2, 4], :] - references[[0, 2, 4], :]
    rmse_xyz = np.sqrt(np.mean(pos_err_all**2, axis=1))
    rmse_pos = np.sqrt(np.mean(pos_err_all**2))
    print('RMSE final x,y,z [m] [LQR]= {}'.format(rmse_xyz))
    print('RMSE final posicion global [m] [LQR]= {}'.format(rmse_pos))


    ### LOOP TD3 con LQR:
    system_TD3 = Quadrotor3D(Ts=Ts)
    x_current_td3 = references[:, 0].copy()
    system_TD3.x = list(x_current_td3)

    lqr_td3 = copy.deepcopy(lqr)
    prev_action = None
    prev_u_td3 = None
    action = np.ones(act_dim)
    obs = build_obs_tuning(x_current_td3, references[:, 0], system_TD3.obs_norm)

    x_list_td3, u_list_td3 = [], []
    q_diag_td3_hist, r_diag_td3_hist = [], []
    total_reward_td3 = 0.0
    pos_sq_err_hist_td3 = []

    for K in range(N_steps):
        ref = references[:, K]

        action = agent.act(torch.from_numpy(obs).float().to(device), noise_scale=0.0)
        Q_td3_k, R_td3_k = apply_action_to_lqr(lqr_td3, action, Q_lqr, R_lqr)
        u_opt_td3 = lqr_td3(x_current_td3, ref)

        system_TD3.step(u_opt_td3)
        x_current_td3 = np.array(system_TD3.x)

        total_reward_td3 += reward_function_complex(ref, x_current_td3, u_opt_td3, prev_u_td3, f_hover)

        pos_sq_err_hist_td3.append((x_current_td3[[0, 2, 4]] - ref[[0, 2, 4]]) ** 2)
        x_list_td3.append(x_current_td3.copy())
        u_list_td3.append(u_opt_td3.copy())
        q_diag_td3_hist.append(np.diag(Q_td3_k).copy())
        r_diag_td3_hist.append(np.diag(R_td3_k).copy())

        obs = build_obs_tuning(x_current_td3, ref, system_TD3.obs_norm)
        prev_action = action
        prev_u_td3 = u_opt_td3

    # estados y acciones al sistema (no del TD3)
    x_list_td3 = np.array(x_list_td3).T 
    u_list_td3 = np.array(u_list_td3).T

    pos_err_td3 = x_list_td3[[0, 2, 4], :] - references[[0, 2, 4], :]
    rmse_td3 = np.sqrt(np.mean(pos_err_td3**2))

    ### print de resultados TD3+LQR
    print('Total reward TD3+LQR: {}'.format(total_reward_td3))
    print('RMSE final posicion global [m] TD3+LQR = {:.4f} m'.format(rmse_td3))
    print('Total reward SOLO LQR: {}'.format(total_reward))
    print('RMSE final posicion global [m] SOLO LQR = {:.4f} m'.format(rmse_pos))

    q_diag_td3_hist = np.array(q_diag_td3_hist)
    r_diag_td3_hist = np.array(r_diag_td3_hist)
    q_diag_final = q_diag_td3_hist[-1]
    r_diag_final = r_diag_td3_hist[-1]
    q_diag_mean = np.mean(q_diag_td3_hist, axis=0)
    r_diag_mean = np.mean(r_diag_td3_hist, axis=0)

    np.set_printoptions(precision=4, suppress=True)
    print('\n=== Matrices LQR aprendidas por TD3 ===')
    print('diag(Q) base        =', np.diag(Q_lqr))
    print('diag(Q) TD3 final   =', q_diag_final)
    print('diag(Q) TD3 promedio=', q_diag_mean)
    print('escala Q final/base =', q_diag_final / np.diag(Q_lqr))
    print('Q final == Q base?  =', np.allclose(q_diag_final, np.diag(Q_lqr), atol=1e-8))

    print('diag(R) base        =', np.diag(R_lqr))
    print('diag(R) TD3 final   =', r_diag_final)
    print('diag(R) TD3 promedio=', r_diag_mean)
    print('escala R final/base =', r_diag_final / np.diag(R_lqr))
    print('R final == R base?  =', np.allclose(r_diag_final, np.diag(R_lqr), atol=1e-8))

    plt.figure(figsize=(15, 7))
    plt.subplot(4, 1, 1)
    plt.plot(references[0, :], 'k--', linewidth=2, label='Reference')
    plt.plot(x_list[0, :], label='LQR', linewidth=1.5)
    plt.plot(x_list_td3[0, :], label='TD3+LQR', linewidth=1.5, alpha=0.8)
    plt.grid(True)
    plt.legend()
    plt.ylabel('X [m]')

    plt.subplot(4, 1, 2)
    plt.plot(references[2, :], 'k--', linewidth=2, label='Reference')
    plt.plot(x_list[2, :], label='LQR', linewidth=1.5)
    plt.plot(x_list_td3[2, :], label='TD3+LQR', linewidth=1.5, alpha=0.8)
    plt.grid(True)
    plt.legend()
    plt.ylabel('Y [m]')

    plt.subplot(4, 1, 3)
    plt.plot(references[4, :], 'k--', linewidth=2, label='Reference')
    plt.plot(x_list[4, :], label='LQR', linewidth=1.5)
    plt.plot(x_list_td3[4, :], label='TD3+LQR', linewidth=1.5, alpha=0.8)
    plt.grid(True)
    plt.legend()
    plt.ylabel('Z [m]')

    plt.subplot(4, 1, 4)
    plt.plot(u_list[0, :], label='LQR - u1', linewidth=1.5)
    plt.plot(u_list_td3[0, :], label='TD3+LQR - u1', linewidth=1.5, alpha=0.8)
    plt.plot(u_list[1, :], label='LQR - u2', linewidth=1.5)
    plt.plot(u_list_td3[1, :], label='TD3+LQR - u2', linewidth=1.5, alpha=0.8)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.ylabel('Control [N]')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lqr_vs_td3_lqr_tracking_xyz_u.png'), dpi=200, bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(references[0, :], references[2, :], references[4, :],
            color='k', linewidth=2.5, linestyle='--', label='Reference')
    ax.plot(x_list[0, :], x_list[2, :], x_list[4, :],
            color='tomato', linewidth=2, label='LQR trajectory')
    ax.plot(x_list_td3[0, :], x_list_td3[2, :], x_list_td3[4, :],
            color='limegreen', linewidth=2, label='TD3+LQR trajectory', linestyle=':')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Quadrotor Trajectories — Reference vs LQR vs TD3+LQR')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lqr_vs_td3_lqr_trajectory_3d_comparison.png'), dpi=200, bbox_inches='tight')
    plt.show()