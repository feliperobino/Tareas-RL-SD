import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from scripts.Quadrotor3D import Quadrotor3D
from scripts.QuadrotorTrajectories import QuadrotorTrajectory

from LQR import LQR, transform_references
from TD3 import TD3Agent, TD3Config, ReplayBuffer, linear_schedule, set_seed

from reps_algorithm import RepsAlgorithm

### DOMAIN RANDOMIZATION
Ts = 0.01
known_params = {
    "m": 0.027,  # kg
    "L": 0.0397,  # m
    "g": 9.8,  # m/s²
}

# rangos de randomización por ep
unknown_params = {
    "Ixx": [0.6e-5, 3.0e-5],
    "Iyy": [0.6e-5, 3.0e-5],
    "Izz": [1.0e-5, 5.0e-5],
    "KF": [2.0e-10, 4.5e-10],
    "KM": [4.0e-12, 1.3e-11],
}

def sample_dr_params(rng, known, unknown):
    out = dict(known)
    # iteramos para cada uno y agregamos ruido
    for k, (low, high) in unknown.items():
        out[k] = rng.uniform(low=low, high=high)

    return out

rng = np.random.default_rng(0)

### DOMAIN ADAPTATION
def init_da_state(unknown):
    keys = list(unknown.keys())
    lows = np.array([unknown[k][0] for k in keys], dtype=np.float64)
    highs = np.array([unknown[k][1] for k in keys], dtype=np.float64)
    mean = 0.5 * (lows + highs)
    std = 0.5 * (highs - lows)
    std_min = 0.05 * (highs - lows)
    return {
        "keys": keys,
        "low": lows,
        "high": highs,
        "mean": mean,
        "std": std,
        "std_min": std_min,
    }


def sample_da_params(rng, known, da_state):
    sample_vec = rng.normal(loc=da_state["mean"], scale=da_state["std"])
    sample_vec = np.clip(sample_vec, da_state["low"], da_state["high"])
    out = dict(known)
    for i, k in enumerate(da_state["keys"]):
        out[k] = float(sample_vec[i])
    return out, sample_vec


def update_da_distribution(reps, da_state, rewards, samples, epsilon=0.5, smooth=0.2):
    rewards = np.asarray(rewards, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    if rewards.size < 2 or samples.shape[0] < 2:
        return da_state

    new_mean, new_std = reps.optimize(rewards=rewards, samples=samples, epsilon=epsilon)
    da_state["mean"] = (1.0 - smooth) * da_state["mean"] + smooth * new_mean
    da_state["std"] = (1.0 - smooth) * da_state["std"] + smooth * new_std
    da_state["std"] = np.maximum(da_state["std"], da_state["std_min"])
    da_state["mean"] = np.clip(da_state["mean"], da_state["low"], da_state["high"])
    return da_state




### ENVIRONMENT
env_train = Quadrotor3D(Ts=Ts) # con DR
env_eval = Quadrotor3D(Ts=Ts) # fijo, para eval

### TRAJECTORY
traj_gen = QuadrotorTrajectory(sample_time=Ts)
refs_raw = traj_gen.helix(length=12, num_cycles=3, radius=1, rise=0.5,  center=(0., 0., 0.5))
references = transform_references(refs_raw)

## LQR BASE (pi_h)
lqr_model_params = sample_dr_params(rng, known_params, unknown_params)
lqr_model = Quadrotor3D(Ts=Ts, params=lqr_model_params)

Q_lqr = np.diag([
        5., 2.,  # x,  ẋ
        5., 2.,  # y,  ẏ
        5., 2.,  # z,  ż
        1., 1., 0.1,  # φ,  θ,  ψ
        0.1, 0.1, 0.1,  # p,  q,  r
    ])
R_lqr = np.eye(4) * 0.01

f_hover = known_params['m'] * known_params['g'] / 4.0
u_min = 0.0
u_max = f_hover * 2.5

lqr = LQR(lqr_model, n=12, m=4, Q=Q_lqr, R=R_lqr, u_lims=[u_min, u_max]) 


### CONTROLADOR RESIDUAL (pi_r) TD3
### reward function residual
def reward_function(state, action, next_state, reference):
    # minimizar error de ref
    pos = state[[0, 2, 4]] # x, y, z
    ref_pos = reference[[0, 2, 4]]
    dist = np.linalg.norm(pos - ref_pos)

    action_penalty =  0.001 * np.linalg.norm(action) # castigando acción grande

    reward = -dist - action_penalty
    return reward

### Agente
delta_max = 0.5 * f_hover # maximo cambio permitido al LQR

agent = TD3Agent(
    reward_function=reward_function,
    obs_dim=12*2 + 4, # estado actual + referencia + acción del LQR
    act_dim=4,
    act_low=-delta_max * np.ones(4),
    act_high=delta_max * np.ones(4),
    cfg=TD3Config(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    max_pos_error=2.0,
    max_angle_deg=60.0,
)


def train_residual( agent, lqr, env, references, known_params, unknown_params, rng, steps=250_000, seed=0, eval_every=1_000,
                   domain_adapt=False, da_adapt_every=10, da_batch_size=32, da_epsilon=0.5):
    set_seed(seed)
    
    # dims
    obs_dim = 12 * 2 + 4
    act_dim = 4
    max_steps = references.shape[1]

    # init replay buffer
    rb = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=agent.cfg.replay_size)

    reps = RepsAlgorithm()
    da_state = init_da_state(unknown_params)

    def _rollout_return_fixed_policy(params_rollout):
        state_roll, _ = env.reset(x0=references[:, 0], params=params_rollout)
        t_roll = 0
        ep_ret_roll = 0.0
        for _ in range(max_steps - 1):
            ref_roll = references[:, t_roll]
            u_lqr_roll = lqr(state_roll, ref_roll)
            obs_roll = agent.build_obs(state_roll, ref_roll, u_lqr_roll, env.obs_norm, f_hover)
            delta_u_roll = agent.act(obs_roll, noise_scale=0.0)
            u_total_roll = np.clip(u_lqr_roll + delta_u_roll, u_min, u_max)

            next_state_roll, _ = env.step(u_total_roll)
            t_roll = min(env.t, max_steps - 1)
            next_ref_roll = references[:, t_roll]

            reward_roll = reward_function(state_roll, u_total_roll, next_state_roll, next_ref_roll)
            unstable_roll = agent.is_unstable(next_state_roll, next_ref_roll)
            traj_done_roll = env.t >= max_steps
            if unstable_roll:
                reward_roll -= 10.0 * (max_steps - env.t)

            ep_ret_roll += float(reward_roll)
            state_roll = next_state_roll
            if traj_done_roll or unstable_roll:
                break

        return ep_ret_roll

    # random params de cada episodio
    # INIT RANDOM PROCESS N FOR EPXLORATION
    if domain_adapt: # DOMAIN ADAPTATION
        params, current_da_sample = sample_da_params(rng, known_params, da_state)
    else:
        params = sample_dr_params(rng, known_params, unknown_params)
        current_da_sample = None
    state, _ = env.reset(x0=references[:, 0], params=params)

    t_idx = 0
    ref = references[:, t_idx]
    u_lqr = lqr(state, ref)
    obs = agent.build_obs(state, ref, u_lqr, env.obs_norm, f_hover)
    ep_return = 0.0
    ep_len = 0
    ep_num = 0
    best_return = -np.inf
    best_actor_state = None
    ep_length_history = []
    window_info = []
    metrics = {
        "episode_returns": [],
        "update_steps":    [],
        "critic_loss":     [],
        "actor_loss":      [],
        "q1_mean":         [],
        "q1_std":          [],
        "target_mean":     [],
        "target_std":      [],
    }
    file_suffix = "_domain_adapt" if domain_adapt else ""

    for t in range(1, steps + 1):
        # GET u_t = pi_theta(s_t) + noise(t)
        noise_frac = linear_schedule(
            t,
            steps,
            agent.cfg.exploration_start_frac,
            agent.cfg.exploration_end_frac,
        )
        noise_scale = noise_frac * (agent.act_high_np - agent.act_low_np)

        if t < agent.cfg.start_steps:
            delta_u = np.random.uniform(agent.act_low_np, agent.act_high_np).astype(np.float32)
        else:
            delta_u = agent.act(obs, noise_scale=noise_scale)

        # GET u'_t = u_t + pi_h(s_t) (con pi_h == LQR)
        u_total = np.clip(u_lqr + delta_u, u_min, u_max)

        # GET s_{t+1} ~ p( - | s_t, u'_t)
        next_state, _ = env.step(u_total)

        t_idx = min(env.t, max_steps - 1)
        next_ref = references[:, t_idx]

        reward = reward_function(state, u_total, next_state, next_ref)

        traj_done = env.t >= max_steps
        unstable = agent.is_unstable(next_state, next_ref)
        if unstable:
            reward -= 10.0 * (max_steps - env.t) # penalty sobre fun reward base para castigar inestabilidad

        done = traj_done or unstable
        next_u_lqr = lqr(next_state, next_ref)
        next_obs = agent.build_obs(next_state, next_ref, next_u_lqr, env.obs_norm, f_hover)

        ## STORE (s_t, u_t, s_t+1, r_t) INTO REPLAY BUFFER
        rb.add(obs, delta_u, reward, next_obs, float(done))

        state = next_state
        ref = next_ref
        u_lqr = next_u_lqr
        obs = next_obs

        ep_return += float(reward)
        ep_len += 1

        if t >= agent.cfg.update_after and t % agent.cfg.update_every == 0 and rb.size >= agent.cfg.batch_size:
            info = agent.update(rb)
            window_info.append(info)

        if done: # ep terminado
            ep_num += 1
            print(
                f"[residual-train] step={t:>7d} ep={ep_num:>4d} "
                f"return={ep_return:>9.2f} len={ep_len:>4d}"
            )

            if domain_adapt and ep_num % da_adapt_every == 0:
                rewards_batch = []
                samples_batch = []
                for _ in range(da_batch_size):
                    params_eval, sample_eval = sample_da_params(rng, known_params, da_state)
                    ret_eval = _rollout_return_fixed_policy(params_eval)
                    rewards_batch.append(ret_eval)
                    samples_batch.append(sample_eval)

                if len(rewards_batch) >= 2:
                    da_state = update_da_distribution(
                        reps=reps,
                        da_state=da_state,
                        rewards=rewards_batch,
                        samples=samples_batch,
                        epsilon=da_epsilon,
                    )
                    print(
                        f"[domain-adapt] ep={ep_num:>4d} mean={da_state['mean']} std={da_state['std']}"
                    )

            # guardamos model si es mejor q el q tenemos
            if ep_return > best_return:
                best_return = ep_return
                best_actor_state = {
                    k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()
                }

            # nueva randomización para el próximo episodio
            if domain_adapt:
                params, current_da_sample = sample_da_params(rng, known_params, da_state)
            else:
                params = sample_dr_params(rng, known_params, unknown_params)
                current_da_sample = None
            state, _ = env.reset(x0=references[:, 0], params=params)
            t_idx = 0
            ref = references[:, t_idx]
            u_lqr = lqr(state, ref)
            obs = agent.build_obs(state, ref, u_lqr, env.obs_norm, f_hover)

            metrics["episode_returns"].append(ep_return)
            ep_length_history.append(ep_len)
            ep_return = 0.0
            ep_len = 0

        if t % eval_every == 0:

            if window_info:
                cl = float(np.mean([i["critic_loss"] for i in window_info]))
                al_v = [i["actor_loss"] for i in window_info if not np.isnan(i["actor_loss"])]
                al = float(np.mean(al_v)) if al_v else float("nan")
                qm = float(np.mean([i["q1_mean"] for i in window_info]))
                qs = float(np.mean([i["q1_std"] for i in window_info]))
                tm = float(np.mean([i["target_mean"] for i in window_info]))
                ts = float(np.mean([i["target_std"] for i in window_info]))

                metrics["update_steps"].append(t)
                metrics["critic_loss"].append(cl)
                metrics["actor_loss"].append(al)
                metrics["q1_mean"].append(qm)
                metrics["q1_std"].append(qs)
                metrics["target_mean"].append(tm)
                metrics["target_std"].append(ts)

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'p3'))
    os.makedirs(results_dir, exist_ok=True)
    model_name = "best_td3_residual_actor_domain_adaptation.pt" if domain_adapt else "best_td3_residual_actor.pt"
    best_model_path = os.path.join(results_dir, model_name)
    if best_actor_state is not None:
        torch.save(best_actor_state, best_model_path)
        print(f"[residual-train] best model saved to {best_model_path} | best_return={best_return:.2f}")
    else:
        print("[residual-train] no completed episode with improvement; existing checkpoint was kept.")


    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(ep_length_history, label='Len')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(metrics['episode_returns'], label='Return')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'td3_residual_train_episode_returns{file_suffix}.png'), dpi=200, bbox_inches='tight')
    plt.show()

    return metrics


def evaluate_residual(agent, lqr, env, references, known_params, f_hover, u_min, u_max, domain_adapt=False):
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'p3'))
    os.makedirs(results_dir, exist_ok=True)
    file_suffix = "_domain_adapt" if domain_adapt else ""

    agent.actor.eval()
    act_dim = 4
    max_steps = references.shape[1]

    state, _ = env.reset(x0=references[:, 0])
    env_lqr = copy.deepcopy(env)
    state_lqr, _ = env_lqr.reset(x0=references[:, 0])
    t_idx = 0

    states   = []
    states_lqr = []
    refs     = []
    u_totals = []
    rewards  = []
    pos_errors = []
    unstable = False

    for k in range(max_steps - 1):
        ref    = references[:, t_idx]
        u_lqr  = lqr(state, ref)
        obs    = agent.build_obs(state, ref, u_lqr, env.obs_norm, f_hover)
        delta_u = agent.act(obs, noise_scale=0.0)
        u_total = np.clip(u_lqr + delta_u, u_min, u_max)

        u_lqr_only = lqr(state_lqr, ref)

        next_state, _ = env.step(u_total)
        next_state_lqr, _ = env_lqr.step(u_lqr_only)
        t_idx = min(env.t, max_steps - 1)
        next_ref = references[:, t_idx]

        reward = reward_function(state, u_total, next_state, next_ref)
        traj_done = env.t >= max_steps
        unstable  = agent.is_unstable(next_state, next_ref)
        if unstable:
            reward -= 10.0 * (max_steps - env.t)

        states.append(np.asarray(next_state))
        states_lqr.append(np.asarray(next_state_lqr))
        refs.append(np.asarray(next_ref))
        u_totals.append(np.asarray(u_total))
        rewards.append(float(reward))
        pos_errors.append(float(np.linalg.norm(next_state[[0, 2, 4]] - next_ref[[0, 2, 4]])))

        state = next_state
        state_lqr = next_state_lqr
        if traj_done or unstable:
            break

    states = np.asarray(states)
    states_lqr = np.asarray(states_lqr)
    refs = np.asarray(refs)
    actions = np.asarray(u_totals)
    rewards = np.asarray(rewards)
    pos_errors = np.asarray(pos_errors)

    total_return = float(np.sum(rewards)) if len(rewards) > 0 else 0.0
    mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
    rmse_pos = float(np.sqrt(np.mean(pos_errors ** 2))) if len(pos_errors) > 0 else np.nan

    t = np.arange(len(states)) * env.Ts


    plt.figure(figsize=(15, 7))

    plt.subplot(3, 1, 1)
    plt.plot(t, refs[:, 0], "k--", label="x ref")
    plt.plot(t, states_lqr[:, 0], label="x LQR")
    plt.plot(t, states[:, 0], label="x Residual")
    plt.ylabel("x [m]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, refs[:, 2], "k--", label="y ref")
    plt.plot(t, states_lqr[:, 2], label="y LQR")
    plt.plot(t, states[:, 2], label="y Residual")
    plt.ylabel("y [m]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, refs[:, 4], "k--", label="z ref")
    plt.plot(t, states_lqr[:, 4], label="z LQR")
    plt.plot(t, states[:, 4], label="z Residual")
    plt.ylabel("z [m]")
    plt.xlabel("t [s]")
    plt.grid(True)
    plt.legend()

    plt.suptitle(
        f"Residual eval — return={total_return:.2f}, "
        f"RMSE pos={rmse_pos:.3f}, steps={len(states)}, unstable={unstable}"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'td3_residual_eval_xyz{file_suffix}.png'), dpi=200, bbox_inches='tight')

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
        states_lqr[:, 0],
        states_lqr[:, 2],
        states_lqr[:, 4],
        linewidth=1.5,
        label="LQR trajectory",
    )

    ax.plot(
        states[:, 0],
        states[:, 2],
        states[:, 4],
        linewidth=1.5,
        label="Residual trajectory",
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Quadrotor Residual — Reference vs Followed Trajectory")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'td3_residual_eval_trajectory_3d{file_suffix}.png'), dpi=200, bbox_inches='tight')

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

    plt.suptitle("Residual actions and rewards")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'td3_residual_eval_actions_rewards{file_suffix}.png'), dpi=200, bbox_inches='tight')
    plt.show()

    print(
        f"[residual-eval] return={total_return:.2f} | "
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


def evaluate_da_vs_dr(agent, lqr, env, references, known_params, f_hover, u_min, u_max):
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'p3'))
    os.makedirs(results_dir, exist_ok=True)

    dr_model_path = os.path.join(results_dir, "best_td3_residual_actor.pt")
    da_model_path = os.path.join(results_dir, "best_td3_residual_actor_domain_adaptation.pt")

    if not os.path.exists(dr_model_path) or not os.path.exists(da_model_path):
        print("[compare-da-dr] missing checkpoint(s); run/evaluate skipped.")
        print(f"[compare-da-dr] required: {dr_model_path}")
        print(f"[compare-da-dr] required: {da_model_path}")
        return None

    def _rollout_with_loaded_actor(actor_state):
        agent.actor.load_state_dict(actor_state)
        agent.actor.eval()
        max_steps = references.shape[1]

        state, _ = env.reset(x0=references[:, 0])
        t_idx = 0

        states = []
        refs = []
        actions = []
        rewards = []
        pos_errors = []
        unstable = False

        for _ in range(max_steps - 1):
            ref = references[:, t_idx]
            u_lqr = lqr(state, ref)
            obs = agent.build_obs(state, ref, u_lqr, env.obs_norm, f_hover)
            delta_u = agent.act(obs, noise_scale=0.0)
            u_total = np.clip(u_lqr + delta_u, u_min, u_max)

            next_state, _ = env.step(u_total)
            t_idx = min(env.t, max_steps - 1)
            next_ref = references[:, t_idx]

            reward = reward_function(state, u_total, next_state, next_ref)
            traj_done = env.t >= max_steps
            unstable = agent.is_unstable(next_state, next_ref)
            if unstable:
                reward -= 10.0 * (max_steps - env.t)

            states.append(np.asarray(next_state))
            refs.append(np.asarray(next_ref))
            actions.append(np.asarray(u_total))
            rewards.append(float(reward))
            pos_errors.append(float(np.linalg.norm(next_state[[0, 2, 4]] - next_ref[[0, 2, 4]])))

            state = next_state
            if traj_done or unstable:
                break

        states = np.asarray(states)
        refs = np.asarray(refs)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        pos_errors = np.asarray(pos_errors)

        return {
            "states": states,
            "refs": refs,
            "actions": actions,
            "rewards": rewards,
            "rmse_pos": float(np.sqrt(np.mean(pos_errors ** 2))) if len(pos_errors) > 0 else np.nan,
            "return": float(np.sum(rewards)) if len(rewards) > 0 else 0.0,
            "unstable": bool(unstable),
        }

    actor_backup = copy.deepcopy(agent.actor.state_dict())
    dr_state = torch.load(dr_model_path, map_location=agent.device)
    da_state = torch.load(da_model_path, map_location=agent.device)

    dr_rollout = _rollout_with_loaded_actor(dr_state)
    da_rollout = _rollout_with_loaded_actor(da_state)

    agent.actor.load_state_dict(actor_backup)

    n_plot = min(len(dr_rollout["states"]), len(da_rollout["states"]), len(da_rollout["refs"]))
    t = np.arange(n_plot) * env.Ts
    refs = da_rollout["refs"][:n_plot]
    dr_states = dr_rollout["states"][:n_plot]
    da_states = da_rollout["states"][:n_plot]

    plt.figure(figsize=(15, 7))
    plt.subplot(3, 1, 1)
    plt.plot(t, refs[:, 0], "k--", label="x ref")
    plt.plot(t, dr_states[:, 0], label="x DR")
    plt.plot(t, da_states[:, 0], label="x DA")
    plt.ylabel("x [m]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, refs[:, 2], "k--", label="y ref")
    plt.plot(t, dr_states[:, 2], label="y DR")
    plt.plot(t, da_states[:, 2], label="y DA")
    plt.ylabel("y [m]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, refs[:, 4], "k--", label="z ref")
    plt.plot(t, dr_states[:, 4], label="z DR")
    plt.plot(t, da_states[:, 4], label="z DA")
    plt.ylabel("z [m]")
    plt.xlabel("t [s]")
    plt.grid(True)
    plt.legend()

    plt.suptitle(
        f"DA vs DR — RMSE DR={dr_rollout['rmse_pos']:.3f}, RMSE DA={da_rollout['rmse_pos']:.3f}"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'td3_residual_eval_da_vs_dr_xyz_domain_adapt.png'), dpi=200, bbox_inches='tight')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(refs[:, 0], refs[:, 2], refs[:, 4], "k--", linewidth=2, label="Reference")
    ax.plot(dr_states[:, 0], dr_states[:, 2], dr_states[:, 4], linewidth=1.5, label="DR trajectory")
    ax.plot(da_states[:, 0], da_states[:, 2], da_states[:, 4], linewidth=1.5, label="DA trajectory")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Quadrotor Residual — DA vs DR")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'td3_residual_eval_da_vs_dr_trajectory_3d_domain_adapt.png'), dpi=200, bbox_inches='tight')
    plt.show()

    print(
        f"[compare-da-dr] DR return={dr_rollout['return']:.2f} rmse={dr_rollout['rmse_pos']:.3f} unstable={dr_rollout['unstable']}"
    )
    print(
        f"[compare-da-dr] DA return={da_rollout['return']:.2f} rmse={da_rollout['rmse_pos']:.3f} unstable={da_rollout['unstable']}"
    )

    return {
        "dr": dr_rollout,
        "da": da_rollout,
    }

def plot_residual_vs_LQR(agent, lqr, env, references, known_params, f_hover, u_min, u_max):

    residual_traj = evaluate_residual(agent, lqr, env, references, known_params, f_hover, u_min, u_max)
    lqr_traj = lqr.simulate(references)

    

if __name__ == "__main__":
    asd = input("Entrenar [y/n]? ")
    train = False

    efg = input("Usar Domain Randomization [1] o Domain Adaptation [2]? ")
    domain_adapt = False
    if efg == "2":
        domain_adapt = True

    if asd.lower() == "y":
        train = True

    if train:
        train_residual(
            agent=agent,
            lqr=lqr,
            env=copy.deepcopy(env_train),
            references=references,
            known_params=known_params,
            unknown_params=unknown_params,
            rng=rng,
            steps=250_000,
            seed=0,
            domain_adapt=domain_adapt,
        )
    else:
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'p3'))
        model_name = "best_td3_residual_actor_domain_adaptation.pt" if domain_adapt else "best_td3_residual_actor.pt"
        best_model_path = os.path.join(results_dir, model_name)
        actor_state = torch.load(best_model_path, map_location=agent.device)
        agent.actor.load_state_dict(actor_state)

    ## plot trajs y errores
    if domain_adapt:
        evaluate_da_vs_dr(
            agent=agent,
            lqr=lqr,
            env=copy.deepcopy(env_eval),
            references=references,
            known_params=known_params,
            f_hover=f_hover,
            u_min=u_min,
            u_max=u_max,
        )
    else:
        evaluate_residual(
            agent=agent,
            lqr=lqr,
            env=copy.deepcopy(env_eval),
            references=references,
            known_params=known_params,
            f_hover=f_hover,
            u_min=u_min,
            u_max=u_max,
            domain_adapt=domain_adapt,
        )