from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ddpg import DDPG
from practice_env import make_env, make_person_params
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory, Transition


def load_workouts(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    workouts = data["workouts"]
    return sorted(workouts, key=lambda w: float(w["level"]))


def workout_action(raw_action: np.ndarray, obs: np.ndarray, workouts: list[dict], env) -> tuple[np.ndarray, dict]:
    # obs[3] is normalized benchmark proxy in measurement_mode="physiology".
    current_level = float(np.clip(obs[3], 0.0, 1.0))
    unlocked = [w for w in workouts if float(w["level"]) <= current_level]
    candidates = unlocked if unlocked else [workouts[0]]

    target = float(raw_action[0])
    selected = min(candidates, key=lambda w: abs(float(w["intensity_score"]) - target))
    action = np.array([float(selected["intensity_score"])], dtype=np.float32)
    action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
    return action, selected


def get_recent_type_counts(actions: list[float], workouts: list[dict], window: int = 7) -> dict[str, int]:
    if not actions or not workouts:
        return {}
    type_counts: dict[str, int] = {}
    for action_value in actions[-window:]:
        selected_workout = min(workouts, key=lambda w: abs(float(w["intensity_score"]) - float(action_value)))
        workout_type = selected_workout.get("type")
        if workout_type is not None:
            workout_type = str(workout_type)
            type_counts[workout_type] = type_counts.get(workout_type, 0) + 1

    return type_counts


def get_diversidad(actions, obs, workouts):
    if not actions or not workouts:
        return 0
    
    return len(get_recent_type_counts(actions, workouts))

def reward(actions, obs, performance, prev_benchmark_score, base_hr, base_spo2, workouts, omnisciente):
    dia, resting_hr, spo2, benchmark_score, w_ratio = obs
    P_t = performance[-1]
    P_t_1 = performance[-2] if len(performance) > 1 else 0
    B_t = benchmark_score
    B_t_1 = prev_benchmark_score

    D_t = get_diversidad(actions, obs, workouts)
    H_t = sum(1 for a in actions[-7:] if a > 2.5)

    repeat_penalty = 0.0
    type_concentration_penalty = 0.0
    consecutive_repeat_penalty = 0.0

    if actions and workouts:
        recent = actions[-7:]
        id_counts: dict[str, int] = {}
        recent_ids: list[str] = []
        type_counts = get_recent_type_counts(actions, workouts)
        for a in recent:
            selected = min(workouts, key=lambda w: abs(float(w["intensity_score"]) - float(a)))
            wid = selected.get("id")
            if wid is not None:
                recent_ids.append(str(wid))
                id_counts[str(wid)] = id_counts.get(str(wid), 0) + 1

        if id_counts:
            max_count = max(id_counts.values())
            repeat_penalty = float(max(0, max_count - 1))

        if type_counts:
            max_type_count = max(type_counts.values())
            type_concentration_penalty = float(max(0, max_type_count - 1))

        if len(recent_ids) > 1:
            consecutive_repeat_penalty = float(sum(1 for i in range(1, len(recent_ids)) if recent_ids[i] == recent_ids[i - 1]))

    if omnisciente:
        d_P = P_t - P_t_1 # derivada
        i_P = float(np.sum(np.asarray(performance, dtype=np.float32))) # integral (acumulado) para mejorar largo plazo
        # y evitar loops de subebaja q maximizan derivada
        r_t = d_P + 3.0 * i_P
        r_t -= max(0, H_t - 5)
        r_t += 2.0 * D_t
        r_t -= 4.0 * repeat_penalty
        r_t -= 3.0 * type_concentration_penalty
        r_t -= 6.0 * consecutive_repeat_penalty

        return r_t
    
    else: 
        r_t = (B_t - B_t_1)
        r_t -= max(0, H_t - 5)
        r_t -= 0.1 * abs(actions[-1] - actions[-2]) if len(actions) > 1 else 0
        r_t -= 0.2 * (max(0, resting_hr - base_hr) + max(0, base_spo2 - spo2))
        r_t += 2.0 * D_t
        r_t -= 4.0 * repeat_penalty
        r_t -= 3.0 * type_concentration_penalty
        r_t -= 6.0 * consecutive_repeat_penalty

        return r_t


def running_baseline(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    if window <= 0 or len(values) <= window:
        return float(np.mean(values))
    return float(np.mean(values[-window:]))


def rollout_greedy_policy(env, agent, workouts, device, seed: int | None = None):
    state, _ = env.reset(seed=seed)
    state_t = torch.from_numpy(np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)).to(device)

    times: list[int] = []
    chosen_intensity: list[float] = []
    selected_types: list[str] = []
    resting_hr: list[float] = []
    spo2: list[float] = []
    benchmark_score: list[float] = []
    performance: list[float] = []
    benchmark_available: list[float] = []
    intensity_diff: list[float] = []

    done = False
    step = 0
    while not done:
        raw_action = agent.calc_action(state_t, action_noise=None).detach().cpu().numpy()[0]
        action, selected = workout_action(raw_action, state, workouts, env)
        next_state, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        times.append(step)
        chosen_intensity.append(float(action[0]))
        resting_hr.append(float(next_state[1]))
        spo2.append(float(next_state[2]))
        benchmark_score.append(float(next_state[3]))
        performance.append(float(env.true_performance))
        benchmark_available.append(float(env.benchmark_available))
        selected_types.append(str(selected.get("type", "")))
        if len(chosen_intensity) > 1:
            intensity_diff.append(abs(float(action[0]) - chosen_intensity[-2]))
        else:
            intensity_diff.append(0.0)

        state = np.asarray(next_state, dtype=np.float32)
        state_t = torch.from_numpy(np.expand_dims(state, axis=0)).to(device)
        step += 1

    return {
        "t": times,
        "intensity": chosen_intensity,
        "selected_types": selected_types,
        "intensity_diff": intensity_diff,
        "resting_hr": resting_hr,
        "spo2": spo2,
        "benchmark_score": benchmark_score,
        "performance": performance,
        "benchmark_available": benchmark_available,
    }


def train_single(device, recovery_speed: str, recovery_tag: str, omnisciente: bool, suffix: str) -> None:
    person_mode = "random_once"
    seed = 0
    params = make_person_params(
        person_mode=person_mode,
        seed=seed,
        athletic_level="low",
        recovery_speed=recovery_speed,
        fitness_retention="high",
    )

    env = make_env(params=params, seed=seed, horizon=365, reward_mode="measured_delta")

    root = Path(__file__).resolve().parent
    workouts = load_workouts(root / "rl_workouts_500.json")

    torch.manual_seed(seed)
    np.random.seed(seed)

    gamma = 0.99
    tau = 0.001
    hidden_size = (400, 300)
    replay_size = 100000
    batch_size = 128
    warmup = 1000
    train_steps = 25000

    agent = DDPG(
        gamma=gamma,
        tau=tau,
        hidden_size=hidden_size,
        num_inputs=env.observation_space.shape[0],
        action_space=env.action_space,
        checkpoint_dir=str(root / "results" / "ddpg_practice"),
    )
    memory = ReplayMemory(replay_size)
    ou_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.zeros(env.action_space.shape[-1], dtype=np.float32),
        sigma=0.2 * np.ones(env.action_space.shape[-1], dtype=np.float32),
    )

    state, _ = env.reset(seed=seed)
    state_t = torch.from_numpy(np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)).to(device)
    base_hr = float(state[1])
    base_spo2 = float(state[2])

    episode_return = 0.0
    episode_returns: list[float] = []
    episode_mean_abs_performance: list[float] = []
    episode_mean_intensity: list[float] = []
    critic_losses: list[float] = []
    actor_losses: list[float] = []
    used_workouts: list[str] = []
    actions_history: list[float] = []
    performance_history: list[float] = [float(env.true_performance)]
    prev_benchmark_score = float(env.last_benchmark_score)
    hr_history: list[float] = [float(state[1])]
    spo2_history: list[float] = [float(state[2])]
    baseline_window = 100
    episode_abs_performance_history: list[float] = []
    episode_intensity_history: list[float] = []

    for step in range(1, train_steps + 1):
        if step <= warmup:
            action = env.action_space.sample().astype(np.float32)
            chosen = {"id": "warmup_random"}
        else:
            raw_action = agent.calc_action(state_t, ou_noise).detach().cpu().numpy()[0]
            action, chosen = workout_action(raw_action, state, workouts, env)

        next_state, env_reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        base_hr = running_baseline(hr_history, baseline_window)
        base_spo2 = running_baseline(spo2_history, baseline_window)
        actions_history.append(float(action[0]))
        performance_history.append(float(env.true_performance))
        episode_abs_performance_history.append(abs(float(env.true_performance)))
        episode_intensity_history.append(abs(float(action[0])))
        custom_reward = reward(actions_history, next_state, performance_history, prev_benchmark_score, base_hr, base_spo2, workouts, omnisciente)
        prev_benchmark_score = float(env.last_benchmark_score)
        hr_history.append(float(next_state[1]))
        spo2_history.append(float(next_state[2]))

        action_t = torch.from_numpy(np.expand_dims(np.asarray(action, dtype=np.float32), axis=0)).to(device)
        done_t = torch.tensor([float(done)], dtype=torch.float32, device=device)
        reward_t = torch.tensor([float(custom_reward)], dtype=torch.float32, device=device)
        next_state_t = torch.from_numpy(np.expand_dims(np.asarray(next_state, dtype=np.float32), axis=0)).to(device)

        memory.push(state_t, action_t, done_t, next_state_t, reward_t)

        if len(memory) >= batch_size and step > warmup:
            transitions = memory.sample(batch_size)
            batch = Transition(*zip(*transitions))
            value_loss, policy_loss = agent.update_params(batch)
            critic_losses.append(value_loss)
            actor_losses.append(policy_loss)

        state = np.asarray(next_state, dtype=np.float32)
        state_t = next_state_t
        episode_return += float(custom_reward)
        used_workouts.append(chosen["id"])

        if done:
            episode_returns.append(episode_return)
            episode_mean_abs_performance.append(float(np.mean(episode_abs_performance_history)) if episode_abs_performance_history else 0.0)
            episode_mean_intensity.append(float(np.mean(episode_intensity_history)) if episode_intensity_history else 0.0)
            episode_return = 0.0
            ou_noise.reset()
            state, _ = env.reset()
            state_t = torch.from_numpy(np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)).to(device)
            actions_history = []
            performance_history = [float(env.true_performance)]
            prev_benchmark_score = float(env.last_benchmark_score)
            hr_history = [float(state[1])]
            spo2_history = [float(state[2])]
            episode_abs_performance_history = []
            episode_intensity_history = []

        if step % 5000 == 0:
            print(
                f"step={step} episodes={len(episode_returns)} "
                f"mean_last10={np.mean(episode_returns[-10:]) if episode_returns else 0.0:.3f}"
            )

    agent.save_checkpoint(train_steps, memory)

    rollout = rollout_greedy_policy(env, agent, workouts, device, seed=seed)

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    axes[0].plot(rollout["t"], rollout["intensity"], color="tab:orange")
    axes[0].set_title("Greedy Policy: Workout Intensity")
    axes[0].set_ylabel("intensity")
    axes[0].grid(True)
    selected_types = rollout.get("selected_types", None) # plot categories elegidas
    if selected_types:
        preferred_order = ["G", "C", "W"]
        present = [t for t in preferred_order if t in set(selected_types)]
        if not present:
            present = list(dict.fromkeys(selected_types))
        mapping = {t: i for i, t in enumerate(present)}
        type_codes = [mapping.get(t, np.nan) for t in selected_types]
        ax2 = axes[0].twinx()
        ax2.plot(rollout["t"], type_codes, color="tab:gray", linestyle="--", alpha=0.9)
        ax2.set_ylabel("workout type")
        ticks = list(mapping.values())
        labels = list(mapping.keys())
        ax2.set_yticks(ticks)
        ax2.set_yticklabels(labels)
        ax2.set_ylim(-0.5, max(ticks) + 0.5)

    axes[1].plot(rollout["t"], rollout["resting_hr"], color="tab:red")
    axes[1].set_title("Measured Resting HR")
    axes[1].set_ylabel("HR")
    axes[1].grid(True)

    axes[2].plot(rollout["t"], rollout["spo2"], color="tab:blue")
    axes[2].set_title("Measured SpO2")
    axes[2].set_ylabel("SpO2")
    axes[2].grid(True)

    axes[3].plot(rollout["t"], rollout["benchmark_score"], color="tab:green")
    axes[3].set_title("Measured Benchmark Score")
    axes[3].set_ylabel("benchmark")
    axes[3].grid(True)

    axes[4].plot(rollout["t"], rollout["performance"], color="tab:purple")
    axes[4].set_title("True Performance")
    axes[4].set_ylabel("performance")
    axes[4].grid(True)

    axes[5].plot(rollout["t"], rollout["intensity_diff"], color="tab:brown")
    axes[5].set_title("|w_t - w_t-1|")
    axes[5].set_xlabel("Time step")
    axes[5].set_ylabel("intensity change")
    axes[5].grid(True)

    plt.tight_layout()
    plt.savefig(root / "results" / "ddpg_practice" / f"policy_rollout_{suffix}.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(episode_returns, color="tab:blue")
    axes[0].set_title("DDPG Episode Return")
    axes[0].set_ylabel("Return")
    axes[0].grid(True)

    axes[1].plot(episode_mean_abs_performance, color="tab:green")
    axes[1].set_title("Mean Absolute Performance per Episode")
    axes[1].set_ylabel("|performance|")
    axes[1].grid(True)

    axes[2].plot(episode_mean_intensity, color="tab:orange")
    axes[2].set_title("Mean Workout Intensity per Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("intensity")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(root / "results" / "ddpg_practice" / f"episode_metrics_{suffix}.png", dpi=150)
    plt.close()

    if actor_losses and critic_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(actor_losses, label="actor_loss", alpha=0.7)
        plt.plot(critic_losses, label="critic_loss", alpha=0.7)
        plt.title("DDPG Losses")
        plt.xlabel("Update")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(root / "results" / "ddpg_practice" / f"losses_{suffix}.png", dpi=150)
        plt.close()

    unique, counts = np.unique(np.asarray(used_workouts, dtype=object), return_counts=True)
    ranking = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n[{suffix}] Top workouts usados:")
    for wid, n in ranking:
        print(f"  {wid}: {n}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recovery_configs = [("high", "high"), ("med", "medium"), ("low", "low")]
    omnisciente_options = [False, True]
    iteration = 1
    total = len(recovery_configs) * len(omnisciente_options)
    
    for recovery_tag, recovery_speed in recovery_configs:
        for omni in omnisciente_options:
            suffix = recovery_tag
            if omni:
                suffix += "_omni"
    
            print(f"\n{'====================================================================='}")
            print(f"Iteration {iteration}/{total}: recovery={recovery_tag}, omnisciente={omni}")
            train_single(device, recovery_speed, recovery_tag, omni, suffix)
            iteration += 1

if __name__ == "__main__":
    main()