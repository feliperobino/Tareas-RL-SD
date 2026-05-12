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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    person_mode = "random_once"
    seed = 0
    params = make_person_params(
        person_mode=person_mode,
        seed=seed,
        athletic_level="low",
        recovery_speed="high",
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

    episode_return = 0.0
    episode_returns: list[float] = []
    critic_losses: list[float] = []
    actor_losses: list[float] = []
    used_workouts: list[str] = []

    for step in range(1, train_steps + 1):
        if step <= warmup:
            action = env.action_space.sample().astype(np.float32)
            chosen = {"id": "warmup_random"}
        else:
            raw_action = agent.calc_action(state_t, ou_noise).detach().cpu().numpy()[0]
            action, chosen = workout_action(raw_action, state, workouts, env)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        action_t = torch.from_numpy(np.expand_dims(np.asarray(action, dtype=np.float32), axis=0)).to(device)
        done_t = torch.tensor([float(done)], dtype=torch.float32, device=device)
        reward_t = torch.tensor([float(reward)], dtype=torch.float32, device=device)
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
        episode_return += float(reward)
        used_workouts.append(chosen["id"])

        if done:
            episode_returns.append(episode_return)
            episode_return = 0.0
            ou_noise.reset()
            state, _ = env.reset()
            state_t = torch.from_numpy(np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)).to(device)

        if step % 5000 == 0:
            print(
                f"step={step} episodes={len(episode_returns)} "
                f"mean_last10={np.mean(episode_returns[-10:]) if episode_returns else 0.0:.3f}"
            )

    agent.save_checkpoint(train_steps, memory)

    plt.figure(figsize=(10, 5))
    plt.plot(episode_returns)
    plt.title("DDPG Episode Return (PracticeEnv + workouts_500)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(root / "results" / "ddpg_practice" / "episode_return.png", dpi=150)
    plt.close()

    if actor_losses and critic_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(actor_losses, label="actor_loss")
        plt.plot(critic_losses, label="critic_loss")
        plt.title("DDPG Losses")
        plt.xlabel("Update")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(root / "results" / "ddpg_practice" / "losses.png", dpi=150)
        plt.close()

    unique, counts = np.unique(np.asarray(used_workouts, dtype=object), return_counts=True)
    ranking = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)[:10]
    print("Top workouts usados:")
    for wid, n in ranking:
        print(f"  {wid}: {n}")


if __name__ == "__main__":
    main()