import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from ddpg import DDPG
from practice_env import make_env, make_person_params

# Create logger
logger = logging.getLogger('test')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="PracticeEnv", help="Env name used only for checkpoint folder naming")
parser.add_argument("--render", default=False, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, help="Render the steps")
parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument("--save_dir", default="./saved_models/", help="Dir. path to load a model")
parser.add_argument("--episodes", default=100, type=int, help="Num. of test episodes")
parser.add_argument("--person_mode", default="random_once", type=str,
                    help="Person sampling mode for PracticeEnv (fixed, random_once, random_each_episode)")
parser.add_argument("--athletic_level", default="low", type=str)
parser.add_argument("--recovery_speed", default="high", type=str)
parser.add_argument("--fitness_retention", default="high", type=str)
parser.add_argument("--workouts_json", default="rl_workouts_500.json", type=str)
parser.add_argument("--horizon", default=365, type=int)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gamma = 0.99
tau = 0.001
hidden_size = (400, 300)


def select_workout_action(raw_action, observation, workouts, env):
    current_level = float(np.clip(observation[3], 0.0, 1.0))
    unlocked = [w for w in workouts if float(w["level"]) <= current_level]
    candidates = unlocked if unlocked else [workouts[0]]
    selected = min(candidates, key=lambda w: abs(float(w["intensity_score"]) - float(raw_action[0])))
    action = np.array([float(selected["intensity_score"])], dtype=np.float32)
    action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
    return action, selected


if __name__ == "__main__":

    logger.info("Using device: {}".format(device))

    root = Path(__file__).resolve().parent
    workouts_path = root / args.workouts_json
    with workouts_path.open("r", encoding="utf-8") as f:
        workouts = sorted(json.load(f)["workouts"], key=lambda w: float(w["level"]))

    params = make_person_params(
        person_mode=args.person_mode,
        seed=args.seed,
        athletic_level=args.athletic_level,
        recovery_speed=args.recovery_speed,
        fitness_retention=args.fitness_retention,
    )
    env = make_env(
        params=params,
        seed=args.seed,
        horizon=args.horizon,
        reward_mode="measured_delta",
        measurement_mode="physiology",
        measurement_noise=0.0,
        sensor_noise=0.0,
        effort_penalty=0.0,
    )

    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir = args.save_dir + args.env

    agent = DDPG(
        gamma,
        tau,
        hidden_size,
        env.observation_space.shape[0],
        env.action_space,
        checkpoint_dir=checkpoint_dir,
    )

    agent.load_checkpoint()
    agent.set_eval()

    returns = list()
    for _ in range(args.episodes):
        state, _ = env.reset()
        state = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(device)
        episode_return = 0.0
        while True:
            if args.render:
                env.render()

            raw_action = agent.calc_action(state, action_noise=None)
            action, _ = select_workout_action(raw_action.cpu().numpy()[0], state.squeeze(0).cpu().numpy(), workouts, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            episode_return += reward

            state = torch.from_numpy(np.asarray(next_state, dtype=np.float32)).unsqueeze(0).to(device)

            if done:
                logger.info(episode_return)
                returns.append(episode_return)
                break

    mean = np.mean(returns)
    variance = np.var(returns)
    logger.info("Score (on %d episodes): %s +/- %s", args.episodes, mean, variance)