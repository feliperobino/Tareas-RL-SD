import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ddpg import DDPG
from practice_env import make_env, make_person_params
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory, Transition

# Create logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

# Libdom raises an error if this is not set to true on Mac OSX
# see https://github.com/openai/spinningup/issues/16 for more information
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Parse given arguments
# gamma, tau, hidden_size, replay_size, batch_size, hidden_size are taken from the original paper
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="PracticeEnv",
                help="Environment name used only for checkpoint folder naming")
parser.add_argument("--render_train", default=False, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
                    help="Render the training steps (default: False)")
parser.add_argument("--render_eval", default=False, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
                    help="Render the evaluation steps (default: False)")
parser.add_argument("--load_model", default=False, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
                    help="Load a pretrained model (default: False)")
parser.add_argument("--save_dir", default="./saved_models/",
                    help="Dir. path to save and load a model (default: ./saved_models/)")
parser.add_argument("--seed", default=0, type=int,
                    help="Random seed (default: 0)")
parser.add_argument("--person_mode", default="random_once", type=str,
                help="Person sampling mode for PracticeEnv (fixed, random_once, random_each_episode)")
parser.add_argument("--athletic_level", default="low", type=str)
parser.add_argument("--recovery_speed", default="high", type=str)
parser.add_argument("--fitness_retention", default="high", type=str)
parser.add_argument("--workouts_json", default="rl_workouts_500.json", type=str)
parser.add_argument("--horizon", default=365, type=int)
parser.add_argument("--timesteps", default=1e6, type=int,
                    help="Num. of total timesteps of training (default: 1e6)")
parser.add_argument("--batch_size", default=64, type=int,
                    help="Batch size (default: 64; OpenAI: 128)")
parser.add_argument("--replay_size", default=1e6, type=int,
                    help="Size of the replay buffer (default: 1e6; OpenAI: 1e5)")
parser.add_argument("--gamma", default=0.99,
                    help="Discount factor (default: 0.99)")
parser.add_argument("--tau", default=0.001,
                    help="Update factor for the soft update of the target networks (default: 0.001)")
parser.add_argument("--noise_stddev", default=0.2, type=float,
                    help="Standard deviation of the OU-Noise (default: 0.2)")
parser.add_argument("--hidden_size", nargs=2, default=[400, 300], type=tuple,
                    help="Num. of units of the hidden layers (default: [400, 300]; OpenAI: [64, 64])")
parser.add_argument("--n_test_cycles", default=10, type=int,
                    help="Num. of episodes in the evaluation phases (default: 10; OpenAI: 20)")
args = parser.parse_args()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using {}".format(device))

if __name__ == "__main__":

    # Define the directory where to save and load models
    checkpoint_dir = args.save_dir + args.env
    writer = SummaryWriter('runs/run_1')

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

    reward_threshold = np.inf

    # Set random seed for all used libraries where possible
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def select_workout_action(raw_action, observation):
        current_level = float(np.clip(observation[3], 0.0, 1.0))
        unlocked = [w for w in workouts if float(w["level"]) <= current_level]
        candidates = unlocked if unlocked else [workouts[0]]
        selected = min(candidates, key=lambda w: abs(float(w["intensity_score"]) - float(raw_action[0])))
        action = np.array([float(selected["intensity_score"])], dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
        return action, selected

    # Define and build DDPG agent
    hidden_size = tuple(args.hidden_size)
    agent = DDPG(args.gamma,
                 args.tau,
                 hidden_size,
                 env.observation_space.shape[0],
                 env.action_space,
                 checkpoint_dir=checkpoint_dir
                 )

    # Initialize replay memory
    memory = ReplayMemory(int(args.replay_size))

    # Initialize OU-Noise
    nb_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                            sigma=float(args.noise_stddev) * np.ones(nb_actions))

    # Define counters and other variables
    start_step = 0
    # timestep = start_step
    if args.load_model:
        # Load agent if necessary
        start_step, memory = agent.load_checkpoint()
    timestep = start_step // 10000 + 1
    rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    epoch = 0
    t = 0
    time_last_checkpoint = time.time()

    # Start training
    logger.info('Train agent on PracticeEnv with workouts_500')
    logger.info('Doing {} timesteps'.format(args.timesteps))
    logger.info('Start at timestep {0} with t = {1}'.format(timestep, t))
    logger.info('Start training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

    while timestep <= args.timesteps:
        ou_noise.reset()
        epoch_return = 0

        state, _ = env.reset(seed=args.seed if timestep == 0 else None)
        state = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(device)
        while True:
            if args.render_train:
                env.render()

            raw_action = agent.calc_action(state, ou_noise).cpu().numpy()[0]
            action, _ = select_workout_action(raw_action, state.squeeze(0).cpu().numpy())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            timestep += 1
            epoch_return += reward

            mask = torch.tensor([done], dtype=torch.float32, device=device)
            reward = torch.tensor([reward], dtype=torch.float32, device=device)
            next_state = torch.from_numpy(np.asarray(next_state, dtype=np.float32)).unsqueeze(0).to(device)
            action = torch.from_numpy(np.asarray(action, dtype=np.float32)).unsqueeze(0).to(device)

            memory.push(state, action, mask, next_state, reward)

            state = next_state

            epoch_value_loss = 0
            epoch_policy_loss = 0

            if len(memory) > args.batch_size:
                transitions = memory.sample(args.batch_size)
                # Transpose the batch
                # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
                batch = Transition(*zip(*transitions))

                # Update actor and critic according to the batch
                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss

            if done:
                break

        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)
        writer.add_scalar('epoch/return', epoch_return, epoch)

        # Test every 10th episode (== 1e4) steps for a number of test_epochs epochs
        if timestep >= 10000 * t:
            t += 1
            test_rewards = []
            for _ in range(args.n_test_cycles):
                state, _ = env.reset()
                state = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(device)
                test_reward = 0
                while True:
                    if args.render_eval:
                        env.render()

                    raw_action = agent.calc_action(state)  # Selection without noise
                    action, _ = select_workout_action(raw_action.cpu().numpy()[0], state.squeeze(0).cpu().numpy())

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)
                    test_reward += reward

                    next_state = torch.from_numpy(np.asarray(next_state, dtype=np.float32)).unsqueeze(0).to(device)

                    state = next_state
                    if done:
                        break
                test_rewards.append(test_reward)

            mean_test_rewards.append(np.mean(test_rewards))

            for name, param in agent.actor.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            for name, param in agent.critic.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            writer.add_scalar('test/mean_test_return', mean_test_rewards[-1], epoch)
            logger.info("Epoch: {}, current timestep: {}, last reward: {}, "
                        "mean reward: {}, mean test reward {}".format(epoch,
                                                                      timestep,
                                                                      rewards[-1],
                                                                      np.mean(rewards[-10:]),
                                                                      np.mean(test_rewards)))

            # Save if the mean of the last three averaged rewards while testing
            # is greater than the specified reward threshold
            # TODO: Option if no reward threshold is given
            if np.mean(mean_test_rewards[-3:]) >= reward_threshold:
                agent.save_checkpoint(timestep, memory)
                time_last_checkpoint = time.time()
                logger.info('Saved model at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

        epoch += 1

    agent.save_checkpoint(timestep, memory)
    logger.info('Saved model at endtime {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    logger.info('Stopping training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    env.close()