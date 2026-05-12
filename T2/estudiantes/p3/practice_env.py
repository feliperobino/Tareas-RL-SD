from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces






@dataclass(frozen=True)
class PracticeParams:
    alpha: float = 0.94
    beta: float = 0.82
    lambda_fitness: float = 0.85
    mu_fatigue: float = 1.65
    k_fitness: float = 1.0
    k_fatigue: float = 1.0
    p0: float = 0.0
    b0: float = 1.0
    rho: float = 0.95
    theta: float = 0.5
    f0: float = 0.0
    g0: float = 0.0

    # Relative-effort gains. The model still follows the paper idea that
    # practice acts through z = w / b, but these coefficients let people differ
    # in how much fitness and fatigue they generate from the same relative effort.
    k_adapt_fitness: float = 1.0
    k_adapt_fatigue: float = 1.0

    # Absolute-load fatigue. This prevents all constant actions from becoming
    # equivalent after adaptation, because large raw loads remain objectively costly.
    k_abs_fatigue: float = 0.12
    abs_fatigue_power: float = 2.2

    # Absolute-load fitness response. This prevents tiny raw actions from producing
    # almost the same fitness gain as meaningful practice. It is a saturating response
    # in load = w / w_max.
    fitness_half_load: float = 0.15
    fitness_load_power: float = 2.0


class ProfileGenerator:
    def __init__(
        self,
        default: bool = False,
        athletic_level: str = "medium",
        recovery_speed: str = "medium",
        fitness_retention: str = "medium",
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.params = PracticeParams()

        if default:
            return

        self.athletic_level = athletic_level
        self.recovery_speed = recovery_speed
        self.fitness_retention = fitness_retention

        self._set_initial_condition()
        self._set_athletic_response()
        self._set_fitness_retention()
        self._set_recovery_speed()

    def _update_params(self, **kwargs):
        p = self.params
        self.params = PracticeParams(**{**p.__dict__, **kwargs})

    def _set_initial_condition(self):
        if self.athletic_level == "low":
            self._update_params(
                f0=float(self.rng.uniform(0.0, 0.4)),
                b0=float(self.rng.uniform(0.45, 0.85)),
                g0=float(self.rng.uniform(0.0, 0.2)),
            )
        elif self.athletic_level == "medium":
            self._update_params(
                f0=float(self.rng.uniform(0.4, 1.2)),
                b0=float(self.rng.uniform(0.85, 1.45)),
                g0=float(self.rng.uniform(0.05, 0.45)),
            )
        elif self.athletic_level == "high":
            self._update_params(
                f0=float(self.rng.uniform(1.2, 2.6)),
                b0=float(self.rng.uniform(1.45, 2.60)),
                g0=float(self.rng.uniform(0.10, 0.85)),
            )
        else:
            raise ValueError('athletic_level must be "low", "medium", or "high".')

    def _set_athletic_response(self):
        """Set parameters that determine how raw load maps to useful fitness.

        Low-level people need a larger fraction of w_max before the practice is
        useful, and they fatigue more from the same absolute load. High-level people
        are more efficient and can tolerate larger raw loads.
        """
        if self.athletic_level == "low":
            self._update_params(
                k_fitness=float(self.rng.uniform(0.75, 1.00)),
                k_fatigue=float(self.rng.uniform(1.10, 1.45)),
                k_adapt_fitness=float(self.rng.uniform(0.75, 1.00)),
                k_adapt_fatigue=float(self.rng.uniform(1.05, 1.35)),
                k_abs_fatigue=float(self.rng.uniform(0.12, 0.28)),
                fitness_half_load=float(self.rng.uniform(0.18, 0.30)),
                fitness_load_power=float(self.rng.uniform(2.0, 3.0)),
                lambda_fitness=float(self.rng.uniform(0.75, 0.92)),
                mu_fatigue=float(self.rng.uniform(1.55, 2.20)),
                theta=float(self.rng.uniform(0.35, 0.60)),
            )
        elif self.athletic_level == "medium":
            self._update_params(
                k_fitness=float(self.rng.uniform(0.90, 1.15)),
                k_fatigue=float(self.rng.uniform(0.90, 1.25)),
                k_adapt_fitness=float(self.rng.uniform(0.90, 1.15)),
                k_adapt_fatigue=float(self.rng.uniform(0.85, 1.15)),
                k_abs_fatigue=float(self.rng.uniform(0.08, 0.20)),
                fitness_half_load=float(self.rng.uniform(0.12, 0.22)),
                fitness_load_power=float(self.rng.uniform(1.7, 2.5)),
                lambda_fitness=float(self.rng.uniform(0.78, 0.95)),
                mu_fatigue=float(self.rng.uniform(1.35, 1.95)),
                theta=float(self.rng.uniform(0.40, 0.70)),
            )
        elif self.athletic_level == "high":
            self._update_params(
                k_fitness=float(self.rng.uniform(1.05, 1.35)),
                k_fatigue=float(self.rng.uniform(0.70, 1.05)),
                k_adapt_fitness=float(self.rng.uniform(1.05, 1.35)),
                k_adapt_fatigue=float(self.rng.uniform(0.65, 1.00)),
                k_abs_fatigue=float(self.rng.uniform(0.04, 0.14)),
                fitness_half_load=float(self.rng.uniform(0.08, 0.16)),
                fitness_load_power=float(self.rng.uniform(1.4, 2.2)),
                lambda_fitness=float(self.rng.uniform(0.82, 0.98)),
                mu_fatigue=float(self.rng.uniform(1.20, 1.75)),
                theta=float(self.rng.uniform(0.45, 0.80)),
            )
        else:
            raise ValueError('athletic_level must be "low", "medium", or "high".')

    def _set_fitness_retention(self):
        if self.fitness_retention == "high":
            self._update_params(
                alpha=float(self.rng.uniform(0.965, 0.990)),
                rho=float(self.rng.uniform(0.955, 0.990)),
            )
        elif self.fitness_retention == "medium":
            self._update_params(
                alpha=float(self.rng.uniform(0.925, 0.965)),
                rho=float(self.rng.uniform(0.920, 0.965)),
            )
        elif self.fitness_retention == "low":
            self._update_params(
                alpha=float(self.rng.uniform(0.875, 0.925)),
                rho=float(self.rng.uniform(0.870, 0.925)),
            )
        else:
            raise ValueError('fitness_retention must be "low", "medium", or "high".')

    def _set_recovery_speed(self):
        if self.recovery_speed == "high":
            self._update_params(
                beta=float(self.rng.uniform(0.45, 0.65)),
                k_fatigue=float(self.params.k_fatigue * self.rng.uniform(0.75, 0.95)),
                k_adapt_fatigue=float(self.params.k_adapt_fatigue * self.rng.uniform(0.75, 0.95)),
                k_abs_fatigue=float(self.params.k_abs_fatigue * self.rng.uniform(0.65, 0.90)),
            )
        elif self.recovery_speed == "medium":
            self._update_params(
                beta=float(self.rng.uniform(0.65, 0.82)),
                k_fatigue=float(self.params.k_fatigue * self.rng.uniform(0.90, 1.10)),
                k_adapt_fatigue=float(self.params.k_adapt_fatigue * self.rng.uniform(0.90, 1.10)),
                k_abs_fatigue=float(self.params.k_abs_fatigue * self.rng.uniform(0.90, 1.10)),
            )
        elif self.recovery_speed == "low":
            self._update_params(
                beta=float(self.rng.uniform(0.82, 0.95)),
                k_fatigue=float(self.params.k_fatigue * self.rng.uniform(1.05, 1.35)),
                k_adapt_fatigue=float(self.params.k_adapt_fatigue * self.rng.uniform(1.05, 1.35)),
                k_abs_fatigue=float(self.params.k_abs_fatigue * self.rng.uniform(1.10, 1.50)),
            )
        else:
            raise ValueError('recovery_speed must be "low", "medium", or "high".')


def get_random_profile(seed=None) -> PracticeParams:
    rng = np.random.default_rng(seed)
    level = rng.choice(["low", "medium", "high"])
    recovery = rng.choice(["low", "medium", "high"])
    fitness = rng.choice(["low", "medium", "high"])
    return ProfileGenerator(
        default=False,
        athletic_level=str(level),
        recovery_speed=str(recovery),
        fitness_retention=str(fitness),
        seed=seed,
    ).params




def get_random_profile(seed=None):
    rng = np.random.default_rng(seed)
    level = rng.choice(["low", "medium", "high"])
    recovery = rng.choice(["low", "medium", "high"])
    fitness = rng.choice(["low", "medium", "high"])
    return ProfileGenerator(
        default=False,
        athletic_level=str(level),
        recovery_speed=str(recovery),
        fitness_retention=str(fitness),
        seed=seed,
    ).params



def make_fixed_params():
    """Deterministic reference person: same dynamics and initial condition."""
    return PracticeParams(
        alpha=0.92,
        beta=0.70,
        lambda_fitness=0.8,
        mu_fatigue=1.4,
        k_fitness=1.0,
        k_fatigue=1.0,
        p0=0.0,
        b0=1.0,
        rho=0.98,
        theta=0.5,
        f0=0.0,
        g0=0.0,
    )


def make_random_params(
    seed: Optional[int] = None,
    athletic_level: Optional[str] = None,
    recovery_speed: Optional[str] = None,
    fitness_retention: Optional[str] = None,
):
    """
    Create one random person using ProfileGenerator from practice_env_fixed.py.

    If a category is None, it is sampled from {"low", "medium", "high"}.
    If you pass a category, that category is fixed and only the numerical values
    inside that category are sampled.
    """
    rng = np.random.default_rng(seed)
    choices = ["low", "medium", "high"]

    if athletic_level is None:
        athletic_level = str(rng.choice(choices))
    if recovery_speed is None:
        recovery_speed = str(rng.choice(choices))
    if fitness_retention is None:
        fitness_retention = str(rng.choice(choices))

    return ProfileGenerator(
        default=False,
        athletic_level=athletic_level,
        recovery_speed=recovery_speed,
        fitness_retention=fitness_retention,
        seed=seed,
    ).params

def make_person_params(
        person_mode: str = "fixed",
        seed: Optional[int] = None,
        fixed_params = None,
        athletic_level: Optional[str] = None,
        recovery_speed: Optional[str] = None,
        fitness_retention: Optional[str] = None,
):
    """
    Select the person used by the environment.

    person_mode options:
      - "fixed": use make_fixed_params(), unless fixed_params is provided.
      - "random_once": sample one random person and keep it fixed for the whole run.
      - "random_each_episode": train_td3 will sample a new person at each episode.

    Use person_seed/seed to make the sampled person reproducible.
    """
    if fixed_params is not None:
        return fixed_params

    if person_mode == "fixed":
        return make_fixed_params()

    if person_mode in {"random_once", "random_each_episode"}:

        if athletic_level is None and recovery_speed is None and fitness_retention is None:
            return get_random_profile(seed=seed)

        else:
            return make_random_params(
                seed=seed,
                athletic_level=athletic_level,
                recovery_speed=recovery_speed,
                fitness_retention=fitness_retention,
            )

    raise ValueError('person_mode must be "fixed", "random_once", or "random_each_episode".')



def make_env(
    params,
    seed: int,
    horizon: int = 365,
    reward_mode: str = "measured_delta",
    measurement_mode: str = "physiology",
    measurement_noise: float = 0.0,
    sensor_noise: float = 0.0,
    effort_penalty: float = 0.0,
):
    env = PracticeEnv(
        horizon=horizon,
        params=params,
        measurement_mode=measurement_mode,
        measurement_noise=measurement_noise,
        sensor_noise=sensor_noise,
        reward_mode=reward_mode,
        effort_penalty=effort_penalty,
        seed=seed,
    )
    env.action_space.seed(seed)
    return env



class PracticeEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        horizon: int = 90,
        w_max: float = 3.0,
        params: Optional[PracticeParams] = None,
        adaptation: str = "max",
        reward_mode: str = "measured_delta",
        measurement_mode: str = "physiology",
        measurement_noise: float = 0.5,
        sensor_noise: float = 0.6,
        proxy_scale: float = 10.0,
        benchmark_interval: int = 7,
        benchmark_at_reset: bool = True,
        effort_penalty: float = 0.0,
        terminal_weight: float = 1.0,
        physio_ema_alpha: float = 0.15,
        reward_hr_weight: float = 1.0,
        reward_spo2_weight: float = 1.0,
        reward_physio_weight: float = 1.0,
        reward_benchmark_weight: float = 5.0,
        reward_clip: Optional[float] = None,
        final_min_intensity: Optional[float] = None,
        b_min: float = 0.25,
        z_max: float = 5.0,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.horizon = int(horizon)
        self.w_max = float(w_max)
        self.params = params if params is not None else PracticeParams()
        self.adaptation = adaptation
        self.reward_mode = reward_mode
        self.measurement_mode = measurement_mode
        self.measurement_noise = float(measurement_noise)
        self.sensor_noise = float(sensor_noise)
        self.proxy_scale = float(proxy_scale)
        self.benchmark_interval = int(benchmark_interval)
        self.benchmark_at_reset = bool(benchmark_at_reset)
        self.effort_penalty = float(effort_penalty)
        self.terminal_weight = float(terminal_weight)
        self.physio_ema_alpha = float(physio_ema_alpha)
        self.reward_hr_weight = float(reward_hr_weight)
        self.reward_spo2_weight = float(reward_spo2_weight)
        self.reward_physio_weight = float(reward_physio_weight)
        self.reward_benchmark_weight = float(reward_benchmark_weight)
        self.reward_clip = reward_clip
        self.final_min_intensity = final_min_intensity
        self.b_min = float(b_min)
        self.z_max = float(z_max)
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([self.w_max], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space.seed(seed)

        if self.measurement_mode == "state":
            self.observation_space = spaces.Box(
                low=np.array([0.0, -np.inf, -np.inf, 0.0, 0.0, -np.inf], dtype=np.float32),
                high=np.array([1.0, np.inf, np.inf, np.inf, 1.0, np.inf], dtype=np.float32),
                dtype=np.float32,
            )
        elif self.measurement_mode == "physiology":
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0, -np.inf, 0.0], dtype=np.float32),
                high=np.array([1.0, 150.0, 150.0, np.inf, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        elif self.measurement_mode in {"direct", "benchmark", "mixed"}:
            obs_dim = 1 if self.measurement_mode in {"direct", "benchmark"} else 3
            self.observation_space = spaces.Box(
                low=-np.inf * np.ones(obs_dim, dtype=np.float32),
                high=np.inf * np.ones(obs_dim, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            raise ValueError(
                'measurement_mode must be "state", "physiology", "direct", "benchmark", or "mixed".'
            )

        self.reset(seed=seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space.seed(seed)

        # Do NOT overwrite self.params here. This keeps the same person/dynamics.
        p = self.params

        self.t = 0
        self.F = float(p.f0)
        self.G = float(p.g0)
        self.b = max(float(p.b0), self.b_min)
        self.last_w = 0.0
        self.last_raw_effort = 0.0
        self.last_effort = 0.0

        self.old_true_performance = self._performance(self.F, self.G)
        self.true_performance = self.old_true_performance
        self.benchmark_score = 0.0
        self.last_benchmark_score = 0.0
        self.benchmark_available = 0.0
        self.days_since_benchmark = self.benchmark_interval
        self.resting_hr = 60.0
        self.spo2 = 98.0
        self.measured_performance = self._measure(self.true_performance)


        self.hr_ema = float(self.resting_hr)
        self.spo2_ema = float(self.spo2)
        self.prev_physiology_score = self._physiology_reward_score()
        self.prev_benchmark_score_for_reward = float(self.last_benchmark_score)
        self.last_physiology_score = float(self.prev_physiology_score)
        self.last_physiology_delta = 0.0
        self.last_benchmark_delta = 0.0

        return self._obs(), self._info(reward=0.0)

    def step(self, action: Any):
        if self.t >= self.horizon:
            raise RuntimeError("Episode finished. Call reset() before step().")

        self.old_true_performance = self.true_performance
        w = self._action_to_intensity(action)

        if self.final_min_intensity is not None and self.t == self.horizon - 1:
            w = max(w, float(self.final_min_intensity))

        p = self.params
        raw_z = w / max(self.b, self.b_min)
        z = float(np.clip(raw_z, 0.0, self.z_max))
        load = float(np.clip(w / max(self.w_max, 1e-8), 0.0, 1.0))

        fitness_load_response = (load ** p.fitness_load_power) / (
            load ** p.fitness_load_power
            + p.fitness_half_load ** p.fitness_load_power
            + 1e-8
        )

        self.F = float(
            p.alpha * self.F
            + p.k_adapt_fitness * fitness_load_response * z ** p.lambda_fitness
        )
        self.G = float(
            p.beta * self.G
            + p.k_adapt_fatigue * z ** p.mu_fatigue
            + p.k_abs_fatigue * load ** p.abs_fatigue_power
        )
        self.b = float(max(self._next_base(w, self.b), self.b_min))
        self.last_w = float(w)
        self.last_raw_effort = float(raw_z)
        self.last_effort = float(z)
        self.t += 1

        self.true_performance = self._performance(self.F, self.G)
        self.measured_performance = self._measure(self.true_performance)
        self._update_reward_smoothers()

        terminated = self.t >= self.horizon
        truncated = False

        reward = self._compute_reward(w=w, terminated=terminated)

        info = self._info(reward=reward)
        return self._obs(), float(reward), terminated, truncated, info

    def _action_to_intensity(self, action: Any) -> float:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != 1:
            raise ValueError("action must be a scalar or an array with one element.")
        return float(np.clip(action_array[0], 0.0, self.w_max))

    def _next_base(self, w: float, b: float) -> float:
        p = self.params
        if self.adaptation == "geometric":
            h = max(w, self.b_min) ** p.theta * max(b, self.b_min) ** (1.0 - p.theta)
        elif self.adaptation == "max":
            h = max(w, b, self.b_min)
        else:
            raise ValueError('adaptation must be "geometric" or "max".')
        return p.rho * h

    def _performance(self, F: float, G: float) -> float:
        p = self.params
        return float(p.p0 + p.k_fitness * F - p.k_fatigue * G)

    def _update_reward_smoothers(self) -> None:
        # Exponential moving average to reduce sensor-noise sensitivity.
        a = float(np.clip(self.physio_ema_alpha, 0.0, 1.0))
        self.hr_ema = float((1.0 - a) * self.hr_ema + a * self.resting_hr)
        self.spo2_ema = float((1.0 - a) * self.spo2_ema + a * self.spo2)

    def _physiology_reward_score(self) -> float:
        # Higher is better. This uses only observable physiology, not true performance.
        # Lower resting HR is rewarded, but scaled softly to avoid excessive incentives.
        hr_score = (60.0 - self.hr_ema) / 10.0

        # SpO2 is mostly a safety/recovery marker. Values below ~97 become negative;
        # values above ~97 become mildly positive.
        spo2_score = (self.spo2_ema - 97.0) / 2.0

        score = self.reward_hr_weight * hr_score + self.reward_spo2_weight * spo2_score
        return float(np.clip(score, -5.0, 5.0))

    def _compute_reward(self, w: float, terminated: bool) -> float:
        # Keep the old true-performance rewards available for debugging/comparison.
        current_physiology_score = self._physiology_reward_score()
        physiology_delta = current_physiology_score - self.prev_physiology_score

        benchmark_delta = 0.0
        if self.benchmark_available:
            benchmark_delta = (
                self.benchmark_score - self.prev_benchmark_score_for_reward
            ) / max(self.proxy_scale, 1e-8)

        if self.reward_mode == "measured_delta":
            reward = self.true_performance - self.old_true_performance

        elif self.reward_mode == "terminal":
            reward = self.true_performance if terminated else 0.0

        elif self.reward_mode == "hybrid":
            reward = self.true_performance - self.old_true_performance
            if terminated:
                reward += self.terminal_weight * self.true_performance

        elif self.reward_mode == "physio_dense":
            # Dense reward from smoothed physiology only.
            reward = self.reward_physio_weight * physiology_delta

        elif self.reward_mode == "benchmark_sparse":
            # Sparse reward only when a new benchmark is available.
            reward = self.reward_benchmark_weight * benchmark_delta if self.benchmark_available else 0.0

        elif self.reward_mode == "physio_hybrid":
            # Recommended default: small dense physiological shaping + sparse benchmark correction.
            reward = self.reward_physio_weight * physiology_delta
            if self.benchmark_available:
                reward += self.reward_benchmark_weight * benchmark_delta

        else:
            raise ValueError(
                'reward_mode must be "measured_delta", "terminal", "hybrid", '
                '"physio_dense", "benchmark_sparse", or "physio_hybrid".'
            )

        reward -= self.effort_penalty * (w / self.w_max) ** 2

        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self.last_physiology_score = float(current_physiology_score)
        self.last_physiology_delta = float(physiology_delta)
        self.last_benchmark_delta = float(benchmark_delta)
        self.prev_physiology_score = float(current_physiology_score)

        if self.benchmark_available:
            self.prev_benchmark_score_for_reward = float(self.benchmark_score)

        return float(reward)

    def _measure(self, true_performance: float):
        if self.measurement_mode == "direct":
            return [float(true_performance + self.rng.normal(0.0, self.measurement_noise))]

        benchmark_noise_std = self.measurement_noise * (1.0 + 0.15 * abs(self.G))
        should_measure_benchmark = (
            self.measurement_mode in {"benchmark", "mixed"}
            or (self.measurement_mode == "physiology" and self._is_benchmark_day())
        )

        if should_measure_benchmark:
            self.benchmark_score = float(
                self.proxy_scale * np.tanh(true_performance / self.proxy_scale)
                + self.rng.normal(0.0, benchmark_noise_std)
            )
            self.last_benchmark_score = self.benchmark_score
            self.benchmark_available = 1.0
            self.days_since_benchmark = 0
        else:
            self.benchmark_score = self.last_benchmark_score
            self.benchmark_available = 0.0
            self.days_since_benchmark = min(self.days_since_benchmark + 1, self.benchmark_interval)

        self.resting_hr = float(
            60.0
            - 6.0 * np.tanh(self.F / self.proxy_scale)
            + 8.0 * np.tanh(self.G / self.proxy_scale)
            + self.rng.normal(0.0, self.sensor_noise)
        )

        fitness_signal = np.tanh(self.F / self.proxy_scale)
        fatigue_signal = np.tanh(self.G / self.proxy_scale)
        acute_effort_signal = np.tanh(self.last_effort / max(self.z_max, 1e-8))

        self.spo2 = float(
            np.clip(
                97.5
                + 0.8 * fitness_signal
                - 2.2 * fatigue_signal
                - 0.7 * acute_effort_signal
                + self.rng.normal(0.0, 0.08 * self.sensor_noise),
                90.0,
                100.0,
            )
        )

        if self.measurement_mode == "benchmark":
            return [self.benchmark_score]
        if self.measurement_mode == "physiology":
            return [self.resting_hr, self.spo2]
        if self.measurement_mode == "mixed":
            return [self.resting_hr, self.spo2, self.benchmark_score]
        return [true_performance]

    def _is_benchmark_day(self) -> bool:
        if self.t == 0:
            return self.benchmark_at_reset
        return self.t % self.benchmark_interval == 0

    def _obs(self) -> np.ndarray:
        if self.measurement_mode == "state":
            return np.array(
                [
                    self.t / self.horizon,
                    self.F,
                    self.G,
                    self.b,
                    self.last_w / self.w_max,
                    self.true_performance,
                ],
                dtype=np.float32,
            )

        if self.measurement_mode == "physiology":
            return np.array(
                [
                    self.t / self.horizon,
                    self.resting_hr,
                    self.spo2,
                    self.last_benchmark_score / self.proxy_scale,
                    self.last_w / self.w_max,
                ],
                dtype=np.float32,
            )

        return np.array(self.measured_performance, dtype=np.float32)

    def _info(self, reward: float) -> dict:
        return {
            "t": self.t,
            "F": self.F,
            "G": self.G,
            "b": self.b,
            "last_w": self.last_w,
            "raw_effort": self.last_raw_effort,
            "effort": self.last_effort,
            "old_true_performance": self.old_true_performance,
            "true_performance": self.true_performance,
            "measured_performance": self.measured_performance,
            "benchmark_score": self.benchmark_score,
            "last_benchmark_score": self.last_benchmark_score,
            "benchmark_available": self.benchmark_available,
            "days_since_benchmark": self.days_since_benchmark,
            "resting_hr": self.resting_hr,
            "spo2": self.spo2,
            "hr_ema": self.hr_ema,
            "spo2_ema": self.spo2_ema,
            "physiology_score": self.last_physiology_score,
            "physiology_delta": self.last_physiology_delta,
            "benchmark_delta": self.last_benchmark_delta,
            "reward": float(reward),
        }
