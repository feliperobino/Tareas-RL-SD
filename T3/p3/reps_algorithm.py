import numpy as np
from scipy.optimize import minimize_scalar

class RepsAlgorithm:
    def __init__(self):
        pass

    def reps_dual(self, eta: float, rewards: np.ndarray, epsilon: float) -> float:
        if eta <= 0:
            return np.inf
        r_max = np.max(rewards)
        stabilized = (rewards - r_max) / eta
        log_mean_exp = np.log(np.mean(np.exp(stabilized))) + r_max / eta
        return eta * epsilon + eta * log_mean_exp


    def solve_eta(self, rewards: np.ndarray, epsilon: float) -> float:
        result = minimize_scalar(
            self.reps_dual,
            bounds=(1e-8, 1e3),
            method="bounded",
            args=(rewards, epsilon),
        )
        if not result.success:
            raise RuntimeError("No se pudo optimizar eta en REPS.")
        return result.x


    def reps_weights(self, rewards: np.ndarray, eta: float) -> np.ndarray:
        z = (rewards - np.max(rewards)) / eta
        w = np.exp(z)
        w /= np.sum(w)
        return w


    def weighted_mean_and_std(self, samples: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
        samples = np.asarray(samples)
        weights = np.asarray(weights)

        weights = weights / np.sum(weights)
        weights = weights[:, None]
        mean = np.sum(weights * samples, axis=0)
        var = np.sum(weights * (samples - mean) ** 2, axis=0)
        std = np.sqrt(var)
        return mean, std

    def optimize(self,
        rewards, # In this case rewards could be 1/distance
        samples, # Samples are the parameters in vector form (n_trajectories, n_unknown_params)
        epsilon: float = 0.5):

        # Solve Eta
        eta = self.solve_eta(rewards, epsilon)

        # Get weights
        weights = self.reps_weights(rewards, eta)

        # New Gaussian
        new_mean, new_std = self.weighted_mean_and_std(samples, weights)

        return new_mean, new_std

