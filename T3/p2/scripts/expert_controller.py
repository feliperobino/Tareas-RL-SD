import numpy as np


class PIDController:


    def __init__(self, Kp=0.8, Ki=0.05, Kd=0.2,
                 output_min=-2.0, output_max=2.0,
                 dt=0.25):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_min = output_min
        self.output_max = output_max
        self.dt = dt

        self._integral   = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0

    def step(self, error: float) -> float:
        self._integral   += error * self.dt
        derivative        = (error - self._prev_error) / self.dt
        self._prev_error  = error

        u = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        if u > self.output_max:
            self._integral -= error * self.dt
            u = self.output_max
        elif u < self.output_min:
            self._integral -= error * self.dt
            u = self.output_min

        return float(u)




class WineExpertController:


    T_START = 18.0
    T_PEAK = 22.0
    T_FINAL = 16.0
    T_RAMP_UP_END = 36.0
    T_HOLD_END = 200.0

    N_LOW_THRESHOLD = 60.0
    N_RESCUE_DOSE = 40.0
    N_COOLDOWN = 8.0

    SLOWDOWN_WINDOW = 8
    SLOWDOWN_EXPECTED = 1.5
    SLOWDOWN_DOSE = 30.0

    INOC_T = 8.0
    INOC_FRAC = 0.15
    INOC_MAX = 70.0

    def __init__(self):
        self.reset()

    def reset(self):
        self._last_add_t = -999.0
        self._did_inoc = False
        self._S_history = []
        self._N0_observed = None


    def setpoint(self, t: float) -> float:
        if t <= self.T_RAMP_UP_END:
            return self.T_START + (self.T_PEAK - self.T_START) * t / self.T_RAMP_UP_END
        elif t <= self.T_HOLD_END:
            return self.T_PEAK
        else:
            frac = min((t - self.T_HOLD_END) / (336.0 - self.T_HOLD_END), 1.0)
            return self.T_PEAK + frac * (self.T_FINAL - self.T_PEAK)


    def _cooldown_ok(self, t):
        return (t - self._last_add_t) >= self.N_COOLDOWN

    def _dS_dt(self):
        w = self._S_history[-self.SLOWDOWN_WINDOW:]
        if len(w) < 2:
            return 999.0
        return (w[0] - w[-1]) / (len(w) * 0.25)

    def _nitrogen_decision(self, obs: np.ndarray, t: float) -> float:
        S, E, X, Xd, N = obs[0], obs[1], obs[2], obs[3], obs[4]

        if self._N0_observed is None:
            self._N0_observed = float(N)

        self._S_history.append(float(S))
        if len(self._S_history) > 20:
            self._S_history.pop(0)

        if S < 10.0:
            return 0.0

        if not self._did_inoc and t >= self.INOC_T and self._cooldown_ok(t):
            dose = float(np.clip(self.INOC_FRAC * self._N0_observed,
                                 10.0, self.INOC_MAX))
            if self._N0_observed > 250.0:
                dose *= 0.5  # already N-rich: smaller boost
            self._did_inoc = True
            self._last_add_t = t
            return dose

        if N < self.N_LOW_THRESHOLD and t > 5.0 and self._cooldown_ok(t):
            self._last_add_t = t
            return self.N_RESCUE_DOSE

        if S > 50.0 and t > 12.0 and self._cooldown_ok(t):
            rate = self._dS_dt()
            if rate < self.SLOWDOWN_EXPECTED:
                severity = max(0.0, self.SLOWDOWN_EXPECTED - rate) / self.SLOWDOWN_EXPECTED
                dose = float(np.clip(self.SLOWDOWN_DOSE * (0.5 + severity), 15.0, 50.0))
                self._last_add_t = t
                return dose

        return 0.0

    def __call__(self, obs: np.ndarray, t: float) -> np.ndarray:
        T_sp = float(np.clip(self.setpoint(t), 8.0, 38.0))
        N_add = self._nitrogen_decision(obs, t)
        return np.array([T_sp, N_add], dtype=np.float32)

