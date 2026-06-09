"""
State variables (8):
  S   : residual sugar [g/L]           (initial ~230 g/L)
  E   : ethanol [% v/v]                (initial 0)
  X   : viable yeast biomass [g/L]     (initial 0.05)
  Xd  : dead yeast biomass [g/L]       (initial 0)
  N   : yeast-assimilable nitrogen [mg/L] (initial 200)
  CO2 : dissolved CO2 [g/L]
  T   : must temperature [°C]
  t   : time [h]

Actions (2):
  u_T  : temperature setpoint change [°C/h], clipped [-2, +2]
  u_N  : nitrogen addition [mg/L per step], clipped [0, 50]
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Monod-type sugar uptake
Vmax0   = 2.50
Ks      = 0.50
KI_E    = 130.0
Ki_X    = 0.80

# Arrhenius for temperature on Vmax  (T in °C, Tref=20 °C)
Ea_R    = 5500.0
Tref    = 20.0

# Yield coefficients
Y_XS    = 0.050
Y_ES    = 0.475
Y_CO2S  = 0.465

# Nitrogen limitation
KN      = 5.0
alpha_N = 0.0015
N_crit  = 50.0

# Yeast death
kd0     = 0.008
kd_E    = 0.0008
kd_T    = 0.0035
T_death = 35.0

# Fermentation enthalpy (heat generation)
dH_S    = 590.0
rho_cp  = 2360.0
V_tank  = 10000.0
UA_cool = 80000000.0

# Simulation
DT      = 0.25
T_MAX   = 336.0


@dataclass
class FermentationState:
    S:   float = 230.0
    E:   float = 0.0
    X:   float = 0.05
    Xd:  float = 0.0
    N:   float = 200.0
    CO2: float = 0.0
    T:   float = 18.0
    t:   float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.S, self.E, self.X, self.Xd,
                         self.N, self.CO2, self.T, self.t], dtype=np.float32)

    @classmethod
    def from_array(cls, arr):
        return cls(*arr.tolist())


class WineFermentationEnv:

    def __init__(self,
                 S0: float = 230.0,
                 T0: float = 18.0,
                 N0: float = 200.0,
                 noise_std: float = 0.0,
                 random_seed: Optional[int] = None):
        self.S0    = S0
        self.T0    = T0
        self.N0    = N0
        self.noise_std = noise_std
        self.rng   = np.random.default_rng(random_seed)

        self._T_setpoint: float = T0
        self._state: FermentationState = FermentationState()
        self._done:  bool = False
        self._trajectory: list = []


    def reset(self, S0=None, T0=None, N0=None) -> np.ndarray:
        S0 = S0 if S0 is not None else self.S0
        T0 = T0 if T0 is not None else self.T0
        N0 = N0 if N0 is not None else self.N0
        self._state = FermentationState(S=S0, E=0.0, X=0.05, Xd=0.0,
                                        N=N0, CO2=0.0, T=T0, t=0.0)
        self._T_setpoint = T0
        self._done = False
        self._trajectory = [self._state.to_array().copy()]
        return self._obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, bool, dict]:

        assert not self._done, "Call reset() before stepping again."


        self._T_setpoint = float(np.clip(action[0], 8.0, 38.0))
        N_add            = float(np.clip(action[1], 0.0, 50.0))

        self._state = self._integrate(self._state, self._T_setpoint, N_add)

        if self.noise_std > 0:
            obs = self._obs() + self.rng.normal(0, self.noise_std, 8).astype(np.float32)
        else:
            obs = self._obs()

        self._trajectory.append(self._state.to_array().copy())

        done = self._check_done()
        self._done = done

        info = {
            "t": self._state.t,
            "T_setpoint": self._T_setpoint,
            "dryness": self._state.S < 2.0,
        }
        return obs, done, info

    def get_trajectory(self) -> np.ndarray:
        return np.array(self._trajectory, dtype=np.float32)

    def get_final_state(self) -> FermentationState:
        return self._state


    def _obs(self) -> np.ndarray:
        return self._state.to_array()

    def _check_done(self) -> bool:
        s = self._state
        if s.t >= T_MAX:
            return True
        if s.S < 1.0:
            return True
        if s.X < 0.001 and s.S > 5.0:
            return True
        return False

    def _integrate(self, state: FermentationState,
                   T_sp: float, N_add: float) -> FermentationState:
        y = np.array([state.S, state.E, state.X, state.Xd,
                      state.N, state.CO2, state.T], dtype=float)

        def dydt(y_):
            return self._odes(y_, T_sp, N_add)

        k1 = dydt(y)
        k2 = dydt(y + 0.5 * DT * k1)
        k3 = dydt(y + 0.5 * DT * k2)
        k4 = dydt(y +       DT * k3)
        y_new = y + (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Physical constraints
        y_new[0] = max(y_new[0], 0.0)   # S >= 0
        y_new[1] = max(y_new[1], 0.0)   # E >= 0
        y_new[2] = max(y_new[2], 0.0)   # X >= 0
        y_new[3] = max(y_new[3], 0.0)   # Xd >= 0
        y_new[4] = max(y_new[4], 0.0)   # N >= 0
        y_new[5] = np.clip(y_new[5], 0.0, 2.0)  # CO2 [0, 2] g/L
        y_new[6] = np.clip(y_new[6], 5.0, 40.0) # T [5, 40] °C

        ns = FermentationState(
            S=y_new[0], E=y_new[1], X=y_new[2], Xd=y_new[3],
            N=y_new[4], CO2=y_new[5], T=y_new[6],
            t=state.t + DT
        )
        return ns

    def _odes(self, y, T_sp, N_add):

        S, E, X, Xd, N, CO2, T = y
        S  = max(S,  0.0)
        E  = max(E,  0.0)
        X  = max(X,  0.0)
        N  = max(N,  0.0)

        T_K    = T + 273.15
        Tref_K = Tref + 273.15
        f_T    = np.exp(Ea_R * (1.0/Tref_K - 1.0/T_K))
        f_T    = np.clip(f_T, 0.0, 5.0)

        f_E = max(0.0, 1.0 - (E / KI_E) ** 2.5)

        f_N = N / (KN + N)

        f_S = S / (Ks + S)

        mu_S = Vmax0 * f_T * f_E * f_N * f_S

        r_S   = mu_S * X                        # sugar consumption  [g/L/h]
        r_X   = Y_XS * r_S                      # biomass growth     [g/L/h]
        r_E   = Y_ES * r_S                      # ethanol production [g/L/h]
        r_CO2 = Y_CO2S * r_S                    # CO2 production     [g/L/h]

        kd = kd0 + kd_E * E + kd_T * max(0.0, T - T_death)
        r_death = kd * X                        # [g/L/h]

        r_N   = alpha_N * r_S * 200.0          # mg/L/h

        Q_ferment = dH_S * r_S * V_tank         # J/h  heat generated
        Q_cool    = UA_cool * (T - T_sp)        # J/h  heat removed
        dT_dt     = (Q_ferment - Q_cool) / (rho_cp * V_tank)

        dS   = -r_S
        dE   =  r_E
        dX   =  r_X - r_death
        dXd  =  r_death
        dN   = -r_N + N_add / DT
        dCO2 =  r_CO2 - 0.15 * CO2
        dT   =  dT_dt

        return np.array([dS, dE, dX, dXd, dN, dCO2, dT])


