from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import numpy as np
from gymnasium import spaces
import gymnasium as gym

@dataclass
class DWPTParams:

    # Electrical values
    vdc: float = 430.0
    L: float = 135e-6
    C: float = 33e-9
    r_primary: float = 0.1
    r_secondary: float = 0.1
    r_load_dc: float = 5.2
    p_rated: float = 30e3

    # Frequency window
    f_min: float = 79e3
    f_max: float = 90e3

    # Coupling / geometry
    k_max: float = 0.26
    coil_spacing_m: float = 1.0
    coupling_sigma_m: float = 0.54

    # Vehicle / episode
    speed_mps: float = 10.0
    dt: float = 5e-4
    episode_length_s: float = 0.20

    # Approximate diode-rectifier AC equivalent load
    use_equivalent_ac_load: bool = True

    # Dynamic smoothing
    tau_meas: float = 2e-3
    tau_vout: float = 4e-3

    # Actuator slew limits
    df_max_per_step: float = 250.0
    dshare_max_per_step: float = 0.05



class DWPTContinuousShareEnv(gym.Env):

    def __init__(self, params: Optional[DWPTParams] = None, seed: Optional[int] = None,
                 system_knowledge='full'):
        self.p = DWPTParams() if params is None else params
        self.rng = np.random.default_rng(seed)
        self.system_knowledge = system_knowledge
        if self.system_knowledge == 'full':
            self.obs_func = self._get_state
        elif self.system_knowledge == 'no_comm':
            self.obs_func = self._get_obs_no_comm
        elif self.system_knowledge == 'comm':
            self.obs_func = self._get_obs_comm
        else:
            raise ValueError(f"System knowledge {self.system_knowledge} not recognized.")

        self.reset(seed)

        obs = self.obs_func()


        self.observation_space = spaces.Box(
            shape=(len(obs),),
            low=-np.inf,
            high=np.inf,
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

    @property
    def vfund_nominal(self) -> float:
        return 4.0 * self.p.vdc / np.pi

    @property
    def r_ac_equiv(self) -> float:
        if self.p.use_equivalent_ac_load:
            return (8.0 / np.pi**2) * self.p.r_load_dc
        return self.p.r_load_dc

    def coupling_profile(self, x: float) -> Tuple[float, float]:
        d = self.p.coil_spacing_m
        s = self.p.coupling_sigma_m

        k1 = self.p.k_max * np.exp(-((x - 0.0) / s) ** 2)
        k2 = self.p.k_max * np.exp(-((x - d) / s) ** 2)
        return float(k1), float(k2)

    def solve_network(self, f_sw: float, k1: float, k2: float, share: float) -> Dict[str, float]:

        w = 2.0 * np.pi * f_sw
        L = self.p.L
        C = self.p.C

        M1 = k1 * L
        M2 = k2 * L

        # Resonant net reactance of each branch
        X = w * L - 1.0 / (w * C)

        # Branch impedances
        Zp = self.p.r_primary + 1j * X
        Zs = self.p.r_secondary + self.r_ac_equiv + 1j * X

        # Fundamental source amplitudes distributed by the sharing factor
        V1 = (1.0 - share) * self.vfund_nominal
        V2 = share * self.vfund_nominal

        A = np.array([
            [Zp, 0.0 + 0.0j, -1j * w * M1],
            [0.0 + 0.0j, Zp, -1j * w * M2],
            [-1j * w * M1, -1j * w * M2, Zs],
        ], dtype=np.complex128)

        b = np.array([V1, V2, 0.0], dtype=np.complex128)

        I1, I2, Is = np.linalg.solve(A, b)

        # Average AC input power.
        # With peak phasors, average power is 0.5 * Re{V * I*}.
        P_in = 0.5 * np.real(V1 * np.conj(I1) + V2 * np.conj(I2))

        # Power dissipated in the equivalent AC load.
        P_out_target = 0.5 * self.r_ac_equiv * (np.abs(Is) ** 2)

        # Efficiency metric used for reward and analysis.
        eta_target = float(np.clip(P_out_target / max(P_in, 1e-6), 0.0, 1.0))

        # Capacitor-voltage magnitudes are useful as stress proxies:
        # |Vc| = |I| / (ωC)
        Vc1 = np.abs(I1) / max(w * C, 1e-12)
        Vc2 = np.abs(I2) / max(w * C, 1e-12)
        Vcs = np.abs(Is) / max(w * C, 1e-12)

        phase1_deg = -np.degrees(np.angle(I1))
        phase2_deg = -np.degrees(np.angle(I2))

        phase_active_mag = (1.0 - share) * abs(phase1_deg) + share * abs(phase2_deg)

        return {
            "I1_mag": float(np.abs(I1)),
            "I2_mag": float(np.abs(I2)),
            "Is_mag": float(np.abs(Is)),
            "phase1_deg": float(phase1_deg),
            "phase2_deg": float(phase2_deg),
            "phase_active_mag_deg": float(phase_active_mag),
            "P_in_target": float(max(P_in, 0.0)),
            "P_out_target": float(max(P_out_target, 0.0)),
            "eta_target": float(eta_target),
            "Vc1": float(Vc1),
            "Vc2": float(Vc2),
            "Vcs": float(Vcs),
        }

    def _lowpass(self, x: float, x_target: float, tau: float) -> float:
        alpha = self.p.dt / max(tau, self.p.dt)
        alpha = np.clip(alpha, 0.0, 1.0)
        return float(x + alpha * (x_target - x))


    def reset(self, seed=None, obs_format='dict', variable=False):
        super().reset(seed=seed)  # THIS sets self.np_random
        self.t = 0.0
        self.x_pos = 0.0
        self.f_sw = 86.5e3
        self.share = 0.05
        if variable:
            self.p.r_load_dc = np.random.choice(np.linspace(4, 7, 1000))
            self.p.speed_mps = np.random.choice(np.linspace(5, 12, 1000))
        else:
            self.p.r_load_dc = 5.1
            self.p.speed_mps = 10
        # print('Load: {} | Car speed: {}'.format(self.p.r_load_dc, self.p.speed_mps))


        self.k1, self.k2 = self.coupling_profile(self.x_pos)
        sol = self.solve_network(self.f_sw, self.k1, self.k2, self.share)

        # Initialize filtered states to steady values
        self.i1 = sol["I1_mag"]
        self.i2 = sol["I2_mag"]
        self.isec = sol["Is_mag"]
        self.pin = sol["P_in_target"]
        self.pout = sol["P_out_target"]
        self.eta = sol["eta_target"]
        self.phase1_deg = sol["phase1_deg"]
        self.phase2_deg = sol["phase2_deg"]
        self.phase_active_mag_deg = sol["phase_active_mag_deg"]
        self.vc1 = sol["Vc1"]
        self.vc2 = sol["Vc2"]
        self.vcs = sol["Vcs"]

        # DC output voltage estimated from DC load power relation:
        self.vout = np.sqrt(max(self.pout, 0.0) * self.p.r_load_dc)

        return self.obs_func(obs_format=obs_format), None


    def _get_state(self, obs_format='dict'):

        dicto_out =  {
            "time": float(self.t),
            "x_pos": float(self.x_pos), #Vehicle position
            "v_pos": float(self.p.speed_mps), # Vehicle speed
            "k1": float(self.k1), # Magnetic coupling coefficient between primary coil 1 and the secondary coil
            "k2": float(self.k2),   # Magnetic coupling coefficient between primary coil 2 and the secondary coil
            "f_sw": float(self.f_sw),  # Inverter switching frequency [Hz] (absolute)
            "share": float(self.share), # Power distribution between coild (absolute)
            "p_in": float(self.pin), # Input Power
            "p_out": float(self.pout), # Output Power
            "eta": float(self.eta), # Efficiency
            "i1": float(self.i1), # Phase of the current in primary coil 1 [degrees]
            "i2": float(self.i2), # Phase of the current in primary coil 2 [degrees]
            "i_secondary": float(self.isec), # Current in secondary coil
            "phase1_deg": float(self.phase1_deg), # Phase of the current in the primary coil
            "phase2_deg": float(self.phase2_deg),  # Phase of the current in the secondary coil
            "phase_active_mag_deg": float(self.phase_active_mag_deg),  # Weighted magnitude of the active phase (based on power sharing)
            "v_out": float(self.vout), # Estimated DC output voltage at the load (battery) [V]
            "v_cap_1": float(self.vc1), # Capacitor voltage in the resonant circuit of coil 1 [V]
            "v_cap_2": float(self.vc2), # Capacitor voltage in the resonant circuit of coil 2 [V]
        }
        if obs_format == 'dict':
            return dicto_out
        else:
            return np.array(list(dicto_out.values()))


    def _get_obs_no_comm(self, obs_format='dict'):
        dicto_out =  {
            "time": float(self.t),
            "x_pos": float(self.x_pos),
            "v_pos": float(self.p.speed_mps),
            "f_sw": float(self.f_sw),
            "share": float(self.share),
            "p_in": float(self.pin),
            "i1": float(self.i1),
            "i2": float(self.i2),
            "phase1_deg": float(self.phase1_deg),
            "phase2_deg": float(self.phase2_deg),
            "v_cap_1": float(self.vc1),
            "v_cap_2": float(self.vc2),
        }
        if obs_format == 'dict':
            return dicto_out
        else:
            return np.array(list(dicto_out.values()))

    def _get_obs_comm(self, obs_format='dict'):
        dicto_out = {
        "time": float(self.t),
        "f_sw": float(self.f_sw),
        "share": float(self.share),
        "x_pos": float(self.x_pos),
        "v_pos": float(self.p.speed_mps),
        "i1": float(self.i1),
        "i2": float(self.i2),
        "phase1_deg": float(self.phase1_deg),
        "phase2_deg": float(self.phase2_deg),
        "p_in": float(self.pin),
        "v_cap_1": float(self.vc1),
        "v_cap_2": float(self.vc2),
        "i_secondary": float(self.isec),
        "v_out": float(self.vout),
        "p_out": float(self.pout),
        "eta": float(self.eta),
        }
        if obs_format == 'dict':
            return dicto_out
        else:
            return np.array(list(dicto_out.values()))
        
    #######################################################
    ######              REWARD FUNCTION              ######
    #######################################################
    def compute_reward(self):
        if self.system_knowledge == 'no_comm':
            return self.compute_reward_no_comm()
        else:
            return self.compute_reward_full()
    
    def compute_reward_full(self):
        power_term = self.pout / self.p.p_rated
        efficiency_term = self.eta
        phase_penalty = phase_band_penalty(self.phase_active_mag_deg)

        reward = (
            1.0 * power_term
            + 0.15 * efficiency_term
            - 0.25 * phase_penalty
        )

        return float(reward)
    
    def compute_reward_no_comm(self):
        #solo vemos variables del primario (ni p_out no eta)
        power_term = self.pin / self.p.p_rated
        
        phase_penalty = phase_band_penalty(self.phase_active_mag_deg)
        
        capacitor_penalty = (
            max(0.0, self.vc1 - 6500.0) / 6500.0
            + max(0.0, self.vc2 - 6500.0) / 6500.0
        )
        
        reward = (
            1.0 * power_term
            - 0.25 * phase_penalty
            - 0.05 * capacitor_penalty
        )
        
        return float(reward)
    #######################################################
    #######################################################

    def step(self, action):
        """
        Advance the simulator by one time step
        """
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size != 2:
            raise ValueError("Acción debe ser de dimensión dos: [a_freq, a_share].")

        a_freq = float(np.clip(a[0], -1.0, 1.0))
        a_share = float(np.clip(a[1], -1.0, 1.0))

        # 1) Apply actuator increments with slew limits and hard bounds
        df = a_freq * self.p.df_max_per_step
        dshare = a_share * self.p.dshare_max_per_step

        self.f_sw = float(np.clip(self.f_sw + df, self.p.f_min, self.p.f_max))
        self.share = float(np.clip(self.share + dshare, 0.0, 1.0))

        # 2) Move the vehicle / receiver
        self.t += self.p.dt
        self.x_pos += self.p.speed_mps * self.p.dt

        # 3) Update coupling based on receiver position
        self.k1, self.k2 = self.coupling_profile(self.x_pos)

        # 4) Solve electrical network at current operating point
        sol = self.solve_network(self.f_sw, self.k1, self.k2, self.share)

        # 5) Filter measurements to create realistic smooth dynamics
        self.i1 = self._lowpass(self.i1, sol["I1_mag"], self.p.tau_meas)
        self.i2 = self._lowpass(self.i2, sol["I2_mag"], self.p.tau_meas)
        self.isec = self._lowpass(self.isec, sol["Is_mag"], self.p.tau_meas)
        self.pin = self._lowpass(self.pin, sol["P_in_target"], self.p.tau_meas)
        self.pout = self._lowpass(self.pout, sol["P_out_target"], self.p.tau_meas)
        self.eta = self._lowpass(self.eta, sol["eta_target"], self.p.tau_meas)
        self.phase1_deg = self._lowpass(self.phase1_deg, sol["phase1_deg"], self.p.tau_meas)
        self.phase2_deg = self._lowpass(self.phase2_deg, sol["phase2_deg"], self.p.tau_meas)
        self.phase_active_mag_deg = self._lowpass(
            self.phase_active_mag_deg, sol["phase_active_mag_deg"], self.p.tau_meas
        )
        self.vc1 = self._lowpass(self.vc1, sol["Vc1"], self.p.tau_meas)
        self.vc2 = self._lowpass(self.vc2, sol["Vc2"], self.p.tau_meas)
        self.vcs = self._lowpass(self.vcs, sol["Vcs"], self.p.tau_meas)

        # 6) Estimate DC-side output voltage from output power and DC load
        vout_target = np.sqrt(max(self.pout, 0.0) * self.p.r_load_dc)
        self.vout = self._lowpass(self.vout, vout_target, self.p.tau_vout)

        obs = self.obs_func()

        # Agregado por mi para que funcione PPO de github! 
        # debe retornar obs, reward, terminated, truncated, info
        reward = self.compute_reward()

        terminated = bool(self.t >= self.p.episode_length_s)

        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info



def phase_band_penalty(phase_deg: float, low: float = 10.0, high: float = 15.0) -> float:

    phase_deg = abs(float(phase_deg))

    if phase_deg < low:
        return ((low - phase_deg) / max(low, 1e-9)) ** 2
    elif phase_deg > high:
        return ((phase_deg - high) / max(high, 1e-9)) ** 2
    else:
        return 0.0


def find_optimal_operating_point(
    env,
    k1: float,
    k2: float,
    n_freq: int = 121,
    n_share: int = 101,
    phase_low: float = 10.0,
    phase_high: float = 15.0,
    w_power: float = 1.0,
    w_eff: float = 0.15,
    w_phase: float = 0.25,
    w_vcap: float = 0.0,
):


    freqs = np.linspace(env.p.f_min, env.p.f_max, n_freq)
    shares = np.linspace(0.0, 1.0, n_share)

    best = None

    for f in freqs:
        for share in shares:
            sol = env.solve_network(f_sw=f, k1=k1, k2=k2, share=share)

            p_out = sol["P_out_target"]
            eta = sol["eta_target"]
            phase = sol["phase_active_mag_deg"]

            phase_pen = phase_band_penalty(
                phase,
                low=phase_low,
                high=phase_high,
            )

            # Optional capacitor stress penalty
            vcap_active = max(
                (1.0 - share) * sol["Vc1"],
                share * sol["Vc2"],
            )
            vcap_pen = max(0.0, vcap_active - 6500.0) / 6500.0

            score = (
                w_power * (p_out / env.p.p_rated)
                + w_eff * eta
                - w_phase * phase_pen
                - w_vcap * vcap_pen
            )

            if best is None or score > best["score_opt"]:
                best = {
                    "f_opt": float(f),
                    "share_opt": float(share),
                    "phase_opt": float(phase),
                    "p_out_opt": float(p_out),
                    "eta_opt": float(eta),
                    "score_opt": float(score),
                    "phase_penalty": float(phase_pen),
                    "vcap_active": float(vcap_active),
                }

    return best


def oracle_controller(
    obs: Dict[str, float],
    env,
    n_action_freq: int = 41,
    n_action_share: int = 41,
    phase_low: float = 10.0,
    phase_high: float = 15.0,
    w_power: float = 1.0,
    w_eff: float = 0.15,
    w_phase: float = 0.25,
    w_slew_f: float = 0.01,
    w_slew_share: float = 0.01,
):


    action_freqs = np.linspace(-1.0, 1.0, n_action_freq)
    action_shares = np.linspace(-1.0, 1.0, n_action_share)

    best = None

    # Predict next position/coupling
    x_next = obs["x_pos"] + env.p.speed_mps * env.p.dt
    k1_next, k2_next = env.coupling_profile(x_next)

    for a_freq in action_freqs:
        for a_share in action_shares:

            # Apply same actuator logic as env.step()
            f_next = np.clip(
                obs["f_sw"] + a_freq * env.p.df_max_per_step,
                env.p.f_min,
                env.p.f_max,
            )

            share_next = np.clip(
                obs["share"] + a_share * env.p.dshare_max_per_step,
                0.0,
                1.0,
            )

            sol = env.solve_network(
                f_sw=f_next,
                k1=k1_next,
                k2=k2_next,
                share=share_next,
            )

            p_out = sol["P_out_target"]
            eta = sol["eta_target"]
            phase = sol["phase_active_mag_deg"]

            phase_pen = phase_band_penalty(
                phase,
                low=phase_low,
                high=phase_high,
            )

            score = (
                w_power * (p_out / env.p.p_rated)
                + w_eff * eta
                - w_phase * phase_pen
                - w_slew_f * abs(a_freq)
                - w_slew_share * abs(a_share)
            )

            if best is None or score > best["score"]:
                best = {
                    "action": np.array([a_freq, a_share], dtype=float),
                    "score": float(score),
                    "f_next": float(f_next),
                    "share_next": float(share_next),
                    "p_out": float(p_out),
                    "eta": float(eta),
                    "phase": float(phase),
                    "phase_penalty": float(phase_pen),
                }

    return best["action"]



