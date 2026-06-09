import numpy as np
from functools import lru_cache
from typing import Optional

_SEED = 42
_rng  = np.random.default_rng(_SEED)


_P = {
    # Anthocyanin extraction from skins
    "anth_k_extr":   0.042,   # extraction rate constant [1/h]
    "anth_k_deg":    0.0025,  # degradation rate [1/h] — increases at high T and cold T
    "anth_S_max":    480.0,   # max extractable [mg/L]
    "anth_EtOH_opt": 70.0,    # optimal EtOH for extraction [g/L]
    "anth_EtOH_sig": 55.0,
    "anth_T_opt":    21.5,    # optimal T — Expert runs at ~22°C, Cool at ~17°C
    "anth_T_sig":    3.5,     # narrow peak: both cold (<18) and warm (>26) reduce extraction

    # Tannin (proanthocyanidin) extraction — seeds + skins
    "tann_k_skins":  0.018,
    "tann_k_seeds":  0.008,
    "tann_k_poly":   0.0003,  # tannin polymerization / precipitation
    "tann_S_skins":  800.0,   # max from skins [mg/L]
    "tann_S_seeds":  600.0,   # max from seeds [mg/L]
    "tann_EtOH_seed":85.0,    # EtOH threshold for seed tannin release
    "tann_EtOH_sig": 30.0,
    "tann_T_seed":   28.0,    # seed tannins extracted MORE at high T (harsh)
    "tann_T_seed_sig": 6.0,   # temperature sensitivity of seed extraction

    # Catechin (monomeric flavan-3-ols)
    "cat_k_extr":    0.025,
    "cat_S_max":     180.0,
    "cat_EtOH_opt":  60.0,
    "cat_EtOH_sig":  40.0,

    # Color (absorbance at 520 nm, proxy)
    "color_anth_coeff": 0.0021,
    "color_copig_tann": 0.0004,

    # Volatile acidity — driven by thermal stress above 22°C and fermentation speed
    # Target range: ~0.30 g/L (cool, slow) to ~1.20 g/L (warm, fast)
    "va_base":       0.28,
    "va_T_thresh":   22.0,    # onset of VA production (lowered from 28°C)
    "va_T_coeff":    0.0015,  # per °C above threshold, per hour
    "va_speed_coeff":0.001,   # faster fermentations produce more VA
    "va_stuck_pen":  0.8,

    # Glycerol [g/L] — longer, cooler fermentations yield more glycerol
    # (yeast glyceropyruvic pathway favored at lower T and osmotic stress)
    # Target range: ~7.5 (warm/fast) to ~10.5 (cool/slow)
    "gly_base":      7.5,
    "gly_dur_coeff": 0.030,   # +glycerol per hour of fermentation
    "gly_T_coeff":  -0.06,    # negative: cooler → more glycerol
    "gly_N_coeff":   0.003,   # more N -> slightly more glycerol
}

_W = {
    "anthocyanin":      +0.30,   # dominant — more is always better up to saturation
    "color_intensity":  +0.15,   # monotone — directly linked to anthocyanin
    "tannin_balance":   +0.10,   # bell-shaped: too little=thin, too much=harsh
    "catechin":         +0.08,   # monotone up
    "glycerol":         +0.08,   # bell at 9.0 g/L
    "volatile_acidity": -0.22,   # penalty: higher VA = worse wine
    "ethanol_balance":  +0.05,   # peaks ~13.8 %v/v
    "sugar_residual":   -0.07,   # penalty: residual sugar is a fault
    "duration_penalty": -0.10,   # bell: too fast OR too slow fermentations penalized
}

_SCORE_MAX_RAW = sum(v for v in _W.values() if v > 0) / sum(abs(v) for v in _W.values())


def maceration_extraction(trajectory: np.ndarray) -> dict:
    """
    trajectory : np.ndarray, shape (N_steps, 8)
        Fermentation trajectory from WineFermentationEnv.get_trajectory().
        Columns: [S, E, X, Xd, N, CO2, T, t]

    Returns
    -------
    dict with keys:
        anthocyanin      : float  [mg/L]    — red color pigments
        tannin           : float  [mg/L]    — total proanthocyanidins
        catechin         : float  [mg/L]    — monomeric flavanols
        color_intensity  : float  [AU]      — abs 420+520+620 nm proxy
        glycerol         : float  [g/L]     — body/texture contributor
        volatile_acidity : float  [g/L]     — acetic acid proxy (fault indicator)
        ethanol_pct      : float  [%v/v]    — final ethanol
        residual_sugar   : float  [g/L]
        fermentation_h   : float  [h]       — total duration
    """
    traj = np.asarray(trajectory, dtype=float)
    n    = len(traj)
    dt   = 0.25  # h

    S_arr = traj[:, 0]
    E_arr = traj[:, 1]   # g/L ethanol
    T_arr = traj[:, 6]

    T_integral = float(np.trapezoid(T_arr, dx=dt))           # °C·h
    E_integral = float(np.trapezoid(E_arr, dx=dt))           # g/L·h

    # Final values
    S_f = float(S_arr[-1])
    E_f = float(E_arr[-1])
    T_f = float(T_arr[-1])
    duration = n * dt


    anth = 0.0
    for i in range(n - 1):
        E_i = E_arr[i];  T_i = T_arr[i]
        f_E = 1.0 / (1.0 + np.exp(-(_P["anth_EtOH_sig"]**-1) *
                                     (E_i - _P["anth_EtOH_opt"])))
        f_T = np.exp(-0.5 * ((T_i - _P["anth_T_opt"]) / _P["anth_T_sig"]) ** 2)
        driving = _P["anth_k_extr"] * f_E * f_T * (_P["anth_S_max"] - max(anth, 0))
        degrad  = _P["anth_k_deg"] * anth * (1.0 + 0.05 * max(T_i - 28.0, 0))
        anth   += (driving - degrad) * dt

    anth = float(np.clip(anth, 0, _P["anth_S_max"]))

    tann_skin = 0.0; tann_seed = 0.0
    for i in range(n - 1):
        E_i = E_arr[i]; T_i = T_arr[i]
        # Skin tannins: early extraction, moderate EtOH
        f_skin = 0.5 + 0.5 * np.tanh((E_i - 30.0) / 25.0)
        drv_sk = _P["tann_k_skins"] * f_skin * T_i/22.0 * \
                 (_P["tann_S_skins"] - max(tann_skin, 0))
        # Seed tannins: need higher EtOH AND are favored by high T (harsh extraction)
        f_seed = 1.0 / (1.0 + np.exp(-(_P["tann_EtOH_sig"]**-1) *
                                       (E_i - _P["tann_EtOH_seed"])))
        f_seed_T = 0.3 + 0.7 * _sigmoid(T_i - _P["tann_T_seed"],
                                          scale=1.0/_P["tann_T_seed_sig"])
        drv_sd = _P["tann_k_seeds"] * f_seed * f_seed_T * (_P["tann_S_seeds"] - max(tann_seed, 0))
        # Polymerization / precipitation of total tannin pool
        total_t = tann_skin + tann_seed
        poly    = _P["tann_k_poly"] * total_t ** 1.5 * dt

        tann_skin += drv_sk * dt
        tann_seed += drv_sd * dt
        tann_skin  = max(tann_skin - poly * 0.6, 0)
        tann_seed  = max(tann_seed - poly * 0.4, 0)

    tannin = float(np.clip(tann_skin + tann_seed, 0, 1400.0))

    cat = 0.0
    for i in range(n - 1):
        E_i = E_arr[i]; T_i = T_arr[i]
        f_E = np.exp(-0.5 * ((E_i - _P["cat_EtOH_opt"]) / _P["cat_EtOH_sig"]) ** 2)
        f_T = 0.8 + 0.2 * (T_i / 25.0)
        drv = _P["cat_k_extr"] * f_E * f_T * (_P["cat_S_max"] - max(cat, 0))
        cat += drv * dt

    catechin = float(np.clip(cat, 0, _P["cat_S_max"]))

    copigmentation = _P["color_copig_tann"] * tannin * anth / 500.0
    color_intensity = float(_P["color_anth_coeff"] * anth + copigmentation)


    T_mean = float(np.mean(T_arr))
    N_mean = float(np.mean(traj[:, 4]))
    glycerol = float(np.clip(
        _P["gly_base"]
        + _P["gly_dur_coeff"] * duration
        + _P["gly_T_coeff"]   * T_mean
        + _P["gly_N_coeff"]   * N_mean,
        4.0, 14.0
    ))


    T_stress  = float(np.mean(np.clip(T_arr - _P["va_T_thresh"], 0, None)))
    mean_rate = (230.0 - S_f) / max(duration, 1.0)
    stuck     = 1.0 if S_f > 5.0 else 0.0
    va = float(np.clip(
        _P["va_base"]
        + _P["va_T_coeff"]    * T_stress * duration
        + _P["va_speed_coeff"]* mean_rate
        + _P["va_stuck_pen"]  * stuck,
        0.20, 2.50
    ))

    ethanol_pct = float(E_f / 7.89)

    return {
        "anthocyanin":      anth,
        "tannin":           tannin,
        "catechin":         catechin,
        "color_intensity":  color_intensity,
        "glycerol":         glycerol,
        "volatile_acidity": va,
        "ethanol_pct":      ethanol_pct,
        "residual_sugar":   S_f,
        "fermentation_h":   duration,
    }



def _hidden_utility(profile: dict, noise_std: float = 0.05) -> float:

    sw  = sum(abs(v) for v in _W.values())
    smr = sum(v for v in _W.values() if v > 0) * 10.0 / sw  # max achievable raw

    def ss_mono(x, low, sat):
        if x <= low: return 0.0
        return min(1.0, (x - low) / (sat - low + 1e-9))

    anth_score  = ss_mono(profile["anthocyanin"],     low=30,   sat=260)
    color_score = ss_mono(profile["color_intensity"], low=0.05, sat=0.70)
    cat_score   = ss_mono(profile["catechin"],         low=60,   sat=165)

    tann_score = _bell_score(profile["tannin"],       center=330, width=120)
    gly_score  = _bell_score(profile["glycerol"],     center=9.0, width=1.5)
    eth_score  = _bell_score(profile["ethanol_pct"],  center=13.8, width=1.3)
    dur_score  = _bell_score(profile["fermentation_h"], center=60, width=22)

    va_score    = _sigmoid_score(profile["volatile_acidity"], low=0.28, opt=0.55, high=1.15)
    sugar_score = _sigmoid_score(profile["residual_sugar"],   low=0.10, opt=2.50, high=20.0)

    raw = (
        _W["anthocyanin"]      * anth_score  +
        _W["color_intensity"]  * color_score +
        _W["tannin_balance"]   * tann_score  +
        _W["catechin"]         * cat_score   +
        _W["glycerol"]         * gly_score   +
        _W["volatile_acidity"] * va_score    +
        _W["ethanol_balance"]  * eth_score   +
        _W["sugar_residual"]   * sugar_score +
        _W["duration_penalty"] * dur_score
    ) * 10.0 / sw

    return float(np.clip(raw / smr * 10.0, 0.0, 10.0))


def human_preference(profile_A: dict, profile_B: dict,
                     noise_std: float = 0.15,
                     seed: Optional[int] = None) -> dict:

    rng_  = np.random.default_rng(seed)

    uA = _hidden_utility(profile_A) + rng_.normal(0, noise_std)
    uB = _hidden_utility(profile_B) + rng_.normal(0, noise_std)

    # Bradley-Terry model: P(A > B) = σ(uA - uB)
    p_A = _sigmoid(uA - uB, scale=1.5)

    winner = 'A' if rng_.random() < p_A else 'B'

    return {
        "winner":           winner,
        "score_A":          float(np.clip(uA, 0, 10)),
        "score_B":          float(np.clip(uB, 0, 10)),
        "margin":           float(abs(uA - uB)),
        "preference_prob":  float(p_A),
    }


def get_wine_quality_score(profile: dict, noise_std: float = 0.0) -> float:
    return _hidden_utility(profile, noise_std=noise_std)


def _sigmoid(x, scale=1.0):
    return 1.0 / (1.0 + np.exp(-scale * x))

def _sigmoid_score(x, low, opt, high):
    if x <= low:
        return 0.0
    elif x <= opt:
        return (x - low) / (opt - low)
    else:
        return max(0.0, 1.0 - (x - opt) / (high - opt + 1e-9))

def _bell_score(x, center, width):
    return float(np.exp(-0.5 * ((x - center) / width) ** 2))


