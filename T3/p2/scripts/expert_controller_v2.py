import numpy as np

class SmoothController:

    T_START = 18.0
    T_MID = 20.0
    T_PEAK = 22.0

    FEED_INTERVAL = 6.0 # H
    MICRO_DOSE = 5.0

    N_TARGET = 220.0
    N_MAX_CORRECTION = 8.0

    CONVERSION_STOP = 0.5 # conversion más de 50 -> no alimentar

    def __init__(self):
        self.reset()

    def reset(self):
        self._S0 = None
        self._last_feed_t = -999.9
        
        self._early_rates = []
        self._S_history = []

    #### 
    ## TEMPERATURA IGUAL PERO MÁS SUAVE LA CURVA

    def setpoint(self, t):
        # 18 -> 20
        if t <= 15:
            return 18.0 + (20.0 - 18.0) * t / 15.0
        # mantener 20
        elif t <= 40:
            return 20.0
        # 20 -> 16
        else:
            return 20.0 - (4.0* t-170.00) / 200.0
        
    #####
    # NITRÓGENO

    # utils
    def sugar_conversion(self, s):
        if self._S0 is None:
            return 0.0
        
        return np.clip(
            (self._S0 - s) / max(self._S0, 1e-6),
            0.0,
            1.0
        )
            
    def dS_dt(self):
        if len(self._S_history) < 2:
            return None
        
        window = self._S_history[-8:]

        if len(window) < 2:
            return None
        
        return (window[0] - window[-1]) / (len(window) * 0.25)
    

    # feeding
    def nitrogen_pol(self, obs, t):

        S, E, X, Xd, N = obs[0], obs[1], obs[2], obs[3], obs[4]

        if self._S0 is None:
            self._S0 = float(S)

        conv = self.sugar_conversion(S)

        # guardar history
        self._S_history.append(float(S))

        if len(self._S_history) > 20:
            self._S_history.pop(0)

        rate = self.dS_dt()

        # early rates
        if t <= 24 and rate is not None:
            self._early_rates.append(rate)

        ### REGLAS
        # AGREGAR PEAK INICIAL
        if t <= 15 and (t - self._last_feed_t) >= self.FEED_INTERVAL:
            self._last_feed_t = t
            return np.clip(
                0.15 * self._S0,
                0.0, 15.0
            )

        # 1. NO ALIMENTAR MÁS CUANDO CONVERSIÓN > 50%
        if conv > self.CONVERSION_STOP:
            return 0.0
        
        dose = 0.0
        
        # 2. DAR MICRODOSIS CADA X HORAS
        if (t - self._last_feed_t) >= self.FEED_INTERVAL:
            dose += self.MICRO_DOSE
            self._last_feed_t = t

        # 3. MANTENER N CERCA DEL TARGET
        if N < self.N_TARGET:

            correction = 0.2 * (self.N_TARGET - N)

            dose += np.clip(
                correction,
                0.0, self.N_MAX_CORRECTION
            )

        # 2. DESACELERAR FERMENTACIÓN
        if (t > 24 and len(self._early_rates) > 10 and rate is not None):
            baseline = np.mean(self._early_rates)
            if baseline > 1e-6:
                ratio = rate / baseline

                if ratio < 0.70:
                    dose += 5.0
                elif ratio < 0.85:
                    dose += 2.5

        return float(np.clip(dose, 0.0, 15.0))
        

    # call
    def __call__(self, obs, t):
        T_sp = self.setpoint(t)
        N_feed = self.nitrogen_pol(obs, t)

        return np.array([T_sp, N_feed], dtype=np.float32)
