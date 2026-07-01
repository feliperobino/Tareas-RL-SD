import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sympy as sy
import numpy as np
from scipy.signal import cont2discrete



# Conversions
MGDL_TO_MMOL = 1.0 / 18.0182   # mg/dL  → mmol/L
MMOL_TO_MGDL = 18.0182          # mmol/L → mg/dL
G_TO_MMOL    = 1000.0 / 180.155 # g/min  → mmol/min

def sim_to_hov_u(insulin_U_min, meal_g_per_min):
    return np.array([insulin_U_min * 1000.0,       # U/min → mU/min
                     meal_g_per_min * G_TO_MMOL])  # g/min → mmol/min

def cgm_to_mmol(cgm_mg_dL):
    return float(cgm_mg_dL) * MGDL_TO_MMOL



class HovorkaModel:
    """
    States (x):
        x[0] = x1  (insulin action on glucose transport/uptake)     [1/min]
        x[1] = x2  (insulin action on glucose disposal)              [1/min]
        x[2] = x3  (insulin action on EGP suppression)               [1/min]
        x[3] = Q1  (glucose in accessible compartment)               [mmol]
        x[4] = Q2  (glucose in non-accessible compartment)           [mmol]
        x[5] = D1  (CHO in gut compartment 1)                        [mmol]
        x[6] = D2  (CHO in gut compartment 2)                        [mmol]
        x[7] = S1  (insulin in subcutaneous compartment 1)           [mU]
        x[8] = S2  (insulin in subcutaneous compartment 2)           [mU]
        x[9] = I   (plasma insulin)                                  [mU/L]
        x[10]= C   (filtered glucose for sensor)                   [mmol/L]

    """
    def __init__(self,
                 Ts=5.0,                 # integration step in minutes
                 BW=70.0,               # body weight [kg]
                 EGPo=0.0161,           # endogenous glucose production [mmol/kg/min]
                 F01=0.0097,            # non–insulin-dependent glucose flux [mmol/kg/min]
                 VG=0.16,               # glucose distribution volume [L/kg]
                 VI=0.12,               # insulin distribution volume [L/kg]
                 k12=0.066,             # transfer between Q2->Q1 [1/min]
                 ka1=0.006, ka2=0.06, ka3=0.03,  # insulin action decays [1/min]
                 ke=0.138,              # insulin elimination [1/min]
                 AG=0.8, tG=40.0,       # gut bioavailability, time constant [min]
                 tI=55.0,               # insulin absorption time constant [min]
                 # Measurement filter
                 ka_int=0.073,
                 SI_scale=0.35):
        # Save step and BW
        self.Ts = float(Ts)
        self.BW = float(BW)

        # Scale per-kg parameters
        self.EGPo = EGPo * BW     # mmol/min
        self.F01  = F01  * BW     # mmol/min
        self.VG   = VG   * BW     # L
        self.VI   = VI   * BW     # L

        # Dynamics parameters
        self.k12 = k12
        self.ka1, self.ka2, self.ka3 = ka1, ka2, ka3
        self.ke = ke

        # Absorption
        self.AG = AG
        self.tG = float(tG)
        self.tI = float(tI)

        # Measurement
        self.ka_int = ka_int


        self.Sf1 = SI_scale * 51.2e-4
        self.Sf2 = SI_scale * 8.2e-4
        self.Sf3 = SI_scale * 520e-4


        # Initial conditions
        G0 = 6.5
        Q10 = G0 * self.VG
        Q20 = Q10
        I0  = 10.0                      # mU/L, basal insulin
        x0 = np.zeros(11, dtype=float)
        self.x_discrete =  np.zeros(11, dtype=float).reshape(-1, 1)
        x0[:] = [0.0, 0.0, 0.0, Q10, Q20, 0.0, 0.0, 0.0, 0.0, I0, G0]
        self.state = x0

    def g_per_min_to_mmol_per_min(self, D_g_per_min):
        # 1 mmol glucose = 180.155 mg
        return (D_g_per_min * 1000.0) / 180.155

    def U_per_h_to_mU_per_min(self, u_U_per_h):
        return (u_U_per_h * 1000.0) / 60.0


    def get_basal(self, G=6.1):
        x1, x2, x3, Q2, D1, D2, S1, S2, I, C, u_U_per_h = sy.symbols('x1, x2, x3, Q2, D1, D2, S1, S2, I, C, u_U_per_h')

        Q1 = G*self.VG

        # We need to find this
        u_mU_min = self.U_per_h_to_mU_per_min(u_U_per_h)


        # Considering no meal
        D_g_this_step = 0
        D_g_per_min = float(D_g_this_step) / self.Ts
        DG_rate = self.g_per_min_to_mmol_per_min(D_g_per_min)

        # Gut absorption
        dD1 = self.AG * DG_rate - D1 / self.tG
        dD2 = D1 / self.tG - D2 / self.tG
        UG = D2 / self.tG  # mmol/min

        # Insulin absorption
        dS1 = u_mU_min - S1 / self.tI
        dS2 = S1 / self.tI - S2 / self.tI
        UI = S2 / self.tI  # mU/min

        # Plasma insulin
        dI = UI / self.VI - self.ke * I

        # Remote insulin actions
        kb1 = self.ka1 * self.Sf1
        kb2 = self.ka2 * self.Sf2
        kb3 = self.ka3 * self.Sf3
        dx1 = -self.ka1 * x1 + kb1 * I
        dx2 = -self.ka2 * x2 + kb2 * I
        dx3 = -self.ka3 * x3 + kb3 * I

        Fc01 = self.F01 * (G / 4.5)
        FR = 0.003 * (G - 9.0) * self.VG if G >= 9.0 else 0.0

        # Glucose subsystem
        dQ1 = -(Fc01 + x1 * Q1) + self.k12 * Q2 - FR + UG + self.EGPo * (1.0 - x3)
        dQ2 = (x1 * Q1) - (self.k12 + x2) * Q2

        # Measurement filter
        dC = self.ka_int * (G - C)

        solution = sy.solve([dx1, dx2, dx3, dQ1, dQ2, dD1, dD2, dS1, dS2, dI, dC], [x1, x2, x3, Q2, D1, D2, S1, S2, I, C, u_U_per_h], dict=True)

        index = np.argmax(np.array([elem[u_U_per_h] for elem in solution]))
        u =  max(np.array([elem[u_U_per_h] for elem in solution]))
        solution = solution[index]
        x0 = np.array([solution[x1], solution[x2], solution[x3], solution[C]*self.VG, solution[Q2], solution[D1],
                       solution[D2], solution[S1], solution[S2], solution[I], solution[C]]).astype(float)
        return u, x0


    def get_basal_more_info(self, G, u_basal_U_h):

        u_mU_min = self.U_per_h_to_mU_per_min(u_basal_U_h)

        S1 = u_mU_min * self.tI
        S2 = S1
        I  = u_mU_min / (self.VI * self.ke)

        x1 = self.Sf1 * I
        x2 = self.Sf2 * I
        x3 = self.Sf3 * I

        Q1 = G * self.VG
        Q2 = x1 * Q1 / (self.k12 + x2)

        C = G

        D1, D2 = 0.0, 0.0

        Fc01 = self.F01 * (G / 4.5) if G > 4.5 else self.F01
        FR   = 0.003 * (G - 9.0) * self.VG if G >= 9.0 else 0.0

        EGPo        = (Fc01 + x1 * Q1 - self.k12 * Q2 + FR) / (1.0 - x3)
        EGPo_per_kg = EGPo / self.BW
        self.EGPo = EGPo

        x0 = np.array([x1, x2, x3, Q1, Q2, D1, D2, S1, S2, I, C], dtype=float)
        return x0, EGPo, EGPo_per_kg



    def _rhs(self, x, t, u_U_per_h, D_g_this_step):
        x1, x2, x3, Q1, Q2, D1, D2, S1, S2, I, C = x

        G = Q1 / self.VG

        u_mU_min = self.U_per_h_to_mU_per_min(u_U_per_h)
        D_g_per_min = float(D_g_this_step) / self.Ts
        DG_rate = self.g_per_min_to_mmol_per_min(D_g_per_min)

        # Gut absorption
        dD1 = self.AG * DG_rate - D1 / self.tG
        dD2 = D1 / self.tG - D2 / self.tG
        UG  = D2 / self.tG # mmol/min

        # Insulin absorption
        dS1 = u_mU_min - S1 / self.tI
        dS2 = S1 / self.tI - S2 / self.tI
        UI  = S2 / self.tI  # mU/min

        # Plasma insulin
        dI  = UI / self.VI - self.ke * I

        # Remote insulin actions
        kb1 = self.ka1 * self.Sf1
        kb2 = self.ka2 * self.Sf2
        kb3 = self.ka3 * self.Sf3
        dx1 = -self.ka1 * x1 + kb1 * I
        dx2 = -self.ka2 * x2 + kb2 * I
        dx3 = -self.ka3 * x3 + kb3 * I

        if G <= 4.5:
            Fc01 = self.F01
        else:
            Fc01 = self.F01 * (G / 4.5)
        FR = 0.003 * (G - 9.0) * self.VG if G >= 9.0 else 0.0

        # Glucose subsystem
        dQ1 = -(Fc01 + x1 * Q1) + self.k12 * Q2 - FR + UG + self.EGPo * (1.0 - x3)
        dQ2 = (x1 * Q1) - (self.k12 + x2) * Q2

        # Measurement filter
        dC = self.ka_int * (G - C)

        return np.array([dx1, dx2, dx3, dQ1, dQ2, dD1, dD2, dS1, dS2, dI, dC], dtype=float)

    def step(self, u=0.0, D=0.0):
        tspan = np.linspace(0.0, self.Ts, 2)
        sol = odeint(self._rhs, self.state, tspan, args=(u, D))
        self.state = sol[-1]
        return self.state.copy()

    def get_linear_system(self, x_eq, u_eq):
        x1, x2, x3, Q1, Q2, D1, D2, S1, S2, I, C = sy.symbols('x1, x2, x3, Q1, Q2, D1, D2, S1, S2, I, C')
        u_mU_min, DG_rate = sy.symbols('u_mU_min, DG_rate')

        u_eq_min = self.U_per_h_to_mU_per_min(u_eq[0])

        D_g_per_min = float(u_eq[1]) / self.Ts
        dg_eq_min = self.g_per_min_to_mmol_per_min(D_g_per_min)
        self.u_eq = [u_eq_min, dg_eq_min]
        self.x_eq = x_eq

        x = sy.Matrix([x1, x2, x3, Q1, Q2, D1, D2, S1, S2, I, C])
        states = [x1, x2, x3, Q1, Q2, D1, D2, S1, S2, I, C]
        u = sy.Matrix([u_mU_min, DG_rate])

        G = Q1 / self.VG

        # Gut absorption
        dD1 = self.AG * DG_rate - D1 / self.tG
        dD2 = D1 / self.tG - D2 / self.tG
        UG = D2 / self.tG  # mmol/min

        # Insulin absorption
        dS1 = u_mU_min - S1 / self.tI
        dS2 = S1 / self.tI - S2 / self.tI
        UI = S2 / self.tI  # mU/min

        # Plasma insulin
        dI = UI / self.VI - self.ke * I

        # Remote insulin actions
        kb1 = self.ka1 * self.Sf1
        kb2 = self.ka2 * self.Sf2
        kb3 = self.ka3 * self.Sf3
        dx1 = -self.ka1 * x1 + kb1 * I
        dx2 = -self.ka2 * x2 + kb2 * I
        dx3 = -self.ka3 * x3 + kb3 * I

        Fc01 = self.F01 * (G / 4.5)
        FR = 0.0

        # Glucose subsystem
        dQ1 = -(Fc01 + x1 * Q1) + self.k12 * Q2 - FR + UG + self.EGPo * (1.0 - x3)
        dQ2 = (x1 * Q1) - (self.k12 + x2) * Q2

        # Measurement filter
        dC = self.ka_int * (G - C)

        # Getting the matrices
        f = sy.Matrix([dx1, dx2, dx3, dQ1, dQ2, dD1, dD2, dS1, dS2, dI, dC])
        g = sy.Matrix([C])

        A = f.jacobian(x)
        B = f.jacobian(u)
        C = g.jacobian(x)
        D = g.jacobian(u)

        # Evaluating at the equilibrium points
        A_eq = np.array(A.subs(
            [(states[i], x_eq[i]) for i in range(len(states))] + [(u_mU_min, u_eq_min), (DG_rate, dg_eq_min)]).evalf())
        B_eq = np.array(B.subs(
            [(states[i], x_eq[i]) for i in range(len(states))] + [(u_mU_min, u_eq_min), (DG_rate, dg_eq_min)]).evalf())
        C_eq = np.array(C.subs(
            [(states[i], x_eq[i]) for i in range(len(states))] + [(u_mU_min, u_eq_min), (DG_rate, dg_eq_min)]).evalf())
        D_eq = np.array(D.subs(
            [(states[i], x_eq[i]) for i in range(len(states))] + [(u_mU_min, u_eq_min), (DG_rate, dg_eq_min)]).evalf())

        # Discretization
        self.Ad, self.Bd, self.Cd, self.Dd, dt = cont2discrete((A_eq, B_eq, C_eq, D_eq), self.Ts, method='zoh')
        self.Ad = self.Ad.astype(float)
        self.Bd = self.Bd.astype(float)
        self.Cd = self.Cd.astype(float)
        self.Dd = self.Dd.astype(float)

        self.x_discrete =  np.zeros(11, dtype=float).reshape(-1, 1)

        return self.Ad, self.Bd, self.Cd, self.Dd, dt


    def discrete_step(self, u=0.0, D=0.0):
        u_min = self.U_per_h_to_mU_per_min(u)
        D_rate = self.g_per_min_to_mmol_per_min(float(D) / self.Ts)

        u_centered = u_min - self.u_eq[0]
        d_centered = D_rate - self.u_eq[1]
        u_vec = np.array([u_centered, d_centered]).reshape(-1, 1)

        self.x_discrete = self.Ad @ self.x_discrete.reshape(-1, 1) + self.Bd @ u_vec
        x_out = self.x_discrete + self.x_eq.reshape(-1, 1)

        return x_out.squeeze()

    def simulate(self, minutes, u_profile=None, D_profile=None):
        n_steps = int(np.ceil(minutes / self.Ts))
        t_rec = np.zeros(n_steps + 1)
        G_rec = np.zeros(n_steps + 1)
        DG_rec = np.zeros(n_steps + 1)
        I_rec = np.zeros(n_steps + 1)
        U_rec = np.zeros(n_steps)
        D_rec = np.zeros(n_steps)

        x = self.state.copy()
        G_rec[0] = x[3] / self.VG
        DG_rec[0] = x[3] / self.VG
        I_rec[0] = x[9]

        for k in range(n_steps):
            t0 = k * self.Ts
            u = float(u_profile(k, t0)) if u_profile is not None else 0.0
            D = float(D_profile(k, t0)) if D_profile is not None else 0.0
            self.step(u=u, D=D)
            x_discrete = self.discrete_step(u=u, D=D)
            x = self.state
            t_rec[k+1] = t0 + self.Ts
            G_rec[k+1] = x[3] / self.VG
            DG_rec[k+1] = x_discrete[3] / self.VG
            I_rec[k+1] = x[9]
            U_rec[k]   = u
            D_rec[k]   = D

        return {
            "t_min": t_rec,
            "G_mmol_L": G_rec,
            "DG_mmol_L": DG_rec,
            "I_mU_L": I_rec,
            "u_U_per_h": U_rec,
            "D_g_step": D_rec,
        }





if __name__ == "__main__":

    model = HovorkaModel(Ts=5.0, BW=70.0)
    u_basal, x0 = model.get_basal(G=6.1)
    model.state = x0
    Ad, Bd, Cd, Dd, dt = model.get_linear_system(x_eq=x0, u_eq=[u_basal, 0])

    def u_prof(k, t):
        bolo = 0.5 if abs(t - 60.0) < 1e-9 else 0.0
        return u_basal + bolo

    def D_prof(k, t):
        return 10.0 if abs(t - 60.0) < 1e-9 else 0.0

    out = model.simulate(500, u_profile=u_prof, D_profile=D_prof)
    t = out["t_min"]
    G = out["G_mmol_L"]
    DG = out['DG_mmol_L']

    plt.figure()
    plt.plot(t, G, label="G [mmol/L]")
    plt.plot(t, DG, label="DG [mmol/L]")
    plt.xlabel("Time [min]")
    plt.ylabel("Glucose [mmol/L]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
