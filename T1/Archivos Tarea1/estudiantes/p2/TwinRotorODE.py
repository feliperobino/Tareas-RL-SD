import numpy as np
from scipy.integrate import odeint
import sympy as sy
from scipy.signal import cont2discrete
import control


class TwinRotor:
    def __init__(self, params, Ts=0.01, x0=None, disturbance_time=[],
                 disturbance_strength=0):
        self.p = params
        self.Ts = Ts
        if x0 is None:
            x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.x = list(x0)
        self.xd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.disturbance_time = disturbance_time
        self.disturbance_strength = disturbance_strength
        self.t = 0

    def xd_func(self, x, t, u):

        p = self.p

        xd0 = x[2]
        xd1 = x[3]
        xd2 = (1/p['Jp']) * (p['Kpp']*x[4] + p['Kpt']*x[5]
                              - p['mgl']*np.cos(x[0])*np.cos(x[1])
                              - p['Bp']*x[2]
                              + p['JrOmega']*x[3])
        xd3 = (1/p['Jy']) * (p['Kyp']*x[4] + p['Kyt']*x[5]
                              - p['By']*x[3]
                              - p['JrOmega']*x[2])
        xd4 = (1/p['Tm']) * (-x[4] + u[0])
        xd5 = (1/p['Tt']) * (-x[5] + u[1])

        return [xd0, xd1, xd2, xd3, xd4, xd5]

    def sim(self, u=None):

        if u is None:
            u = list(self.get_eq_inputs())
        self.x = np.array(self.x, dtype=float)
        t = np.linspace(0, self.Ts, 2)
        sol = odeint(self.xd_func, self.x, t, args=(u,))

        if len(self.disturbance_time) > 0:
            if self.disturbance_time[0] <= self.t < self.disturbance_time[1]:
                progress = (self.t - self.disturbance_time[0]) / (self.disturbance_time[1] - self.disturbance_time[0])
                intensity = np.sin(np.pi * progress)
                d = [self.disturbance_strength * intensity, self.disturbance_strength * 0.5 * intensity]
            else:
                d = [0, 0]
        else:
            d = [0, 0]

        self.x = list(sol[-1, :])
        y = [self.x[0] + np.random.randn() * 0.01, self.x[1] + np.random.randn() * 0.005]
        self.x[2] += d[0]
        self.x[3] += d[1]

        self.t += 1
        return self.x.copy(), y


    def get_eq_point(self, x0=0, x1=1):
        x2, x3, x4, x5, u0, u1 = sy.symbols('x2 x3 x4 x5 u0 u1')

        p = self.p

        xd0 = x2
        xd1 = x3
        xd2 = (1 / p['Jp']) * (p['Kpp'] * x4 + p['Kpt'] * x5
                               - p['mgl'] * np.cos(x0) * np.cos(x1)
                               - p['Bp'] * x2
                               + p['JrOmega'] * x3)  #
        xd3 = (1 / p['Jy']) * (p['Kyp'] * x4 + p['Kyt'] * x5
                               - p['By'] * x3
                               - p['JrOmega'] * x2)
        xd4 = (1 / p['Tm']) * (-x4 + u0)
        xd5 = (1 / p['Tt']) * (-x5 + u1)

        solution = sy.solve([xd0, xd1, xd2, xd3, xd4, xd5], [x2, x3, x4, x5, u0, u1], dict=True)
        x_eq = [x0, x1, solution[0][x2], solution[0][x3], solution[0][x4], solution[0][x5]]
        u_eq = [solution[0][u0], solution[0][u1]]
        return x_eq, u_eq


    def get_linear_system(self, x_eq, u_eq):
        x0, x1, x2, x3, x4, x5 = sy.symbols('x0 x1 x2 x3 x4 x5')
        u0, u1 = sy.symbols('u0 u1')

        x = sy.Matrix([x0, x1, x2, x3, x4, x5])
        u = sy.Matrix([u0, u1])

        p = self.p
        xd0 = x2
        xd1 = x3
        xd2 = (1 / p['Jp']) * (p['Kpp'] * x4 + p['Kpt'] * x5
                               - p['mgl'] * sy.cos(x0) * sy.cos(x1)
                               - p['Bp'] * x2
                               + p['JrOmega'] * x3)
        xd3 = (1 / p['Jy']) * (p['Kyp'] * x4 + p['Kyt'] * x5
                               - p['By'] * x3
                               - p['JrOmega'] * x2)
        xd4 = (1 / p['Tm']) * (-x4 + u0)
        xd5 = (1 / p['Tt']) * (-x5 + u1)



        # Getting the matrices
        f = sy.Matrix([xd0, xd1, xd2, xd3, xd4, xd5])
        g = sy.Matrix([x0, x1])

        A = f.jacobian(x)
        B = f.jacobian(u)
        C = g.jacobian(x)
        D = g.jacobian(u)

        # Evaluating at the equilibrium points
        A_eq = np.array(A.subs([(x0, x_eq[0]), (x1, x_eq[1]), (x2, x_eq[2]), (x3, x_eq[3]), (x4, x_eq[4]), (x5, x_eq[5]), (u0, u_eq[0]), (u1, u_eq[1])]).evalf())
        B_eq = np.array(B.subs([(x0, x_eq[0]), (x1, x_eq[1]), (x2, x_eq[2]), (x3, x_eq[3]), (x4, x_eq[4]), (x5, x_eq[5]), (u0, u_eq[0]), (u1, u_eq[1])]).evalf())
        C_eq = np.array(C.subs([(x0, x_eq[0]), (x1, x_eq[1]), (x2, x_eq[2]), (x3, x_eq[3]), (x4, x_eq[4]), (x5, x_eq[5]), (u0, u_eq[0]), (u1, u_eq[1])]).evalf())
        D_eq = np.array(D.subs([(x0, x_eq[0]), (x1, x_eq[1]), (x2, x_eq[2]), (x3, x_eq[3]), (x4, x_eq[4]), (x5, x_eq[5]), (u0, u_eq[0]), (u1, u_eq[1])]).evalf())

        # Discretization
        Ad, Bd, Cd, Dd, dt = cont2discrete((A_eq, B_eq, C_eq, D_eq), self.Ts, method='zoh')

        obs = control.obsv(Ad, Cd)
        rank = np.linalg.matrix_rank(obs)
        print(rank)
        print(A.shape)
        print(f"Is observable?: {rank == A.shape[0]}")

        eigvals = np.linalg.eigvals(Ad)
        print("Max |eigenvalue|:", np.max(np.abs(eigvals)))

        return Ad, Bd, Cd, Dd, dt


    def sim_discrete(self, u, A, B, C, D, x_eq, u_eq):
        xk = np.array(self.xd, dtype=float).reshape(-1, 1)
        u = np.array(u, dtype=float).reshape(-1, 1) - np.array(u_eq, dtype=float).reshape(-1, 1)
        x_k1 = A @ xk + B @ u
        y_k = C @ xk + D @ u
        self.xd = x_k1
        x_out = x_k1.copy() + np.array(x_eq, dtype=float).reshape(-1, 1)
        return x_out.squeeze().tolist()


