import numpy as np
from scipy.integrate import odeint
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt

class Quadrotor3D:
    """3D quadrotor (Crazyflie CF2X) nonlinear ODE model.

    State vector (12):
        x = [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]

    Input vector (4):
        u = [f1, f2, f3, f4]   individual motor thrusts [N]

    Motor layout (top view, X-config):
        2(CCW)  1(CW)
            \\  /
             \\/
             /\\
            /  \\
        3(CW)  4(CCW)
    """

    def __init__(self, params=None, Ts=0.01, x0=None,
                 disturbance_time=[], disturbance_strength=0.0):
        if params is None:
            self.p = {
                'm':   0.027,
                'Ixx': 1.4e-5,
                'Iyy': 1.4e-5,
                'Izz': 2.17e-5,
                'L':   0.0397,
                'KF':  3.16e-10,
                'KM':  7.94e-12,
                'g':   9.8,
                            }
        else:
            self.p = params

        self.Ts = Ts
        if x0 is None:
            x0 = [0.0] * 12
        self.x  = list(x0)
        self.xd = [0.0] * 12
        self.t  = 0
        self.disturbance_time     = disturbance_time
        self.disturbance_strength = disturbance_strength

        # Normalization vector
        self.obs_norm = np.array([
              2.0,        # x       [m]
              2.0,        # x_dot   [m/s]
              2.0,        # y       [m]
              2.0,        # y_dot   [m/s]
              2.0,        # z       [m]
              2.0,        # z_dot   [m/s]
              np.pi/4,    # phi     [rad]
              np.pi/4,    # theta   [rad]
              np.pi,      # psi     [rad]
              5.0,        # p       [rad/s]
              5.0,        # q       [rad/s]
              5.0,        # r       [rad/s]
        ])


    def reset(self, x0=None, params=None):
        if params is None:
            pass
        else:
            self.p = params

        if x0 is None:
            x0 = [0.0] * 12
        self.x = list(x0)
        self.xd = [0.0] * 12
        self.t = 0

        y = [self.x[0] + np.random.randn() * 5e-3,  # x
             self.x[2] + np.random.randn() * 5e-3,  # y
             self.x[4] + np.random.randn() * 5e-3,  # z
             self.x[6] + np.random.randn() * 2e-3,  # phi
             self.x[7] + np.random.randn() * 2e-3,  # theta
             self.x[8] + np.random.randn() * 1e-3]  # psi

        return np.array(x0), np.array(y)

    @staticmethod
    def _rot_xyz(phi, theta, psi):
        cp, sp = np.cos(phi),   np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi),   np.sin(psi)
        return np.array([
            [ ct*cy,  sp*st*cy - cp*sy,  cp*st*cy + sp*sy],
            [ ct*sy,  sp*st*sy + cp*cy,  cp*st*sy - sp*cy],
            [-st,     sp*ct,              cp*ct            ]
        ])

    @staticmethod
    def _euler_kinematics(phi, theta):
        cp, sp = np.cos(phi),   np.sin(phi)
        ct, tt = np.cos(theta), np.tan(theta)
        return np.array([
            [1,  sp*tt,    cp*tt ],
            [0,  cp,      -sp    ],
            [0,  sp/ct,    cp/ct ]
        ])


    def xd_func(self, x, t, u):
        p      = self.p
        m, g   = p['m'], p['g']
        Ixx, Iyy, Izz = p['Ixx'], p['Iyy'], p['Izz']
        L      = p['L']
        gamma  = p['KM'] / p['KF']

        x_dot, y_dot, z_dot = x[1], x[3], x[5]
        phi, theta, psi     = x[6], x[7], x[8]
        p_b, q_b, r_b       = x[9], x[10], x[11]

        f1, f2, f3, f4 = u[0], u[1], u[2], u[3]
        T = f1 + f2 + f3 + f4

        R   = self._rot_xyz(phi, theta, psi)
        acc = R @ np.array([0., 0., T]) / m - np.array([0., 0., g])

        sqrt2L = L / np.sqrt(2.0)
        tau_x  = sqrt2L * ( f1 + f2 - f3 - f4)
        tau_y  = sqrt2L * (-f1 + f2 + f3 - f4)
        tau_z  = gamma  * (-f1 + f2 - f3 + f4)

        omega    = np.array([p_b, q_b, r_b])
        J        = np.diag([Ixx, Iyy, Izz])
        Jinv     = np.diag([1/Ixx, 1/Iyy, 1/Izz])
        rate_dot = Jinv @ (np.array([tau_x, tau_y, tau_z]) - np.cross(omega, J @ omega))

        W       = self._euler_kinematics(phi, theta)
        ang_dot = W @ omega

        return [
            x_dot,        acc[0],
            y_dot,        acc[1],
            z_dot,        acc[2],
            ang_dot[0],   ang_dot[1],  ang_dot[2],
            rate_dot[0],  rate_dot[1], rate_dot[2]
        ]


    def step(self, u=None):
        if u is None:
            u = self.hover_input()

        f_hover = self.p['m'] * self.p['g'] / 4.0
        f_max = 3 * f_hover
        u = np.clip(u, 0.0, f_max)

        self.x = np.array(self.x, dtype=float)
        sol = odeint(self.xd_func, self.x, [0, self.Ts], args=(u,))

        if self.disturbance_time and self.disturbance_time[0] <= self.t < self.disturbance_time[1]:
            progress  = ((self.t - self.disturbance_time[0]) /
                         (self.disturbance_time[1] - self.disturbance_time[0]))
            intensity = np.sin(np.pi * progress)
            d = self.disturbance_strength * intensity
        else:
            d = 0.0

        self.x     = list(sol[-1, :])
        self.x[9] += d    # kick roll rate p

        y = [self.x[0] + np.random.randn() * 5e-3,   # x
             self.x[2] + np.random.randn() * 5e-3,   # y
             self.x[4] + np.random.randn() * 5e-3,   # z
             self.x[6] + np.random.randn() * 2e-3,   # phi
             self.x[7] + np.random.randn() * 2e-3,   # theta
             self.x[8] + np.random.randn() * 1e-3]   # psi

        self.t += 1
        return np.array(self.x.copy()), np.array(y)


    def hover_input(self):
        f = self.p['m'] * self.p['g'] / 4.0
        return [f, f, f, f]

    def get_eq_point(self, z_ref=1.0):
        x_eq = [0., 0., 0., 0., z_ref, 0., 0., 0., 0., 0., 0., 0.]
        u_eq = self.hover_input()
        return x_eq, u_eq

    def get_linear_system(self, x_eq, u_eq, eps=1e-6):
        nx, nu = 12, 4
        x_eq = np.array(x_eq, dtype=float)
        u_eq = np.array(u_eq, dtype=float)

        def f(x, u):
            return np.array(self.xd_func(x, 0.0, u))

        A = np.zeros((nx, nx))
        for j in range(nx):
            xp, xm = x_eq.copy(), x_eq.copy()
            xp[j] += eps;  xm[j] -= eps
            A[:, j] = (f(xp, u_eq) - f(xm, u_eq)) / (2 * eps)

        B = np.zeros((nx, nu))
        for j in range(nu):
            up, um = u_eq.copy(), u_eq.copy()
            up[j] += eps;  um[j] -= eps
            B[:, j] = (f(x_eq, up) - f(x_eq, um)) / (2 * eps)

        C = np.zeros((6, 12))

        C[0, 0] = 1.0  # x
        C[1, 2] = 1.0  # y
        C[2, 4] = 1.0  # z
        C[3, 6] = 1.0  # phi
        C[4, 7] = 1.0  # theta
        C[5, 8] = 1.0  # psi


        D = np.zeros((nx, nu))
        Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), self.Ts, method='zoh')
        return Ad, Bd, C



if __name__ == '__main__':

    # Physical parameters
    params = {
        'm':   0.027,      # kg
        'Ixx': 1.4e-5,     # kg·m²
        'Iyy': 1.4e-5,     # kg·m²
        'Izz': 2.17e-5,    # kg·m²
        'L':   0.0397,     # m    (arm length)
        'KF':  3.16e-10,   # N·s²/rad²  (thrust coefficient)
        'KM':  7.94e-12,   # N·m·s²/rad²  (moment coefficient)
        'g':   9.8,        # m/s²
    }

    Ts    = 0.01      # (100 Hz)
    N     = 500
    t_vec = np.arange(N) * Ts

    STATE_LABELS = [
        'x [m]',     'x_dot [m/s]',
        'y [m]',     'y_dot [m/s]',
        'z [m]',     'z_dot [m/s]',
        'phi [rad]', 'theta [rad]', 'psi [rad]',
        'p [rad/s]', 'q [rad/s]',  'r [rad/s]'
    ]

    # Scenario 1: pure hover at z = 1 m
    system = Quadrotor3D(params, Ts)
    x_eq, u_eq = system.get_eq_point(z_ref=1.0)
    system.x = list(x_eq)

    states_hover = np.zeros((N, 12))
    for k in range(N):
        states_hover[k, :], _ = system.step(u=u_eq)

    fig1, axes1 = plt.subplots(3, 4, figsize=(20, 10))
    fig1.suptitle('Quadrotor 3D — Hover at z = 1 m', fontsize=13)
    for i, ax in enumerate(axes1.flat):
        ax.plot(t_vec, states_hover[:, i], linewidth=1.2, color='steelblue')
        ax.set_ylabel(STATE_LABELS[i])
        ax.set_xlabel('t [s]')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Scenario 2: ascent step — 20 % extra thrust
    system2 = Quadrotor3D(params, Ts, x0=[0.]*12)
    u_step  = [f * 1.2 for f in u_eq]

    states_step = np.zeros((N, 12))
    for k in range(N):
        states_step[k, :], _ = system2.step(u=u_step)

    fig2, axes2 = plt.subplots(3, 4, figsize=(20, 10))
    fig2.suptitle('Quadrotor 3D — Vertical step (20 % thrust increase from rest)', fontsize=13)
    for i, ax in enumerate(axes2.flat):
        ax.plot(t_vec, states_step[:, i], linewidth=1.2, color='darkorange')
        ax.set_ylabel(STATE_LABELS[i])
        ax.set_xlabel('t [s]')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

