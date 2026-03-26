from TwinRotorODE import TwinRotor
import numpy as np
import matplotlib.pyplot as plt
from KF import DiscreteKalmanFilter


params = {
        'Jp': 0.038,
        'Jy': 0.043,
        'mgl': 0.32,
        'Kpp': 0.204,
        'Kyp': 0.051,
        'Kpt': 0.011,
        'Kyt': 0.072,
        'Bp': 0.05,
        'By': 0.03,
        'Tm': 0.11,
        'Tt': 0.10,
        'JrOmega': 0.02,
    }

Ts = 0.01

# Instantiating the system
system = TwinRotor(params, Ts)

# Get equilibrium point for a desired reference
reference = [0, np.pi/4]
x_eq, u_eq = system.get_eq_point(x0=reference[0], x1=reference[1])
system.x = x_eq
y_eq = np.array([x_eq[0], x_eq[1]]).reshape(-1, 1)

# Given the equilibrium point, get the discretized linear matrices
Ad, Bd, Cd, Dd, dt = system.get_linear_system(x_eq, u_eq)

# Instantiate the Kalman Filter
Q = np.diag([1e-4, 1e-4, 1e-2, 1e-2, 1e-4, 1e-4])
R = np.eye(Cd.shape[0]) * 0.001 ** 2
P0 = np.eye(Ad.shape[0]) * 10
kf = DiscreteKalmanFilter.create(Ad, Bd, Cd, Q, R, P_init=P0, D=Dd, x_init=np.zeros((6, 1)))


##### Simulation #######
u_hover = [u_eq[0] - 0.01, u_eq[1]] # Small perturbation

N = 2000 # Steps
states = np.zeros((N, 6))
states_discrete = np.zeros((N, 6))
states_estimated = np.zeros((N, 6))
measurements = np.zeros((N, 2))
for k in range(N):
    states[k, :], measurements[k, :] = system.sim(u=u_hover)
    states_discrete[k, :] = system.sim_discrete(u_hover, Ad, Bd, Cd, Dd, x_eq, u_eq)

    x_hat = kf.step(u=np.array(u_hover).reshape(-1, 1) - np.array(u_eq).reshape(-1, 1), y=np.array(measurements[k, :]).reshape(-1, 1) - y_eq.reshape(-1, 1))["x_filt"]
    states_estimated[k, :] = (x_hat + np.array(x_eq).reshape(-1, 1)).squeeze()  # filtered state estimate at time k


##### Plotting #####
t_vec = np.arange(N) * Ts
labels = ['δφ [rad]', 'δψ [rad]', 'δφ̇ [rad/s]', 'δψ̇ [rad/s]', 'δVm [V]', 'δVt [V]']

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
fig.suptitle('Twin Rotor — Deviation from equilibrium',
             fontsize=13)
for i, ax in enumerate(axes.flat):
    ax.plot(t_vec, states[:, i], linewidth=1.2, label='NonLinear')
    ax.plot(t_vec, states_discrete[:, i], linewidth=1.2, label='Linear')
    ax.plot(t_vec, states_estimated[:, i], linewidth=1.2, label='KF est')
    ax.set_ylabel(labels[i])
    ax.set_xlabel('t [s]')
    ax.grid(True, alpha=0.3)
    ax.legend()
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize=(18, 8))
ax.plot(t_vec, measurements, linewidth=1.2, label='Measurements')
ax.set_ylabel(labels[i])
ax.set_xlabel('t [s]')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()