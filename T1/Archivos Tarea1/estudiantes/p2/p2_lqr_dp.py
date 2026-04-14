from TwinRotorODE import TwinRotor
import numpy as np
import control


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
system = TwinRotor(params, Ts)

# modelo linealizado para el eq
x_eq, u_eq = system.get_eq_point(x0=0, x1=np.pi / 4)
Ad, Bd, Cd, Dd, dt = system.get_linear_system(x_eq, u_eq)

# Matrices de costo
Q = np.diag([1e-4, 1e-4, 1e-2, 1e-2, 1e-4, 1e-4])
R = np.eye(Bd.shape[1]) * 0.001 ** 2


# ricatti dp
def riccati_dp(A, B, Q, R, max_iter=20000, tol=1e-7):
    P = Q.copy()
    K = None
    history = []

    for k in range(max_iter):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ K
        P_next = 0.5 * (P_next + P_next.T)

        diff = np.linalg.norm(P_next - P, ord='fro')
        history.append(diff)
        P = P_next

        if diff < tol:
            return K, P, k + 1, np.array(history), True

    return K, P, max_iter, np.array(history), False


K_dp, P_dp, iters, history, converged = riccati_dp(Ad, Bd, Q, R)
K_dlqr, P_dlqr, E_dlqr = control.dlqr(Ad, Bd, Q, R)
K_dlqr = np.asarray(K_dlqr)
P_dlqr = np.asarray(P_dlqr)
E_dlqr = np.asarray(E_dlqr)

err_abs = np.linalg.norm(K_dp - K_dlqr, ord='fro')
err_rel = err_abs / max(np.linalg.norm(K_dlqr, ord='fro'), 1e-16)

# imprimir resultados

np.set_printoptions(precision=8, suppress=True)
print("=== Discrete LQR via dynamic programming ===")
print(f"Converged: {converged}")
print(f"Iterations until convergence: {iters}")
print(f"Final Riccati delta: {history[-1]:.3e}")
print("K from dynamic programming:")
print(K_dp)
print("\nK from control.dlqr:")
print(K_dlqr)
print("\n||K_dp - K_dlqr||:")
print(err_abs)
print("Relative difference:")
print(err_rel)
print("\nClosed-loop eigenvalues from control.dlqr:")
print(E_dlqr)
