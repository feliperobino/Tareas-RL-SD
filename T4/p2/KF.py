import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


class DiscreteKalmanFilter:
    def __init__(self, A, B, C, D, Q, R, P, x_eq, u_eq, y_eq,  x_init=None, eps=1e-12):
        if x_init is None:
            self.x = np.zeros(Q.shape[0])
        self.x = x_init
        self.P = P
        self.Q = Q
        self.R = R
        self.eps = eps
        self.x_eq = np.array(x_eq).reshape(-1, 1)
        self.u_eq = np.array(u_eq).reshape(-1, 1)
        self.y_eq = np.array(y_eq).reshape(-1, 1)
        self.A = A
        self.B = B
        self.C = C
        self.D = D


    def predict(self, u: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Time update:
            x^- = A x + B u
            P^- = A P A^T + Q
        """
        u = np.asarray(u, dtype=float)
        nu = self.B.shape[1]

        if u.shape == (nu,):
            u = u.reshape(nu, 1)
        if u.shape != (nu, 1):
            raise ValueError(f"u must be (nu,1); got {u.shape}")

        x_pred = self.A @ self.x + self.B @ u
        P_pred = self.A @ self.P @ self.A.T + self.Q
        P_pred = _sym(P_pred)

        self.x = x_pred
        self.P = P_pred

        x_pred_out = (x_pred + self.x_eq).squeeze()

        return x_pred_out

    def update(self, y: np.ndarray, u: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Measurement update:
            y_hat = C x^- + D u
            S = C P^- C^T + R
            K = P^- C^T S^{-1}
            x^+ = x^- + K (y - y_hat)
            P^+ = (I - K C) P^- (I - K C)^T + K R K^T
        """
        y = np.asarray(y, dtype=float)
        ny = self.C.shape[0]

        if y.shape == (ny,):
            y = y.reshape(ny, 1)
        if y.shape != (ny, 1):
            raise ValueError(f"y must be (ny,1); got {y.shape}")

        if u is None:
            u = np.zeros((self.B.shape[1], 1), dtype=float)
        else:
            u = np.asarray(u, dtype=float)
            nu = self.B.shape[1]
            if u.shape == (nu,):
                u = u.reshape(nu, 1)
            if u.shape != (nu, 1):
                raise ValueError(f"u must be (nu,1); got {u.shape}")

        y_hat = self.C @ self.x + self.D @ u

        S = self.C @ self.P @ self.C.T + self.R
        S = _sym(S)

        S = S + self.eps * np.eye(S.shape[0])

        K = self.P @ self.C.T @ np.linalg.solve(S, np.eye(S.shape[0]))

        innov = y - y_hat
        x_filt = self.x + K @ innov

        I = np.eye(self.P.shape[0])
        IKC = I - K @ self.C

        P_filt = IKC @ self.P @ IKC.T + K @ self.R @ K.T
        P_filt = _sym(P_filt)

        # Store filtered
        self.x = x_filt
        self.P = P_filt

        return {
            "x_filt": x_filt,
            "P_filt": P_filt,
            "K": K,
            "innov": innov,
            "y_hat": y_hat,
            "S": S,
        }

    def step(
        self,
        u: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:


        u = np.array(u).reshape(-1, 1) - self.u_eq
        y = np.array(y).reshape(-1, 1) - self.y_eq

        out = {}
        self.predict(u)
        #out.update(self.predict(u))

        if y is not None:
            out.update(self.update(y, u=u))

        x_est = (out["x_filt"] + self.x_eq).squeeze()

        return x_est



