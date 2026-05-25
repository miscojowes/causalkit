"""
Fast NOTEARS implementations.

Two strategies:
1. notears_lbfgs: L-BFGS-B + doubled variables (accurate, moderate speed)
2. notears_adam: Adam + scipy expm (fast but less accurate)
"""

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import torch


# ═══════════════════════════════════════════════════════════════════════
#  Strategy 1: L-BFGS-B + doubled variables (official NOTEARS approach)
#  Accurate, handles L1 properly, moderate speed
# ═══════════════════════════════════════════════════════════════════════

def notears_lbfgs(
    X: np.ndarray,
    lambda_1: float = 0.01,
    max_iter: int = 10,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.1,
    lbfgs_maxiter: int = 20,
) -> np.ndarray:
    """NOTEARS with L-BFGS-B and doubled variables (official approach).

    From Zheng et al. (2018): https://github.com/xunzheng/notears

    Args:
        X: Data (n, d), centered
        lambda_1: L1 penalty
        max_iter: Max augmented Lagrangian iterations
        rho_max: Max penalty parameter
        w_threshold: Prune edges with |w| < threshold
        lbfgs_maxiter: Max L-BFGS iterations per call

    Returns:
        W: (d, d) weight matrix
    """
    n, d = X.shape
    X = X - X.mean(axis=0, keepdims=True)

    def _loss(W):
        M = X @ W
        R = X - M
        loss = 0.5 / n * (R ** 2).sum()
        G_loss = -1.0 / n * X.T @ R
        return loss, G_loss

    def _h(W):
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        return (w[: d * d] - w[d * d :]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda_1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda_1, -G_smooth + lambda_1), axis=None)
        return obj, g_obj

    w_est = np.zeros(2 * d * d)
    rho = 1.0
    alpha = 0.0
    h = np.inf

    bnds = [(0, 0) if i == j else (0, None)
            for _ in range(2) for i in range(d) for j in range(d)]

    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(
                _func, w_est, method="L-BFGS-B", jac=True, bounds=bnds,
                options={"maxiter": lbfgs_maxiter},
            )
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h and h < np.inf:
                rho *= 10
            else:
                break
        w_est = w_new
        h = h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0.0
    return W_est


# ═══════════════════════════════════════════════════════════════════════
#  Strategy 2: Adam + scipy expm (fast, good for bootstrapping)
#  Each iteration is fast, needs more iterations
# ═══════════════════════════════════════════════════════════════════════

def notears_adam(
    X: np.ndarray,
    lambda_1: float = 0.01,
    max_iter: int = 20,
    lr: float = 5e-3,
    rho_max: float = 1e8,
    w_threshold: float = 0.01,
) -> np.ndarray:
    """NOTEARS with Adam optimizer + scipy expm for fast acyclicity.

    Args:
        X: Data (n, d), centered
        lambda_1: L1 penalty
        max_iter: Max outer iterations
        lr: Learning rate
        rho_max: Max penalty parameter
        w_threshold: Prune edges

    Returns:
        W: (d, d) weight matrix
    """
    n, d = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    X_t = torch.from_numpy(X).float()

    W = torch.zeros(d, d, requires_grad=True)
    W.data.add_(torch.randn(d, d) * 1e-4)

    optimizer = torch.optim.Adam([W], lr=lr)
    rho = 1.0
    alpha = 0.0
    h = np.inf
    best_h = np.inf
    best_W = None

    for outer in range(max_iter):
        for _ in range(15):
            optimizer.zero_grad()
            X_pred = X_t @ W.T
            recon = 0.5 / n * torch.sum((X_t - X_pred) ** 2)
            l1 = lambda_1 * torch.sum(torch.abs(W))

            W_np = W.detach().numpy()
            try:
                h_val = np.trace(slin.expm(W_np * W_np)) - d
            except Exception:
                h_val = 0.0

            h_t = torch.tensor(h_val, dtype=torch.float32)
            loss = recon + l1 + alpha * h_t + 0.5 * rho * h_t ** 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], 5.0)
            optimizer.step()

        W_np = W.detach().numpy()
        try:
            h_new = np.trace(slin.expm(W_np * W_np)) - d
        except Exception:
            h_new = float("inf")

        if not np.isnan(h_new) and h_new < best_h:
            best_h = h_new
            best_W = W_np.copy()

        if h_new > 0.25 * h and h < np.inf:
            rho = min(rho * 10, rho_max)
        alpha += rho * h_new
        h = h_new

        if h <= 1e-8:
            best_W = W_np.copy()
            break

    if best_W is None:
        return W.detach().numpy()
    best_W[np.abs(best_W) < w_threshold] = 0.0
    return best_W


# ═══════════════════════════════════════════════════════════════════════
#  Bootstrap wrapper (works with any notears_* function)
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_notears(
    X: np.ndarray,
    n_bootstraps: int = 50,
    lambda_1: float = 0.01,
    max_iter: int = 10,
    w_threshold: float = 0.1,
    method: str = "lbfgs",
    seed: int = 42,
) -> tuple:
    """Bootstrapped NOTEARS with uncertainty.

    Args:
        X: Data matrix (n, d)
        n_bootstraps: Number of bootstrap samples
        lambda_1: L1 penalty
        max_iter: Max iterations per run
        w_threshold: Edge pruning threshold per run
        method: 'lbfgs' or 'adam'
        seed: Random seed

    Returns:
        (P, S, W_list): edge probs, edge stds, individual weights
    """
    from sklearn.utils import resample

    notears_fn = notears_lbfgs if method == "lbfgs" else notears_adam

    d = X.shape[1]
    W_list = []
    n_failed = 0

    for i in range(n_bootstraps):
        try:
            X_boot = resample(X, random_state=seed + i)
            X_boot = X_boot - X_boot.mean(axis=0, keepdims=True)
            W_i = notears_fn(
                X_boot,
                lambda_1=lambda_1,
                max_iter=max_iter,
                w_threshold=w_threshold,
            )
            if not np.isnan(W_i).any():
                W_list.append(W_i)
            else:
                n_failed += 1
        except Exception:
            n_failed += 1

    if len(W_list) == 0:
        return np.zeros((d, d)), np.zeros((d, d)), []

    W_stack = np.array(W_list)
    W_abs = np.abs(W_stack)
    P = np.mean(W_abs > 0, axis=0)
    S = np.std(W_abs, axis=0)
    np.fill_diagonal(P, 0.0)
    np.fill_diagonal(S, 0.0)

    return P, S, W_list
