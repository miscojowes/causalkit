"""
Fast NOTEARS implementations with L2 prior support.

Two strategies:
1. notears_lbfgs: L-BFGS-B + doubled variables (accurate, supports priors)
2. notears_adam: Adam + scipy expm (fast, supports priors)

Both accept prior_matrix for L2 regularization toward domain knowledge.
"""

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import torch
import warnings


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
    prior_matrix: np.ndarray = None,
    lambda_prior: float = 0.0,
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
        prior_matrix: Prior knowledge matrix [0,1] where 1 = likely edge, 0 = likely absent
        lambda_prior: L2 penalty strength for prior deviation

    Returns:
        W: (d, d) weight matrix
    """
    n, d = X.shape
    X = X - X.mean(axis=0, keepdims=True)

    if prior_matrix is not None:
        prior_matrix = np.asarray(prior_matrix, dtype=float)
        # Symmetrize: prior[i,j] = mean of (i,j) and (j,i) since NOTEARS
        # uses undirected prior strength
        prior_matrix = (prior_matrix + prior_matrix.T) / 2.0

    def _loss(W):
        M = X @ W
        R = X - M
        loss = 0.5 / n * (R ** 2).sum()
        G_loss = -1.0 / n * X.T @ R
        # L2 prior penalty: penalize weights deviating from prior expectations
        if prior_matrix is not None and lambda_prior > 0:
            # For prior=0 (edge unlikely): penalize |W|^2
            # For prior close to 1 (edge likely): penalize (|W| - mu)^2 where mu > 0
            prior_penalty = lambda_prior * np.sum(prior_matrix * W**2)
            loss += prior_penalty
            G_loss += 2 * lambda_prior * prior_matrix * W
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

    best_h = np.inf
    best_W = None

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

        # Track best DAG
        if h < best_h:
            best_h = h
            best_W = _adj(w_est).copy()

        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = best_W if best_W is not None else _adj(w_est)
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
    prior_matrix: np.ndarray = None,
    lambda_prior: float = 0.0,
) -> np.ndarray:
    """NOTEARS with Adam optimizer + scipy expm for fast acyclicity.

    Args:
        X: Data (n, d), centered
        lambda_1: L1 penalty
        max_iter: Max outer iterations
        lr: Learning rate
        rho_max: Max penalty parameter
        w_threshold: Prune edges
        prior_matrix: Prior knowledge matrix [0,1]
        lambda_prior: L2 penalty strength for prior deviation

    Returns:
        W: (d, d) weight matrix
    """
    n, d = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    X_t = torch.from_numpy(X).float()

    if prior_matrix is not None:
        prior_t = torch.from_numpy(np.asarray(prior_matrix, dtype=float)).float()
        prior_t = (prior_t + prior_t.T) / 2.0
    else:
        prior_t = None

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

            # L2 prior penalty
            if prior_t is not None and lambda_prior > 0:
                l2_prior = lambda_prior * torch.sum(prior_t * W**2)
            else:
                l2_prior = 0.0

            W_np = W.detach().numpy()
            try:
                h_val = np.trace(slin.expm(W_np * W_np)) - d
            except Exception:
                h_val = 0.0

            h_t = torch.tensor(h_val, dtype=torch.float32)
            loss = recon + l1 + l2_prior + alpha * h_t + 0.5 * rho * h_t ** 2
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
#  Bootstrap wrapper with optional priors
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_notears(
    X: np.ndarray,
    n_bootstraps: int = 50,
    lambda_1: float = 0.01,
    max_iter: int = 10,
    w_threshold: float = 0.1,
    method: str = "lbfgs",
    seed: int = 42,
    prior_matrix: np.ndarray = None,
    lambda_prior: float = 0.0,
) -> tuple:
    """Bootstrapped NOTEARS with uncertainty and optional priors.

    Args:
        X: Data matrix (n, d)
        n_bootstraps: Number of bootstrap samples
        lambda_1: L1 penalty
        max_iter: Max iterations per run
        w_threshold: Edge pruning threshold per run
        method: 'lbfgs' or 'adam'
        seed: Random seed
        prior_matrix: Prior knowledge matrix [0,1]
        lambda_prior: L2 penalty strength for prior deviation

    Returns:
        (P, S, W_list, W_abs_list): edge probs, edge stds, weights, abs weights
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
                prior_matrix=prior_matrix,
                lambda_prior=lambda_prior,
            )
            if not np.isnan(W_i).any():
                W_list.append(W_i)
            else:
                n_failed += 1
        except Exception:
            n_failed += 1

    if len(W_list) == 0:
        return np.zeros((d, d)), np.zeros((d, d)), [], []

    W_stack = np.array(W_list)
    W_abs = np.abs(W_stack)
    P = np.mean(W_abs > 0, axis=0)
    S = np.std(W_abs, axis=0)
    np.fill_diagonal(P, 0.0)
    np.fill_diagonal(S, 0.0)

    return P, S, W_list, W_abs


# ═══════════════════════════════════════════════════════════════════════
#  Probability calibration: Platt scaling for bootstrap proportions
# ═══════════════════════════════════════════════════════════════════════

def calibrate_bootstrap_proportions(
    P_raw: np.ndarray,
    W_binary_val: np.ndarray,
) -> tuple:
    """Calibrate bootstrap proportions using Platt scaling on validation data.

    Maps raw proportions P_raw to calibrated probabilities using:
        P_cal = 1 / (1 + exp(-(a * logit(P_raw) + b)))

    Args:
        P_raw: Raw bootstrap proportions (d, d)
        W_binary_val: True binary adjacency matrix for validation

    Returns:
        (P_cal, a, b): calibrated probabilities, Platt parameters
    """
    from sklearn.linear_model import LogisticRegression

    d = P_raw.shape[0]
    # Collect all edge proportions and ground truth
    probs = []
    labels = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            probs.append(P_raw[i, j])
            labels.append(1 if W_binary_val[i, j] > 0.5 else 0)

    probs = np.array(probs).reshape(-1, 1)
    labels = np.array(labels)

    # Filter out constant features
    valid = (probs > 0).ravel() & (probs < 1).ravel()
    if valid.sum() < 5:
        # Not enough variation, return raw
        return P_raw, 0.0, 0.0

    # Fit Platt (logistic regression on logit)
    eps = 1e-6
    logit_p = np.log((probs + eps) / (1 - probs + eps))

    try:
        lr = LogisticRegression(C=1.0, solver='lbfgs')
        lr.fit(logit_p[valid].reshape(-1, 1), labels[valid])

        a = lr.coef_[0, 0]
        b = lr.intercept_[0]

        # Apply to all edges
        logit_all = np.log((probs + eps) / (1 - probs + eps))
        P_cal_flat = 1 / (1 + np.exp(-(a * logit_all.ravel() + b)))

        P_cal = P_raw.copy()
        idx = 0
        for i in range(d):
            for j in range(d):
                if i != j:
                    P_cal[i, j] = P_cal_flat[idx]
                    idx += 1

        return P_cal, a, b
    except Exception:
        return P_raw, 0.0, 0.0


def expected_calibration_error(
    P_est: np.ndarray,
    W_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        P_est: Estimated edge probabilities (d, d)
        W_true: True binary adjacency matrix (d, d)
        n_bins: Number of probability bins

    Returns:
        ECE score (lower is better, 0 = perfect calibration)
    """
    d = P_est.shape[0]
    probs = []
    labels = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            probs.append(P_est[i, j])
            labels.append(1 if W_true[i, j] > 0.5 else 0)

    probs = np.array(probs)
    labels = np.array(labels)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            avg_prob = probs[mask].mean()
            avg_label = labels[mask].mean()
            ece += mask.sum() * abs(avg_label - avg_prob)

    return ece / max(len(probs), 1)


def brier_score(
    P_est: np.ndarray,
    W_true: np.ndarray,
) -> float:
    """Compute Brier Score (mean squared error of probability prediction).

    Args:
        P_est: Estimated edge probabilities (d, d)
        W_true: True binary adjacency matrix (d, d)

    Returns:
        Brier score (lower is better)
    """
    d = P_est.shape[0]
    total = 0.0
    count = 0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            total += (P_est[i, j] - W_true[i, j]) ** 2
            count += 1
    return total / count
