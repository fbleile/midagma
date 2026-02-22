# src/methods/notears/linear.py

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid

# NEW (match your dagma integration)
from src.utils.notreks import TrekRegularizer, trek_value_grad
import torch
from typing import Optional


def notears_linear(
    X,
    lambda1,
    loss_type,
    max_iter=100,
    h_tol=1e-8,
    rho_max=1e16,
    w_threshold=0.3,
    *,
    trek_reg: Optional[TrekRegularizer] = None,   # NEW
):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian."""
    def _loss(W):
        M = X @ W
        if loss_type == "l2":
            R = X - M
            loss = 0.5 / X.shape[0] * (R**2).sum()
            G_loss = -1.0 / X.shape[0] * X.T @ R
        elif loss_type == "logistic":
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == "poisson":
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError("unknown loss type")
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

        # --- NEW: trek penalty (value + grad) ---
        trek_val = 0.0
        trek_grad = 0.0
        if trek_reg is not None and trek_reg.enabled() and trek_reg.mode == "opt":
            trek_val, trek_grad = trek_value_grad(
                W,
                trek_reg,
                torch_dtype=torch.double,
                device=torch.device("cpu"),
            )
        # augmented Lagrangian (unchanged) + trek term
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        if trek_reg is not None and trek_reg.enabled() and trek_reg.mode == "opt":
            obj = obj + float(trek_reg.weight) * trek_val

        # smooth gradient part
        G_smooth = G_loss + (rho * h + alpha) * G_h
        if trek_reg is not None and trek_reg.enabled() and trek_reg.mode == "opt":
            G_smooth = G_smooth + float(trek_reg.weight) * trek_grad

        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]

    if loss_type == "l2":
        X = X - np.mean(X, axis=0, keepdims=True)

    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


if __name__ == '__main__':
    from notears import utils
    utils.set_random_seed(1)

    n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    np.savetxt('W_true.csv', W_true, delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
    assert utils.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)

