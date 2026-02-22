# src/methods/midagma/linear.py

from __future__ import annotations

import logging
import time
import typing
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from scipy.special import expit as sigmoid
from tqdm.auto import tqdm

from utils.notreks import TrekRegularizer, trek_value_grad  # same as in your current file
from utils.logger import LogConfig, StructuredLogger, build_default_logger


__all__ = ["MiDagmaLinear"]


def _indicator_from_pairs_np(
    I: typing.Union[np.ndarray, typing.Sequence[typing.Tuple[int, int]]],
    d: int,
    dtype: type,
) -> np.ndarray:
    """
    Build S in {0,1}^{dxd} from pairs I (m,2).
    Interprets I as directed pairs (i,j). If you want undirected, symmetrize outside.
    """
    S = np.zeros((d, d), dtype=dtype)
    if I is None:
        return S
    I_np = np.asarray(I, dtype=np.int64)
    if I_np.size == 0:
        return S
    if I_np.ndim != 2 or I_np.shape[1] != 2:
        raise ValueError("I must be array-like of shape (m,2).")
    rows = I_np[:, 0]
    cols = I_np[:, 1]
    S[rows, cols] = 1.0
    return S


class MiDagmaLinear:
    """
    Same as your DagmaLinear, but replaces the DAGMA logdet penalty h(W)
    with a logdet penalty on A(W) of size (2d,2d):

        A = [[W2, w*S],
             [I , W2^T]]

    and h(W) = -logdet(s I_{2d} - A) + 2d log(s).

    Everything else (score, l1, optimizer, trek_reg) stays the same.
    """

    def __init__(
        self,
        loss_type: str,
        *,
        I_pairs: Optional[typing.Union[np.ndarray, typing.Sequence[typing.Tuple[int, int]]]] = None,
        w: float = 1.0,
        verbose: bool = False,
        dtype: type = np.float64,
        trek_reg: Optional[TrekRegularizer] = None,
        logger=None,
        log_cfg=None,
    ) -> None:
        losses = ["l2", "logistic"]
        assert loss_type in losses, f"loss_type should be one of {losses}"
        self.loss_type = loss_type
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None

        # midagma-specific
        self.I_pairs = I_pairs
        self.w = float(w)

        # optional trek regularizer (unchanged)
        self.trek_reg = trek_reg

        # torch settings for trek_value_grad (unchanged)
        import torch
        self._torch_dtype = torch.double
        self._device = torch.device("cpu")

        # logging
        self._logger = logger or build_default_logger(level=logging.INFO if verbose else logging.WARNING)
        self._log_cfg = log_cfg or LogConfig(enabled=verbose)
        self._slog = StructuredLogger(self._logger, self._log_cfg)

        # will be initialized in fit()
        self.S = None  # (d,d) indicator


    # -------------------------
    # score stays identical
    # -------------------------
    def _score(self, W: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        if self.loss_type == "l2":
            dif = self.Id - W
            rhs = self.cov @ dif
            loss = 0.5 * np.trace(dif.T @ rhs)
            G_loss = -rhs
        else:  # logistic
            R = self.X @ W
            loss = 1.0 / self.n * (np.logaddexp(0, R) - self.X * R).sum()
            G_loss = (1.0 / self.n * self.X.T) @ sigmoid(R) - self.cov
        return loss, G_loss


    # -------------------------
    # midagma logdet on A(W)
    # -------------------------
    def _build_A(self, W: np.ndarray) -> np.ndarray:
        d = W.shape[0]
        W2 = W * W

        if self.S is None:
            raise RuntimeError("S not initialized; call fit() first.")
        S = self.S

        I_d = self.Id
        top = np.concatenate([W2, self.w * S], axis=1)          # (d,2d)
        bot = np.concatenate([I_d, W2.T], axis=1)              # (d,2d)
        A = np.concatenate([top, bot], axis=0)                 # (2d,2d)
        return A


    def _h(self, W: np.ndarray, s: float = 1.0) -> typing.Tuple[float, np.ndarray]:
        """
        h(W) = -logdet( s I_{2d} - A(W) ) + 2d log s
        grad: G_h = d h / dW
        """
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be square")
        d = W.shape[0]
        A = self._build_A(W)

        M = s * np.eye(2 * d, dtype=self.dtype) - A

        # value
        sign, logabsdet = la.slogdet(M)
        if sign <= 0:
            # outside domain -> let caller handle; return inf-ish and NaN grad
            return np.inf, np.full_like(W, np.nan)

        h = -logabsdet + (2 * d) * np.log(s)

        # gradient wrt A: G_A = M^{-T}
        Minv = sla.inv(M)
        G_A = Minv.T

        # map gradient to W2 (dxd):
        # W2 appears in top-left, and in bottom-right as transpose
        G_tl = G_A[:d, :d]
        G_br = G_A[d:, d:]
        G_W2 = G_tl + G_br.T

        # chain W2 = W*W
        G_h = 2.0 * W * G_W2
        # G_h = W * G_tl.T
        return h, G_h


    def _func(self, W: np.ndarray, mu: float, s: float = 1.0):
        score, _ = self._score(W)
        h, _ = self._h(W, s)

        trek_val, _ = trek_value_grad(
            W,
            self.trek_reg,
            torch_dtype=self._torch_dtype,
            device=self._device,
        )

        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h

        tr = self.trek_reg
        if tr is not None and tr.enabled() and tr.mode == "opt":
            obj = obj + tr.weight * trek_val

        return obj, score, h, trek_val


    def _adam_update(self, grad: np.ndarray, iter: int, beta_1: float, beta_2: float) -> np.ndarray:
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        return m_hat / (np.sqrt(v_hat) + 1e-8)


    def minimize(
        self,
        W: np.ndarray,
        mu: float,
        max_iter: int,
        s: float,
        lr: float,
        tol: float = 1e-10,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
        pbar: typing.Optional[tqdm] = None,
    ) -> typing.Tuple[np.ndarray, bool]:
        t0 = time.time()
        stage = getattr(self, "_stage", 0)

        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        self.vprint(f"\n\nMinimize (MiDAGMA) -- mu:{mu} -- lr:{lr} -- s:{s} -- l1:{self.lambda1} -- iters:{max_iter}")

        mask_inc = np.zeros((self.d, self.d), dtype=self.dtype)
        if self.inc_c is not None:
            mask_inc[self.inc_r, self.inc_c] = -2 * mu * self.lambda1

        mask_exc = np.ones((self.d, self.d), dtype=self.dtype)
        if self.exc_c is not None:
            mask_exc[self.exc_r, self.exc_c] = 0.0

        for iter in range(1, max_iter + 1):
            # --- domain check via M^{-1} >= 0 (same spirit as DAGMA), but on 2d system ---
            A = self._build_A(W)
            M2 = s * np.eye(2 * self.d, dtype=self.dtype) - A
            try:
                Minv2 = sla.inv(M2) + 1e-16
            except la.LinAlgError:
                return W, False

            while np.any(Minv2 < 0):
                if iter == 1 or s <= 0.9:
                    self.vprint(f"W went out of domain for s={s} at iter={iter}")
                    return W, False
                W += lr * grad
                lr *= 0.5
                if lr <= 1e-16:
                    return W, True
                W -= lr * grad
                A = self._build_A(W)
                M2 = s * np.eye(2 * self.d, dtype=self.dtype) - A
                Minv2 = sla.inv(M2) + 1e-16
                self.vprint(f"Learning rate decreased to lr={lr}")

            # score grad (unchanged)
            if self.loss_type == "l2":
                G_score = -mu * self.cov @ (self.Id - W)
            else:
                G_score = mu / self.n * self.X.T @ sigmoid(self.X @ W) - mu * self.cov

            # h grad (new)
            _, G_h = self._h(W, s)

            Gobj = (
                G_score
                + mu * self.lambda1 * np.sign(W)
                + G_h
                + mask_inc * np.sign(W)
            )

            G_trek_norm = None
            if self.trek_reg is not None and self.trek_reg.enabled():
                trek_val, trek_grad = trek_value_grad(
                    W,
                    self.trek_reg,
                    torch_dtype=self._torch_dtype,
                    device=self._device,
                )
                if self.trek_reg.mode == "opt":
                    Gobj = Gobj + self.trek_reg.weight * trek_grad
                    trek_weight = float(getattr(self.trek_reg, "weight", 0.0))
                    G_trek_norm = float(np.linalg.norm(trek_weight * trek_grad)) if trek_grad is not None else 0.0

            # diagnostics
            Gobj_norm = float(np.linalg.norm(Gobj))
            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            grad_norm = float(np.linalg.norm(grad))

            W -= lr * grad
            W *= mask_exc

            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, score, h, trek_val = self._func(W, mu, s)

                if self._log_cfg.enabled:
                    trek_name = self.trek_reg.name if self.trek_reg is not None else "none"
                    trek_mode = self.trek_reg.mode if self.trek_reg is not None else "off"
                    trek_weight = float(self.trek_reg.weight) if self.trek_reg is not None else 0.0
                    trek_cfg = {}
                    if self.trek_reg is not None:
                        trek_cfg = {k: v for k, v in self.trek_reg.cfg.items() if k != "I"}

                    self._slog.emit("minimize.checkpoint", {
                        "iter": int(iter),
                        "stage": int(stage),
                        "elapsed_sec": float(time.time() - t0),
                        "obj_total": float(obj_new),
                        "score_datafit": float(score),

                        "reg_dag_name": "midagma_logdet_on_A",
                        "reg_dag_value": float(h),
                        "reg_dag_cfg": {"s": float(s), "w": float(self.w), "pairs": "I_pairs"},

                        "reg_trek_name": trek_name,
                        "reg_trek_value": float(trek_val),
                        "reg_trek_cfg": trek_cfg,
                        "trek_mode": trek_mode,
                        "trek_weight": trek_weight,

                        "mu": float(mu),
                        "lr": float(lr),

                        "w_norm": float(np.linalg.norm(W)),
                        "w_abs_sum": float(np.abs(W).sum()),
                        "max_abs_w": float(np.abs(W).max()),

                        "grad_raw_norm": float(Gobj_norm),
                        "grad_step_norm": float(grad_norm),
                        "step_norm": float(lr * grad_norm),
                        "grad_trek_norm": G_trek_norm,
                    })

                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    if pbar is not None:
                        pbar.update(max_iter - iter + 1)
                    break
                obj_prev = obj_new

            if pbar is not None:
                pbar.update(1)

        return W, True


    def fit(
        self,
        X: np.ndarray,
        lambda1: float = 0.03,
        w_threshold: float = 0.3,
        T: int = 5,
        mu_init: float = 1.0,
        mu_factor: float = 0.2,
        s: typing.Union[typing.List[float], float] = [1.0, 0.9, 0.8, 0.7, 0.6],
        warm_iter: int = 3e4,
        max_iter: int = 6e4,
        lr: float = 0.0003,
        checkpoint: int = 1000,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
        exclude_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None,
        include_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None,
    ) -> np.ndarray:
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)

        if self.loss_type == "l2":
            self.X = self.X - X.mean(axis=0, keepdims=True)

        # build S once (depends on d)
        self.S = _indicator_from_pairs_np(self.I_pairs, d=self.d, dtype=self.dtype)

        self.exc_r, self.exc_c = None, None
        self.inc_r, self.inc_c = None, None

        if exclude_edges is not None:
            if type(exclude_edges) is tuple and type(exclude_edges[0]) is tuple and np.all(np.array([len(e) for e in exclude_edges]) == 2):
                self.exc_r, self.exc_c = zip(*exclude_edges)
            else:
                raise ValueError("blacklist should be a tuple of edges, e.g., ((1,2), (2,3))")

        if include_edges is not None:
            if type(include_edges) is tuple and type(include_edges[0]) is tuple and np.all(np.array([len(e) for e in include_edges]) == 2):
                self.inc_r, self.inc_c = zip(*include_edges)
            else:
                raise ValueError("whitelist should be a tuple of edges, e.g., ((1,2), (2,3))")

        self.cov = X.T @ X / float(self.n)
        self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)

        mu = mu_init
        if isinstance(s, list):
            if len(s) < T:
                self.vprint(f"Length of s is {len(s)}, using last value for t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif isinstance(s, (int, float)):
            s = T * [float(s)]
        else:
            raise ValueError("s should be a list, int, or float.")

        with tqdm(total=(T - 1) * int(warm_iter) + int(max_iter)) as pbar:
            for i in range(int(T)):
                self.vprint(f"\nIteration -- {i+1}:")
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, success = self.minimize(
                        self.W_est.copy(),
                        mu,
                        inner_iters,
                        s[i],
                        lr=lr_adam,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        pbar=pbar,
                    )
                    if success is False:
                        self.vprint("Retrying with larger s")
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                mu *= mu_factor

        self.h_final, _ = self._h(self.W_est, s=1.0)
        self.score_final, _ = self._score(self.W_est)
        self.W_est[np.abs(self.W_est) < w_threshold] = 0.0

        self._slog.close()
        return self.W_est
