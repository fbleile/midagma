# src/methods/kds/trek_jax.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Literal

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp

from src.utils.notreks import TrekMode, TrekRegularizerNames, CyclePenalty, PSTPenalty, PerronMethod, TCCVersion, Agg


def _pairs_to_jax(I: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
    I_np = np.asarray(I, dtype=np.int64)
    if I_np.size == 0:
        return jnp.asarray([], dtype=jnp.int32), jnp.asarray([], dtype=jnp.int32)
    if I_np.ndim != 2 or I_np.shape[1] != 2:
        raise ValueError("I must be array-like of shape (m,2)")
    return jnp.asarray(I_np[:, 0], dtype=jnp.int32), jnp.asarray(I_np[:, 1], dtype=jnp.int32)


def _matrix_power(A: jnp.ndarray, p: int) -> jnp.ndarray:
    if p < 0:
        raise ValueError("p must be >= 0")
    d = A.shape[0]
    I = jnp.eye(d, dtype=A.dtype)
    def body(k, out):
        return out @ A
    return jax.lax.cond(
        p == 0,
        lambda _: I,
        lambda _: jax.lax.fori_loop(1, p, body, A),
        operand=None,
    )


def _series_I_minus_log_I_minus_W(W: jnp.ndarray, K: int, s: float = 1.0) -> jnp.ndarray:
    # I - log(I - W) = I + sum_{k=1..K} W^k/(k*s^k)
    d = W.shape[0]
    I = jnp.eye(d, dtype=W.dtype)

    def step(carry, k):
        Wk = carry
        term = Wk / (k * (s ** k))
        Wk_next = Wk @ W
        return Wk_next, term

    W1 = W
    _, terms = jax.lax.scan(step, W1, jnp.arange(1, K + 1, dtype=jnp.int32))
    return I + jnp.sum(terms, axis=0)


def pst_mat_jax(
    W: jnp.ndarray,
    seq: PSTPenalty = "exp",
    *,
    K_log: Optional[int] = None,
    eps_inv: float = 1e-8,
    s: float = 1.0,
) -> jnp.ndarray:
    seq = str(seq).lower().strip()
    if seq not in {"exp", "log", "inv", "binom"}:
        raise ValueError("seq must be one of {'exp','log','inv','binom'}")

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square")
    d = W.shape[0]

    # notreks convention: W2 = W ⊙ W
    W2 = jnp.square(W)
    I = jnp.eye(d, dtype=W.dtype)

    if seq == "exp":
        F = jsp.expm(W2)
        return F.T @ F

    if seq == "inv":
        A = I - W2
        if eps_inv > 0:
            A = A + eps_inv * I
        X = jsp.solve(A, I, assume_a="gen")
        return X.T @ X

    if seq == "log":
        if K_log is None:
            K_log = 2 * int(d)
        F = _series_I_minus_log_I_minus_W(W2, K=int(K_log), s=float(s))
        return F.T @ F

    # seq == "binom"
    F = _matrix_power(I + W2, p=int(d))
    return F.T @ F


def pst_jax(
    W: jnp.ndarray,
    I_pairs: Any,
    *,
    seq: PSTPenalty = "exp",
    K_log: Optional[int] = None,
    eps_inv: float = 1e-8,
    s: float = 1.0,
    agg: Agg = "mean",
) -> jnp.ndarray:
    rows, cols = _pairs_to_jax(I_pairs)
    if rows.size == 0:
        return jnp.asarray(0.0, dtype=W.dtype)

    H = pst_mat_jax(W, seq=seq, K_log=K_log, eps_inv=eps_inv, s=s)
    vals = H[rows, cols]

    agg = str(agg).lower().strip()
    if agg == "mean":
        return jnp.mean(vals)
    if agg == "sum":
        return jnp.sum(vals)
    if agg == "max":
        return jnp.max(vals)
    if agg == "lse":
        return jax.nn.logsumexp(vals)
    if agg == "none":
        return vals
    raise ValueError("agg must be one of {'mean','sum','max','lse','none'}")


# ---- Optional: TCC in JAX (matches your block A/B construction) ----

def _indicator_from_pairs_jax(I_pairs: Any, d: int, dtype) -> jnp.ndarray:
    I_np = np.asarray(I_pairs, dtype=np.int64)
    S = jnp.zeros((d, d), dtype=dtype)
    if I_np.size == 0:
        return S
    if I_np.ndim != 2 or I_np.shape[1] != 2:
        raise ValueError("I must be shape (m,2)")
    rows = jnp.asarray(I_np[:, 0], dtype=jnp.int32)
    cols = jnp.asarray(I_np[:, 1], dtype=jnp.int32)
    return S.at[rows, cols].set(1.0)


def _spectral_radius_power(A: jnp.ndarray, n_iter: int = 30, eps: float = 1e-12) -> jnp.ndarray:
    d = A.shape[0]
    v = jnp.ones((d,), dtype=A.dtype)

    def body(_, v):
        Av = A @ v
        v = Av / (jnp.linalg.norm(Av) + eps)
        return v

    v = jax.lax.fori_loop(0, int(n_iter), body, v)
    # Rayleigh quotient
    num = jnp.dot(v, A @ v)
    den = jnp.dot(v, v) + eps
    return num / den


def tcc_spectral_jax(
    W: jnp.ndarray,
    I_pairs: Any,
    *,
    w: float = 1.0,
    version: TCCVersion = "approx_trek_graph",
    n_iter: int = 30,
    eps: float = 1e-12,
) -> jnp.ndarray:
    # matches notreks: W2 = W⊙W, A/B blocks
    d = W.shape[0]
    W2 = jnp.square(W)
    S = _indicator_from_pairs_jax(I_pairs, d=d, dtype=W.dtype)

    I_d = jnp.eye(d, dtype=W.dtype)
    zero = jnp.zeros_like(S)
    bot = jnp.concatenate([I_d, W2.T], axis=1)

    A = jnp.concatenate(
        [jnp.concatenate([W2, float(w) * S], axis=1), bot],
        axis=0,
    )
    B = jnp.concatenate(
        [jnp.concatenate([W2, zero], axis=1), bot],
        axis=0,
    )

    rho_A = _spectral_radius_power(A, n_iter=n_iter, eps=eps)

    if version == "DAG_learning":
        return rho_A

    if version == "exact_trek_graph":
        rho_B = _spectral_radius_power(B, n_iter=n_iter, eps=eps)
        return rho_A - rho_B

    if version == "exact_original_graph":
        rho_W2 = _spectral_radius_power(W2, n_iter=n_iter, eps=eps)
        return rho_A - rho_W2

    if version == "approx_trek_graph":
        # same spirit as your Rayleigh LB baseline but using u≈v from power iter:
        # use v from power-iter implicitly by recomputing and reusing it is possible;
        # simplest: compute rho_B directly (still fast for small d)
        rho_B = _spectral_radius_power(B, n_iter=n_iter, eps=eps)
        return rho_A - rho_B

    raise ValueError(f"Unknown TCC version: {TCCVersion}")


def trek_value_jax(W: jnp.ndarray, tr: Any) -> jnp.ndarray:
    """
    JAX analogue of src.utils.notreks.trek_value(W, tr).
    Expects the same TrekRegularizer layout:
      tr.name, tr.enabled(), tr.mode, tr.cfg["I"], and for PST cfg["seq"], cfg["kwargs"].
    """
    if tr is None or not tr.enabled():
        return jnp.asarray(0.0, dtype=W.dtype)

    name = str(tr.name).lower().strip()
    cfg: Dict[str, Any] = tr.cfg if tr.cfg is not None else {}

    I_pairs = cfg.get("I", None)
    if I_pairs is None or len(I_pairs) == 0:
        return jnp.asarray(0.0, dtype=W.dtype)

    if name == "pst":
        seq = cfg.get("seq", "exp")
        kwargs = cfg.get("kwargs", {}) or {}
        # support your agg if present, default to mean
        agg = kwargs.pop("agg", "mean")
        return pst_jax(W, I_pairs, seq=seq, agg=agg, **kwargs)

    elif name == "tcc":
        # only spectral branch here (logdet can be added similarly if you need it)
        w = float(cfg.get("w", 1.0))
        n_iter = int(cfg.get("n_iter", 30))
        eps = float(cfg.get("eps", 1e-12))
        version = str(cfg.get("version", "approx_trek_graph"))
        return tcc_spectral_jax(W, I_pairs, w=w, version=version, n_iter=n_iter, eps=eps)
    
    elif name == "none":
        return 0.0
    
    else:
        raise ValueError(f"Unknown trek regularizer: {TrekRegularizerNames}")
