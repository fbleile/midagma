# -*- coding: utf-8 -*-

from __future__ import annotations

import typing
import numpy as np
from jax import random

from src.utils.graphs import is_dag_adj  # your helper


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def is_dag(B: np.ndarray) -> bool:
    try:
        import igraph as ig
    except Exception as e:
        raise ImportError("DAG checks require `python-igraph`. Install it.") from e
    G = ig.Graph.Adjacency(B.tolist())
    return G.is_dag()

def _noise_scale_vec(d: int, noise_scale) -> np.ndarray:
    if noise_scale is None:
        return np.ones(d, dtype=float)
    if np.isscalar(noise_scale):
        return float(noise_scale) * np.ones(d, dtype=float)
    noise_scale = np.asarray(noise_scale, dtype=float)
    if noise_scale.shape != (d,):
        raise ValueError("noise_scale must be scalar or length d")
    return noise_scale


def simulate_parameter(
    key,
    B: np.ndarray,
    w_ranges: typing.List[typing.Tuple[float, float]] = ((-2.0, -0.5), (0.5, 2.0)),
) -> np.ndarray:
    """
    Sample weights W consistent with adjacency B.

    For every edge (i,j):
      1) choose interval index
      2) sample uniform weight in that interval

    Fully deterministic w.r.t. key.
    """
    W = np.zeros_like(B, dtype=float)

    rows, cols = np.where(B == 1)
    m = len(rows)

    if m == 0:
        return W

    # split once, vectorized sampling (cleaner + faster)
    key_int, key_u = random.split(key)

    # choose which interval for each edge
    idx = np.array(
        random.randint(key_int, shape=(m,), minval=0, maxval=len(w_ranges))
    )

    # sample uniforms in [0,1]
    u = np.array(random.uniform(key_u, shape=(m,)))

    lows = np.array([w_ranges[k][0] for k in idx])
    highs = np.array([w_ranges[k][1] for k in idx])

    W[rows, cols] = lows + (highs - lows) * u
    return W



def simulate_linear_sem_acyclic(
    key,
    W: np.ndarray,
    *,
    n: int,
    sem_type: str,
    noise_scale: typing.Optional[typing.Union[float, typing.List[float]]] = None,
) -> np.ndarray:
    import igraph as ig
    from jax import random
    import numpy as np

    d = W.shape[0]
    scale_vec = _noise_scale_vec(d, noise_scale)

    # must be DAG
    B = (W != 0).astype(np.int64)
    if not is_dag_adj(B):
        raise ValueError("W must be a DAG for acyclic simulation")

    G = ig.Graph.Weighted_Adjacency(W.tolist())
    order = list(G.topological_sorting())

    X = np.zeros((n, d), dtype=float)

    def _simulate_single_equation(*, key, X_pa: np.ndarray, w_pa: np.ndarray, scale: float) -> np.ndarray:
        # X_pa: (n, k), w_pa: (k,)
        lin = X_pa @ w_pa if w_pa.size > 0 else np.zeros(n, dtype=float)

        if sem_type == "gauss":
            z = np.asarray(random.normal(key, shape=(n,))) * scale
            return lin + z

        if sem_type == "exp":
            # sample exponential with mean=scale: -scale * log(U)
            u = np.asarray(random.uniform(key, shape=(n,), minval=1e-12, maxval=1.0))
            z = -scale * np.log(u)
            return lin + z

        if sem_type == "gumbel":
            # gumbel(0, scale): -scale * log(-log(U))
            u = np.asarray(random.uniform(key, shape=(n,), minval=1e-12, maxval=1.0 - 1e-12))
            z = -scale * np.log(-np.log(u))
            return lin + z

        if sem_type == "uniform":
            # uniform(-scale, scale)
            u = np.asarray(random.uniform(key, shape=(n,)))
            z = (2.0 * u - 1.0) * scale
            return lin + z

        if sem_type == "logistic":
            # Bernoulli(sigmoid(lin))
            p = 1.0 / (1.0 + np.exp(-lin))
            u = np.asarray(random.uniform(key, shape=(n,)))
            return (u < p).astype(float)

        if sem_type == "poisson":
            # Poisson(exp(lin)) — JAX has random.poisson but wants rate
            rate = np.exp(lin)
            z = np.asarray(random.poisson(key, lam=rate, shape=rate.shape))
            return z.astype(float)

        raise ValueError("unknown sem type")

    key_loop = key
    for j in order:
        key_loop, k_j = random.split(key_loop)

        parents = G.neighbors(j, mode=ig.IN)
        parents = list(parents)

        X_pa = X[:, parents] if len(parents) > 0 else np.zeros((n, 0), dtype=float)
        w_pa = W[parents, j] if len(parents) > 0 else np.zeros((0,), dtype=float)

        X[:, j] = _simulate_single_equation(
            key=k_j,
            X_pa=X_pa,
            w_pa=w_pa,
            scale=float(scale_vec[j]),
        )

    return X


def simulate_linear_sem_cyclic_gauss(
    key: random.PRNGKey,
    W: np.ndarray,
    *,
    n: int,
    noise_scale: typing.Optional[typing.Union[float, typing.List[float]]] = None,
    jitter: float = 1e-8,
) -> np.ndarray:
    """
    Cyclic linear Gaussian SEM:
        X = (I - W^T)^{-1} Z
    Requires (I - W^T) invertible and stable-ish.
    """

    d = W.shape[0]
    scale_vec = _noise_scale_vec(d, noise_scale)

    I = np.eye(d, dtype=float)
    A = I - W.T

    # sample Z ~ N(0, diag(scale^2))
    key, subk = random.split(key)

    Z = np.asarray(random.normal(key, shape=(n, d))) * scale_vec[None, :]

    # solve A X^T = Z^T  => X = Z @ A^{-T}
    # use solve for stability; add tiny jitter if singular-ish
    try:
        X = np.linalg.solve(A + jitter * I, Z.T).T
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "Cyclic Gaussian SEM failed: (I - W^T) singular/ill-conditioned. "
            "Try smaller weights or increase jitter."
        ) from e

    return X


def simulate_linear_sem(
    key: random.PRNGKey,
    W: np.ndarray,
    *,
    n: int,
    sem_type: str,
    noise_scale: typing.Optional[typing.Union[float, typing.List[float]]] = None,
    allow_cyclic: bool = True,
) -> np.ndarray:
    """
    Unified linear SEM:
      - if DAG -> topological simulation for many noise types
      - if cyclic and allow_cyclic:
            only supports Gaussian via stable solve
    """
    B = (W != 0).astype(np.int64)
    dag = is_dag_adj(B)

    if dag:
        return simulate_linear_sem_acyclic(key, W, n=n, sem_type=sem_type, noise_scale=noise_scale)

    if not allow_cyclic:
        raise ValueError("Graph is cyclic but allow_cyclic=False")

    if sem_type != "gauss":
        raise ValueError("Cyclic linear SEM implemented only for sem_type='gauss' (stable solve).")

    return simulate_linear_sem_cyclic_gauss(key, W, n=n, noise_scale=noise_scale)

def simulate_nonlinear_sem_acyclic(
    B: np.ndarray,
    n: int,
    sem_type: str,
    noise_scale: typing.Optional[typing.Union[float, typing.List[float]]] = None,
) -> np.ndarray:
    import igraph as ig

    def _simulate_single_equation(X, scale):
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == "mlp":
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=(pa_size, hidden))
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=(hidden,))
            W2[np.random.rand(hidden) < 0.5] *= -1
            return sigmoid(X @ W1) @ W2 + z
        if sem_type == "mim":
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            return np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        raise ValueError("unknown sem type")

    d = B.shape[0]
    scale_vec = np.ones(d) if noise_scale is None else (float(noise_scale) * np.ones(d) if np.isscalar(noise_scale) else np.asarray(noise_scale))
    if len(scale_vec) != d:
        raise ValueError("noise_scale must be scalar or length d")

    if not is_dag(B):
        raise ValueError("B must be DAG for nonlinear acyclic simulation")

    X = np.zeros((n, d))
    G = ig.Graph.Adjacency(B.tolist())
    order = G.topological_sorting()
    for j in order:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X


def simulate_nonlinear_sem_cyclic(
    B: np.ndarray,
    n: int,
    sem_type: str,
    noise_scale: typing.Optional[typing.Union[float, typing.List[float]]] = None,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    relax: float = 0.5,
) -> np.ndarray:
    """
    Cyclic nonlinear SEM via fixed-point iteration:
      X <- (1-relax)*X + relax*f(X) + noise
    This needs contraction-ish behavior; otherwise it may not converge.
    """
    d = B.shape[0]
    scale_vec = np.ones(d) if noise_scale is None else (float(noise_scale) * np.ones(d) if np.isscalar(noise_scale) else np.asarray(noise_scale))
    if len(scale_vec) != d:
        raise ValueError("noise_scale must be scalar or length d")

    # pre-sample parent function parameters per node for reproducibility
    rng = np.random.default_rng(0)
    parents = [np.where(B[:, j] != 0)[0] for j in range(d)]

    # create random node-wise functions
    node_params = []
    for j in range(d):
        pa = parents[j]
        p = len(pa)
        if p == 0:
            node_params.append(None)
            continue
        if sem_type == "mlp":
            hidden = 50
            W1 = rng.normal(scale=0.2, size=(p, hidden))
            W2 = rng.normal(scale=0.2, size=(hidden,))
            node_params.append(("mlp", W1, W2))
        elif sem_type == "mim":
            w1 = rng.normal(scale=0.2, size=(p,))
            w2 = rng.normal(scale=0.2, size=(p,))
            w3 = rng.normal(scale=0.2, size=(p,))
            node_params.append(("mim", w1, w2, w3))
        else:
            raise ValueError("unknown sem type for cyclic nonlinear")

    def f_j(Xpa, params):
        if params is None:
            return np.zeros((n,))
        kind = params[0]
        if kind == "mlp":
            _, W1, W2 = params
            return sigmoid(Xpa @ W1) @ W2
        if kind == "mim":
            _, w1, w2, w3 = params
            return np.tanh(Xpa @ w1) + np.cos(Xpa @ w2) + np.sin(Xpa @ w3)
        raise RuntimeError("bad params")

    X = np.zeros((n, d))
    Z = np.random.normal(scale=scale_vec, size=(n, d))

    for it in range(max_iter):
        X_new = X.copy()
        for j in range(d):
            pa = parents[j]
            base = f_j(X[:, pa], node_params[j])
            X_new[:, j] = (1.0 - relax) * X[:, j] + relax * (base + Z[:, j])

        diff = np.max(np.abs(X_new - X))
        X = X_new
        if diff < tol:
            return X

    raise RuntimeError(f"cyclic nonlinear SEM did not converge (diff={diff:g}). Try smaller relax or fewer edges.")


def simulate_nonlinear_sem(
    B: np.ndarray,
    n: int,
    sem_type: str,
    noise_scale: typing.Optional[typing.Union[float, typing.List[float]]] = None,
    *,
    allow_cyclic: bool = True,
) -> np.ndarray:
    if is_dag(B):
        return simulate_nonlinear_sem_acyclic(B, n, sem_type, noise_scale)
    if not allow_cyclic:
        raise ValueError("Graph is cyclic but allow_cyclic=False")
    return simulate_nonlinear_sem_cyclic(B, n, sem_type, noise_scale)
