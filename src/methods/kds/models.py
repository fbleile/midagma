## -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type

import numpy as np

# stadion / jax
import jax
import jax.numpy as jnp
from jax import random
import jax.scipy.linalg as jsp_linalg
from jax.tree_util import tree_map as _tree_map


from src.utils.notreks_jax import trek_value_jax

from src.methods.kds.stadion.models import LinearSDE, MLPSDE

Pairs = np.ndarray  # shape (m,2), int indices (i,j)

# ---------------------------------------------------------------------
# Extract adjacency proxy from stadion params (JAX, differentiable)
# ---------------------------------------------------------------------


def adjacency_from_param_linear(param: Any, *, zero_diag: bool = True) -> jnp.ndarray:
    """
    Extract adjacency proxy from stadion LinearSDE parameters.

    stadion LinearSDE defines the drift component-wise as:
        f_j(x) = x @ w_j + b_j
    where w_j is the j-th row of the weight matrix and:
        W[j, i] = effect of x_i on f_j

    Your convention is W[parent, child], i.e. W_adj[i, j] corresponds to i -> j,
    so we return |W|^T.

    Works for:
      - param as a dict-like: {"weights": (d,d), ...}
      - param as the per-mechanism pytree used inside regularize_sparsity:
          param["weights"] has shape (d, d) with axis-0 indexing mechanisms j.
    """
    # 1) pull weights out
    if isinstance(param, dict) or hasattr(param, "__getitem__"):
        if "weights" in param:
            W = jnp.asarray(param["weights"])
        elif "w" in param:
            # some variants name it "w"
            W = jnp.asarray(param["w"])
        else:
            raise KeyError("param must contain key 'weights' (or 'w').")
    else:
        raise TypeError(f"Unsupported param type: {type(param)}")

    # 2) sanity check
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"Expected square weight matrix, got shape {W.shape}.")

    # 3) convert to your parent->child convention: W_adj[i,j] = |W[j,i]|
    W_adj = jnp.abs(W).T

    # 4) optionally drop self-loops
    if zero_diag:
        d = W_adj.shape[0]
        W_adj = W_adj * (1.0 - jnp.eye(d, dtype=W_adj.dtype))

    return W_adj


def _get_drift_fn(model: Any, param: Any):
    """
    Returns a callable drift(x) -> R^d.
    Tries common stadion naming patterns.
    """
    # common: model.drift(x, param)
    if hasattr(model, "drift") and callable(getattr(model, "drift")):
        return lambda x: model.drift(x, param)
    # common: model.f(x, param)
    if hasattr(model, "f") and callable(getattr(model, "f")):
        return lambda x: model.f(x, param)
    # sometimes: model._drift(x, param)
    if hasattr(model, "_drift") and callable(getattr(model, "_drift")):
        return lambda x: model._drift(x, param)

    raise AttributeError("Could not find drift function on model (tried drift/f/_drift).")

def adjacency_from_param_mlp_via_jacobian(
    model: Any,
    param: Any,
    *,
    x0: Optional[jnp.ndarray] = None,
    zero_diag: bool = True,
) -> jnp.ndarray:
    # infer dimension
    d = int(getattr(model, "n_vars", 0) or getattr(model, "d", 0) or 0)
    if d == 0:
        # infer from a weights matrix if present
        try:
            d = int(_get_weights_from_param(param).shape[0])
        except Exception as e:
            raise ValueError("Could not infer dimension d for MLPSDE adjacency extraction.") from e

    if x0 is None:
        x0 = jnp.zeros((d,), dtype=jnp.float32)

    drift = _get_drift_fn(model, param)

    J = jax.jacobian(drift)(x0)   # (d,d)
    W_adj = jnp.abs(J).T
    if zero_diag:
        W_adj = W_adj * (1.0 - jnp.eye(d, dtype=W_adj.dtype))
    return W_adj



# ---------------------------------------------------------------------
# stadion model subclasses that "infuse" trek penalty into regularize_sparsity
# ---------------------------------------------------------------------

def _tr_get(tr, key, default=None):
    if tr is None:
        return default
    if isinstance(tr, dict):
        return tr.get(key, default)
    return getattr(tr, key, default)

class LinearSDEWithTrek(LinearSDE):
    def __init__(self, *, trek_reg=None, **kwargs):
        super().__init__(**kwargs)
        self._trek_reg = trek_reg  # MUST NOT contain arrays (store hashable I!)

    def regularize_sparsity(self, param):
        reg = super().regularize_sparsity(param)
        tr = self._trek_reg
        if tr is None:
            return reg

        # "enabled" is config-like; keep it pure python
        enabled = _tr_get(tr, "enabled", True)
        if callable(enabled):
            enabled = bool(enabled())
        if not bool(enabled):
            return reg

        if _tr_get(tr, "mode", "opt") != "opt":
            return reg

        W_adj = adjacency_from_param_linear(param)
        trek_val = trek_value_jax(W_adj, tr)   # should be JAX scalar

        # DON'T do float(...) inside traced code; keep it JAX
        w = jnp.asarray(_tr_get(tr, "weight", 1.0), dtype=getattr(reg, "dtype", jnp.float32))
        return reg + w * trek_val


class MLPSDEWithTrek(MLPSDE):
    def __init__(self, *, trek_reg: Any = None, **kwargs):
        super().__init__(**kwargs)
        self._trek_reg = trek_reg

    def regularize_sparsity(self, param):
        reg = super().regularize_sparsity(param)

        tr = self._trek_reg
        if tr is None:
            return reg
        if hasattr(tr, "enabled") and callable(tr.enabled) and not tr.enabled():
            return reg
        if getattr(tr, "mode", "opt") != "opt":
            return reg

        W_adj = adjacency_from_param_mlp_via_jacobian(self, param)
        trek_val = trek_value_jax(W_adj, tr)
        return reg + float(getattr(tr, "weight", 1.0)) * trek_val
