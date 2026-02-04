# -*- coding: utf-8 -*-
import numpy as np
from jax import random
import jax.numpy as jnp

def is_binary_adjacency(B: np.ndarray) -> bool:
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        return False
    if not np.issubdtype(B.dtype, np.integer):
        return False
    vals = np.unique(B)
    return np.all((vals == 0) | (vals == 1))


def count_edges(B: np.ndarray) -> int:
    return int(B.sum())


def is_dag_adj(B: np.ndarray) -> bool:
    """
    Return True iff adjacency matrix B represents a DAG.
    Prefers igraph if available, otherwise uses networkx.
    """
    try:
        import igraph as ig
        g = ig.Graph.Adjacency(B.tolist(), mode=ig.ADJ_DIRECTED)
        return bool(g.is_dag())
    except Exception:
        try:
            import networkx as nx
            G = nx.from_numpy_array(B, create_using=nx.DiGraph)
            return nx.is_directed_acyclic_graph(G)
        except Exception as e:
            raise ImportError(
                "Need `python-igraph` or `networkx` installed to check DAG-ness."
            ) from e


def permute_adjacency(key: random.PRNGKey, B: np.ndarray) -> np.ndarray:
    d = B.shape[0]
    perm = np.asarray(random.permutation(key, d))
    return B[np.ix_(perm, perm)]


def max_edges(d: int, acyclic: bool) -> int:
    return d * (d - 1) // 2 if acyclic else d * (d - 1)


def weighted_sample_without_replacement(
    key: random.PRNGKey,
    weights: np.ndarray,
    m: int,
) -> np.ndarray:
    """
    Efraimidis–Spirakis (A-ExpJ style):
      sample keys = U^(1/w) and take top-m.
    weights must be strictly positive.
    Returns: np.ndarray of indices (shape (m,))
    """
    w = jnp.asarray(weights, dtype=jnp.float32)
    if (w <= 0).any():
        raise ValueError("weights must be strictly positive for weighted sampling without replacement")

    # U in (0,1)
    U = random.uniform(key, shape=w.shape, minval=1e-12, maxval=1.0 - 1e-12)
    keys = U ** (1.0 / w)

    # take top-m keys (unsorted indices are fine)
    idx = jnp.argpartition(keys, -m)[-m:]
    return np.asarray(idx)


def ascii_adj(B: np.ndarray, max_d: int = 40) -> str:
    """
    Render adjacency as ASCII:
      '.' = 0, '#' = 1
    Truncates if d > max_d.
    """
    d = B.shape[0]
    if d > max_d:
        B = B[:max_d, :max_d]
        d = max_d
    lines = []
    for i in range(d):
        lines.append("".join("#" if B[i, j] else "." for j in range(d)))
    return "\n".join(lines)


def perm_by_topological(B: np.ndarray) -> np.ndarray:
    """
    Permute nodes by a topological order (DAG only).
    Falls back to identity if topo order not available.
    """
    try:
        import igraph as ig
        g = ig.Graph.Adjacency(B.tolist(), mode=ig.ADJ_DIRECTED)
        order = g.topological_sorting()
        order = np.asarray(order, dtype=int)
        return B[np.ix_(order, order)]
    except Exception:
        try:
            import networkx as nx
            G = nx.from_numpy_array(B, create_using=nx.DiGraph)
            order = list(nx.topological_sort(G))
            order = np.asarray(order, dtype=int)
            return B[np.ix_(order, order)]
        except Exception:
            return B


def perm_by_degree(B: np.ndarray) -> np.ndarray:
    deg = (B.sum(axis=0) + B.sum(axis=1)).astype(np.int64)  # in+out
    order = np.argsort(-deg)  # descending
    return B[np.ix_(order, order)]


def perm_by_sbm_blocks(B: np.ndarray) -> np.ndarray:
    """
    Try to find communities/blocks (SBM) and permute accordingly.
    Uses igraph community detection if available.
    Falls back to degree ordering.
    """
    try:
        import igraph as ig
        g = ig.Graph.Adjacency(B.tolist(), mode=ig.ADJ_DIRECTED)
        ug = g.as_undirected(combine_edges="max")
        # fast-ish community methods; infomap tends to work well
        try:
            cl = ug.community_infomap()
        except Exception:
            cl = ug.community_multilevel()
        membership = np.asarray(cl.membership, dtype=int)
        order = np.argsort(membership, kind="stable")
        return B[np.ix_(order, order)]
    except Exception:
        return perm_by_degree(B)


def permute_for_visual(gt: str, B: np.ndarray) -> np.ndarray:
    gt = gt.strip()
    if gt.endswith("_acyclic"):
        return perm_by_topological(B)
    if gt.startswith("sbm"):
        return perm_by_sbm_blocks(B)
    if gt.startswith("scale_free"):
        return perm_by_degree(B)
    # ER / sparse: degree ordering makes patterns easier to see
    return perm_by_degree(B)

