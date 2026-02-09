# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal, Optional, Tuple
import numpy as np
from src.utils.graphs import *
from jax import random

from definitions import DEFAULT_GRAPH_TYPES

GraphType = Literal["ER", "ER_acyclic", "scale_free", "scale_free_acyclic", "sbm", "sbm_acyclic"]

def sample_erdos_renyi(
    key: random.PRNGKey,
    *,
    d: int,
    s0: int,
    acyclic: bool,
) -> np.ndarray:
    """
    Erdős–Rényi with *exact* edge count: G(d, s0) on directed graphs (no self-loops).
    If acyclic=True, sample from strictly lower triangle (DAG by construction), then permute.
    """
    s0 = int(s0)
    s0_max = max_edges(d, acyclic)
    if s0 < 0 or s0 > s0_max:
        raise ValueError(f"Requested s0={s0} edges, feasible range [0, {s0_max}] for d={d}, acyclic={acyclic}")

    B = np.zeros((d, d), dtype=np.int64)
    if s0 == 0:
        return permute_adjacency(key, B)

    if acyclic:
        rows, cols = np.tril_indices(d, k=-1)             # i>j only
    else:
        rows, cols = np.where(~np.eye(d, dtype=bool))     # all off-diagonal

    n_cand = len(rows)
    key, subk = random.split(key)

    # sample s0 distinct indices without replacement
    idx = np.asarray(random.permutation(subk, n_cand)[:s0])

    B[rows[idx], cols[idx]] = 1
    return permute_adjacency(key, B)


def sample_scale_free(
    key: random.PRNGKey,
    *,
    d: int,
    s0: int,
    acyclic: bool,
    power: float = 1.0,
) -> np.ndarray:
    """
    Scale-free-ish directed graph with exact total edge count s0.
    Deterministic via JAX keys.
    """
    try:
        import igraph as ig
    except Exception as e:
        raise ImportError("scale_free requires python-igraph.") from e

    s0 = int(s0)
    s0_max = max_edges(d, acyclic)
    if s0 < 0 or s0 > s0_max:
        raise ValueError(f"s0 must be in [0, {s0_max}]")

    if d <= 1:
        return np.zeros((d, d), dtype=np.int64)

    key, k_flip, k_drop, k_add, k_perm = random.split(key, 5)

    # --------------------------------------------------
    # 1) BA graph (igraph is CPU anyway → deterministic)
    # --------------------------------------------------
    m0 = int(np.clip(round(s0 / max(d - 1, 1)), 1, max(d - 1, 1)))
    g = ig.Graph.Barabasi(n=d, m=m0, directed=True, power=power)

    B = np.array(g.get_adjacency().data, dtype=np.int64).T
    np.fill_diagonal(B, 0)

    # --------------------------------------------------
    # 2) random flips (cycles)
    # --------------------------------------------------
    if not acyclic:
        flip = np.asarray(random.bernoulli(k_flip, 0.5, (d, d)))
        B = (flip * B + (1 - flip) * B.T)
        B[B > 1] = 1

    if acyclic:
        B = np.tril(B, k=-1)

    # --------------------------------------------------
    # 3) adjust to exact s0
    # --------------------------------------------------
    cur = int(B.sum())

    if cur > s0:
        r, c = np.where(B == 1)
        perm = np.asarray(random.permutation(k_drop, len(r)))
        drop = perm[: (cur - s0)]
        B[r[drop], c[drop]] = 0

    elif cur < s0:
        if acyclic:
            rows, cols = np.tril_indices(d, k=-1)
        else:
            rows, cols = np.where(~np.eye(d, dtype=bool))

        missing = np.where(B[rows, cols] == 0)[0]
        need = s0 - cur
        perm = np.asarray(random.permutation(k_add, len(missing)))
        add = missing[perm[:need]]
        B[rows[add], cols[add]] = 1

    return permute_adjacency(k_perm, B)



def sample_sbm(
    key: random.PRNGKey,
    *,
    d: int,
    s0: int,
    n_blocks: int = 5,
    damp: float = 0.1,
    acyclic: bool = False,
) -> np.ndarray:
    """
    SBM with exact edge count s0 using weighted sampling.
    """
    s0 = int(s0)
    s0_max = max_edges(d, acyclic)
    if s0 < 0 or s0 > s0_max:
        raise ValueError(f"s0 must be in [0, {s0_max}]")

    key, k_perm, k_split, k_sample, k_final = random.split(key, 5)

    n_blocks = min(max(1, n_blocks), d)

    # --------------------------------------------------
    # random partition
    # --------------------------------------------------
    perm = np.asarray(random.permutation(k_perm, d))

    if n_blocks > 1:
        split_pts = np.sort(
            np.asarray(random.choice(k_split, d - 1, (n_blocks - 1,), replace=False)) + 1
        )
    else:
        split_pts = []

    blocks = np.split(perm, split_pts)

    block_id = np.empty(d, dtype=np.int64)
    for bi, b in enumerate(blocks):
        block_id[b] = bi

    # --------------------------------------------------
    # candidates
    # --------------------------------------------------
    if acyclic:
        rows, cols = np.tril_indices(d, k=-1)
    else:
        rows, cols = np.where(~np.eye(d, dtype=bool))

    same_block = (block_id[rows] == block_id[cols])
    w = np.where(same_block, 1.0, float(damp))
    w = np.maximum(w, 1e-12)

    # --------------------------------------------------
    # weighted sample without replacement (JAX trick)
    # --------------------------------------------------
    U = random.uniform(k_sample, shape=w.shape)
    keys = U ** (1.0 / w)
    idx = np.argpartition(np.asarray(keys), -s0)[-s0:]

    B = np.zeros((d, d), dtype=np.int64)
    B[rows[idx], cols[idx]] = 1

    return permute_adjacency(k_final, B)



def sample_graph(*, key: random.PRNGKey, d: int, s0: int, graph_type: str) -> np.ndarray:
    gt = graph_type.strip()
    if gt == "ER":
        return sample_erdos_renyi(key, d=d, s0=int(s0), acyclic=False)
    if gt == "ER_acyclic":
        return sample_erdos_renyi(key, d=d, s0=int(s0), acyclic=True)
    if gt == "scale_free":
        edges_per_var = max(1, int(round(s0 / max(d, 1))))
        return sample_scale_free(key, d=d, s0=int(s0), acyclic=False)
    if gt == "scale_free_acyclic":
        edges_per_var = max(1, int(round(s0 / max(d, 1))))
        return sample_scale_free(key, d=d, s0=int(s0), acyclic=True)

    if gt == "sbm":
        edges_per_var = max(1, int(round(s0 / max(d, 1))))
        return sample_sbm(key, d=d, s0=int(s0), acyclic=False)
    if gt == "sbm_acyclic":
        edges_per_var = max(1, int(round(s0 / max(d, 1))))
        return sample_sbm(key, d=d, s0=int(s0), acyclic=True)

    raise ValueError(f"Unknown graph_type: {graph_type}")


if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        description="Smoke test for graph samplers.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    parser.add_argument("--d", type=int, default=20, help="Number of nodes")
    parser.add_argument("--s0", type=int, default=60, help="Density proxy (edges ~ s0 for ER, etc.)")
    parser.add_argument("--n_trials", type=int, default=20, help="How many graphs per type")
    args = parser.parse_args()

    key = random.PRNGKey(int(args.seed))

    graph_types = DEFAULT_GRAPH_TYPES

    print("\n=== graph sampler smoke test ===")
    print(f"seed={args.seed}  d={args.d}  s0={args.s0}  n_trials={args.n_trials}\n")

    # Header
    print(f"{'graph_type':18s} | {'edges(avg)':>10s} {'min':>6s} {'max':>6s} | {'dag_rate':>8s} | checks")
    print("-" * 80)

    any_fail = False

    for gt in graph_types:
        edges = []
        dag_flags = []
        bad_msgs = []

        for t in range(args.n_trials):
            B = sample_graph(key=key, d=args.d, s0=args.s0, graph_type=gt)
            # --- print ONE example adjacency (permuted for clarity) ---
            if t == 0 and False:
                B_ex = permute_for_visual(gt, B)
                
                print(f"\n[{gt}] example adjacency (permuted for visual clarity)")
                print(f"edges={int(B_ex.sum())}, dag={is_dag_adj(B_ex) if gt.endswith('_acyclic') else 'n/a'}")
                print(ascii_adj(B_ex, max_d=min(args.d, 40)))
                print()

            # --- basic invariants ---
            if B.shape != (args.d, args.d):
                bad_msgs.append("shape")
            if not is_binary_adjacency(B):
                bad_msgs.append("not_binary_int")
            if not np.all(np.diag(B) == 0):
                bad_msgs.append("diag_nonzero")
            m = count_edges(B)
            if m != args.s0:
                bad_msgs.append(f"edge_count({m}!={args.s0})")


            edges.append(count_edges(B))

            # --- DAG checks ---
            try:
                is_dag = is_dag_adj(B)
            except ImportError:
                is_dag = None
            dag_flags.append(is_dag)

        # stats
        e_avg = float(np.mean(edges)) if edges else float("nan")
        e_min = int(np.min(edges)) if edges else -1
        e_max = int(np.max(edges)) if edges else -1

        # dag rate (ignore None)
        dag_known = [x for x in dag_flags if x is not None]
        dag_rate = float(np.mean(dag_known)) if dag_known else float("nan")

        # hard expectations:
        # - *_acyclic must be DAG for all trials (if dag check available)
        # - non-acyclic: we just want *some* cyclic graphs over trials (weak check)
        if dag_known:
            if gt.endswith("_acyclic"):
                if not all(dag_known):
                    bad_msgs.append("acyclic_but_not_dag")
            else:
                # weak check: avoid a bug where we accidentally always sample DAGs
                if all(dag_known):
                    bad_msgs.append("always_dag?(weak)")

        checks = "OK" if len(set(bad_msgs)) == 0 else "FAIL: " + ",".join(sorted(set(bad_msgs)))
        if checks != "OK":
            any_fail = True

        dag_rate_str = f"{dag_rate:0.2f}" if dag_known else "n/a"
        print(f"{gt:18s} | {e_avg:10.1f} {e_min:6d} {e_max:6d} | {dag_rate_str:>8s} | {checks}")

    print("\nNotes:")
    print(textwrap.dedent("""\
      - 'always_dag?(weak)' is a weak warning: it can happen by chance for small d/s0.
        Increase --n_trials or --s0 or use d larger if you want stronger evidence of cycles.
      - DAG checks require python-igraph or networkx. If neither is installed, dag_rate will be 'n/a'.
    """))

    if any_fail:
        raise SystemExit("Some graph sampler checks failed.")
    print("All checks passed.")
