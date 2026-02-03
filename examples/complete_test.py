# -*- coding: utf-8 -*-

#!/usr/bin/env python
# coding: utf-8
"""
complete_test.py

Pipeline test harness for:
  - data generation (DAGMA utils)
  - I construction (oracle no-trek or MI tests)
  - base algorithms (pluggable; starts with DagmaLinear)
  - trek regularizers (none, PST, TCC-spectral, TCC-logdet)
  - summary table output (SHD + NNZ + a few accuracy stats)

No visualization. Minimal logging.

Example:
  python full_small_test.py

Notes:
  - Make sure your notreks package and dagma are importable.
  - This file intentionally avoids deep logging; it prints just a final table.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from dagma import utils
from dagma.linear import DagmaLinear

from notreks.mi_tests import get_I_from_full_pairwise_tests, summarize_I
from notreks.notreks import (
    PSTRegularizer,
    TCCRegularizer,
    pst,
    get_no_trek_pairs,
    trek_cycle_coupling_value_gradW,  # should support cycle_penalty="spectral"|"logdet"
)

from logger import LogConfig, build_default_logger


# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass(frozen=True)
class DataSpec:
    seed: int = 4
    n: int = 500
    d: int = 10
    s0: int = 40
    graph_type: str = "ER"
    sem_type: str = "gauss"


@dataclass(frozen=True)
class ISpec:
    source: str = "oracle"  # "oracle" | "pairwise"
    # pairwise-test settings
    alpha: float = 0.001
    test: str = "spearman"
    num_perm: int = 500
    seed: int = 0
    bonferroni: bool = True
    undirected: bool = False
    # oracle settings
    pst_seq_for_oracle: str = "exp"  # used by get_no_trek_pairs(...) (it calls pst_mat internally)
    cap: int = None


@dataclass(frozen=True)
class AlgoSpec:
    name: str = "dagma_linear"
    loss_type: str = "l2"
    lambda1: float = 0.02
    max_iter: int = int(6e4)
    mu_factor: float = 0.1
    s: float = 2.0


@dataclass(frozen=True)
class TrekRegSpec:
    name: str  # "none" | "pst" | "tcc"
    weight: float = 0.1

    # PST
    seq: str = "log"
    K_log: int = 40
    eps_inv: float = 1e-8
    s: float = 5.0
    agg: str = "mean"  # only used for reporting via pst(...)

    # TCC
    cycle_penalty: str = "spectral"  # "spectral" | "logdet"
    w: float = 100.0
    n_iter: int = 10
    eps: float = 1e-12
    version: str = "C"   # if you expose it; otherwise ignored by your wrapper
    method: str = "eig_numpy"
    # logdet param (used by cycle_penalty="logdet")
    s_logdet: float = 2.0

    # whether trek reg affects optimization:
    mode: str = "opt"  # "opt" | "log"


# -----------------------------
# Helpers: data, I, algorithms
# -----------------------------
def make_logger() -> Tuple[logging.Logger, LogConfig]:
    logger = build_default_logger(level=logging.INFO)
    log_cfg = LogConfig(
        enabled=True,
        print_to_console=False,
        store_csv=False,
        store_jsonl=False,
        keep_in_memory=True,
    )
    return logger, log_cfg


def generate_data(spec: DataSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: X (n,d), B_true (d,d) binary DAG, W_true (d,d) weighted adjacency.
    """
    utils.set_random_seed(spec.seed)
    B_true = utils.simulate_dag(spec.d, spec.s0, spec.graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, spec.n, spec.sem_type)
    return X, B_true, W_true


def build_I(X: np.ndarray, B_true: np.ndarray, I_spec: ISpec) -> np.ndarray:
    d = B_true.shape[0]
    if I_spec.source == "pairwise":
        I = get_I_from_full_pairwise_tests(
            X,
            alpha=I_spec.alpha,
            test=I_spec.test,
            num_perm=I_spec.num_perm,
            seed=I_spec.seed,
            bonferroni=I_spec.bonferroni,
            undirected=I_spec.undirected,
        )

    if I_spec.source == "oracle":
        # "no-trek oracle" from the true graph structure
        W_oracle = torch.as_tensor(B_true, dtype=torch.double, device="cpu")
        I = get_no_trek_pairs(W_oracle, seq=I_spec.pst_seq_for_oracle)
    
    I_full = I.copy()

    if I_spec.cap is not None and isinstance(I_spec.cap, int):
        I = np.asarray(I, dtype=np.int64).reshape(-1, 2)
    
        m = I.shape[0]
        if I_spec.cap < m:
            rng = np.random.default_rng(1)
            idx = rng.choice(m, size=I_spec.cap, replace=False)
            I = I[idx]
    
            print(f"I capped: {m} → {I.shape[0]} pairs")
            print("full:", I_full.tolist())
            print("kept:", I.tolist())


    return I.astype(np.int64, copy=False)

    raise ValueError("ISpec.source must be one of {'oracle','pairwise'}")


# Base algorithm registry (easy to extend)
AlgoRunner = Callable[[np.ndarray, np.ndarray, Any, AlgoSpec, logging.Logger, LogConfig], np.ndarray]
ALGO_REGISTRY: Dict[str, AlgoRunner] = {}


def register_algo(name: str):
    def _decorator(fn: AlgoRunner):
        ALGO_REGISTRY[name] = fn
        return fn
    return _decorator


@register_algo("dagma_linear")
def run_dagma_linear(
    X: np.ndarray,
    B_true: np.ndarray,
    trek_reg: Any,
    algo_spec: AlgoSpec,
    logger: logging.Logger,
    log_cfg: LogConfig,
) -> np.ndarray:
    model = DagmaLinear(
        loss_type=algo_spec.loss_type,
        trek_reg=trek_reg,
        logger=logger,
        log_cfg=log_cfg,
    )
    W_est = model.fit(
        X,
        lambda1=algo_spec.lambda1,
        max_iter=algo_spec.max_iter,
        mu_factor=algo_spec.mu_factor,
        s=algo_spec.s,
    )
    return W_est


def make_trek_reg(I: np.ndarray, tr: TrekRegSpec):
    if tr.name == "none":
        return None

    if tr.name == "pst":
        return PSTRegularizer(
            I=I,
            seq=tr.seq,
            weight=tr.weight,
            kwargs={"K_log": tr.K_log, "eps_inv": tr.eps_inv, "s": tr.s},
            mode=tr.mode,
        )

    if tr.name == "tcc":
        # expects your updated TCCRegularizer supports cycle_penalty="spectral"|"logdet"
        # and passes through needed kwargs to trek_cycle_coupling_value_gradW internally.
        return TCCRegularizer(
            I=I,
            cycle_penalty=tr.cycle_penalty,
            weight=tr.weight,
            w=tr.w,
            n_iter=tr.n_iter,
            eps=tr.eps,
            mode=tr.mode,
            # If you expose these in your class, keep them; otherwise harmless:
            version=getattr(tr, "version", "approx_trek_graph"),
            method=getattr(tr, "method", "eig_torch"),
            s=getattr(tr, "s_logdet", 2.0),
        )

    raise ValueError("TrekRegSpec.name must be one of {'none','pst','tcc'}")


# -----------------------------
# Evaluation
# -----------------------------
def nnz(W_est: np.ndarray) -> int:
    return int(np.sum(W_est != 0))


def shd_from_utils(B_true: np.ndarray, W_est: np.ndarray) -> Dict[str, Any]:
    # dagma.utils.count_accuracy expects boolean adjacency estimate
    
    acc = utils.count_accuracy(B_true, W_est != 0)
    # acc typically contains keys: fdr, tpr, fpr, shd, nnz, precision, recall, etc.
    # We'll just return it as a dict (safe even if keys differ).
    return dict(acc)


def direct_reg_values(
    W_true: np.ndarray,
    W_est: np.ndarray,
    I: np.ndarray,
    tr: TrekRegSpec,
) -> Dict[str, Optional[float]]:
    """
    Optional: compute regularizer value on true and estimated W, independent of model internals.
    Keeps this minimal; only returns what makes sense for the reg.
    """
    out: Dict[str, Optional[float]] = {
        "reg_true": None,
        "reg_est": None,
    }

    if tr.name == "pst":
        Wt = torch.as_tensor(W_true, dtype=torch.double, device="cpu")
        We = torch.as_tensor(W_est, dtype=torch.double, device="cpu")
        out["reg_true"] = float(pst(Wt, I, seq=tr.seq, K_log=tr.K_log, eps_inv=tr.eps_inv, s=tr.s, agg=tr.agg).item())
        out["reg_est"] = float(pst(We, I, seq=tr.seq, K_log=tr.K_log, eps_inv=tr.eps_inv, s=tr.s, agg=tr.agg).item())
        return out

    if tr.name == "tcc":
        Wt = torch.as_tensor(W_true, dtype=torch.double, device="cpu")
        We = torch.as_tensor(W_est, dtype=torch.double, device="cpu")
        val_true, _ = trek_cycle_coupling_value_gradW(
            Wt,
            I,
            w=tr.w,
            cycle_penalty=tr.cycle_penalty,
            version=tr.version,
            method=tr.method,
            n_iter=tr.n_iter,
            eps=tr.eps,
            s=tr.s_logdet,
        )
        val_est, _ = trek_cycle_coupling_value_gradW(
            We,
            I,
            w=tr.w,
            cycle_penalty=tr.cycle_penalty,
            version=tr.version,
            method=tr.method,
            n_iter=tr.n_iter,
            eps=tr.eps,
            s=tr.s_logdet,
        )
        out["reg_true"] = float(val_true.item())
        out["reg_est"] = float(val_est.item())
        return out

    return out


# -----------------------------
# Main run loop
# -----------------------------
def run_suite(
    data_specs: DataSpec,
    i_spec: ISpec,
    algo_specs: List[AlgoSpec],
    trek_specs: List[TrekRegSpec],
) -> pd.DataFrame:
    logger, log_cfg = make_logger()

    # 1) data
    for data_spec in data_specs:
        X, B_true, W_true = generate_data(data_spec)
    
        # 2) I
        I = build_I(X, B_true, i_spec)
        # Lightweight printing: only shape + summary
        print(f"I_source={i_spec.source}  I_shape={I.shape}  (d={data_spec.d})")
        summarize_I(I, d=data_spec.d)
    
        rows: List[Dict[str, Any]] = []
    
        # 3) run all specs
        for algo in algo_specs:
            if algo.name not in ALGO_REGISTRY:
                raise ValueError(f"Unknown algo '{algo.name}'. Registered: {sorted(ALGO_REGISTRY.keys())}")
    
            run_algo = ALGO_REGISTRY[algo.name]
    
            for tr in trek_specs:
                trek_reg = make_trek_reg(I, tr)
    
                W_est = run_algo(
                    X=X,
                    B_true=B_true,
                    trek_reg=trek_reg,
                    algo_spec=algo,
                    logger=logger,
                    log_cfg=log_cfg,
                )
    
                acc = shd_from_utils(B_true, W_est)
                reg_vals = direct_reg_values(W_true, W_est, I, tr)
    
                row = {
                    "algo": algo.name,
                    "trek_reg": tr.name,
                    "cycle_penalty": (tr.cycle_penalty if tr.name == "tcc" else ""),
                    "seq": (tr.seq if tr.name == "pst" else ""),
                    "I_source": i_spec.source,
                    "seed": data_spec.seed,
                    "d": data_spec.d,
                    "n": data_spec.n,
                    "s0": data_spec.s0,
                    "graph": data_spec.graph_type,
                    "sem": data_spec.sem_type,
                    "lambda1": algo.lambda1,
                    "mu_factor": algo.mu_factor,
                    "max_iter": algo.max_iter,
                    "trek_weight": tr.weight,
                    "nnz": nnz(W_est),
                    # pull SHD and a couple common stats if present
                    "shd": acc.get("shd", None),
                    "tpr": acc.get("tpr", None),
                    "fdr": acc.get("fdr", None),
                    "fpr": acc.get("fpr", None),
                    "tp": acc.get("tp", None),
                    "fp": acc.get("fp", None),
                    "fn": acc.get("fn", None),
                    "reg_true": reg_vals["reg_true"],
                    "reg_est": reg_vals["reg_est"],
                }
                rows.append(row)
    
                # Minimal one-line per run
                print(
                    f"[{algo.name} | {tr.name}"
                    + (f":{tr.cycle_penalty}" if tr.name == "tcc" else "")
                    + f"] shd={row['shd']} nnz={row['nnz']} reg_est={row['reg_est']}"
            )

    df = pd.DataFrame(rows)

    # Pretty “spec” label
    def _spec_label(r):
        if r["trek_reg"] == "none":
            return "none"
        if r["trek_reg"] == "pst":
            return f"pst:{r['seq']}"
        if r["trek_reg"] == "tcc":
            return f"tcc:{r['cycle_penalty']}"
        return str(r["trek_reg"])

    df["spec"] = df.apply(_spec_label, axis=1)

    # Sort for readability
    sort_cols = ["algo", "spec", "I_source", "seed"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def default_suite() -> Tuple[DataSpec, ISpec, List[AlgoSpec], List[TrekRegSpec]]:
    data_specs = [
            # DataSpec(seed=4, n=500, d=10, s0=40, graph_type="ER", sem_type="gauss"),
            # DataSpec(seed=4, n=1000, d=10, s0=35, graph_type="ER", sem_type="gauss"),
            # DataSpec(seed=40, n=1000, d=10, s0=35, graph_type="ER", sem_type="gauss"),
            # DataSpec(seed=47, n=500, d=10, s0=35, graph_type="ER", sem_type="gauss"),
            DataSpec(seed=61, n=1000, d=10, s0=30, graph_type="ER", sem_type="gauss"),
            
        ]

    # Switch source to "pairwise" if you want the full MI-tests pipeline
    i_spec = ISpec(source="oracle", pst_seq_for_oracle="exp", cap=1) #  

    algo_specs = [
        AlgoSpec(name="dagma_linear", loss_type="l2", lambda1=0.02, max_iter=int(6e4), mu_factor=0.1, s=1.),
    ]

    trek_specs = [
        TrekRegSpec(
            name="pst",
            weight=10.0,
            seq="exp",
            K_log=40,
            eps_inv=1e-8,
            s=5.0,
            agg="mean",
            mode="log",
        ),
        # TrekRegSpec(
        #     name="pst",
        #     weight=1.0,
        #     seq="exp",
        #     K_log=40,
        #     eps_inv=1e-8,
        #     s=5.0,
        #     agg="mean",
        #     mode="opt",
        # ),
        # TrekRegSpec(
        #     name="pst",
        #     weight=10.0,
        #     seq="inv",
        #     K_log=40,
        #     eps_inv=1e-8,
        #     s=5.0,
        #     agg="mean",
        #     mode="opt",
        # ),
        TrekRegSpec(
            name="tcc",
            cycle_penalty="spectral",
            weight=.01,
            w=10.,
            n_iter=10,
            eps=1e-12,
            version="approx_trek_graph",
            method="eig_torch",
            mode="opt",
        ),
        
        # TrekRegSpec(
        #     name="tcc",
        #     cycle_penalty="spectral",
        #     weight=.001,
        #     w=10.,
        #     n_iter=10,
        #     eps=1e-12,
        #     version="DAG_learning",
        #     method="eig_torch",
        #     mode="opt",
        # ),
        # TrekRegSpec(
        #     name="tcc",
        #     cycle_penalty="logdet",
        #     weight=1.,
        #     w=10.,
        #     eps=1e-12,
        #     s_logdet=2.0, # For logdet stability: bump s_logdet if you see sign flips / nan
        #     version="exact_trek_graph",  # exact baseline
        #     mode="opt",
        # ),
    ]

    return data_specs, i_spec, algo_specs, trek_specs


def print_overview_table(df: pd.DataFrame):
    """
    Output a compact overview table:
      grouped by (algo, spec, I_source, seed) with SHD and NNZ (and some stats).
    """
    cols = [
        "algo", "spec", "I_source", "seed",
        "shd", "nnz",
        "tpr", "fdr", "fpr",
        "tp", "fp", "fn",
        "reg_est",
    ]
    cols = [c for c in cols if c in df.columns]

    # If multiple runs per spec (e.g., multiple seeds), show all rows.
    view = df[cols].copy()

    # Make it cleaner
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 140)

    print("\n==================== SUMMARY TABLE ====================")
    print(view.to_string(index=False))
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Run full small pipeline test (no viz, outputs table).")
    parser.add_argument("--I_source", type=str, default=None, choices=["oracle", "pairwise"], help="Override I source.")
    parser.add_argument("--seed", type=int, default=None, help="Override data seed.")
    parser.add_argument("--d", type=int, default=None, help="Override number of nodes.")
    parser.add_argument("--n", type=int, default=None, help="Override number of samples.")
    args = parser.parse_args()

    data_spec, i_spec, algo_specs, trek_specs = default_suite()

    if args.I_source is not None:
        i_spec = ISpec(**{**asdict(i_spec), "source": args.I_source})
    if args.seed is not None:
        data_spec = DataSpec(**{**asdict(data_spec), "seed": args.seed})
    if args.d is not None:
        data_spec = DataSpec(**{**asdict(data_spec), "d": args.d})
    if args.n is not None:
        data_spec = DataSpec(**{**asdict(data_spec), "n": args.n})

    df = run_suite(data_spec, i_spec, algo_specs, trek_specs)
    print_overview_table(df)


if __name__ == "__main__":
    main()


# So far spectral penalties are better in dense settings, whereas in sparse settings they tend to be very bad