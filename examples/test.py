# src/tests/complete_test.py
#
# Updated pipeline test harness compatible with:
#   - src.methods.method_spec  (per-algo specs)
#   - src.methods.dagma.*      (DAGMA runner via registry)
#   - src.methods.nodags.*     (NODAGS runner via registry)
#
# Key changes vs your ancient version:
#   - No local ALGO_REGISTRY / local AlgoSpec anymore.
#   - Uses the central registry entries: (runner, spec_cls).
#   - Builds spec via spec_cls and your “filter unknown keys” pattern.
#   - Trek reg built via your existing make_trek_reg (or import it).
#
# This file keeps your data+I pipeline and evaluation unchanged.

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np
import pandas as pd
import torch

from jax import random
from src.methods.kds.stadion.models import LinearSDE, MLPSDE
from pprint import pprint

# ---- I construction + notreks ----
from src.utils.mi_tests import get_I_from_full_pairwise_tests, summarize_I
from src.utils.notreks import (
    pst,
    get_no_trek_pairs,
    trek_cycle_coupling_value_gradW,
)
from src.utils.notreks import make_trek_reg  # <- use your canonical builder

# ---- logging ----
from src.utils.logger import LogConfig, build_default_logger

# ---- central algo registry ----
from src.methods.method_spec import ALGO_REGISTRY  # <- your new registry with AlgoEntry(runner, spec_cls)

# ---- DAGMA data utils (unchanged) ----
from src.methods.dagma import utils as dagma_utils

from src.eval.metrics import evaluate_structure, find_best_threshold_for_shd



# -----------------------------
# Config dataclasses (test-level)
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
    pst_seq_for_oracle: str = "exp"
    cap: Optional[int] = None


# NOTE: TrekRegSpec is your YAML-ish spec for make_trek_reg(...)
# Keep it as dict in this test (simpler + matches your framework config style)
TrekCfg = Dict[str, Any]
AlgoCfg = Dict[str, Any]


# -----------------------------
# Helpers: logging, data, I
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
    dagma_utils.set_random_seed(spec.seed)
    B_true = dagma_utils.simulate_dag(spec.d, spec.s0, spec.graph_type)
    W_true = dagma_utils.simulate_parameter(B_true)
    X = dagma_utils.simulate_linear_sem(W_true, spec.n, spec.sem_type)
    return X, B_true, W_true


def build_I(X: np.ndarray, B_true: np.ndarray, I_spec: ISpec) -> np.ndarray:
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
    elif I_spec.source == "oracle":
        W_oracle = torch.as_tensor(B_true, dtype=torch.double, device="cpu")
        I = get_no_trek_pairs(W_oracle, seq=I_spec.pst_seq_for_oracle)
    else:
        raise ValueError("ISpec.source must be one of {'oracle','pairwise'}")

    I = np.asarray(I, dtype=np.int64).reshape(-1, 2)

    if I_spec.cap is not None:
        m = I.shape[0]
        if int(I_spec.cap) < m:
            rng = np.random.default_rng(1)
            idx = rng.choice(m, size=int(I_spec.cap), replace=False)
            I = I[idx]
            print(f"I capped: {m} → {I.shape[0]} pairs")

    return I.astype(np.int64, copy=False)


# -----------------------------
# Eval
# -----------------------------
def nnz(W_est: np.ndarray) -> int:
    return int(np.sum(W_est != 0))


def acc_from_metrics(W_true: np.ndarray, W_est: np.ndarray, *, thr: float = 1e-12) -> Dict[str, Any]:
    return evaluate_structure(W_true, W_est, thr=thr)

def direct_reg_values(
    W_true: np.ndarray,
    W_est: np.ndarray,
    I: np.ndarray,
    trek_cfg: TrekCfg,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"reg_true": None, "reg_est": None}
    name = str(trek_cfg.get("name", "none")).lower().strip()

    if name == "pst":
        seq = str(trek_cfg.get("seq", "exp"))
        K_log = int(trek_cfg.get("K_log", 40))
        eps_inv = float(trek_cfg.get("eps_inv", 1e-8))
        s = float(trek_cfg.get("s", 5.0))
        agg = str(trek_cfg.get("agg", "mean"))

        Wt = torch.as_tensor(W_true, dtype=torch.double, device="cpu")
        We = torch.as_tensor(W_est, dtype=torch.double, device="cpu")
        out["reg_true"] = float(pst(Wt, I, seq=seq, K_log=K_log, eps_inv=eps_inv, s=s, agg=agg).item())
        out["reg_est"] = float(pst(We, I, seq=seq, K_log=K_log, eps_inv=eps_inv, s=s, agg=agg).item())
        return out

    if name == "tcc":
        cycle_penalty = str(trek_cfg.get("cycle_penalty", "spectral"))
        w = float(trek_cfg.get("w", 100.0))
        n_iter = int(trek_cfg.get("n_iter", 10))
        eps = float(trek_cfg.get("eps", 1e-12))
        version = str(trek_cfg.get("version", "approx_trek_graph"))
        method = str(trek_cfg.get("method", "eig_torch"))
        s_logdet = float(trek_cfg.get("s_logdet", 2.0))

        Wt = torch.as_tensor(W_true, dtype=torch.double, device="cpu")
        We = torch.as_tensor(W_est, dtype=torch.double, device="cpu")
        val_true, _ = trek_cycle_coupling_value_gradW(
            Wt, I, w=w, cycle_penalty=cycle_penalty, version=version, method=method, n_iter=n_iter, eps=eps, s=s_logdet
        )
        val_est, _ = trek_cycle_coupling_value_gradW(
            We, I, w=w, cycle_penalty=cycle_penalty, version=version, method=method, n_iter=n_iter, eps=eps, s=s_logdet
        )
        out["reg_true"] = float(val_true.item())
        out["reg_est"] = float(val_est.item())
        return out

    return out


# -----------------------------
# New: build per-algo spec from registry entry (your old filtering style)
# -----------------------------
def build_algo_spec(algo_cfg: AlgoCfg):
    algo_name = str(algo_cfg.get("name", "")).strip()
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algo '{algo_name}'. Registered: {sorted(ALGO_REGISTRY.keys())}")

    entry = ALGO_REGISTRY[algo_name]
    SpecCls = entry.spec_cls

    # supports unknown keys by filtering (your old pattern)
    spec = SpecCls(**{
        k: algo_cfg[k]
        for k in SpecCls.__dataclass_fields__.keys()
        if k in algo_cfg
    })
    return entry, spec


# -----------------------------
# Main suite
# -----------------------------
def run_suite(
    data_specs: List[DataSpec],
    i_spec: ISpec,
    algo_cfgs: List[AlgoCfg],
    trek_cfgs: List[TrekCfg],
) -> pd.DataFrame:
    logger, log_cfg = make_logger()

    rows: List[Dict[str, Any]] = []

    for ds_spec in data_specs:
        X, B_true, W_true = generate_data(ds_spec)

        I = build_I(X, B_true, i_spec)
        print(f"I_source={i_spec.source}  I_shape={I.shape}  (d={ds_spec.d})")
        summarize_I(I, d=ds_spec.d)

        for algo_cfg in algo_cfgs:
            entry, algo_spec = build_algo_spec(algo_cfg)

            for trek_cfg in trek_cfgs:
                trek_reg = make_trek_reg(I, trek_cfg)

                W_est = entry.runner(
                    X,
                    trek_reg,
                    algo_spec,
                    logger,
                    log_cfg,
                )
                # print(W_est, W_true)
                thr_best, acc = find_best_threshold_for_shd(W_true, W_est)

                reg_vals = direct_reg_values(W_true, W_est, I, trek_cfg)
                
                tr_name = str(trek_cfg.get("name", "none"))
                row = {
                    "algo": str(getattr(algo_spec, "name", algo_cfg.get("name"))),
                    "trek_reg": tr_name,
                    "cycle_penalty": (str(trek_cfg.get("cycle_penalty", "")) if tr_name == "tcc" else ""),
                    "seq": (str(trek_cfg.get("seq", "")) if tr_name == "pst" else ""),
                    "I_source": i_spec.source,
                    "seed": ds_spec.seed,
                    "d": ds_spec.d,
                    "n": ds_spec.n,
                    "s0": ds_spec.s0,
                    "graph": ds_spec.graph_type,
                    "sem": ds_spec.sem_type,
                    "trek_weight": float(trek_cfg.get("weight", 0.0)),
                    
                    "thr_best": thr_best,
                
                    # --- structure metrics (everything from src.eval.metrics.evaluate_structure) ---
                    "nnz_true": acc.get("nnz_true", None),
                    "nnz_est": acc.get("nnz_est", None),
                    "shd": acc.get("shd", None),
                    "fhd": acc.get("fhd", None),
                    
                    "fro_rel": acc.get("fro_rel", None),
                
                    "tp": acc.get("tp", None),
                    "fp": acc.get("fp", None),
                    "fn": acc.get("fn", None),
                    "tn": acc.get("tn", None),
                
                    "precision": acc.get("precision", None),
                    "tpr": acc.get("tpr", None),
                    "fdr": acc.get("fdr", None),
                    "fpr": acc.get("fpr", None),
                
                    # --- reg diagnostics ---
                    "reg_true": reg_vals["reg_true"],
                    "reg_est": reg_vals["reg_est"],
                }
                rows.append(row)


                spec_label = (
                    "none" if tr_name == "none"
                    else f"pst:{row['seq']}" if tr_name == "pst"
                    else f"tcc:{row['cycle_penalty']}" if tr_name == "tcc"
                    else tr_name
                )
                print(
                    f"[{row['algo']} | {spec_label}] "
                    f"thr={row['thr_best']:.2e} "
                    f"shd={row['shd']} fhd={row['fhd']:.3f} fro_rel={row['fro_rel']:.3e} "
                    f"nnz={row['nnz_est']}/{row['nnz_true']} "
                    f"tpr={row['tpr']:.2f} fdr={row['fdr']:.2f} "
                    f"reg_est={row['reg_est']:.3g}"
                )



    df = pd.DataFrame(rows)

    def _spec_label(r):
        if r["trek_reg"] == "none":
            return "none"
        if r["trek_reg"] == "pst":
            return f"pst:{r['seq']}"
        if r["trek_reg"] == "tcc":
            return f"tcc:{r['cycle_penalty']}"
        return str(r["trek_reg"])

    df["spec"] = df.apply(_spec_label, axis=1)
    df = df.sort_values(["algo", "spec", "I_source", "seed"]).reset_index(drop=True)
    return df


def default_suite() -> Tuple[List[DataSpec], ISpec, List[AlgoCfg], List[TrekCfg]]:
    data_specs = [
        DataSpec(seed=25, n=1000, d=40, s0=200, graph_type="ER", sem_type="gauss"),
    ]

    i_spec = ISpec(source="oracle", pst_seq_for_oracle="exp")

    # IMPORTANT: algo_cfgs are dicts now; spec_cls filtering will pick correct fields per algo.
    algo_cfgs: List[AlgoCfg] = [
        # # ---------------------------------------------------------
        # # DAGMA — linear NOTEARS-style baseline
        # # ---------------------------------------------------------
        # {
        #     "name": "dagma_linear",
        #     "loss_type": "l2",
        #     "lambda1": 0.02,
        #     "max_iter": int(6e4),
        #     "mu_factor": 0.1,
        #     "s": 1.0,
        # },
        # ---------------------------------------------------------
        # DAGMA — linear NOTEARS-style baseline
        # ---------------------------------------------------------
        {
            "name": "midagma_linear",
            "loss_type": "l2",
            "lambda1": 0.0002,
            "max_iter": int(12e4),
            "mu_factor": 0.1,
            "s": 1.2,
        },
        # # ---------------------------------------------------------
        # # NODAGS — nonlinear flow-based baseline
        # # ---------------------------------------------------------
        # {
        #     "name": "nodags",
        #     "fun_type": "lin-mlp",
        #     "epochs": 100,
        #     "batch_size": 128,
        #     "lr": .001,
        #     "optim": "adam",
        #     "dag_input": True,
        # },
        # # ---------------------------------------------------------
        # # KDS — stadion linear stationary diffusion baseline
        # # (conceptually similar to dagma_linear but SDE-based)
        # # ---------------------------------------------------------
        # {
        #     "name": "kds",
        #     "model": "linear",          # matches LinearSDEWithTrek
        #     "seed": 0,
    
        #     # stadion.fit(...) kwargs
        #     "targets": None,
        #     "bandwidth": 2.0,
        #     "estimator": "linear",
        #     "learning_rate": 3e-3,
        #     "steps": 20_000,
        #     "batch_size": 128,
        #     "reg": 1e-2,
        #     "warm_start_intv": True,
        #     "device": None,
        #     "verbose": 10,
    
        #     # optional constructor kwargs for LinearSDE (usually empty)
        #     "model_kwargs": {},
        # },
    ]


    trek_cfgs: List[TrekCfg] = [
        {"name": "pst", "weight": .1, "seq": "log", "K_log": 40, "eps_inv": 1e-8, "s": 5.0, "agg": "mean", "mode": "off"},
        # {"name": "pst", "weight": .01, "seq": "exp", "K_log": 40, "eps_inv": 1e-8, "s": 5.0, "agg": "mean", "mode": "opt"},
        # {"name": "tcc", "cycle_penalty": "spectral", "version": "approx_trek_graph", "weight": 0.01, "w": 1., "mode": "opt"},
    ]

    return data_specs, i_spec, algo_cfgs, trek_cfgs


def print_overview_table(df: pd.DataFrame):
    cols = [
        "algo", "spec", "I_source", "seed",
        "thr_best",
        "shd", "fro_rel", #  "fhd",
        "nnz_est", "nnz_true",
        #"tpr", "fdr", "fpr",
        # "tp", "fp", "fn",
        "reg_est",
    ]

    cols = [c for c in cols if c in df.columns]

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 140)

    print("\n==================== SUMMARY TABLE ====================")
    print(df[cols].to_string(index=False))
    print("=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full small pipeline test (no viz, outputs table).")
    parser.add_argument("--I_source", type=str, default=None, choices=["oracle", "pairwise"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    data_specs, i_spec, algo_cfgs, trek_cfgs = default_suite()

    # override only the first data spec (matches your old behavior)
    if args.I_source is not None:
        i_spec = ISpec(**{**asdict(i_spec), "source": args.I_source})
    if args.seed is not None:
        data_specs = [DataSpec(**{**asdict(data_specs[0]), "seed": args.seed})]
    if args.d is not None:
        data_specs = [DataSpec(**{**asdict(data_specs[0]), "d": args.d})]
    if args.n is not None:
        data_specs = [DataSpec(**{**asdict(data_specs[0]), "n": args.n})]

    df = run_suite(data_specs, i_spec, algo_cfgs, trek_cfgs)
    print_overview_table(df)

