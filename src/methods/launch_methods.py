# -*- coding: utf-8 -*-
"""
src/methods/launch.py

Runner script launched by ExperimentManager.launch_methods (via SLURM array or local).
It executes exactly ONE (method instance × dataset) run.

Inputs:
  - resolved method instance YAML (already expanded grid)
  - datasets.txt mapping file + dataset_index (1-based)
  - path_data_root containing datasets/<dataset_id>/...
  - path_results output directory (one shared results folder for the experiment)

Outputs (suggested layout):
  <path_results>/
    runs/
      <method_id>/
        <dataset_id>/
          W_est.npy
          meta.yaml
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.utils.yaml import load_yaml, save_yaml
from src.data.dataset import Dataset

# algo registry + spec
from src.methods.method_spec import ALGO_REGISTRY, assert_registry_complete

# optional logging (your existing logger helpers)
from src.utils.logger import LogConfig, build_default_logger

# trek regs: use directly from your notreks package
from src.utils.notreks import PSTRegularizer, TCCRegularizer

def base_method_id(method_id: str) -> str:
    return method_id.split("__", 1)[0]

# -------------------------
# trek reg factory
# -------------------------
def make_trek_reg(I: np.ndarray, cfg: Dict[str, Any]) -> Any:
    """
    cfg example:
      trek_reg:
        name: none | pst | tcc
        weight: 0.1
        # pst params
        seq: log
        K_log: 40
        eps_inv: 1e-8
        s: 5.0
        mode: opt
        # tcc params
        cycle_penalty: spectral|logdet
        w: 100.0
        n_iter: 10
        eps: 1e-12
        version: approx_trek_graph
        method: eig_torch
        s_logdet: 2.0
        mode: opt
    """
    name = str(cfg.get("name", "none")).strip().lower()
    if name in ("none", "", "null"):
        return None

    weight = float(cfg.get("weight", 0.0))

    if name == "pst":
        return PSTRegularizer(
            I=I,
            seq=str(cfg.get("seq", "log")),
            weight=weight,
            kwargs={
                "K_log": int(cfg.get("K_log", 40)),
                "eps_inv": float(cfg.get("eps_inv", 1e-8)),
                "s": float(cfg.get("s", 5.0)),
            },
            mode=str(cfg.get("mode", "opt")),
        )

    if name == "tcc":
        return TCCRegularizer(
            I=I,
            cycle_penalty=str(cfg.get("cycle_penalty", "spectral")),
            weight=weight,
            w=float(cfg.get("w", 100.0)),
            n_iter=int(cfg.get("n_iter", 10)),
            eps=float(cfg.get("eps", 1e-12)),
            mode=str(cfg.get("mode", "opt")),
            # optional passthroughs (keep harmless defaults)
            version=str(cfg.get("version", "approx_trek_graph")),
            method=str(cfg.get("method", "eig_torch")),
            s=float(cfg.get("s_logdet", 2.0)),
        )

    raise ValueError("trek_reg.name must be one of {'none','pst','tcc'}")


# -------------------------
# small helpers
# -------------------------
def read_dataset_id(datasets_file: Path, dataset_index_1based: int) -> str:
    lines = [ln.strip() for ln in datasets_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if dataset_index_1based < 1 or dataset_index_1based > len(lines):
        raise IndexError(
            f"dataset_index={dataset_index_1based} out of range for {datasets_file} (N={len(lines)})"
        )
    return lines[dataset_index_1based - 1]


def out_dir(path_results: Path, method_id: str, dataset_id: str) -> Path:
    return path_results / "runs" / method_id / dataset_id


def build_logger() -> tuple[Any, LogConfig]:
    logger = build_default_logger(level="INFO")
    log_cfg = LogConfig(
        enabled=True,
        print_to_console=False,
        store_csv=False,
        store_jsonl=False,
        keep_in_memory=True,
    )
    return logger, log_cfg


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_id", type=str, required=True)
    parser.add_argument("--method_cfg", type=Path, required=True)

    parser.add_argument("--dataset", type=Path, required=True)
    
    parser.add_argument("--path_data_root", type=Path, required=True)
    parser.add_argument("--path_results", type=Path, required=True)

    parser.add_argument("--descr", type=str, required=False, default="")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    # sanity: make sure expected algos are registered
    assert_registry_complete()

    method_id = str(args.method_id)
    method_cfg_path = Path(args.method_cfg)
    path_data_root = Path(args.path_data_root)
    path_results = Path(args.path_results)
    
    ds_root = Path(args.dataset) 
    dataset_id = ds_root.name

    # output
    run_dir = out_dir(path_results, method_id, dataset_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    W_path = run_dir / "W_est.npy"
    meta_path = run_dir / "meta.yaml"

    if (not args.overwrite) and W_path.exists() and meta_path.exists():
        print(f"[METHOD-RUN] exists -> skip  method_id={method_id} dataset_id={dataset_id} out={run_dir}")
        return

    # load resolved method cfg
    cfg = load_yaml(method_cfg_path)

    algo_cfg = cfg.get("algo", {})
    trek_cfg = cfg.get("trek_reg", {"name": "none"})

    # ---- build algo spec (supports unknown keys by filtering) ----
    algo_name = str(algo_cfg.get("name", "dagma_linear"))
    
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algo '{algo_name}'. Registered: {sorted(ALGO_REGISTRY.keys())}")
    
    entry = ALGO_REGISTRY[algo_name]
    SpecCls = entry.spec_cls
    
    algo_spec = SpecCls(**{
        k: algo_cfg[k]
        for k in SpecCls.__dataclass_fields__.keys()
        if k in algo_cfg
    })
    
    # ---- load dataset ----
    ds_root = path_data_root / str(dataset_id)
    ds = Dataset.load(ds_root)
    
    # ---- build trek regularizer ----
    trek_reg = make_trek_reg(ds.I, trek_cfg)
    
    # ---- run ----
    logger, log_cfg = build_logger()
    
    t0 = time.time()
    W_est = entry.runner(ds.X, trek_reg, algo_spec, logger, log_cfg)
    wall_s = time.time() - t0
    
    np.save(W_path, np.asarray(W_est))

    meta: Dict[str, Any] = {
        "descr": str(args.descr),
        "base_method_id": base_method_id(method_id),
        "method_id": method_id,
        "dataset_id": str(dataset_id),
    
        "paths": {
            "dataset_root": str(ds_root),
            "W_est": "W_est.npy",
        },

        "algo": dict(algo_cfg),
        "trek_reg": dict(trek_cfg),

        "dataset_meta": dict(ds.meta or {}),
        "shapes": {
            "X": list(ds.X.shape),
            "B_true": list(ds.B_true.shape),
            "W_true": list(ds.W_true.shape),
            "I_full": list(ds.I_full.shape),
            "I": list(ds.I.shape),
            "W_est": list(np.asarray(W_est).shape),
        },
        "counts": {
            "edges_true": int(ds.B_true.sum()),
            "I_pairs": int(ds.I.shape[0]),
        },

        "walltime_sec": float(wall_s),
    }

    save_yaml(meta, meta_path)

    print(
        f"[METHOD-RUN] method_id={method_id} dataset_id={dataset_id} "
        f"algo={algo_spec.name} trek={trek_cfg.get('name','none')} "
        f"out={run_dir} wall_s={wall_s:.2f}"
    )


if __name__ == "__main__":
    main()

