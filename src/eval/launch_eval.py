# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.yaml import load_yaml, save_yaml
from src.eval.metrics import evaluate_structure
from src.eval.summarize_eval import summarize_eval


def _find_run_metas(runs_root: Path) -> List[Path]:
    # expects .../runs/**/meta.yaml
    return sorted(runs_root.glob("**/meta.yaml"))


def _resolve_dataset_root(meta: Dict[str, Any]) -> Optional[Path]:
    # preferred: dataset_meta.data_path (you asked for this)
    ds_meta = meta.get("dataset_meta", {}) or {}
    if "data_path" in ds_meta and ds_meta["data_path"] is not None:
        return Path(ds_meta["data_path"])

    # fallback: meta.paths.dataset_root (your launch_methods currently writes this)
    paths = meta.get("paths", {}) or {}
    if "dataset_root" in paths and paths["dataset_root"] is not None:
        return Path(paths["dataset_root"])

    return None


def _dataset_filemap(ds_root: Path) -> Dict[str, str]:
    """
    As requested: append these relative paths to ds_root.
    """
    return {
        "ds_root": ".",
        "meta_yaml": "meta.yaml",
        "true_dir": "true",
        "B_true": "true/B_true.npy",
        "I": "true/I.npy",
        "I_full": "true/I_full.npy",
        "W_true": "true/W_true.npy",
        "X": "X.npy",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path_results", type=Path, required=True)
    p.add_argument("--runs_subdir", type=str, default="runs")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--thr", type=float, default=0.0)
    p.add_argument("--descr", type=str, default="")
    args = p.parse_args()
    
    path_results = Path(args.path_results)
    runs_root = path_results / args.runs_subdir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_csv = out_dir / "eval_long.csv"
    out_meta = out_dir / "eval_meta.yaml"
    
    metas = _find_run_metas(runs_root)
    if not metas:
        raise FileNotFoundError(f"No meta.yaml found under {runs_root}")
    
    rows: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []
    
    for meta_path in metas:
        run_dir = meta_path.parent
        meta = load_yaml(meta_path)
    
        method_id = str(meta.get("method_id", run_dir.parent.name if run_dir.parent else "method"))
        dataset_id = str(meta.get("dataset_id", run_dir.name))
    
        W_est_path = run_dir / "W_est.npy"
        if not W_est_path.exists():
            rejects.append({
                "run_dir": str(run_dir),
                "reason": "missing W_est.npy",
                "method_id": method_id,
                "dataset_id": dataset_id,
            })
            continue
    
        ds_root = _resolve_dataset_root(meta)
        if ds_root is None:
            rejects.append({
                "run_dir": str(run_dir),
                "reason": "cannot resolve dataset root (dataset_meta.data_path or paths.dataset_root missing)",
                "method_id": method_id,
                "dataset_id": dataset_id,
            })
            continue
    
        fmap = _dataset_filemap(ds_root)
        W_true_path = ds_root / fmap["W_true"]
        B_true_path = ds_root / fmap["B_true"]
        I_true_path = ds_root / fmap["I"]
    
        if not W_true_path.exists():
            rejects.append({
                "run_dir": str(run_dir),
                "reason": f"missing W_true at {W_true_path}",
                "method_id": method_id,
                "dataset_id": dataset_id,
                "ds_root": str(ds_root),
            })
            continue
    
        try:
            B_true = np.load(B_true_path)
            W_true = np.load(W_true_path)
            W_est = np.load(W_est_path)
            I = np.load(I_true_path)
        except Exception as e:
            rejects.append({
                "run_dir": str(run_dir),
                "reason": f"failed loading npy: {e}",
                "method_id": method_id,
                "dataset_id": dataset_id,
                "meta_path": str(meta_path),
                "W_true": str(W_true_path),
                "W_est": str(W_est_path),
                "I": str(I_true_path),
            })
            continue
    
        metr = evaluate_structure(B_true, W_true, W_est, I=I, thr=float(args.thr))

        row: Dict[str, Any] = {
            "descr": str(args.descr),
            "method_id": method_id,
            "dataset_id": dataset_id,
            "run_dir": str(run_dir),
            "meta_yaml": str(meta_path),
            "ds_root": str(ds_root),
            "W_true": str(W_true_path),
            "W_est": str(W_est_path),
            "thr": float(args.thr),
        }
        row.update(metr)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    summarize_eval(
        eval_csv=out_csv,
        out_dir=out_dir,
    )
    
    # small pivot summary (median over datasets) for quick view
    if not df.empty:
        pivot = df.pivot_table(
            index="method_id",
            values=["shd", "fhd", "nnz_est", "precision", "fdr", "tpr", "fpr", "tp", "fp", "fn"],
            aggfunc="median",
            dropna=False,
        ).sort_values(by="shd", ascending=True)
        pivot.to_csv(out_dir / "eval_pivot_median.csv")
    
    save_yaml(
        {
            "descr": str(args.descr),
            "path_results": str(path_results),
            "runs_root": str(runs_root),
            "out_csv": str(out_csv),
            "n_runs_found": int(len(metas)),
            "n_rows_written": int(len(rows)),
            "n_rejected": int(len(rejects)),
            "rejected": rejects[:50],  # don’t blow up yaml; keep first 50
        },
        out_meta,
    )
    
    print(f"[EVAL] wrote {len(rows)} rows -> {out_csv}")
    if rejects:
        print(f"[EVAL] rejected {len(rejects)} runs (see {out_meta})")
