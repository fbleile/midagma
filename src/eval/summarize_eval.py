# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

MetricDir = Literal["min", "max"]


DEFAULT_METRIC_DIR: Dict[str, MetricDir] = {
    # lower is better
    "shd": "min",
    "fhd": "min",
    "fdr": "min",
    "fpr": "min",
    "pst_exp": "min",
    # higher is better
    "precision": "max",
    # optional: keep tpr as alias if present
    "tpr": "max",
}

DEFAULT_METRICS = [
    "shd", "fhd", "fro_rel",
    "nnz_true", "nnz_est", "nnz_err",
    "tp", "fp", "fn", "tn",
    "precision", "fdr", "tpr", "fpr",
    "pst_exp",
]

def _ensure_base_method_id(df: pd.DataFrame) -> pd.DataFrame:
    if "base_method_id" not in df.columns:
        df = df.copy()
        df["base_method_id"] = df["method_id"].astype(str).str.split("__", n=1).str[0]
    return df

# the nnz_err is the number of nonzeros error between the groundtruth and the estimation
# it captures a general sparsity bias, if present.
def _add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "nnz_true" in df.columns and "nnz_est" in df.columns and "nnz_err" not in df.columns:
        df["nnz_err"] = df["nnz_est"] - df["nnz_true"]
    return df


def _rank_within_dataset(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    dataset_col: str,
    metric: str,
    direction: MetricDir,
) -> pd.Series:
    """
    Rank inside each dataset for the given metric.
    1.0 = best. Ties get average rank.
    """
    asc = (direction == "min")
    return df.groupby(group_cols + [dataset_col])[metric].rank(method="average", ascending=asc)


def _percent_to_best_within_dataset(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    dataset_col: str,
    metric: str,
    direction: MetricDir,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Score in [0, 1]-ish where best=1.0 in each dataset for each metric.
      - max metrics: score = val / best
      - min metrics: score = best / val
    Handles zeros robustly.
    """
    g = df.groupby(group_cols + [dataset_col])[metric]
    if direction == "max":
        best = g.transform("max")
        val = df[metric]
        # if best==0: score = 1 if val==0 else 0
        return np.where(best.abs() <= eps, np.where(val.abs() <= eps, 1.0, 0.0), val / best)
    else:
        best = g.transform("min")
        val = df[metric]
        # if best==0: score = 1 if val==0 else 0
        return np.where(best.abs() <= eps, np.where(val.abs() <= eps, 1.0, 0.0), best / np.maximum(val, eps))


def _format_best_bold(table: pd.DataFrame, dirs: Dict[str, MetricDir], decimals: int = 3) -> pd.DataFrame:
    """
    Convert numeric table to string with best value per column bolded.
    """
    out = table.copy()

    # format numeric
    for c in out.columns:
        out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.{decimals}f}")

    # bold best
    for c in out.columns:
        if c not in dirs:
            continue
        col = table[c]
        if col.dropna().empty:
            continue
        if dirs[c] == "min":
            best_val = col.min()
            mask = col == best_val
        else:
            best_val = col.max()
            mask = col == best_val
        out.loc[mask, c] = out.loc[mask, c].map(lambda s: f"\\textbf{{{s}}}" if s != "" else s)

    return out


def _write_table_csv_tex(
    table: pd.DataFrame,
    *,
    out_csv: Path,
    out_tex: Path,
    dirs: Dict[str, MetricDir],
    decimals: int = 3,
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=True)

    formatted = _format_best_bold(table, dirs=dirs, decimals=decimals)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(
        formatted.to_latex(escape=False),
        encoding="utf-8",
    )


def _aggregate_tables(
    df: pd.DataFrame,
    *,
    entity_col: str,
    dataset_col: str,
    metrics: List[str],
    dirs: Dict[str, MetricDir],
) -> Dict[str, pd.DataFrame]:
    """
    Return dict of tables: avg, median, rank_mean, score_mean
    Each table indexed by entity_col.
    """
    
    # mean/median over datasets
    avg = df.groupby(entity_col)[metrics].mean(numeric_only=True)
    std = df.groupby(entity_col)[metrics].std(numeric_only=True)
    med = df.groupby(entity_col)[metrics].median(numeric_only=True)

    # ranks + percent-to-best computed per dataset, then averaged
    tmp = df[[entity_col, dataset_col] + metrics].copy()
    for m in metrics:
        direction = dirs.get(m, "min")
        tmp[f"rank__{m}"] = tmp.groupby(dataset_col)[m].rank(method="average", ascending=(direction == "min"))
        tmp[f"score__{m}"] = _percent_to_best_within_dataset(
            tmp, group_cols=[], dataset_col=dataset_col, metric=m, direction=direction
        )

    rank_cols = [f"rank__{m}" for m in metrics]
    score_cols = [f"score__{m}" for m in metrics]

    rank_mean = tmp.groupby(entity_col)[rank_cols].mean(numeric_only=True)
    rank_mean.columns = metrics  # rename back to metric names

    score_mean = tmp.groupby(entity_col)[score_cols].mean(numeric_only=True)
    score_mean.columns = metrics

    return {
        "avg": avg,
        "std": std,
        "median": med,
        "rank_mean": rank_mean,
        "score_mean": score_mean,
    }


def summarize_eval(
    eval_csv: Path,
    out_dir: Path,
    *,
    dataset_col: str = "dataset_id",
    method_col: str = "method_id",
    base_col: str = "base_method_id",
    metrics: List[str] = None,
    metric_dirs: Dict[str, MetricDir] = None,
):
    eval_csv = Path(eval_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(eval_csv)
    df = _ensure_base_method_id(df)
    df = _add_derived_metrics(df)

    metric_dirs = dict(DEFAULT_METRIC_DIR) if metric_dirs is None else dict(metric_dirs)
    metrics = list(DEFAULT_METRICS) if metrics is None else list(metrics)
    
    seen = set()
    metrics = [m for m in metrics if not (m in seen or seen.add(m))]


    # keep only available metrics
    metrics = [m for m in metrics if m in df.columns]

    # -------------------------
    # (1) Hyperparameter evaluation: within each base_method_id, compare method_id variants
    # -------------------------
    hp_root = out_dir / "hp_eval"
    for base, df_base in df.groupby(base_col):
        # entity = method_id inside this base
        tables = _aggregate_tables(
            df_base,
            entity_col=method_col,
            dataset_col=dataset_col,
            metrics=metrics,
            dirs=metric_dirs,
        )

        for name, table in tables.items():
            _write_table_csv_tex(
                table,
                out_csv=hp_root / f"{base}__{name}.csv",
                out_tex=hp_root / f"{base}__{name}.tex",
                dirs=metric_dirs if name in ("avg", "median") else ({m: "min" for m in metrics} if name == "rank_mean" else {m: "max" for m in metrics}),
                decimals=3,
            )

    # -------------------------
    # (2) Method comparison: average over method_id within base per dataset, then compare bases
    # -------------------------
    # first: aggregate within (base, dataset) over variants
    df_bd = df.groupby([base_col, dataset_col])[metrics].mean(numeric_only=True).reset_index()

    tables_base = _aggregate_tables(
        df_bd,
        entity_col=base_col,
        dataset_col=dataset_col,
        metrics=metrics,
        dirs=metric_dirs,
    )

    for name, table in tables_base.items():
        _write_table_csv_tex(
            table,
            out_csv=out_dir / f"method_eval__{name}.csv",
            out_tex=out_dir / f"method_eval__{name}.tex",
            dirs=metric_dirs if name in ("avg", "median") else ({m: "min" for m in metrics} if name == "rank_mean" else {m: "max" for m in metrics}),
            decimals=3,
        )
