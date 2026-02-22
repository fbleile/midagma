# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from src.utils.notreks import pst  # adjust path if needed



def binarize(W: np.ndarray, *, thr: float = 1e-12) -> np.ndarray:
    """Return binary adjacency B where B[i,j]=1 iff |W[i,j]| > thr, with zero diagonal."""
    B = (np.abs(W) > thr).astype(np.int64)
    np.fill_diagonal(B, 0)
    return B


def count_edges(B: np.ndarray) -> int:
    return int(B.sum())


def shd(B_true: np.ndarray, B_hat: np.ndarray) -> int:
    """
    Structural Hamming Distance for directed graphs (no CPDAG business):
    count of entries where B_true != B_hat (excluding diagonal).
    """
    assert B_true.shape == B_hat.shape
    M = (B_true != B_hat).astype(np.int64)
    np.fill_diagonal(M, 0)
    return int(M.sum())


def fhd(B_true: np.ndarray, B_hat: np.ndarray) -> float:
    """
    Fractional Hamming Distance = SHD / (#possible directed edges).
    """
    d = B_true.shape[0]
    denom = d * (d - 1)
    return shd(B_true, B_hat) / float(denom) if denom > 0 else float("nan")


def confusion_counts(B_true: np.ndarray, B_hat: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Return TP, FP, FN, TN over directed edges excluding diagonal.
    """
    assert B_true.shape == B_hat.shape
    d = B_true.shape[0]
    mask = ~np.eye(d, dtype=bool)

    t = B_true[mask].astype(bool)
    h = B_hat[mask].astype(bool)

    tp = int(np.logical_and(t, h).sum())
    fp = int(np.logical_and(~t, h).sum())
    fn = int(np.logical_and(t, ~h).sum())
    tn = int(np.logical_and(~t, ~h).sum())
    return tp, fp, fn, tn


def rates(B_true: np.ndarray, B_hat: np.ndarray) -> Dict[str, float]:
    tp, fp, fn, tn = confusion_counts(B_true, B_hat)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, fdr=fdr, tpr=tpr, fpr=fpr)

def fro_error(W_true, W_est):
    return float(np.linalg.norm(W_est - W_true, ord="fro"))

def fro_error_rel(W_true, W_est):
    denom = np.linalg.norm(W_true, ord="fro")
    return float(np.linalg.norm(W_est - W_true, ord="fro") / (denom + 1e-12))

def pst_exp(W_est: np.ndarray, I: Optional[np.ndarray]) -> float:
    """
    Return scalar PST penalty (exp, mean aggregation) for the given constrained pairs I.

    - W_est: numpy array (d,d)
    - I: numpy array of shape (m,2) with integer node pairs, or None/empty -> NaN
    """
    if I is None:
        return float("nan")
    I = np.asarray(I)
    if I.size == 0:
        return float("nan")
    if I.ndim != 2 or I.shape[1] != 2:
        raise ValueError(f"I must have shape (m,2), got {I.shape}")

    Wt = torch.as_tensor(W_est, dtype=torch.float32)
    It = torch.as_tensor(I, dtype=torch.long)

    with torch.no_grad():
        val = pst(Wt, It, "exp", agg="mean")
    # pst returns a torch scalar tensor
    return float(val.item())
    

def find_best_threshold_for_shd(
    W_true: np.ndarray,
    W_est: np.ndarray,
    *,
    thr_grid: np.ndarray | None = None,
    max_candidates: int = 50,
    include: tuple[float, ...] = (0.0, 1e-12),
) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (thr_best, acc_best) where thr_best minimizes SHD.
    Ties are broken by smaller thr (keeps more edges; stable).
    """
    d = W_true.shape[0]
    assert W_true.shape == W_est.shape

    # Prepare candidates: thresholds at abs(W_est) values (off-diagonal)
    A = np.abs(np.asarray(W_est))
    np.fill_diagonal(A, 0.0)
    vals = A[A > 0].ravel()

    if thr_grid is None:
        if vals.size == 0:
            # Nothing to threshold; only 0 makes sense
            thr_candidates = np.array(list(include), dtype=float)
        else:
            uniq = np.unique(vals)
            # subsample if too many unique values
            if uniq.size > max_candidates:
                # take quantiles (includes smallest/largest)
                qs = np.linspace(0, 0.3, max_candidates)
                uniq = np.quantile(uniq, qs)
                uniq = np.unique(uniq)
            thr_candidates = np.concatenate([np.array(list(include), dtype=float), uniq])
    else:
        thr_candidates = np.asarray(thr_grid, dtype=float)

    # Ensure nonnegative sorted unique
    thr_candidates = np.unique(np.clip(thr_candidates, 0.0, np.inf))
    thr_candidates.sort()

    # Evaluate SHD efficiently (reuse B_true)
    B_true = binarize(W_true, thr=0.0)  # W_true edges are usually exact; 0 is fine

    best_thr = float(thr_candidates[0]) if thr_candidates.size else 0.0
    best_shd = None
    best_acc: Dict[str, Any] = {}

    for thr in thr_candidates:
        # binarize estimate at thr, compute shd
        B_hat = binarize(W_est, thr=float(thr))
        s = shd(B_true, B_hat)

        if best_shd is None or s < best_shd or (s == best_shd and thr < best_thr):
            best_shd = int(s)
            best_thr = float(thr)
            best_acc = evaluate_structure(B_true, W_true, W_est, thr=best_thr)

            # Early stop: can't beat 0
            if best_shd == 0:
                break

    return best_thr, best_acc

def evaluate_structure(
    B_true: np.ndarray,
    W_true: np.ndarray,
    W_est: np.ndarray,
    *,
    I: np.ndarray = None,
    thr: float = 1e-12,
) -> Dict[str, Any]:
    """
    Evaluate W_est against W_true using basic structure metrics.
    """
    B_hat = binarize(W_est, thr=thr)

    out: Dict[str, Any] = {}
    out["nnz_true"] = count_edges(B_true)
    out["nnz_est"] = count_edges(B_hat)
    out["shd"] = shd(B_true, B_hat)
    out["fhd"] = fhd(B_true, B_hat)
    out["fro_rel"] = fro_error_rel(W_true, W_est)
    out["pst_exp"] = pst_exp(W_est, I)
    
    out.update(rates(B_true, B_hat))
    return out

