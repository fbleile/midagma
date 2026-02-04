# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Literal, Dict, Any

import numpy as np
from scipy import stats



@dataclass(frozen=True)
class IndepTestResult:
    i: int
    j: int
    stat: float
    pvalue: float


def _center_gram(K: np.ndarray) -> np.ndarray:
    """Double-center a Gram matrix: H K H."""
    n = K.shape[0]
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    all_mean = K.mean()
    return K - row_mean - col_mean + all_mean


def _rbf_gram(x: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """
    RBF Gram matrix for 1D array x.
    sigma: bandwidth. If None, use median heuristic on pairwise distances.
    """
    x = np.asarray(x).reshape(-1, 1)
    # squared distances
    D2 = (x - x.T) ** 2

    if sigma is None:
        # median heuristic (on off-diagonal distances)
        off = D2[np.triu_indices(D2.shape[0], k=1)]
        med = np.median(off)
        # if constant variable, med=0 -> avoid division by zero
        sigma2 = med if med > 0 else 1.0
    else:
        sigma2 = float(sigma) ** 2
        if sigma2 <= 0:
            sigma2 = 1.0

    return np.exp(-D2 / (2.0 * sigma2))


def hsic_stat(x: np.ndarray, y: np.ndarray, sigma_x: Optional[float] = None, sigma_y: Optional[float] = None) -> float:
    """
    Biased HSIC estimator (sufficient for permutation tests):
      HSIC = (1/n^2) * sum( Kc * Lc )
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = x.shape[0]
    K = _rbf_gram(x, sigma=sigma_x)
    L = _rbf_gram(y, sigma=sigma_y)
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    return float((Kc * Lc).sum() / (n * n))


def _dcor_centered_dist(A: np.ndarray) -> np.ndarray:
    """
    Double-center distance matrix A:
      A_ij - mean_row_i - mean_col_j + mean_all
    """
    row = A.mean(axis=1, keepdims=True)
    col = A.mean(axis=0, keepdims=True)
    allm = A.mean()
    return A - row - col + allm


def dcor_stat(x: np.ndarray, y: np.ndarray) -> float:
    """
    Distance correlation statistic based on distance covariance.
    Returns dCor in [0,1] (up to numerical issues).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = x.shape[0]

    # pairwise absolute distances (1D)
    ax = np.abs(x[:, None] - x[None, :])
    ay = np.abs(y[:, None] - y[None, :])

    Ax = _dcor_centered_dist(ax)
    Ay = _dcor_centered_dist(ay)

    dcov2 = (Ax * Ay).sum() / (n * n)
    dvarx2 = (Ax * Ax).sum() / (n * n)
    dvary2 = (Ay * Ay).sum() / (n * n)

    if dvarx2 <= 0 or dvary2 <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(np.sqrt(dvarx2 * dvary2)))


def permutation_pvalue(
    stat_fn,
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_perm: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Permutation test p-value for dependence statistic.
    Permute y (equivalently permute pairing), recompute statistic.
    Returns (stat_obs, pvalue).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if rng is None:
        rng = np.random.default_rng(0)

    stat_obs = float(stat_fn(x, y))
    n = x.shape[0]

    # permutation distribution
    ge = 0
    for _ in range(num_perm):
        perm = rng.permutation(n)
        stat_perm = float(stat_fn(x, y[perm]))
        if stat_perm >= stat_obs:
            ge += 1

    # +1 smoothing for valid p-values
    p = (ge + 1) / (num_perm + 1)
    return stat_obs, float(p)

def pearson_stat_pvalue(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Pearson correlation test.
    Returns (|r|, pvalue).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    r, p = stats.pearsonr(x, y)
    return float(abs(r)), float(p)


def spearman_stat_pvalue(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Spearman rank correlation test.
    Returns (|rho|, pvalue).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    rho, p = stats.spearmanr(x, y)
    # scipy can return nan if inputs are constant
    if not np.isfinite(rho) or not np.isfinite(p):
        return 0.0, 1.0
    return float(abs(rho)), float(p)


TestName = Literal["hsic", "dcor", "pearson", "spearman"]


def test_pairwise_independence(
    X: np.ndarray,
    pairs: Iterable[Tuple[int, int]],
    *,
    test: TestName = "hsic",
    num_perm: int = 200,
    seed: int = 0,
) -> List[IndepTestResult]:
    """
    Run a bivariate independence test and return (stat, pvalue) per pair.

    - hsic, dcor: permutation p-values (num_perm, seed used)
    - pearson, spearman: analytic p-values from scipy (num_perm ignored)
    """
    X = np.asarray(X)
    pairs = list(pairs)

    rng = np.random.default_rng(seed)

    out: List[IndepTestResult] = []

    if test == "hsic":
        stat_fn = hsic_stat
        for (i, j) in pairs:
            x = X[:, i]
            y = X[:, j]
            stat, p = permutation_pvalue(stat_fn, x, y, num_perm=num_perm, rng=rng)
            out.append(IndepTestResult(i=i, j=j, stat=float(stat), pvalue=float(p)))
        return out

    if test == "dcor":
        stat_fn = dcor_stat
        for (i, j) in pairs:
            x = X[:, i]
            y = X[:, j]
            stat, p = permutation_pvalue(stat_fn, x, y, num_perm=num_perm, rng=rng)
            out.append(IndepTestResult(i=i, j=j, stat=float(stat), pvalue=float(p)))
        return out

    if test == "pearson":
        for (i, j) in pairs:
            stat, p = pearson_stat_pvalue(X[:, i], X[:, j])
            out.append(IndepTestResult(i=i, j=j, stat=stat, pvalue=p))
        return out

    if test == "spearman":
        for (i, j) in pairs:
            stat, p = spearman_stat_pvalue(X[:, i], X[:, j])
            out.append(IndepTestResult(i=i, j=j, stat=stat, pvalue=p))
        return out

    raise ValueError("test must be one of 'hsic', 'dcor', 'pearson', 'spearman'")


def get_I_from_full_pairwise_tests(
    X: np.ndarray,
    *,
    alpha: float = 0.05,
    test: TestName = "hsic",
    num_perm: int = 200,
    seed: int = 0,
    bonferroni: bool = True,
    exclude_diagonal: bool = True,
) -> np.ndarray:
    """
    Test all pairs and return I = {(i,j): fail-to-reject independence}, i.e. p > alpha_eff.

    NOTE: permutation p-values are expensive; num_perm controls runtime.
    """
    X = np.asarray(X)
    n, d = X.shape

    pairs: List[Tuple[int, int]] = []
    for i in range(d):
        for j in range(i + 1, d):
            pairs.append((i, j))

    results = test_pairwise_independence(X, pairs, test=test, num_perm=num_perm, seed=seed)

    m = len(results)
    alpha_eff = (alpha / m) if (bonferroni and m > 0) else alpha

    I: List[Tuple[int, int]] = []
    for r in results:
        if r.pvalue > alpha_eff:
            I.append((r.i, r.j))

    return np.asarray(I, dtype=int)


def summarize_I(I: np.ndarray, d: int, max_show: int = 10) -> None:
    I = np.asarray(I, dtype=int)
    print(f"I size: {len(I)} pairs (d={d})")
    if len(I) == 0:
        return
    print("first pairs:", I[:max_show].tolist(), ("..." if len(I) > max_show else ""))


def _sanity_check():
    """
    Sanity check for:
      - HSIC / dCor: should detect general nonlinear dependence (incl non-monotone)
      - Pearson: should detect linear dependence
      - Spearman: should detect monotone dependence

    We therefore run two groups of scenarios:
      Group 1 (general nonlinear): only HSIC/dCor are expected to always succeed.
      Group 2 (linear/monotone): Pearson/Spearman should succeed too.
    """
    rng = np.random.default_rng(123)

    alpha = 0.05
    seed = 0
    num_perm = 400

    # ---------- helper ----------
    def pvals_for(X, test_name):
        pairs = [(0, 1), (0, 2), (1, 2)]
        res = test_pairwise_independence(X, pairs, test=test_name, num_perm=num_perm, seed=seed)
        return {(r.i, r.j): r.pvalue for r in res}

    def must_dep(p, pair, test_name, scenario):
        if not (p[pair] < alpha):
            raise AssertionError(f"[{test_name}][{scenario}] expected DEP for {pair}, got p={p[pair]:.3g}")

    def must_indep(p, pair, test_name, scenario):
        if not (p[pair] > alpha):
            raise AssertionError(f"[{test_name}][{scenario}] expected INDEP for {pair}, got p={p[pair]:.3g}")

    # ============================================================
    # Group 1: general nonlinear (HSIC/dCor expected to shine)
    # ============================================================

    n1 = 200

    def sc_nonmono_chain():
        # x1 -> x2 (non-monotone), x3 independent
        x1 = rng.standard_normal(n1)
        x2 = np.sin(3.0 * x1) + 0.15 * rng.standard_normal(n1)
        x3 = rng.standard_normal(n1)
        return np.column_stack([x1, x2, x3])

    def sc_two_parents_indep_parents():
        # x1 ⟂ x2, both -> x3
        x1 = rng.standard_normal(n1)
        x2 = rng.standard_normal(n1)
        x3 = np.tanh(x1) + (x2**2 - np.mean(x2**2)) + 0.20 * rng.standard_normal(n1)
        return np.column_stack([x1, x2, x3])

    def sc_full():
        x1 = rng.standard_normal(n1)
        x2 = np.sin(3.0 * x1) + 0.15 * rng.standard_normal(n1)
        x3 = (x1 * x2) + np.cos(x2) + 0.25 * rng.standard_normal(n1)
        return np.column_stack([x1, x2, x3])

    nonlinear_scenarios = [
        ("NL_chain_nonmonotone", sc_nonmono_chain, {"dep": [(0, 1)], "indep": [(0, 2), (1, 2)]}),
        ("NL_two_parents", sc_two_parents_indep_parents, {"dep": [(0, 2), (1, 2)], "indep": [(0, 1)]}),
        ("NL_full", sc_full, {"dep": [(0, 1), (0, 2), (1, 2)], "indep": []}),
    ]

    # ============================================================
    # Group 2: linear + monotone (Pearson/Spearman should succeed)
    # ============================================================

    n2 = 400

    def sc_linear_chain():
        # x1 -> x2 linear, x3 independent
        x1 = rng.standard_normal(n2)
        x2 = 0.9 * x1 + 0.2 * rng.standard_normal(n2)
        x3 = rng.standard_normal(n2)
        return np.column_stack([x1, x2, x3])

    def sc_monotone_chain():
        # x1 -> x2 monotone nonlinear, x3 independent
        x1 = rng.standard_normal(n2)
        x2 = np.tanh(2.0 * x1) + 0.15 * rng.standard_normal(n2)
        x3 = rng.standard_normal(n2)
        return np.column_stack([x1, x2, x3])

    easy_scenarios = [
        ("LIN_chain", sc_linear_chain, {"dep": [(0, 1)], "indep": [(0, 2), (1, 2)]}),
        ("MONO_chain", sc_monotone_chain, {"dep": [(0, 1)], "indep": [(0, 2), (1, 2)]}),
    ]

    print("=== mi_tests sanity check (HSIC/dCor + Pearson/Spearman) ===")
    print(f"alpha={alpha}, num_perm={num_perm}")
    print("Note: Pearson/Spearman are NOT expected to always detect non-monotone nonlinear dependence.\n")

    # ---- HSIC & dCor should pass all nonlinear scenarios ----
    for test_name in ["hsic", "dcor"]:
        print(f"--- test = {test_name} (general nonlinear) ---")
        for name, gen, expect in nonlinear_scenarios:
            X = gen()
            p = pvals_for(X, test_name)
            print(f"{name}: p01={p[(0,1)]:.3f}, p02={p[(0,2)]:.3f}, p12={p[(1,2)]:.3f}")
            for pair in expect["dep"]:
                must_dep(p, pair, test_name, name)
            for pair in expect["indep"]:
                must_indep(p, pair, test_name, name)
            print(f"  [OK] {name}")
        print()

    # ---- Pearson & Spearman should pass the easy scenarios ----
    for test_name in ["pearson", "spearman"]:
        print(f"--- test = {test_name} (linear/monotone) ---")
        for name, gen, expect in easy_scenarios:
            X = gen()
            p = pvals_for(X, test_name)
            print(f"{name}: p01={p[(0,1)]:.3f}, p02={p[(0,2)]:.3f}, p12={p[(1,2)]:.3f}")
            for pair in expect["dep"]:
                must_dep(p, pair, test_name, name)
            for pair in expect["indep"]:
                must_indep(p, pair, test_name, name)
            print(f"  [OK] {name}")
        print()

    print("=== done ===")


if __name__ == "__main__":
    _sanity_check()
