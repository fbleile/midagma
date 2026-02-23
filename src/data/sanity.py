from src.utils.yaml import (
    load_yaml, save_yaml,
    expand_on_keys, grid_choice_filename,
    )
from definitions import DEFAULT_GRAPH_TYPES, DEFAULT_LINEAR_SEM_TYPES, DEFAULT_NONLINEAR_SEM_TYPES, DEFAULT_SEM_TYPES, DATA_GRID_KEYS


def _is_acyclic_graph_type(gt: str) -> bool:
    return gt.endswith("_acyclic")


def _is_nonlinear_sem(st: str) -> bool:
    return st in DEFAULT_NONLINEAR_SEM_TYPES


def _is_linear_sem(st: str) -> bool:
    return st in DEFAULT_LINEAR_SEM_TYPES


def _sanity_check_data(cfg: dict) -> tuple[bool, list[str]]:
    """
    Sanity-check a *resolved* (grid-expanded) data config.

    Supports either:
      - explicit (n, s0), or
      - derived-input knobs epv and/or n_alpha (where n_alpha = n / (epv * d))

    Precedence:
      - If epv is present and s0 missing -> derive s0 = round(epv * d)
      - If n_alpha is present and n missing -> derive n = round(n_alpha * epv * d)
        (epv inferred from s0/d if epv missing but s0 present)
      - If n/s0 explicitly present, we use them (no override)
    """
    reasons: list[str] = []

    # --- pull raw fields (may be missing) ---
    d_raw = cfg.get("d")
    n_raw = cfg.get("n")
    s0_raw = cfg.get("s0")
    epv_raw = cfg.get("epv")
    n_alpha_raw = cfg.get("n_alpha")

    gt = str(cfg.get("graph_type", ""))
    st = str(cfg.get("sem_type", ""))
    is_acyclic = _is_acyclic_graph_type(gt)

    # --- basic presence for d ---
    try:
        d = int(d_raw)
    except Exception:
        d = 0
    if d <= 0:
        reasons.append(f"bad basic dims: d={d_raw} (need int >0)")
        return False, reasons  # cannot proceed meaningfully

    # --- derive/validate epv and s0 ---
    epv: float | None = None
    if epv_raw is not None:
        try:
            epv = float(epv_raw)
        except Exception:
            reasons.append(f"bad epv: epv={epv_raw} (need float)")
    if epv is not None and epv <= 0:
        reasons.append(f"bad epv: epv={epv} (need >0)")

    s0: int | None = None
    if s0_raw is not None:
        try:
            s0 = int(s0_raw)
        except Exception:
            reasons.append(f"bad sparsity: s0={s0_raw} (need int >=0)")

    # If s0 not provided but epv is, derive s0
    if s0 is None and epv is not None:
        s0 = int(round(epv * d))

    # If s0 still unknown, reject (need either s0 or epv)
    if s0 is None:
        reasons.append("missing sparsity: provide s0 or epv (edges-per-var)")
    else:
        if s0 < 0:
            reasons.append(f"bad sparsity: s0={s0} (need >=0)")

    # If epv missing but s0 known, infer epv for later checks
    if epv is None and s0 is not None and d > 0:
        epv = float(s0) / float(d)

    # --- derive/validate n from n_alpha if needed ---
    n_alpha: float | None = None
    if n_alpha_raw is not None:
        try:
            n_alpha = float(n_alpha_raw)
        except Exception:
            reasons.append(f"bad n_alpha: n_alpha={n_alpha_raw} (need float)")
    if n_alpha is not None and n_alpha <= 0:
        reasons.append(f"bad n_alpha: n_alpha={n_alpha} (need >0)")

    n: int | None = None
    if n_raw is not None:
        try:
            n = int(n_raw)
        except Exception:
            reasons.append(f"bad sample size: n={n_raw} (need int >0)")

    # If n not provided but n_alpha is, derive n (requires epv)
    if n is None and n_alpha is not None:
        if epv is None or epv <= 0:
            reasons.append("cannot derive n from n_alpha: epv is missing/invalid (provide epv or s0)")
        else:
            n = int(round(n_alpha * epv * d))

    # If n still unknown, reject (need either n or n_alpha)
    if n is None:
        reasons.append("missing sample size: provide n or n_alpha")
    else:
        if n <= 0:
            reasons.append(f"bad sample size: n={n} (need >0)")

    # --- quick return if already broken ---
    if reasons:
        return False, reasons

    # (C) sample vs dimension (updated to use n_alpha if present)
    # old rule was: n < max(100, 5*d)
    # new: compare n_alpha = n/(epv*d) (only meaningful if epv>0)
    # We'll keep the old rule as a weak floor and add a n_alpha-based check.
    if n < max(100, 5 * d):
        reasons.append(f"too few samples (absolute): n={n} < max(100, 5*d={5*d})")

    # n_alpha-based regime check (only if we have epv)
    if epv is not None and epv > 0:
        alpha_eff = n / (epv * d)
        # thresholds: few < 2, moderate 2..5, many > 5 (your earlier convention)
        if alpha_eff < 2.0:
            reasons.append(
                f"too few samples (scaled): n_alpha={alpha_eff:.2f} < 2.0 "
                f"(n={n}, epv={epv:.2f}, d={d})"
            )

    # (B) density (proxy)
    if d <= 2:
        reasons.append(f"too few variables: d={d} (need >2)")
    else:
        if is_acyclic:
            max_edges = d * (d - 1) // 2
        else:
            max_edges = d * (d - 1)

        if s0 > max_edges:
            reasons.append(f"s0={s0} exceeds theoretical max_edges={max_edges}")

        density = s0 / max_edges if max_edges > 0 else 1.0

        # only reject near-complete graphs
        if is_acyclic and density > 0.75:
            reasons.append(
                f"too dense: density={density:.2f} (>0.75), s0={s0}, max_edges={max_edges}"
            )
        elif not is_acyclic and density > 0.40:
            reasons.append(
                f"too dense: density={density:.2f} (>0.40), s0={s0}, max_edges={max_edges}"
            )

    # (A) graph <-> sem coupling
    # linear cyclic: only gauss supported
    if not is_acyclic and not _is_nonlinear_sem(st) and st != "gauss":
        reasons.append(
            f"cyclic graph requires sem_type='gauss' for linear SEM, got sem_type='{st}'"
        )

    return len(reasons) == 0, reasons