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
    reasons: list[str] = []

    n = int(cfg.get("n", 0))
    d = int(cfg.get("d", 0))
    s0 = int(cfg.get("s0", 0))
    gt = str(cfg.get("graph_type", ""))
    st = str(cfg.get("sem_type", ""))
    is_acyclic = _is_acyclic_graph_type(gt)

    # basic presence
    if n <= 0 or d <= 0:
        reasons.append(f"bad basic dims: n={n}, d={d} (need >0)")
    if s0 < 0:
        reasons.append(f"bad sparsity: s0={s0} (need >=0)")

    # (C) sample vs dimension
    if n < max(100, 5 * d):
        reasons.append(f"too few samples: n={n} < max(100, 5*d={5*d})")

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
                f"too dense: density={density:.2f} (>0.7), "
                f"s0={s0}, max_edges={max_edges}"
            )
        elif not is_acyclic and density > 0.4:
            reasons.append(
                f"too dense: density={density:.2f} (>0.7), "
                f"s0={s0}, max_edges={max_edges}"
            )
    # (A) graph <-> sem coupling
    # linear cyclic: only gauss supported
    if not is_acyclic and not _is_nonlinear_sem(st) and st != "gauss":
        reasons.append(f"cyclic graph requires sem_type='gauss' for linear SEM, got sem_type='{st}'")

    return len(reasons) == 0, reasons