# -*- coding: utf-8 -*-
# src/data/data_spec.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from src.data.indep_spec import ISpec
from src.utils.yaml import *

# -----------------------------
# Core specs
# -----------------------------
@dataclass(frozen=True)
class DataSpec:
    # core dataset shape + graph
    n: int = 500
    d: int = 10
    s0: int = 40                 # IMPORTANT: we treat this as TOTAL number of edges
    graph_type: str = "ER"

    # SEM type (you use e.g. "lin_gauss")
    sem_type: str = "lin_gauss"

    # linear SEM parameters
    noise_scale: float = 1.0
    w_ranges: Tuple[Tuple[float, float], ...] = ((-2.0, -0.5), (0.5, 2.0))


def _pick(cfg: Dict[str, Any], keys: set[str]) -> Dict[str, Any]:
    return {k: cfg[k] for k in keys if k in cfg}


def _parse_w_ranges(x: Any) -> Tuple[Tuple[float, float], ...]:
    """
    Accept YAML list-of-lists like:
      w_ranges:
        - [-2.0, -0.5]
        - [0.5, 2.0]
    """
    if x is None:
        return ((-2.0, -0.5), (0.5, 2.0))
    if not isinstance(x, list) or not all(isinstance(t, (list, tuple)) and len(t) == 2 for t in x):
        raise ValueError("w_ranges must be a list of pairs, e.g. [[-2.0,-0.5],[0.5,2.0]]")
    out: List[Tuple[float, float]] = []
    for low, high in x:
        out.append((float(low), float(high)))
    return tuple(out)


def load_data_config(path: Path) -> tuple[str, int, DataSpec, ISpec]:
    """
    Works with THIS YAML shape (flat top-level keys):
      id, n_datasets, n, d, s0, graph_type, sem_type, noise_scale, w_ranges,
      independencies: {...},
    """
    cfg = load_yaml(path)

    cfg_id = str(cfg.get("id", path.parent.name))
    n_datasets = int(cfg.get("n_datasets", 1))

    # --- DataSpec from top-level keys ---
    data_keys = set(DataSpec.__dataclass_fields__.keys())
    data_dict = _pick(cfg, data_keys)

    # w_ranges needs conversion from list->tuple[tuple[float,float],...]
    if "w_ranges" in cfg:
        data_dict["w_ranges"] = _parse_w_ranges(cfg.get("w_ranges"))

    data_spec = DataSpec(**data_dict)

    # --- ISpec from nested independencies ---
    indep_cfg = cfg.get("independencies", {})
    i_spec = ISpec(**indep_cfg)

    return cfg_id, n_datasets, data_spec, i_spec


def to_yaml_dict(cfg_id: str, n_datasets: int, data_spec: DataSpec, i_spec: ISpec) -> Dict[str, Any]:
    """
    Produce a YAML dict in the SAME format as your example.
    """
    d = {
        "id": cfg_id,
        "n_datasets": int(n_datasets),
        **data_spec.__dict__,
        "w_ranges": [list(t) for t in data_spec.w_ranges],
        "independencies": dict(i_spec.__dict__),
    }
    return d


def save_data_config(path: Path, cfg_id: str, n_datasets: int, data_spec: DataSpec, i_spec: ISpec) -> None:
    save_yaml(to_yaml_dict(cfg_id, n_datasets, data_spec, i_spec), path)

