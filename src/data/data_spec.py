# -*- coding: utf-8 -*-
# src/data/data_spec.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import math

from src.data.indep_spec import ISpec
from src.utils.yaml import *

# -----------------------------
# Core specs
# -----------------------------

@dataclass(frozen=True)
class DataSpec:
    n: int = 500
    d: int = 10

    # Either specify s0 directly OR specify epv and derive s0 = round(epv * d)
    s0: int = 40
    epv: Optional[float] = None   # edges-per-variable (short key for YAML)

    graph_type: str = "ER"
    sem_type: str = "lin_gauss"
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

def _compute_s0_from_epv(epv: float, d: int) -> int:
    # Choose rounding policy; round is usually OK.
    # You can also use math.floor/ceil depending on what you want.
    return int(round(epv * d))

def load_data_config_to_spec(path: Path) -> tuple[str, int, DataSpec, ISpec]:
    cfg = load_yaml(path)
    cfg_id = path.parent.name

    data_keys = set(DataSpec.__dataclass_fields__.keys())
    data_dict = _pick(cfg, data_keys)

    if "w_ranges" in cfg:
        data_dict["w_ranges"] = _parse_w_ranges(cfg.get("w_ranges"))

    # --- derive s0 from epv if provided and s0 not explicitly provided ---
    epv = cfg.get("epv", None)  # allow short key
    if epv is not None and "s0" not in cfg:
        d_val = cfg.get("d")
        if isinstance(d_val, int):
            data_dict["s0"] = _compute_s0_from_epv(float(epv), d_val)
        else:
            raise ValueError(f"Cannot derive s0 from epv when d is not an int (got {type(d_val)}). Expand grid first.")

    data_spec = DataSpec(**data_dict)

    # --- ISpec from nested independencies ---
    indep_cfg = cfg.get("independencies", {})
    i_spec = ISpec(**indep_cfg)

    return cfg_id, data_spec, i_spec


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


def save_data_config(path: Path, data_spec: DataSpec, i_spec: ISpec) -> None:
    save_yaml(to_yaml_dict(data_spec, i_spec), path)

