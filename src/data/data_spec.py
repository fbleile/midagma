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
    # core dataset shape + graph
    n: int = 500
    d: int = 10
    s0: int = 40

    # derived-input knobs (optional)
    epv: Optional[float] = None        # edges per variable: s0 ≈ epv * d
    n_alpha: Optional[float] = None    # n_alpha = n / (epv * d)

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

def _s0_from_epv(epv: float, d: int) -> int:
    return int(round(epv * d))

def _n_from_nalpha(n_alpha: float, epv: float, d: int) -> int:
    return int(round(n_alpha * epv * d))

def load_data_config_to_spec(path: Path) -> tuple[str, int, DataSpec, ISpec]:
    cfg = load_yaml(path)
    cfg_id = path.parent.name

    data_keys = set(DataSpec.__dataclass_fields__.keys())
    data_dict = _pick(cfg, data_keys)

    if "w_ranges" in cfg:
        data_dict["w_ranges"] = _parse_w_ranges(cfg.get("w_ranges"))

    # --- derive s0 from epv if needed ---
    if "s0" not in cfg and cfg.get("epv") is not None:
        d_val = cfg.get("d")
        if not isinstance(d_val, int):
            raise ValueError("Deriving s0 from epv requires concrete d (int). Expand grid first.")
        data_dict["s0"] = _s0_from_epv(float(cfg["epv"]), d_val)

    # --- derive n from n_alpha if needed ---
    if "n" not in cfg and cfg.get("n_alpha") is not None:
        d_val = cfg.get("d")
        if not isinstance(d_val, int):
            raise ValueError("Deriving n from n_alpha requires concrete d (int). Expand grid first.")

        epv = cfg.get("epv")
        if epv is None:
            # allow epv to be inferred from s0 if s0 exists
            s0_val = data_dict.get("s0", cfg.get("s0"))
            if s0_val is None:
                raise ValueError("n_alpha provided but neither epv nor s0 available to infer epv.")
            epv = float(s0_val) / float(d_val)

        data_dict["n"] = _n_from_nalpha(float(cfg["n_alpha"]), float(epv), d_val)

    data_spec = DataSpec(**data_dict)
    
    print(data_spec)

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

