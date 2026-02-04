# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


PathLike = Union[str, Path]


# -------------------------------------------------------
# load
# -------------------------------------------------------

def load_yaml(path: PathLike) -> Dict[str, Any]:
    """
    Load YAML → dict.

    Always returns a dict (empty if file empty).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return {} if data is None else data


# -------------------------------------------------------
# save
# -------------------------------------------------------

def save_yaml(obj: Dict[str, Any], path: PathLike) -> Path:
    """
    Save dict → YAML (pretty, deterministic).

    - creates parent dirs
    - stable key order
    - human readable
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            obj,
            f,
            sort_keys=True,   # reproducible
            default_flow_style=False,
            allow_unicode=True,
        )

    return path
