# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from treksbench.utils.parse import save_yaml

def save_data_folder(folder: Path, arrays: Dict[str, np.ndarray], meta: Optional[Dict[str, Any]] = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(folder / f"{k}.npy", v)
    if meta is not None:
        save_yaml(meta, folder / "meta.yaml")
