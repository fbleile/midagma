# -*- coding: utf-8 -*-

from __future__ import annotations
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Union
import yaml

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(d: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False)

@contextmanager
def timer():
    t0 = time.time()
    yield lambda: (time.time() - t0)
