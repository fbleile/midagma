# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

from definitions import PROJECT_DIR, DATA_DIR

# def descr_root(path_data: Path, descr: str) -> Path:
#     return path_data / descr

# def run_root(path_data: Path, descr: str) -> Path:
#     return descr_root(path_data, descr) / SUBDIR_RUNS_INSIDE_DESCR

# def seed_folder(path_data: Path, descr: str, seed: int, grid_key: Optional[str] = None) -> Path:
#     base = run_root(path_data, descr)
#     if grid_key is None:
#         return base / f"seed_{seed}"
#     return base / f"{grid_key}-seed_{seed}"

# def split_folder(path_data: Path, descr: str, seed: int, split: str, grid_key: Optional[str] = None) -> Path:
#     sf = seed_folder(path_data, descr, seed, grid_key)
#     if split == "train":
#         return sf / FOLDER_TRAIN
#     if split == "test":
#         return sf / FOLDER_TEST
#     raise ValueError("split must be 'train' or 'test'")
    
    
def resolve_from_project(path: Union[str, Path]) -> Path:
    """
    If `path` is relative, interpret it relative to PROJECT_DIR.
    If absolute, keep as-is.
    """
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_DIR / p)