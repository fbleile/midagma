# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union
from src.utils.yaml import save_yaml, load_yaml

import numpy as np

from definitions import (
    FILE_X,
    FILE_B_TRUE,
    FILE_W_TRUE,
    FILE_I,
    FILE_META,
    TRUE_DIR,
)


@dataclass(frozen=True)
class Dataset:
    """
    In-memory representation of one generated dataset + its metadata.

    Stored layout:
      datasets/<dataset_id>/
        X.npy
        meta/
          meta.yaml
          B_true.npy
          W_true.npy
          I_full.npy
          I.npy
    """
    X: np.ndarray
    B_true: np.ndarray
    W_true: np.ndarray
    I_full: np.ndarray
    I: np.ndarray
    meta: Dict[str, Any]

    # --------- convenience ---------
    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    # --------- IO helpers ---------
    @staticmethod
    def paths(ds_root: Union[str, Path]) -> Dict[str, Path]:
        ds_root = Path(ds_root)
        true_dir = ds_root / TRUE_DIR
        return {
            "ds_root": ds_root,
            "true_dir": true_dir,
            "X": ds_root / "X.npy",
            "B_true": true_dir / "B_true.npy",
            "W_true": true_dir / "W_true.npy",
            "I_full": true_dir / "I_full.npy",
            "I": true_dir / "I.npy",
            "meta_yaml": ds_root / "meta.yaml",
        }

    def save(self, ds_root: Union[str, Path], *, overwrite: bool = False) -> Path:

        P = self.paths(ds_root)
        ds_root = P["ds_root"]
        true_dir = P["true_dir"]

        if ds_root.exists() and (not overwrite):
            return ds_root

        ds_root.mkdir(parents=True, exist_ok=True)
        true_dir.mkdir(parents=True, exist_ok=True)

        np.save(P["X"], self.X)
        np.save(P["B_true"], self.B_true.astype(np.int64, copy=False))
        np.save(P["W_true"], self.W_true.astype(np.float64, copy=False))
        np.save(P["I_full"], self.I_full.astype(np.int64, copy=False))
        np.save(P["I"], self.I.astype(np.int64, copy=False))

        save_yaml(self.meta, P["meta_yaml"])
        return ds_root

    @classmethod
    def load(cls, ds_root: Union[str, Path]) -> "Dataset":
        P = cls.paths(ds_root)
        X = np.load(P["X"], allow_pickle=False)
        B_true = np.load(P["B_true"], allow_pickle=False)
        W_true = np.load(P["W_true"], allow_pickle=False)
        I_full = np.load(P["I_full"], allow_pickle=False)
        I = np.load(P["I"], allow_pickle=False)
        meta = load_yaml(P["meta_yaml"])
        return cls(X=X, B_true=B_true, W_true=W_true, I_full=I_full, I=I, meta=meta)
