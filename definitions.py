# -*- coding: utf-8 -*-

# definitions.py
from __future__ import annotations

from pathlib import Path
import os

# ---------- hardware ----------
try:
    import psutil  # optional
    try:
        CPU_COUNT = len(psutil.Process().cpu_affinity())
    except AttributeError:
        CPU_COUNT = psutil.cpu_count(logical=True) or os.cpu_count() or 1
except Exception:
    CPU_COUNT = os.cpu_count() or 1


# ---------- project roots ----------
# file is src/definitions.py -> project root is parents[1] if src is directly under root
PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR  # synonym

# heuristic for cluster detection
IS_CLUSTER = Path("/cluster").is_dir() or bool(os.environ.get("SLURM_JOB_ID", ""))

# ---------- top-level repo subdirs ----------
SUBDIR_CONFIGS = "configs"
SUBDIR_DATA = "data"
SUBDIR_RUNS = "runs"
SUBDIR_RESULTS = "results"
SUBDIR_ASSETS = "assets"
SUBDIR_SLURM_LOGS = "slurm_logs"
SUBDIR_SRC = "src"

CONFIG_DIR = PROJECT_DIR / SUBDIR_CONFIGS
DATA_DIR = PROJECT_DIR / SUBDIR_DATA
RUNS_DIR = PROJECT_DIR / SUBDIR_RUNS
RESULTS_DIR = PROJECT_DIR / SUBDIR_RESULTS
ASSETS_DIR = PROJECT_DIR / SUBDIR_ASSETS
SRC_DIR = PROJECT_DIR / SUBDIR_SRC
SLURM_LOGS_DIR = PROJECT_DIR / SUBDIR_SLURM_LOGS

# ---------- experiment structure ----------
SUBDIR_EXPERIMENTS = SRC_DIR / "experiments"
CONFIG_DATA = "data.yaml"
CONFIG_DATA_GRID = "grid.yaml"
CONFIG_METHODS = "methods.yaml"
ONFIG_METHODS_VALIDATION = "methods_validation.yaml"

# ---------- canonical filenames (numpy storage) ----------
FILE_X = "X.npy"
FILE_B_TRUE = "B_true.npy"
FILE_W_TRUE = "W_true.npy"
FILE_I = "I.npy"
FILE_META = "meta.yaml"
FILE_SPEC_USED = "data_config.used.yaml"

# if you later store CSVs / json:
FILE_SUMMARY_CSV = "summary.csv"
FILE_RESULTS_PARQUET = "results.parquet"

# ---------- standard config filenames inside each config folder ----------
CONFIG_DATA = "data_config.yaml"
CONFIG_METHODS = "methods.yaml"
CONFIG_METHODS_VALIDATION = "methods_validation.yaml"

# ---------- sanity bounds (optional) ----------
NAN_MIN = -1e12
NAN_MAX = +1e12

# ---------- default dataset families (optional convenience) ----------
DEFAULT_DATA_GEN_TYPES = [
    "lin_sem_er",
    "lin_sem_er_acyclic",
    "lin_sem_scale_free",
    "lin_sem_sbm",
    "nonlin_sem_er",
]
