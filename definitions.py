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

LOCAL_STORE_DIR = Path("/cluster/path/to/anonymous/directory")
CLUSTER_GROUP_DIR = Path("/cluster/path/to/anonymous/directory")
CLUSTER_SCRATCH_DIR = Path("/cluster/path/to/anonymous/directory")

IS_CLUSTER = Path("/cluster").is_dir() or bool(os.environ.get("SLURM_JOB_ID", ""))
STORE_DIR = CLUSTER_GROUP_DIR if IS_CLUSTER else LOCAL_STORE_DIR
SCRATCH_STORE_DIR = CLUSTER_SCRATCH_DIR if IS_CLUSTER else LOCAL_STORE_DIR

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
CONFIG_DATA = "data.yaml"
CONFIG_DATA_GRID = "data_grid.yaml"
CONFIG_METHODS = "methods.yaml"
CONFIG_METHODS_BASELINE_HYPER = "methods_baseline_hyper.yaml"
CONFIG_METHODS_NOTREK_HYPER = "methods_notrek_hyper.yaml"
CONFIG_METHODS_HYPERPARAMS = "methods_hyperparams.yaml"

SUBDIR_EXPERIMENTS = SRC_DIR / "experiments"
EXPERIMENT_COMMANDS_LIST = SUBDIR_EXPERIMENTS / "command_list.txt"

# ---------- canonical filenames ----------
FILE_X = "X.npy"
FILE_B_TRUE = "B_true.npy"
FILE_W_TRUE = "W_true.npy"
FILE_I = "I.npy"
FILE_META = "meta.yaml"
TRUE_DIR = "true"

# if you later store CSVs / json:
FILE_SUMMARY_CSV = "summary.csv"
FILE_RESULTS_PARQUET = "results.parquet"

# ---------- sanity bounds ----------
NAN_MIN = -1e12
NAN_MAX = +1e12

# ---------- data generation ----------
DEFAULT_GRAPH_TYPES = [
    "ER", "ER_acyclic",
    "scale_free", "scale_free_acyclic",
    "sbm", "sbm_acyclic",
]

DEFAULT_LINEAR_SEM_TYPES = ["gauss", "exp", "gumbel", "uniform", "logistic", "poisson",]
DEFAULT_NONLINEAR_SEM_TYPES = ["mlp", "mim",]

DEFAULT_SEM_TYPES = (
    DEFAULT_LINEAR_SEM_TYPES
    + DEFAULT_NONLINEAR_SEM_TYPES
)

DATA_GRID_KEYS = ["n", "d", "s0", "graph_type", "sem_type", "min_indeps"]

# ---------- methods ----------
BASELINES = [
    "dagma_linear",
    # "dagma_nonlinear",
    # "kds",
    # "nodags",
]


# cluster
YAML_RUN = "__run__"
DEFAULT_RUN_KWARGS = {"n_cpus": 1, "n_gpus": 0, "length": "short"}