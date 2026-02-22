## -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any, Dict, Iterable, List, Tuple, Protocol, Type, Literal, Optional
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from src.utils.yaml import load_yaml, save_yaml, _set_by_dotpath, _sanitize_token, expand_on_keys, grid_choice_filename

from definitions import BASELINES
from src.utils.logger import LogConfig

# dagma imports (adjust to your project)
from src.methods.dagma.linear import DagmaLinear
from src.methods.nodags.resblock_trainer import resflow_train_test_wrapper
from src.methods.kds.models import LinearSDEWithTrek, MLPSDEWithTrek, adjacency_from_param_linear, adjacency_from_param_mlp_via_jacobian
from src.methods.midagma.linear import MiDagmaLinear
from src.methods.notears.linear import notears_linear

@dataclass(frozen=True)
class MethodInstance:
    """
    One concrete (algo + trek_reg) config after grid expansion.
    """
    method_id: str          # unique id (string)
    block_id: str           # the parent block id from YAML
    cfg: Dict[str, Any]     # resolved config with scalars only on grid axes


def _get_by_dotpath(cfg: Dict[str, Any], dotpath: str) -> Any:
    cur: Any = cfg
    for k in dotpath.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing dotpath '{dotpath}' in method config.")
        cur = cur[k]
    return cur


def _choices(v: Any) -> List[Any]:
    return v if isinstance(v, list) else [v]


def _method_id(block_id: str, choices: Dict[str, Any], *, max_len: int = 180) -> str:
    if not choices:
        return _sanitize_token(block_id)

    parts = [block_id]
    for k in sorted(choices.keys()):
        parts.append(f"{_sanitize_token(k)}={_sanitize_token(choices[k])}")

    s = "__".join(parts)
    return s if len(s) <= max_len else s[:max_len]


def expand_method_block(block: Dict[str, Any]) -> List[MethodInstance]:
    """
    Expand one block with explicit __grid__ dotpaths.
    """
    block_id = str(block.get("id", "method")).strip()
    grid_keys = block.get("__grid__", [])
    if grid_keys is None:
        grid_keys = []
    if not isinstance(grid_keys, list):
        raise ValueError(f"Method block '{block_id}': __grid__ must be a list of dotpaths.")

    # remove meta keys for the resolved cfg
    base = {k: v for k, v in block.items() if k not in ("__grid__",)}

    # collect axes
    axes: List[Tuple[str, List[Any]]] = []
    for dp in grid_keys:
        v = _get_by_dotpath(base, str(dp))
        axes.append((str(dp), _choices(v)))

    if not axes:
        mid = _method_id(block_id, {})
        return [MethodInstance(method_id=mid, block_id=block_id, cfg=base)]

    dps = [dp for dp, _ in axes]
    vals = [vs for _, vs in axes]

    out: List[MethodInstance] = []
    import itertools
    for combo in itertools.product(*vals):
        resolved = {k: v for k, v in base.items()}  # shallow ok; we'll set via dotpath into nested dicts
        # need deep copy to be safe with nested dict reuse
        import copy
        resolved = copy.deepcopy(resolved)

        choices: Dict[str, Any] = {}
        for dp, val in zip(dps, combo):
            _set_by_dotpath(resolved, dp, val)
            choices[dp] = val

        mid = _method_id(block_id, choices)
        out.append(MethodInstance(method_id=mid, block_id=block_id, cfg=resolved))

    return out


def expand_methods_config(cfg: Dict[str, Any]) -> List[MethodInstance]:
    blocks = cfg.get("methods", [])
    if not isinstance(blocks, list):
        raise ValueError("methods.yaml must contain a top-level key: methods: [ ... ]")

    out: List[MethodInstance] = []
    for b in blocks:
        if not isinstance(b, dict):
            raise ValueError("Each entry in methods: must be a dict.")
        out.extend(expand_method_block(b))
    return out


def load_methods_yaml(path) -> Tuple[Dict[str, Any], List[MethodInstance]]:
    cfg = load_yaml(path)
    return cfg, expand_methods_config(cfg)

def expand_methods_grid(methods_yaml_path: Path, out_dir: Path) -> List[Path]:
    grid = load_yaml(methods_yaml_path)
    methods = grid.get("methods", [])
    assert isinstance(methods, list), "methods must be a list"

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_paths: List[Path] = []

    for m in methods:
        assert isinstance(m, dict), "each methods[] entry must be a dict"
        mid = str(m.get("id", "method"))

        # expand scalar lists inside *this* method entry
        candidates = list(expand_on_keys(m, keys=None))

        for resolved, choices in candidates:
            # build filename that includes method id + choices
            fname = grid_choice_filename(choices, prefix=f"methods_{mid}")
            p = out_dir / fname
            save_yaml({"methods": [resolved]}, p)  # write as a single-method file
            cfg_paths.append(p)

    return sorted(cfg_paths)


def assert_registry_complete():
    missing = [n for n in BASELINES if n not in ALGO_REGISTRY]
    if missing:
        raise RuntimeError(f"Missing algo runners for: {missing}")

class AlgoSpecBase(Protocol):
    name: str

# Runner now accepts Any spec (but we keep spec_cls alongside)
AlgoRunner = Callable[[np.ndarray, Any, Any, logging.Logger, LogConfig], np.ndarray]

@dataclass(frozen=True)
class AlgoEntry:
    runner: AlgoRunner
    spec_cls: Type[Any]   # dataclass type for that algo


ALGO_REGISTRY: Dict[str, AlgoEntry] = {}


def register_algo(name: str, *, spec_cls: Type[Any]):
    """
    Register an algo with its spec dataclass.
    """
    def _decorator(fn: AlgoRunner):
        ALGO_REGISTRY[name] = AlgoEntry(runner=fn, spec_cls=spec_cls)
        return fn
    return _decorator

@dataclass(frozen=True)
class DagmaLinearSpec:
    name: str = "dagma_linear"
    loss_type: str = "l2"
    lambda1: float = 0.02
    max_iter: int = int(6e4)
    mu_factor: float = 0.1
    s: float = 2.0

@register_algo("dagma_linear", spec_cls=DagmaLinearSpec)
def run_dagma_linear(
    X: np.ndarray,
    trek_reg: Any,
    algo_spec: DagmaLinearSpec,
    logger: logging.Logger,
    log_cfg: LogConfig,
) -> np.ndarray:
    model = DagmaLinear(
        loss_type=algo_spec.loss_type,
        trek_reg=trek_reg,
        logger=logger,
        log_cfg=log_cfg,
    )
    W_est = model.fit(
        X,
        lambda1=algo_spec.lambda1,
        max_iter=algo_spec.max_iter,
        mu_factor=algo_spec.mu_factor,
        s=algo_spec.s,
    )
    return W_est


@dataclass(frozen=True)
class MiDagmaLinearSpec:
    name: str = "midagma_linear"
    loss_type: str = "l2"
    lambda1: float = 0.02
    max_iter: int = int(6e4)
    mu_factor: float = 0.1
    s: float = 2.0

    # midagma-specific
    w: float = 1.0
    
    seed: int = 0


@register_algo("midagma_linear", spec_cls=MiDagmaLinearSpec)
def run_midagma_linear(
    X: np.ndarray,
    trek_reg: Any,
    algo_spec: MiDagmaLinearSpec,
    logger: logging.Logger,
    log_cfg: LogConfig,
) -> np.ndarray:
    # --- pull I from trek_reg if available ---
    I_pairs = None
    if trek_reg is not None:
        # common patterns in your codebase: trek_reg.cfg["I"] or trek_reg.I
        if hasattr(trek_reg, "cfg") and isinstance(getattr(trek_reg, "cfg"), dict):
            I_pairs = trek_reg.cfg.get("I", None)
        if I_pairs is None and hasattr(trek_reg, "I"):
            I_pairs = getattr(trek_reg, "I")

    model = MiDagmaLinear(
        loss_type=algo_spec.loss_type,
        I_pairs=I_pairs,
        w=algo_spec.w,
        trek_reg=trek_reg,
        logger=logger,
        log_cfg=log_cfg,
    )

    W_est = model.fit(
        X,
        lambda1=algo_spec.lambda1,
        max_iter=algo_spec.max_iter,
        mu_factor=algo_spec.mu_factor,
        s=algo_spec.s,
    )
    return W_est

@dataclass(frozen=True)
class NotearsLinearSpec:
    name: str = "notears_linear"
    loss_type: str = "l2"
    lambda1: float = 0.0

    # NOTEARS params
    max_iter: int = 100
    h_tol: float = 1e-8
    rho_max: float = 1e16
    w_threshold: float = 0.3


@register_algo("notears_linear", spec_cls=NotearsLinearSpec)
def run_notears_linear(
    X: np.ndarray,
    trek_reg: Any,
    algo_spec: NotearsLinearSpec,
    logger: logging.Logger,
    log_cfg: LogConfig,
) -> np.ndarray:
    
    W_est = notears_linear(
        X,
        lambda1=algo_spec.lambda1,
        loss_type=algo_spec.loss_type,
        max_iter=algo_spec.max_iter,
        h_tol=algo_spec.h_tol,
        rho_max=algo_spec.rho_max,
        w_threshold=algo_spec.w_threshold,
        trek_reg=trek_reg,
    )
    return W_est


@dataclass(frozen=True)
class NODAGSSpec:
    name: str = "nodags"

    # model
    fun_type: Literal["mul-mlp", "lin-mlp", "nnl-mlp", "fac-mlp", "gst-mlp"] = "fac-mlp"
    lip_const: float = 0.9
    act_fun: Literal["tanh", "relu", "sigmoid"] = "tanh"
    full_input: bool = False
    n_hidden: int = 1
    n_factors: int = 10

    # training
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    optim: Literal["adam", "sgd"] = "adam"
    upd_lip: bool = True
    n_lip_iter: int = 5

    # regularization / output
    l1_reg: bool = True
    lambda_c: float = 1e-3
    thresh_val: float = 1e-1

    # optional NODAGS internals if you expose them
    n_power_series: Optional[int] = None
    init_var: float = 0.5
    lin_logdet: bool = False
    dag_input: bool = False
    centered: bool = True

@register_algo("nodags", spec_cls=NODAGSSpec)
def run_nodags(
    X: np.ndarray,
    trek_reg: Any,
    algo_spec: NODAGSSpec,
    logger: logging.Logger,
    log_cfg: LogConfig,
) -> np.ndarray:
    # build wrapper (map spec fields -> wrapper kwargs)
    rb = resflow_train_test_wrapper(
        n_nodes=X.shape[1],
        batch_size=algo_spec.batch_size,
        l1_reg=algo_spec.l1_reg,
        lambda_c=algo_spec.lambda_c,
        n_lip_iter=algo_spec.n_lip_iter,
        fun_type=algo_spec.fun_type,
        lip_const=algo_spec.lip_const,
        act_fun=algo_spec.act_fun,
        lr=algo_spec.lr,
        epochs=algo_spec.epochs,
        optim=algo_spec.optim,
        v=getattr(algo_spec, "verbose", False),
        inline=False,
        upd_lip=algo_spec.upd_lip,
        full_input=algo_spec.full_input,
        n_hidden=algo_spec.n_hidden,
        n_factors=algo_spec.n_factors,
        n_power_series=getattr(algo_spec, "n_power_series", None),
        init_var=getattr(algo_spec, "init_var", 0.5),
        lin_logdet=getattr(algo_spec, "lin_logdet", False),
        dag_input=getattr(algo_spec, "dag_input", False),
        thresh_val=algo_spec.thresh_val,
        centered=getattr(algo_spec, "centered", True),
    )

    # observational-only interface in your framework:
    datasets = [np.asarray(X)]
    intervention_sets = [np.array([], dtype=np.int64)]

    rb.train(datasets, intervention_sets, batch_size=algo_spec.batch_size, trek_reg=trek_reg)

    W_est = rb.get_adjacency()

    return W_est


# ---------------------------------------------------------------------
# KDS spec + runner
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class KDSSpec:
    name: str = "kds"

    model: Literal["linear", "mlp"] = "linear"
    seed: int = 0

    # output thresholding (optional)
    thresh_val: float = 1e-12

    # ---- KDS / stadion fit(...) kwargs ----
    # Signature:
    # fit(self, key, x, targets=None, bandwidth=5.0, estimator="linear",
    #     learning_rate=0.003, steps=10000, batch_size=128, reg=0.001,
    #     warm_start_intv=True, device=None, verbose=10)

    targets: Optional[Any] = None               # keep Any: could be array/list/dict depending on stadion usage
    bandwidth: float = 5.0
    estimator: Literal["u-statistic", "v-statistic", "linear"] = "linear"  # "stein" is common; keep if supported in your version
    learning_rate: float = 0.003
    steps: int = 10_000
    batch_size: int = 128
    reg: float = 0.001
    warm_start_intv: bool = True
    device: Optional[Any] = None                # stadion might accept e.g. "cpu"/"gpu" or a jax device handle
    verbose: int = 10

    # ---- optional passthrough for model constructor etc. ----
    # keep this if you already use it for non-fit args
    model_kwargs: Optional[Dict[str, Any]] = None


@register_algo("kds", spec_cls=KDSSpec)
def run_kds(
    X: np.ndarray,
    trek_reg: Any,
    algo_spec: KDSSpec,
    logger: logging.Logger,
    log_cfg: Any,
) -> np.ndarray:
    """
    KDS / stadion runner.

    - trains stadion model (LinearSDE or MLPSDE)
    - trek penalty is injected via overridden regularize_sparsity
    - returns adjacency estimate W_est in your (parent,child) convention
    """
    X_np = np.asarray(X)
    d = int(X_np.shape[1])

    key = random.PRNGKey(int(algo_spec.seed))

    fit_kwargs = {
        "fit": {
            "targets": algo_spec.targets,  # None is fine
            "bandwidth": float(algo_spec.bandwidth),
            "estimator": str(algo_spec.estimator),
            "learning_rate": float(algo_spec.learning_rate),
            "steps": int(algo_spec.steps),
            "batch_size": int(algo_spec.batch_size),
            "reg": float(algo_spec.reg),
            "warm_start_intv": bool(algo_spec.warm_start_intv),
            "device": algo_spec.device,    # can be None
            "verbose": int(algo_spec.verbose),
        }
    }

    # constructor kwargs (optional)
    model_kwargs = dict(algo_spec.model_kwargs or {})

    if algo_spec.model == "linear":
        model = LinearSDEWithTrek(trek_reg=trek_reg, **model_kwargs)
    elif algo_spec.model == "mlp":
        model = MLPSDEWithTrek(trek_reg=trek_reg, **model_kwargs)
    else:
        raise ValueError(f"Unknown algo_spec.model={algo_spec.model!r}")

    # stadion expects JAX arrays; be explicit
    data = jnp.asarray(X_np)

    # fit
    key, subk = random.split(key)
    model.fit(subk, data, **fit_kwargs.get("fit", {}))

    # params -> adjacency
    param = model.param
    if algo_spec.model == "linear":
        W_est = np.asarray(adjacency_from_param_linear(param))
    else:
        W_est = np.asarray(
            adjacency_from_param_mlp_via_jacobian(
                model, param, x0=jnp.zeros((d,), dtype=jnp.float32)
            )
        )

    return W_est