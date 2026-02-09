# -*- coding: utf-8 -*-
from __future__ import annotations

import itertools
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union, Optional

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


# -------------------------------------------------------
# grid expansion (cartesian product over all lists)
# -------------------------------------------------------

def _sanitize_token(x: Any) -> str:
    """
    Make a filename-safe token.
    """
    s = str(x)
    s = s.replace("/", "-").replace("\\", "-").replace(" ", "")
    s = re.sub(r"[^a-zA-Z0-9_.=-]+", "", s)
    return s


def _collect_list_paths(tree: Any, prefix: Tuple[Any, ...] = ()) -> List[Tuple[Tuple[Any, ...], List[Any]]]:
    """
    Collect all paths (as tuples of dict keys) whose value is a list.
    We treat *every* list as a grid axis (cartesian product).
    """
    out: List[Tuple[Tuple[Any, ...], List[Any]]] = []

    if isinstance(tree, dict):
        for k, v in tree.items():
            out.extend(_collect_list_paths(v, prefix + (k,)))
    elif isinstance(tree, list):
        out.append((prefix, tree))

    return out


def _set_by_path(tree: Any, path: Tuple[Any, ...], value: Any) -> None:
    """
    Set tree[path] = value where path is a tuple of keys.
    """
    cur = tree
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value
    
def _set_by_dotpath(cfg: Dict[str, Any], dotpath: str, value: Any) -> None:
    """
    Set cfg["a"]["b"]["c"] = value given "a.b.c".
    Creates intermediate dicts if needed.
    """
    keys = dotpath.split(".")
    cur: Any = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def expand_grid_config(grid: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Expand a YAML dict by taking the cartesian product over *all* list-valued leaves.

    Yields:
      (resolved_config, choices)

    where choices maps "a.b.c" -> chosen_value for each expanded list.
    """
    axes = _collect_list_paths(grid)
    if not axes:
        yield deepcopy(grid), {}
        return

    paths = [p for p, _ in axes]
    values = [vals for _, vals in axes]

    for combo in itertools.product(*values):
        resolved = deepcopy(grid)
        choices: Dict[str, Any] = {}
        for path, val in zip(paths, combo):
            _set_by_path(resolved, path, val)
            choices[".".join(map(str, path))] = val
        yield resolved, choices


def grid_choice_filename(choices: Dict[str, Any], *, prefix: str = "data", max_len: int = 180) -> str:
    """
    Build a stable, readable filename from grid choices.

    Example:
      data_a.b=1__c=foo.yaml
    """
    if not choices:
        return f"{prefix}_default.yaml"

    parts = []
    for k in sorted(choices.keys()):
        parts.append(f"{_sanitize_token(k)}={_sanitize_token(choices[k])}")
    name = "__".join(parts)
    if len(name) > max_len:
        name = name[:max_len]
    return f"{prefix}__{name}.yaml"


def list_yaml_files(dir_path: PathLike, *, exclude: Iterable[str] = ()) -> List[Path]:
    """
    List *.yaml files in a directory, optionally excluding basenames.
    """
    dir_path = Path(dir_path)
    excl = set(exclude)
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.glob("*.yaml") if p.name not in excl])


# -------------------------------------------------------
# grid expansion via explicit __grid__ spec
# -------------------------------------------------------

_GRID_KEY = "__grid__"

def _is_scalar(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None

def _collect_scalar_list_paths(tree: Any, prefix: Tuple[Any, ...] = ()) -> List[Tuple[Tuple[Any, ...], List[Any]]]:
    """
    Collect paths whose value is a list of scalars (or empty).
    We *skip* structural lists like:
      - list of dicts
      - list of lists / tuples
    which covers e.g. w_ranges, hidden_sizes, methods: [...]
    """
    out: List[Tuple[Tuple[Any, ...], List[Any]]] = []

    if isinstance(tree, dict):
        for k, v in tree.items():
            out.extend(_collect_scalar_list_paths(v, prefix + (k,)))
        return out

    if isinstance(tree, list):
        if len(tree) == 0:
            # empty list: treat as structural (do not expand)
            return out

        # expand ONLY if it's a list of scalars
        if all(_is_scalar(x) for x in tree):
            out.append((prefix, tree))
        return out

    return out

def _set_by_path(tree: Any, path: Tuple[Any, ...], value: Any) -> None:
    cur = tree
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value

def expand_on_keys(
    base: Dict[str, Any],
    *,
    keys: Optional[List[str]] = None,
    defaults: Optional[Dict[str, Any]] = None,
) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Expand `base` by cartesian product.

    If keys is provided:
      - expand ONLY those *top-level* keys (same behavior as before)

    If keys is None:
      - expand ALL scalar-list leaves recursively (skips structural lists automatically)

    Returns:
      (resolved_config, choices)
    where choices maps either:
      - key -> chosen_value          (keys provided)
      - dotpath -> chosen_value      (keys None / recursive)
    """
    defaults = {} if defaults is None else defaults

    # -------- mode A: explicit top-level keys --------
    if keys is not None:
        axes: List[Tuple[str, List[Any]]] = []
        for k in keys:
            if k in base:
                v = base[k]
            elif k in defaults:
                v = defaults[k]
            else:
                continue

            if not isinstance(v, list):
                v = [v]
            axes.append((k, v))

        if not axes:
            yield deepcopy(base), {}
            return

        axis_keys = [k for k, _ in axes]
        axis_vals = [vs for _, vs in axes]

        for combo in itertools.product(*axis_vals):
            resolved = deepcopy(base)
            choices: Dict[str, Any] = {}
            for k, v in zip(axis_keys, combo):
                resolved[k] = v
                choices[k] = v
            yield resolved, choices
        return

    # -------- mode B: expand all scalar-list leaves (recursive) --------
    axes2 = _collect_scalar_list_paths(base)
    if not axes2:
        yield deepcopy(base), {}
        return

    paths = [p for p, _ in axes2]
    values = [vals for _, vals in axes2]

    for combo in itertools.product(*values):
        resolved = deepcopy(base)
        choices: Dict[str, Any] = {}
        for path, val in zip(paths, combo):
            _set_by_path(resolved, path, val)
            choices[".".join(map(str, path))] = val
        yield resolved, choices
