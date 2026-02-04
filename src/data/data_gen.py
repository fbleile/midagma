# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from jax import random

from src.utils.rng import np_rng_from_key
from src.data.data_spec import DataSpec, save_yaml, load_data_config
from src.data.indep_spec import ISpec, build_I_full_and_capped
from src.data.graphs import sample_graph
from src.data.sem import simulate_parameter, simulate_linear_sem, simulate_nonlinear_sem
from src.data.dataset import Dataset
from src.utils.timer import timer
from src.data.dataset import Dataset
from src.utils.paths import resolve_from_project


def data_folder(path_data: Union[str, Path], descr: str, seed: int) -> Path:
    path_data = Path(path_data)
    return path_data / descr / "runs" / f"seed_{seed}"


def save_arrays(folder: Path, arrays: Dict[str, np.ndarray]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(folder / f"{k}.npy", v)


def make_data_from_spec(
    spec: DataSpec,
    i_spec: ISpec,
    *,
    key: random.PRNGKey,
    max_tries: int = 2000,
) -> Dataset:
    key_try = key

    for t in range(int(max_tries)):
        key_try, subk = random.split(key_try)

        # split per component (graph/weights/data/indeps)
        subk, k_graph = random.split(subk)
        subk, k_param = random.split(subk)
        subk, k_data  = random.split(subk)

        # graph
        B = sample_graph(key=k_graph, d=spec.d, s0=spec.s0, graph_type=spec.graph_type)

        # weights
        W_true = simulate_parameter(k_param, B, w_ranges=spec.w_ranges)

        # samples
        X = simulate_linear_sem(
            k_data,
            W_true,
            n=spec.n,
            sem_type=spec.sem_type,
            noise_scale=spec.noise_scale,
        )

        # independencies
        I_full, I = build_I_full_and_capped(X=X, B_true=B, I_spec=i_spec)

        if int(I_full.shape[0]) < int(i_spec.min_indeps):
            continue

        meta: Dict[str, Any] = {
            "try": int(t),
            "accepted": True,
        }

        return Dataset(X=X, B_true=B, W_true=W_true, I_full=I_full, I=I, meta=meta)

    raise RuntimeError(
        f"Failed to sample a dataset with >= {i_spec.min_indeps} independencies after {max_tries} tries."
    )


def generate_and_store(
    *,
    descr: str,
    spec: DataSpec,
    i_spec: ISpec,
    data_config_path: Optional[Union[str, Path]],
    path_data: Union[str, Path],
    dataset_id: str,
    dataset_subkey: random.PRNGKey,
    base_seed: int,
) -> Path:
    """
    Stores under:
      data/<descr>/datasets/<dataset_id>/{X.npy, meta/...}
    """

    root = Path(path_data) / descr
    ds_root = root / "datasets" / str(dataset_id)

    # 1) generate dataset object
    with timer() as wall:
        ds: Dataset = make_data_from_spec(spec, i_spec, key=dataset_subkey)
    wall_min = wall() / 60.0

    # 2) enrich meta *before* saving
    #    (keep ds.meta small: only scalars/lists/dicts)
    def _s(p: Path) -> str:
        return str(p)

    meta = dict(ds.meta or {})
    meta.update({
        "descr": str(descr),
        "dataset_id": str(dataset_id),
        "base_seed": int(base_seed),
        "subkey": np.asarray(dataset_subkey).tolist(),
        "walltime_min": float(wall_min),
        "data_spec": dict(spec.__dict__),
        "i_spec": dict(i_spec.__dict__),
        "counts": {
            "edges_true": int(ds.B_true.sum()),
            "I_full_pairs": int(ds.I_full.shape[0]),
            "I_pairs": int(ds.I.shape[0]),
        },
        "shapes": {
            "X": list(ds.X.shape),
            "B_true": list(ds.B_true.shape),
            "W_true": list(ds.W_true.shape),
            "I_full": list(ds.I_full.shape),
            "I": list(ds.I.shape),
        },
        "paths": {k: str(v) for k, v in ds.paths(Path("")).items()},
    })

    ds2 = Dataset(
        X=ds.X,
        B_true=ds.B_true,
        W_true=ds.W_true,
        I_full=ds.I_full,
        I=ds.I,
        meta=meta,
    )

    # 3) save via dataset object
    ds2.save(ds_root)

    # 4) provenance: copy config yaml once (optional)
    if data_config_path is not None:
        cfg_path = Path(data_config_path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "data.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    return ds_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--descr", type=str, required=True)
    parser.add_argument("--data_config_path", type=Path, required=True)
    parser.add_argument("--path_data", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_datasets", type=int, required=False)
    parser.add_argument("--sanity_check_plots", action="store_true")
    args = parser.parse_args()


    data_config_path = resolve_from_project(args.data_config_path)
    path_data = resolve_from_project(args.path_data)

    # load spec
    cfg_id, n_datasets, data_spec, i_spec = load_data_config(data_config_path)
    
    if args.n_datasets is not None:
        n_datasets = args.n_datasets
    
    # NEW: plan dataset ids up-front (append-only), ids are strings
    datasets_dir = (path_data / args.descr / "datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    existing_ids = []
    for p in datasets_dir.iterdir():
        if p.is_dir():
            try:
                existing_ids.append(int(p.name))   # assumes numeric folder names
            except ValueError:
                pass
    
    start_idx = (max(existing_ids) + 1) if existing_ids else 1
    dataset_ids = [str(i) for i in range(start_idx, start_idx + int(n_datasets))]
    
    base_seed = int(args.seed)
    base_key = random.PRNGKey(base_seed)
    keys = random.split(base_key, int(n_datasets))
    
    # CHANGED: enumerate over (dataset_id, subkey)
    for dataset_id, subkey in zip(dataset_ids, keys):
        spec_k = DataSpec(**data_spec.__dict__)
    
        out = generate_and_store(
            descr=args.descr,                 # e.g. "lin_sem_er"
            spec=spec_k,                      # DataSpec (no seed/key anymore)
            i_spec=i_spec,                    # ISpec
            data_config_path=data_config_path,# resolved absolute Path to YAML
            path_data=path_data,              # resolved absolute Path to data root
            dataset_id=dataset_id,            # str id like "44"
            dataset_subkey=subkey,            # jax PRNGKey for this dataset
            base_seed=base_seed,              # the CLI seed (int)
        )

        print(
            f"[DATA-GEN] descr={args.descr} "
            f"dataset_id={dataset_id} "
            f"base_seed={base_seed} "
            f"subkey={list(map(int, np.asarray(subkey).tolist()))} "
            f"config={data_config_path} "
            f"out={out}\n"
            f"  DataSpec: n={spec_k.n} d={spec_k.d} s0={spec_k.s0} graph_type={spec_k.graph_type} sem_type={spec_k.sem_type} "
            f"noise_scale={spec_k.noise_scale}\n"
            f"  ISpec: source={i_spec.source}"
            f"alpha={i_spec.alpha} test={i_spec.test} num_perm={i_spec.num_perm} bonf={i_spec.bonferroni} undirected={i_spec.undirected} "
            f"cap={i_spec.cap} min_indeps={i_spec.min_indeps}"
        )


# !python /Users/fbleile/Projects/midagma/src/experiment/data_gen.py --descr lin_sem_er --data_config_path configs/hyper/data.yaml --path_data data --seed 0 --n_datasets=1