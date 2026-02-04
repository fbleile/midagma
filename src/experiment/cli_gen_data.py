# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from jax import random

from src.data.data_spec import load_data_config, DataSpec
from src.data.data_gen import generate_and_store
from src.utils.paths import resolve_from_project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descr", type=str, required=True)
    parser.add_argument("--data_config_path", type=Path, required=True)
    parser.add_argument("--path_data", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_datasets", type=int, required=False)
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




if __name__ == "__main__":
    main()


# !python /Users/fbleile/Projects/midagma/src/experiment/cli_gen_data.py --descr lin_sem_er --data_config_path configs/hyper/data.yaml --path_data data --seed 0 --n_datasets=1