# -*- coding: utf-8 -*-
import copy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import argparse
import shutil
from pathlib import Path
from jax import random

from src.utils.yaml import load_yaml, expand_on_keys, save_yaml, grid_choice_filename
from src.experiment.launch import generate_run_commands
from src.data.sanity import _sanity_check_data

from src.methods.method_spec import load_methods_yaml, expand_methods_grid

from definitions import (
    PROJECT_DIR,
    CLUSTER_GROUP_DIR,
    CLUSTER_SCRATCH_DIR,
    IS_CLUSTER,
    
    SUBDIR_RESULTS,
    SUBDIR_DATA,
    
    YAML_RUN,
    DEFAULT_RUN_KWARGS,
    
    SLURM_LOGS_DIR,
    
    CONFIG_DIR,
    CONFIG_DATA,
    CONFIG_METHODS_BASELINE_HYPER,
    CONFIG_METHODS_NOTREK_HYPER,
    CONFIG_METHODS,
    
    DATA_DIR,
    
    DEFAULT_SEM_TYPES,
    DEFAULT_GRAPH_TYPES,
    DATA_GRID_KEYS,
)



class ExperimentManager:
    """Tool for clean and reproducible experiment handling via folders"""

    def __init__(self, experiment, seed=0, verbose=True, compute="cluster", dry=True, n_datasets=1, only_methods=None,
                 scratch=False, subdir_results=None):

        self.experiment = experiment # lin-er-acyclic / lin-er / ... / nonlin-er-acyclic
        self.config_path = CONFIG_DIR / self.experiment
        self.project_dir = ((CLUSTER_SCRATCH_DIR if scratch else CLUSTER_GROUP_DIR) if IS_CLUSTER else PROJECT_DIR)
        self.store_path_results = self.project_dir / SUBDIR_RESULTS / self.experiment
        self.store_path_data = self.project_dir / SUBDIR_DATA / self.experiment
        self.key = random.PRNGKey(seed)
        self.seed = seed
        self.compute = compute
        self.verbose = verbose
        self.dry = dry

        self.slurm_logs_dir = SLURM_LOGS_DIR
        Path(self.slurm_logs_dir).mkdir(exist_ok=True)

        self.n_datasets = n_datasets
        
        # Data setup
        def exists_or_none(p):
            return p if p.exists() else None


        self.data_config_path = exists_or_none(self.config_path / CONFIG_DATA)
        self.methods_baseline_hyper_config_path = exists_or_none(self.config_path / CONFIG_METHODS_BASELINE_HYPER)
        self.methods_notreks_hyper_config_path = exists_or_none(self.config_path / CONFIG_METHODS_NOTREK_HYPER)
        self.methods_config_path = exists_or_none(self.config_path / CONFIG_METHODS)
        
        if self.verbose:
            if self.config_path.exists() \
                and self.config_path.is_dir():
                print("experiment:       ", self.experiment, flush=True)
                print("data directory:", self.store_path_data, flush=True, end="\n\n")
                print("results directory:", self.store_path_results, flush=True, end="\n\n")
            else:
                print(f"experiment `{self.experiment}` not specified in `{self.config_path}`."
                      f"check spelling and files")
                exit(1)

        # parse configs
        self.data_config = load_yaml(self.data_config_path) if self.data_config_path else None
        # self.data_grid_config = load_yaml(self.data_grid_config_path) if self.data_grid_config_path else None
        self.methods_config = load_yaml(self.methods_config_path) if self.methods_config_path else None
        
        # Methods setup

        # # adjust configs based on only_methods
        # self.only_methods = only_methods
        # if self.only_methods is not None:
        #     for k in list(self.methods_config.keys()):
        #         if not any([m in k for m in self.only_methods]):
        #             del self.methods_config[k]

        #     if self.methods_validation_config is not None:
        #         for k in list(self.methods_validation_config.keys()):
        #             if not any([m in k for m in self.only_methods]):
        #                 del self.methods_validation_config[k]

    def _inherit_specification(self, subdir, inherit_from):
        if inherit_from is not None:
            v = str(inherit_from.name).split("_")[1:]
            return subdir + "_" + "_".join(v)
        else:
            return subdir


    def _get_name_without_version(self, p):
        return "_".join(p.name.split("_")[:-1])


    def _list_main_folders(self, subdir, root_path=None, inherit_from=None):
        if root_path is None:
            root_path = self.store_path_results
        subdir = self._inherit_specification(subdir, inherit_from)
        if root_path.is_dir():
            return sorted([
                p for p in root_path.iterdir()
                if (p.is_dir() and subdir == self._get_name_without_version(p))
            ])
        else:
            return []


    def _init_folder(self, subdir, root_path=None, inherit_from=None, dry=False, add_logs_folder=False):
        if root_path is None:
            root_path = self.store_path_results
        subdir = self._inherit_specification(subdir, inherit_from)
        existing = self._list_main_folders(subdir, root_path=root_path)
        if existing:
            latest_existing = sorted(existing)[-1]
            suffix = int(latest_existing.stem.rsplit("_", 1)[-1]) + 1
        else:
            suffix = 0
        folder = root_path / (subdir + f"_{suffix:02d}")
        assert not folder.exists(), "Something went wrong. The data foler we initialize should not exist."
        if not dry:
            folder.mkdir(exist_ok=False, parents=True)
            if add_logs_folder:
                (folder / "logs").mkdir(exist_ok=False, parents=True)
        return folder


    def _copy_file(self, from_path, to_path):
        shutil.copy(from_path, to_path)


    def make_data(self, check: bool = False):
        if check:
            assert self.store_path_data.exists(), "folder doesn't exist; run `--data` first"
            paths_data = self._list_main_folders(SUBDIR_DATA, root_path=self.store_path_data)
            assert len(paths_data) > 0, "data not created yet; run `--data` first"
            final_data = list(filter(lambda p: p.name.rsplit("_", 1)[-1] == "final", paths_data))
            if final_data:
                assert len(final_data) == 1
                return final_data[0]
            return paths_data[-1]
    
        assert self.data_config_path is not None, \
            f"Expected data_grid.yaml at:\n{self.config_path / CONFIG_DATA}"
    
        n_datasets = self.n_datasets
        experiment_name = self.experiment.replace("/", "--")
    
        # 1) init data folder: data/<experiment>/data_XX/
        path_data = self._init_folder(SUBDIR_DATA, root_path=self.store_path_data)
    
        # 2) copy grid yaml into data folder root
        if not self.dry:
            self._copy_file(self.data_config_path, path_data / CONFIG_DATA)
        
        # 3) expand only classic keys from grid yaml (ignore structural lists)
        grid = load_yaml(self.data_config_path)
        
        defaults = {
            "graph_type": DEFAULT_GRAPH_TYPES,
            "sem_type": DEFAULT_SEM_TYPES,
        }
        
        candidates = list(expand_on_keys(grid, keys=DATA_GRID_KEYS, defaults=defaults))
        
        # 4) sanity filter + write valid configs into data_XX/configs/
        configs_dir = path_data / "_configs"
        valid_cfg_paths: list[Path] = []
        rejected: list[dict] = []
        
        for resolved, choices in candidates:
            ok, reasons = _sanity_check_data(resolved)
            if not ok:
                rejected.append({
                    "choices": choices,
                    "reasons": reasons,
                    # include a small snapshot of the key fields to eyeball quickly
                    "n": resolved.get("n"),
                    "d": resolved.get("d"),
                    "s0": resolved.get("s0"),
                    "graph_type": resolved.get("graph_type"),
                    "sem_type": resolved.get("sem_type"),
                })
                continue
        
            fname = grid_choice_filename(choices, prefix="data")
            p = configs_dir / fname
            if not self.dry:
                save_yaml(resolved, p)
            valid_cfg_paths.append(p)
        
        cfg_paths = sorted(valid_cfg_paths)
        
        print(f'rejected: {rejected}')
        print(f'len cfg_paths: {cfg_paths}')
        
        # write rejection report for debugging
        if rejected and (not self.dry):
            # keep it near the grid for easy inspection
            save_yaml({"rejected": rejected}, path_data / "sanity_rejects.yaml")
        
        assert len(cfg_paths) > 0, (
            "No valid configs after sanity checks. "
            f"See {path_data / 'sanity_rejects.yaml'} for reasons."
        )

    
        # 5) one slurm array PER config, but with shifted dataset folder ids
        for cfg_idx, cfg_path in enumerate(cfg_paths):
            # dataset folder id is numeric: c cfg_idx + r SLURM_ARRAY_TASK_ID
            cmd = (
                rf"python '{self.project_dir}/src/data/launch_data.py' "
                r"--seed \$SLURM_ARRAY_TASK_ID "
                rf"--data_config_path '{cfg_path}' "
                rf"--path_data '{path_data}' "
                rf"--descr '{experiment_name}-data-{cfg_idx:02d}$SLURM_ARRAY_TASK_ID' "
            )
    
            generate_run_commands(
                array_command=cmd,
                array_indices=range(1, n_datasets + 1),
                mode=self.compute,
                hours=2,
                mins=59,
                n_cpus=2,
                n_gpus=0,
                mem=2000,
                prompt=False,
                dry=self.dry,
                output_path_prefix=self.slurm_logs_dir,
            )
    
        if self.dry and path_data.exists():
            shutil.rmtree(path_data)
    
        return path_data



    def launch_methods(self, *, check: bool = False, method_config: str = 'methods.yaml'):
        # 1) check data exists
        path_data = self.make_data(check=True)
    
        if check:
            paths_results = self._list_main_folders(SUBDIR_RESULTS, inherit_from=path_data)
            assert len(paths_results) > 0, "results not created yet; run `--launch_methods` first"
            return paths_results[-1]
    
        # 2) init results folder
        path_results = self._init_folder(SUBDIR_RESULTS, inherit_from=path_data)
        self._copy_file(method_config, path_results / CONFIG_METHODS)
    
        # 3) list dataset ids (folder names), ignore "_" folders
        data_found = sorted(
            [p for p in path_data.iterdir() if p.is_dir() and not p.name.startswith("_")],
            key=lambda p: p.name,
        )
        dataset_ids = [p.name for p in data_found]
        
        print(f"Found datasets: {dataset_ids}")
        if len(data_found) != self.n_datasets:
            warnings.warn(f"\nNumber of data sets does not match data config "
                f"(got: `{len(data_found)}`, expected `{self.n_datasets}`).\n"
                f"data path: {path_data}\n")
            if len(data_found) < self.n_datasets:
                print("Exiting.")
                return
            else:
                print(f"Taking first {self.n_datasets} data folders")
                data_found = data_found[:self.n_datasets]
        elif self.verbose:
            print(f"\nLaunching experiments for {len(data_found)} data sets.")
            
        dataset_ids = dataset_ids[: self.n_datasets]
        data_found = data_found[: self.n_datasets]
    
        # 4) write datasets mapping file once (SLURM arrays want numeric index)
        mapping_path = path_results / "datasets.txt"
        if not self.dry:
            (path_results / "logs").mkdir(exist_ok=True, parents=True)
            mapping_path.write_text("\n".join(dataset_ids) + "\n", encoding="utf-8")
    
        # 5) expand methods.yaml into concrete configs (one YAML per instance)
        grid = load_yaml(method_config)
        methods = grid.get("methods", [])
        assert isinstance(methods, list), "methods.yaml must contain a top-level list: methods: [...]"
    
        configs_dir = path_results / "_configs_methods"
        cfg_paths: list[Path] = []
    
        for m in methods:
            assert isinstance(m, dict), "each entry in methods[] must be a dict"
            block_id = str(m.get("id", "method"))
    
            # expand ALL scalar-list leaves inside this method entry
            candidates = list(expand_on_keys(m, keys=None))
    
            for resolved, choices in candidates:
                # filename encodes the hyperparam choices => uniqueness + provenance
                fname = grid_choice_filename(choices, prefix=f"{block_id}")
                p = configs_dir / fname
                if not self.dry:
                    save_yaml(resolved, p)
                cfg_paths.append(p)
    
        cfg_paths = sorted(cfg_paths)
        if self.verbose:
            print(f"Expanded to {len(cfg_paths)} method configs in {configs_dir}")
    
        assert len(cfg_paths) > 0, "No expanded method configs produced (check methods.yaml)."
    
        # 6) launch: one job per (method_cfg, dataset_path)
        experiment_name = self.experiment.replace("/", "--")
        
        n_launched = 0
        
        for method_cfg_path in cfg_paths:
            method_id = method_cfg_path.stem
        
            base_cmd = (
                f"python '{self.project_dir}/src/methods/launch_methods.py' "
                f"--method_id '{method_id}' "
                f"--method_cfg '{method_cfg_path}' "
                f"--path_data_root '{path_data}' "
                f"--path_results '{path_results}' "
            )
        
            command_list = []
            for data_path in data_found[: self.n_datasets]:  # clip here if you want
                cmd = base_cmd + (
                    f"--dataset '{data_path}' "
                    f"--descr '{experiment_name}-{method_id}-{data_path.name}' "
                )
                command_list.append(cmd)
        
            cmd_args = dict(
                command_list=command_list,
                array_indices=range(1, len(command_list)+1),
                mode=self.compute,
                dry=self.dry,
                prompt=False,
                output_path_prefix=f"{path_results}/logs/",
                hours=2,
                mins=59,
                n_cpus=2,
                n_gpus=0,
                mem=8000,
            )
        
            n_launched += len(command_list)
            generate_run_commands(**cmd_args)
        
        print(
            f"\nLaunched {n_launched} runs total "
            f"({len(cfg_paths)} method instances × up to {min(len(data_found), self.n_datasets)} datasets)"
        )
        
        if self.dry and path_results.exists():
            shutil.rmtree(path_results)
        
        return path_results


    def make_summary(self):
        """
        Create evaluation summary for all runs of this experiment.
    
        Scrapes:
          results/<experiment>/**/runs/**/meta.yaml
    
        Outputs:
          results/<experiment>/<summary_XX>/
            eval_long.csv
            eval_pivot_median.csv
            eval_meta.yaml
        """
        # init summary folder (like make_data / launch_methods)
        path_results = self.launch_methods(check=True)
    
        path_summary = self._init_folder("summary", root_path=self.store_path_results, dry=self.dry)
        if self.dry:
            # create then remove later; but we still want the command preview
            pass
    
        experiment_name = self.experiment.replace("/", "--")
    
        cmd = (
            f"python '{self.project_dir}/src/eval/launch_eval.py' "
            f"--path_results '{path_results}' "
            f"--runs_subdir 'runs' "
            f"--out_dir '{path_summary}' "
            f"--thr 1e-12 "
            f"--descr '{experiment_name}-summary' "
        )
    
        generate_run_commands(
            command_list=[cmd],
            mode=self.compute,
            hours=0,
            mins=20,
            n_cpus=2,
            n_gpus=0,
            mem=4000,
            prompt=False,
            dry=self.dry,
            output_path_prefix=f"{path_results}/logs/",
        )
    
        if self.dry and path_summary.exists():
            shutil.rmtree(path_summary)
    
        return path_results




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, nargs="?", default="test", help="experiment config folder")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--compute", type=str, default="local")
    parser.add_argument("--seed", type=int)

    parser.add_argument("--data", action="store_true")
    parser.add_argument("--methods_baseline_hyper", action="store_true")
    parser.add_argument("--methods_notreks_hyper", action="store_true")
    parser.add_argument("--methods", action="store_true")
    parser.add_argument("--summary", action="store_true")
    
    parser.add_argument("--launch_type", action="store_true")

    parser.add_argument("--scratch", action="store_true")

    parser.add_argument("--n_datasets", type=int, help="overwrites default specified in config")
    parser.add_argument("--only_methods",nargs="*",type=str,)
    parser.add_argument("--subdir_results", type=str)

    kwargs = parser.parse_args()

    kwargs_sum = sum([
        kwargs.data,
        kwargs.methods_baseline_hyper,
        kwargs.methods_notreks_hyper,
        kwargs.methods,
        kwargs.summary,
    ])
    assert kwargs_sum == 1, f"pass 1 option, got `{kwargs_sum}`"

    exp = ExperimentManager(experiment=kwargs.experiment, compute=kwargs.compute, n_datasets=kwargs.n_datasets,
                            seed=kwargs.seed if kwargs.seed is not None else 0,
                            dry=not kwargs.submit, only_methods=kwargs.only_methods, scratch=kwargs.scratch,
                            subdir_results=kwargs.subdir_results)
    
    if kwargs.data:
        if kwargs.seed is None:
            raise ValueError("Seed undefined")
        _ = exp.make_data()

    elif kwargs.methods_baseline_hyper:
        _ = exp.launch_methods(method_config=exp.methods_baseline_hyper_config_path)

    elif kwargs.methods_notreks_hyper:
        _ = exp.launch_methods(method_config=exp.methods_notreks_hyper_config_path)

    elif kwargs.methods:
        _ = exp.launch_methods(method_config=exp.methods_config_path)

    elif kwargs.summary:
        _ = exp.make_summary()

    else:
        raise ValueError("Unknown option passed")
        
# !python src/experiment/experiment_manager.py acyclic_d20_hyper --data --n_datasets=4 --seed=1 --submit

# !python src/experiment/experiment_manager.py acyclic_d20_large_samples --methods_notreks_hyper --n_datasets=10 --submit

# !python manager.py scm-er --data --submit --n_datasets=50
# !python manager.py scm-er --methods_train_validation --n_datasets 50 --submit --only_methods ours-lnl_u_diag ours-linear_u_diag
# !python manager.py scm-er --summary_train_validation --n_datasets 50 --submit --only_methods ours-lnl_u_diag ours-linear_u_diag
# !python manager.py scm-er --methods --submit --n_datasets=1 --only_methods kds-lnl_u_diag kds-linear_u_diag
# !python manager.py scm-er --summary --submit --n_datasets=1 --only_methods kds-lnl_u_diag kds-linear_u_diag

# problems with gies, llc, nodags,
# seems to be the changed search_theta_test_intv thing