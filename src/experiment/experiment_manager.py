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

from src.utils.yaml import load_yaml

from definitions import (
    PROJECT_DIR,
    CLUSTER_GROUP_DIR,
    CLUSTER_SCRATCH_DIR,
    IS_CLUSTER,
    
    SUBDIR_RESULTS,
    SUBDIR_DATA,
    
    YAML_RUN,
    DEFAULT_RUN_KWARGS,
    
    SLURM_SUBMISSION_MAX,
    SLURM_LOGS_DIR,
    
    CONFIG_DIR,
    CONFIG_DATA,
    CONFIG_DATA_GRID,
    CONFIG_METHODS,
    CONFIG_METHODS_VALIDATION,
)



class ExperimentManager:
    """Tool for clean and reproducible experiment handling via folders"""

    def __init__(self, experiment, seed=0, verbose=True, compute="cluster", dry=True, n_datasets=1, only_methods=None,
                 scratch=False, subdir_results=None):

        self.experiment = experiment # lin-er-acyclic / lin-er / ... / nonlin-er-acyclic
        self.config_path = CONFIG_DIR / self.experiment
        self.store_path_root = ((CLUSTER_SCRATCH_DIR if scratch else CLUSTER_GROUP_DIR) if IS_CLUSTER else PROJECT_DIR)
        self.store_path = self.store_path_root / SUBDIR_RESULTS / self.experiment
        self.key = random.PRNGKey(seed)
        self.seed = seed
        self.compute = compute
        self.verbose = verbose
        self.dry = dry

        self.slurm_logs_dir = SLURM_LOGS_DIR
        Path(self.slurm_logs_dir).mkdir(exist_ok=True)

        self.n_datasets = n_datasets
        
        # Data setup

        self.data_config_path = self.config_path / CONFIG_DATA
        self.data_grid_config_path = self.config_path / CONFIG_DATA_GRID
        self.methods_config_path = self.config_path / CONFIG_METHODS
        self.methods_validation_config_path = self.config_path / CONFIG_METHODS_VALIDATION

        if self.verbose:
            if self.config_path.exists() \
                and self.config_path.is_dir():
                print("experiment:       ", self.experiment, flush=True)
                print("data directory:", self.data_path, flush=True, end="\n\n")
                print("results directory:", self.store_path, flush=True, end="\n\n")
            else:
                print(f"experiment `{self.experiment}` not specified in `{self.config_path}`."
                      f"check spelling and files")
                exit(1)

        # parse configs
        self.data_config = load_yaml(self.data_config_path)
        # TODO self.data_grid_config = load_yaml(self.data_grid_config_path)
        self.methods_config = load_yaml(self.methods_config_path)
        self.methods_validation_config = load_yaml(self.methods_validation_config_path)
        
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
            root_path = self.store_path
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
            root_path = self.store_path
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


    def make_data(self, check=False, grid=False):
        assert self.data_config is not None, \
            f"Error when loading or file not found for data.yaml at path:\n" \
            f"{self.data_config_path}"
        n_datasets = self.n_datasets

        # init data folder
        path_data = self._init_folder(SUBDIR_DATA)
        self._copy_file(self.data_config_path, path_data / config_file_name)
        if self.dry:
            shutil.rmtree(path_data)

        # launch runs that generate data
        experiment_name = kwargs.experiment.replace("/", "--")
        cmd = f"python '{PROJECT_DIR}/src/data/data_gen.py' " \
              r"--seed \$SLURM_ARRAY_TASK_ID " \
              f"--data_config_path '{data_config_path}' " \
              f"--path_data '{path_data}' "

        if grid:
            grid_ids = range(len(self.data_grid_config))
        else:
            grid_ids = [None]
        
        # define throttle to respect job submition limit
        if len(grid_ids) > SLURM_SUBMISSION_MAX:
            raise RuntimeError(
                    f"Slurm submission limit exceeded: "
                    f"planned {grid_ids} grid points, but the max allowed is {SLURM_SUBMISSION_MAX}. "
                    "Execution stopped."
                )

        for grid_id in grid_ids:
            cmd_final = cmd
            if grid_id is not None:
                cmd_final += f"--grid_id {grid_id} "
                cmd_final += rf"--descr '{experiment_name}-data-{grid_id}-\$SLURM_ARRAY_TASK_ID' "
            else:
                cmd_final += rf"--descr '{experiment_name}-data-\$SLURM_ARRAY_TASK_ID' "

            generate_run_commands(
                array_command=cmd_final,
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

        print(f"\nLaunched {n_datasets * len(grid_ids)} runs total ({len(grid_ids)} grid options)")
        return path_data


    def launch_methods(self, train_validation=False, check=False, select_results=None):
        # check data has been generated
        path_data = self.make_data(check=True)

        if check:
            paths_results = self._list_main_folders(EXPERIMENT_PREDS, inherit_from=path_data)
            assert len(paths_results) > 0, "results not created yet; run `--launch_methods` first"
            selected = "final" if select_results is None else select_results
            final_results = list(filter(lambda p: p.name.rsplit("_", 1)[-1] == selected, paths_results))
            if final_results:
                assert len(final_results) == 1
                return final_results[0]
            else:
                return paths_results[-1]

        # select method config depending on whether we do train_validation or testing
        if train_validation:
            assert self.methods_validation_config is not None, \
                f"Error when loading or file not found for methods_validation.yaml at path:\n" \
                f"{self.methods_validation_config_path}"
            methods_config = self.methods_validation_config
            methods_config_path = self.methods_validation_config_path
            config_file_name = EXPERIMENT_CONFIG_METHODS_VALIDATION

        else:
            methods_config = self.methods_config
            methods_config_path = self.methods_config_path
            config_file_name = EXPERIMENT_CONFIG_METHODS

        # init results folder
        path_results = self._init_folder(EXPERIMENT_PREDS, inherit_from=path_data)
        self._copy_file(methods_config_path, path_results / config_file_name)
        if self.dry:
            shutil.rmtree(path_results)

        # print data sets expected and found
        data_found = sorted(
            [p for p in path_data.iterdir() if p.is_dir()],
            key=lambda p: int(p.name)
        )

        print(f"Found data seeds: {[int(p.name) for p in data_found]}")
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

        n_launched, n_methods = 0, 0
        path_data_root = data_found[0].parent

        # launch runs that execute methods
        print("baseline methods:\n")
        experiment_name = kwargs.experiment.replace("/", "--")
        for k, (method, hparams) in enumerate(methods_config.items()):

            n_methods += 1
            seed_indices = sorted([int(p.name) for p in data_found])

            # if possible convert to range for shorter slurm command
            if seed_indices == list(range(seed_indices[0], seed_indices[-1] + 1)):
                seed_indices = range(seed_indices[0], seed_indices[-1] + 1)

            cmd = f"python '{PROJECT_DIR}/experiment/methods.py' " \
                  f"--method {method} " \
                  r"--seed \$SLURM_ARRAY_TASK_ID " \
                  r"--data_id \$SLURM_ARRAY_TASK_ID " \
                  f"--path_results '{path_results}' " \
                  f"--path_data_root '{path_data_root}' " \
                  f"--path_methods_config '{methods_config_path}' " \
                  rf"--descr '{experiment_name}-{method}-run-\$SLURM_ARRAY_TASK_ID' "
            if train_validation:
                cmd += f"--train_validation "
                
            print()
            assert YAML_RUN in hparams or hparams is None, f"Add `__run__` specification of `{method}` method in yaml"
            run_kwargs = hparams[YAML_RUN] if hparams is not None else DEFAULT_RUN_KWARGS
            cmd_args = dict(
                array_indices=seed_indices,
                mode=self.compute,
                dry=self.dry,
                prompt=False,
                output_path_prefix=f"{path_results}/logs/",
                **run_kwargs,
            )
            # create log directory here already in case there is a failure before folder creation in script
            if not self.dry:
                (path_results / "logs").mkdir(exist_ok=True, parents=True)

            # 1 job for each dataset
            n_launched += len(seed_indices)
            generate_run_commands(array_command=cmd, **cmd_args)

        print(f"\nLaunched {n_launched} runs total ({n_methods} methods)")
        return path_results


    def make_data_summary(self):
        # check results have been generated
        path_data = self.make_data(check=True)

        # init results folder
        path_plots = self._init_folder(EXPERIMENT_DATA_SUMMARY, inherit_from=path_data)
        if self.dry:
            shutil.rmtree(path_plots)

        # print data sets expected and found
        if self.data_grid_config is not None:
            n_datasets = self.n_datasets_grid * len(self.data_grid_config)

        else:
            n_datasets = self.n_datasets

        data_found = sorted([p for p in path_data.iterdir() if p.is_dir()])
        print(f"Found data seeds: {[int(p.name) for p in data_found]}")
        if len(data_found) != n_datasets:
            warnings.warn(f"\nNumber of data sets does not match data config "
                f"(got: `{len(data_found)}`, expected `{n_datasets}`).\n"
                f"data path: {path_data}\n")
            if len(data_found) < n_datasets:
                print("Exiting.")
                # return
            else:
                print(f"Taking first {n_datasets} data folders")

        elif self.verbose:
            print(f"\nLaunching summary experiment for {len(data_found)} data sets.")

        # create summary
        experiment_name = kwargs.experiment.replace("/", "--")
        cmd = f"python '{PROJECT_DIR}/experiment/data_summary.py' " \
              f"--path_data {path_data} " \
              f"--path_plots '{path_plots}' " \
              f"--descr '{experiment_name}-{path_plots.parts[-1]}' "

        generate_run_commands(
            command_list=[cmd],
            array_throttle=SLURM_SUBMISSION_MAX,
            mode=self.compute,
            hours=23,
            mins=59,
            n_cpus=2,
            n_gpus=0,
            mem=4000,
            prompt=False,
            dry=self.dry,
            output_path_prefix=self.slurm_logs_dir,
        )
        return path_plots


    def make_summary(self, train_validation=False, inject_hyperparams=False, wasser_eps=None, select_results=None):
        # check results have been generated
        path_data = self.make_data(check=True)
        path_results = self.launch_methods(check=True, train_validation=train_validation, select_results=select_results)

        # init results folder
        subdir = EXPERIMENT_SUMMARY_VALIDATION if train_validation else EXPERIMENT_SUMMARY
        path_plots = self._init_folder(EXPERIMENT_SUMMARY, inherit_from=path_results)
        if self.dry:
            shutil.rmtree(path_plots)

        # select method config depending on whether we do train_validation or testing
        if train_validation:
            assert self.methods_validation_config is not None, \
                f"Error when loading or file not found for methods_validation.yaml at path:\n" \
                f"{self.methods_validation_config_path}"
            methods_config = self.methods_validation_config
            methods_config_path = self.methods_validation_config_path
            suffix =  "_train_validation"

        else:
            methods_config = self.methods_config
            methods_config_path = self.methods_config_path
            suffix = ""
        
        # print results expected and found
        results = sorted([p for p in path_results.iterdir()])
        results_found = {}
        for j, (method, _) in enumerate(methods_config.items()):
            n_expected = self.n_datasets
            results_found[method] = list(filter(lambda p: p.name.rsplit("_", 1)[0] == method + suffix, results))
            warn = not len(results_found[method]) == n_expected
            print(f"{method + ':':50s}"
                  f"{len(results_found[method]):3d}/{n_expected}\t\t"
                  f"{'(!)' if warn else ''}"
                  f"\t{[int(p.stem.rsplit('_', 1)[1]) for p in results_found[method]] if j == 0 else ''}")

        # create summary
        experiment_name = kwargs.experiment.replace("/", "--")
        cmd = f"python '{PROJECT_DIR}/experiment/summary.py' " \
              f"--methods_config_path {methods_config_path} " \
              f"--path_data {path_data} " \
              f"--path_plots '{path_plots}' " \
              f"--path_results '{path_results}' " \
              f"--descr '{experiment_name}-{path_plots.parts[-1]}' "
              
        if train_validation:
            cmd += f"--train_validation "
        if train_validation and inject_hyperparams:
            cmd += f"--inject_hyperparams "
        if wasser_eps is not None:
            cmd += f"--wasser_eps {wasser_eps} "
        if self.only_methods is not None:
            cmd += f"--only_methods " + " ".join(self.only_methods) + " "

        generate_run_commands(
            command_list=[cmd],
            array_throttle=SLURM_SUBMISSION_MAX,
            mode=self.compute,
            hours=11,
            mins=59,
            n_cpus=4,
            n_gpus=0,
            mem=4000,
            prompt=False,
            dry=self.dry,
            output_path_prefix=self.slurm_logs_dir,
        )
        return path_plots



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, nargs="?", default="test", help="experiment config folder")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--compute", type=str, default="local")

    parser.add_argument("--data", action="store_true")
    parser.add_argument("--data_grid", action="store_true")
    parser.add_argument("--methods_train_validation", action="store_true")
    parser.add_argument("--methods", action="store_true")
    parser.add_argument("--summary_data", action="store_true")
    parser.add_argument("--summary_train_validation", action="store_true")
    parser.add_argument("--summary", action="store_true")
    
    parser.add_argument("--inject_hyperparams", action="store_true")

    parser.add_argument("--scratch", action="store_true")

    parser.add_argument("--n_datasets", type=int, help="overwrites default specified in config")
    parser.add_argument("--only_methods",nargs="*",type=str,)
    parser.add_argument("--wasser_eps", type=float)
    parser.add_argument("--select_results", type=str)
    parser.add_argument("--subdir_results", type=str)

    kwargs = parser.parse_args()

    kwargs_sum = sum([
        kwargs.data_grid,
        kwargs.data,
        kwargs.methods_train_validation,
        kwargs.methods,
        kwargs.summary_data,
        kwargs.summary_train_validation,
        kwargs.summary,
    ])
    assert kwargs_sum == 1, f"pass 1 option, got `{kwargs_sum}`"

    exp = ExperimentManager(experiment=kwargs.experiment, compute=kwargs.compute, n_datasets=kwargs.n_datasets,
                            dry=not kwargs.submit, only_methods=kwargs.only_methods, scratch=kwargs.scratch,
                            subdir_results=kwargs.subdir_results)

    if kwargs.data or kwargs.data_grid:
        _ = exp.make_data(grid=kwargs.data_grid)

    elif kwargs.methods or kwargs.methods_train_validation:
        _ = exp.launch_methods(train_validation=kwargs.methods_train_validation)

    elif kwargs.summary_data:
        _ = exp.make_data_summary()

    elif kwargs.summary or kwargs.summary_train_validation:
        _ = exp.make_summary(
                train_validation=kwargs.summary_train_validation,
                inject_hyperparams=kwargs.inject_hyperparams,
                wasser_eps=kwargs.wasser_eps,
                select_results=kwargs.select_results
            )

    else:
        raise ValueError("Unknown option passed")

# !python manager.py scm-er --data --submit --n_datasets=50
# !python manager.py scm-er --methods_train_validation --n_datasets 50 --submit --only_methods ours-lnl_u_diag ours-linear_u_diag
# !python manager.py scm-er --summary_train_validation --n_datasets 50 --submit --only_methods ours-lnl_u_diag ours-linear_u_diag
# !python manager.py scm-er --methods --submit --n_datasets=1 --only_methods kds-lnl_u_diag kds-linear_u_diag
# !python manager.py scm-er --summary --submit --n_datasets=1 --only_methods kds-lnl_u_diag kds-linear_u_diag

# problems with gies, llc, nodags,
# seems to be the changed search_theta_test_intv thing