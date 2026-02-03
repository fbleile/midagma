#!/usr/bin/env python
# coding: utf-8

from dagma import utils
from dagma.linear import DagmaLinear
from dagma.nonlinear import DagmaMLP, DagmaNonlinear

from notreks.mi_tests import get_I_from_full_pairwise_tests, summarize_I
from notreks.notreks import (
    PSTRegularizer,
    TCCRegularizer,   # <-- NEW
    pst,
    trek_cycle_coupling_value_gradW,
    get_no_trek_pairs,
)
from logger import LogConfig, build_default_logger
import logging
import torch

# ---- Generate data ----
utils.set_random_seed(4)
n, d, s0 = 500, 10, 40
graph_type, sem_type = "ER", "gauss"

B_true = utils.simulate_dag(d, s0, graph_type)
W_true = utils.simulate_parameter(B_true)
X = utils.simulate_linear_sem(W_true, n, sem_type)

print(X.shape)
print(B_true)

logger = build_default_logger(level=logging.INFO)  # doesn’t matter; printing disabled

log_cfg = LogConfig(
    enabled=True,
    print_to_console=False,
    store_csv=False,
    store_jsonl=False,
    keep_in_memory=True,
)

# ---- Build I from pairwise tests on full data ----
I_test = get_I_from_full_pairwise_tests(
    X,
    alpha=0.001,
    test="spearman",
    num_perm=500,
    seed=0,
    bonferroni=True,
    undirected=False,
)
print("I_test:", I_test.shape)
summarize_I(I_test, d=d)

# ---- or build I from "no trek" oracle (your helper) ----
I = get_no_trek_pairs(torch.as_tensor(B_true, dtype=torch.double, device="cpu"))
print("I_no_trek:", I.shape)
summarize_I(I, d=d)

# ============================================================
# 1) PST trek regularizer run
# ============================================================
trek_pst = PSTRegularizer(
    I=I,
    seq="log",
    weight=1.0,
    kwargs={"K_log": 40, "eps_inv": 1e-8, "s":5.},
    mode="log",   # 'opt' or 'log'
)

model_lin_pst = DagmaLinear(
    loss_type="l2",
    trek_reg=trek_pst,
    logger=logger,
    log_cfg=log_cfg,
)

W_est_pst = model_lin_pst.fit(X, lambda1=0.02, max_iter=6e4, mu_factor=0.1)
acc_pst = utils.count_accuracy(B_true, W_est_pst != 0)
print("Linear acc (PST):", acc_pst)

model_lin_pst._slog.visualize(include=[
    "obj_total",
    "score_datafit",
    "reg_dag_value",
    "reg_trek_value",
    "grad_score_norm",
    "grad_dag_norm",
    "grad_trek_norm",
])
print("run dir (PST):", model_lin_pst._slog.run_dir)

print("pst(W_true):", pst(torch.as_tensor(W_true, dtype=torch.double, device="cpu"), trek_pst.cfg["I"], agg="mean").item())
print("pst(W_est_pst):", pst(torch.as_tensor(W_est_pst, dtype=torch.double, device="cpu"), trek_pst.cfg["I"], agg="mean").item())


# ============================================================
# 2) Spectral-radius coupling trek regularizer run
# ============================================================
trek_sr = TCCRegularizer(
    I=I,
    cycle_penalty="spectral",
    version="approx_trek_graph",
    weight=1.,
    w=100.0,
    n_iter=10,
    eps=1e-12,
    mode="opt",  # set 'opt' if you want it to affect optimization
)

model_lin_sr = DagmaLinear(
    loss_type="l2",
    trek_reg=trek_sr,
    logger=logger,
    log_cfg=log_cfg,
)

W_est_sr = model_lin_sr.fit(X, lambda1=0.02, max_iter=6e4, mu_factor=0.1)
acc_sr = utils.count_accuracy(B_true, W_est_sr != 0)
print("Linear acc (SR coupling):", acc_sr)

model_lin_sr._slog.visualize(include=[
    "obj_total",
    "score_datafit",
    "reg_dag_value",
    "reg_trek_value",
    "grad_score_norm",
    "grad_dag_norm",
    "grad_trek_norm",
])
print("run dir (SR):", model_lin_sr._slog.run_dir)

# optional: direct value check on the final W (independent of model internals)
val_true, _ = trek_cycle_coupling_value_gradW(torch.as_tensor(W_true, dtype=torch.double, device="cpu"), trek_sr.cfg["I"], w=trek_sr.cfg["w"], n_iter=trek_sr.cfg["n_iter"], eps=trek_sr.cfg["eps"])
val_est, _  = trek_cycle_coupling_value_gradW(torch.as_tensor(W_est_sr, dtype=torch.double, device="cpu"), trek_sr.cfg["I"], w=trek_sr.cfg["w"], n_iter=trek_sr.cfg["n_iter"], eps=trek_sr.cfg["eps"])
print("sr_coupling(W_true):", float(val_true.item()))
print("sr_coupling(W_est_sr):", float(val_est.item()))
