# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as sla

def safe_inv_M(W: np.ndarray, s: float, *, eps: float = 1e-12, debug: bool = True):
    d = W.shape[0]
    I = np.eye(d, dtype=W.dtype)

    # 1) basic checks
    if debug:
        print("W stats:",
              "min", np.nanmin(W), "max", np.nanmax(W),
              "has_nan", np.isnan(W).any(),
              "has_inf", np.isinf(W).any())
        print("s =", s)

    A = s * I - (W * W)

    if debug:
        print("A = sI - W*W stats:",
              "min", np.nanmin(A), "max", np.nanmax(A),
              "has_nan", np.isnan(A).any(),
              "has_inf", np.isinf(A).any())

        # conditioning / singularity diagnostics
        try:
            condA = np.linalg.cond(A)
            print("cond(A) =", condA)
        except Exception as e:
            print("cond(A) failed:", repr(e))

        try:
            sign, logdet = np.linalg.slogdet(A)
            print("slogdet(A): sign =", sign, "logabsdet =", logdet)
        except Exception as e:
            print("slogdet(A) failed:", repr(e))

        # smallest singular value is a very direct “how close to singular”
        try:
            svals = np.linalg.svd(A, compute_uv=False)
            print("sigma_min(A) =", float(np.min(svals)), "sigma_max(A) =", float(np.max(svals)))
        except Exception as e:
            print(W)
            print("svd(A) failed:", repr(e))

    # 2) robust inversion: use solve (more stable) + ridge if needed
    try:
        M = sla.solve(A, I, assume_a="gen", check_finite=True)
        if debug and (np.isnan(M).any() or np.isinf(M).any()):
            print("M from solve has NaN/Inf -> will ridge and retry")
            raise ValueError("solve produced NaN/Inf")
        return M

    except Exception as e:
        if debug:
            print("solve(A,I) failed:", repr(e))

        # ridge BEFORE inversion
        A_reg = A + eps * I
        if debug:
            print(f"Retry with ridge eps={eps:g}")
        M = sla.solve(A_reg, I, assume_a="gen", check_finite=True)
        if debug:
            print("after ridge: has_nan", np.isnan(M).any(), "has_inf", np.isinf(M).any())
        return M
