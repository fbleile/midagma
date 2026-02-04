# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from jax import random

def np_rng_from_key(key: random.PRNGKey) -> np.random.Generator:
    # stable deterministic conversion; 32-bit is enough
    seed = int(random.bits(key, 32))
    return np.random.default_rng(seed)
