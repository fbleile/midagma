import functools
from typing import NamedTuple, Any

import jax
from jax import numpy as jnp
from jax import random
import numpy as onp
import tensorflow as tf
import tensorflow_datasets as tfds


class Batch(NamedTuple):
    """
    Data batch container for gradient descent on the KDS.

    Args:
        x (ndarray): Data vector [..., d]
        env_indicator (ndarray): One-hot indicator vector of shape ``[n_envs,]``
            encoding which environment this batch belongs to
    """
    x: Any
    env_indicator: Any

    dtype = dict(
        x=jnp.float32,
        env_indicator=jnp.int32,
    )

    @staticmethod
    def get_sharding(sharding):
        """
        Defines how to shard the batch across devices.
    
        We use replicated sharding (PartitionSpec()) so everything is replicated.
        Works with modern JAX sharding objects like NamedSharding.
        """
        return dict(
            x=sharding,
            env_indicator=sharding,
        )





def _dtype_to_tensor_spec(v):
    return tf.TensorSpec(None, v) if isinstance(v, tf.dtypes.DType) else v


def _tensor_spec_to_dtype(v):
    return v.dtype if isinstance(v, tf.TensorSpec) else v


def structured_py_function(func, inp, t_out, name=None):
    """
    Workaround for allowing `tf.py_function` that returns structured elements like dict or NamedTuple
    Taken from https://github.com/tensorflow/tensorflow/issues/27679#issuecomment-522578000
    """
    def wrapped_func(*flat_inp):
        reconstructed_inp = tf.nest.pack_sequence_as(inp, flat_inp, expand_composites=True)
        result = func(*reconstructed_inp)
        return tf.nest.flatten(result, expand_composites=True)
    flat_t_out = tf.nest.flatten(t_out, expand_composites=True)
    flat_out = tf.py_function(
        func=wrapped_func,
        inp=tf.nest.flatten(inp, expand_composites=True),
        Tout=[_tensor_spec_to_dtype(v) for v in flat_t_out],
        name=name)
    spec_out = tf.nest.map_structure(_dtype_to_tensor_spec, t_out, expand_composites=True)
    out = tf.nest.pack_sequence_as(spec_out, flat_out, expand_composites=True)
    return out


def _double_cache_and_shard(ds, sharding):
    """
    Keep at least two batches on the accelerator, for processing and memory loading of batches to occur in parallel.
    Adapted from https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/dataset.py
    """
    shard_tree = Batch.get_sharding(sharding)
    shard_leaves, shard_treedef = jax.tree_util.tree_flatten(shard_tree)
    batch = None

    for next_batch in ds:
        assert next_batch is not None

        # shard Batch pytree across devices (replicated if sharding is replicated)
        leaves, treedef = jax.tree_util.tree_flatten(next_batch)
        assert treedef == shard_treedef
        next_batch = jax.tree_util.tree_unflatten(
            treedef,
            [jax.device_put(x, shard) for x, shard in zip(leaves, shard_leaves)]
        )

        if batch is not None:
            yield Batch(**batch)

        batch = next_batch

    if batch is not None:
        yield Batch(**batch)



def _sample_kds_batch(_, *, rng, x, batch_size):

    # sample random env
    n_envs = len(x)
    env = rng.choice(n_envs, replace=True)
    env_indicator = jnp.eye(n_envs)[env]
    x = x[env]

    # sample batch of observations
    if x.shape[0] > batch_size:
        x = rng.choice(x, size=(batch_size,), replace=False)

    # yield dict that initializes a Batch object
    return dict(
        x=x,
        env_indicator=env_indicator,
    )

from typing import Any, Iterator


def _make_jax_batcher(seed: int, x: Any, batch_size: int) -> Iterator[Batch]:
    """
    Infinite generator yielding stadion Batch objects:
      Batch(x=[B,d], env_indicator=[n_envs])
    Uses jax.random only.
    """
    key = random.PRNGKey(int(seed))

    # normalize x to list-of-env arrays
    if isinstance(x, (list, tuple)):
        xs = [jnp.asarray(a) for a in x]
    else:
        x = jnp.asarray(x)
        xs = [x] if x.ndim == 2 else [x[i] for i in range(x.shape[0])]

    n_envs = int(len(xs))
    if n_envs <= 0:
        raise ValueError("x must contain at least one environment")

    while True:
        key, k_env, k_perm = random.split(key, 3)

        env = random.randint(k_env, shape=(), minval=0, maxval=n_envs)  # scalar int
        Xenv = xs[int(env)]
        m = int(Xenv.shape[0])

        B = int(min(int(batch_size), m))
        perm = random.permutation(k_perm, m)
        idx = perm[:B]

        xb = Xenv[idx]
        env_indicator = jnp.eye(n_envs, dtype=jnp.int32)[env]

        yield Batch(x=xb, env_indicator=env_indicator)

def make_dataloader(seed, device, x, batch_size):
    # device arg kept for signature compatibility; we don't need it here
    yield from _make_jax_batcher(seed=int(seed), x=x, batch_size=int(batch_size))


# def make_dataloader(seed, sharding, x, batch_size):
#     """
#     Create a dataloader for gradient descent optimization of the KDS.

#     Args:
#         seed (int)
#         sharding: a JAX Sharding object (e.g., NamedSharding(mesh, PartitionSpec()))
#         x (Dataset): ndarray [n,d], ndarray [m,n,d], or list of [n_i,d]
#         batch_size (int)

#     Return:
#         generator yielding Batch objects
#     """
#     rng = onp.random.default_rng(seed)
#     f = functools.partial(_sample_kds_batch, rng=rng, x=x, batch_size=batch_size)

#     ds = tf.data.Dataset.from_tensor_slices([0]).repeat(None)  # loop indefinitely
#     ds = ds.map(
#         lambda *args: structured_py_function(func=f, inp=[*args], t_out=Batch.dtype),
#         deterministic=True,
#         num_parallel_calls=1,
#     )
#     ds = ds.prefetch(1)
#     ds = tfds.as_numpy(ds)
#     ds = _double_cache_and_shard(ds, sharding)
#     yield from ds
