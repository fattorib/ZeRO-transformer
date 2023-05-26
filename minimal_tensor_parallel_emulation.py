import argparse
from functools import partial
from time import time

import jax
import numpy as np
from jax.sharding import Mesh
from tqdm import tqdm
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
from typing import Any
from jax.lax import with_sharding_constraint
from typing import Callable


def parse():
    parser = argparse.ArgumentParser(
        description="pjit/jit training emulator & benchmarker"
    )
    parser.add_argument("--emulation", default=False, action="store_true")
    parser.add_argument("--iter", default=10, type=int)
    parser.add_argument("--dp", default=8, type=int)
    parser.add_argument("--mp", default=1, type=int)
    args = parser.parse_args()
    return args


def train_step(
    params: Any,
    batch: jnp.array,
    batch_spec: Any = None,
    grad_fn: Callable = None,
    dp_axis_size: int = None,
    per_device_parallelism: int = None,
):
    """
    Computes loss/grads for a single batch of data, optionally with gradient accumulation
    """
    batch_size = jnp.shape(batch)[0]
    microbatch_size = dp_axis_size * per_device_parallelism
    num_micro_steps = batch_size // microbatch_size
    assert num_micro_steps * microbatch_size == batch_size

    # reshape to add a microbatch dimension
    batch = batch.reshape((num_micro_steps, microbatch_size) + batch.shape[1:])
    batch = with_sharding_constraint(
        batch, batch_spec
    )  # keep dp sharding for microbatches

    # accumulate gradients
    def cumul_minibatch_step(carry, x_y):
        cumul_loss, cumul_grads = carry
        minibatch = x_y
        loss, grads = grad_fn(to_bf16(params), minibatch)
        cumul_grads = jax.tree_map(jnp.add, cumul_grads, grads)

        return (cumul_loss + loss, cumul_grads), None

    grad_init = to_bf16(jax.tree_util.tree_map(jnp.zeros_like, params))

    (loss, grads), _ = jax.lax.scan(
        cumul_minibatch_step, init=(jnp.zeros(()), grad_init), xs=batch
    )

    metrics = {
        "train/loss": loss,
        "train/ppl": jnp.exp(loss),
    }

    return grads, metrics


# Emulating 8 TPU cores
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == "__main__":

    rng = jax.random.PRNGKey(23)
    args = parse()

    if args.emulation:
        print("Emulating 8 TPU cores")
        grad_acc_steps = 8
        batch_size = 128
        d_model = 1536
        n_layer = 8
        num_iter = args.iter

        def to_bf16(t):
            return t

        import contextlib

        maybe_profiler = contextlib.nullcontext()

    else:
        grad_acc_steps = 64
        batch_size = 512
        d_model = 2048
        n_layer = 16
        num_iter = args.iter

        # only works on TPU
        def to_bf16(t):
            return jax.tree_map(
                lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t
            )

        maybe_profiler = jax.profiler.trace(
            "jax-trace-pjit", create_perfetto_link=False
        )

    # Setting up device mesh
    mesh = Mesh(np.array(jax.devices()).reshape(args.dp), ("dp"))

    # indicates batch dim is split across dp axis
    batch_sharding = NamedSharding(mesh, P("dp", None))
    no_shard = NamedSharding(mesh, None)

    def create_mini_model(rng):
        # create mini model that does a sequence of matmul + residual
        params = jax.random.normal(rng, shape=(n_layer, d_model, d_model))
        return params

    model_params = create_mini_model(rng)

    def fwd(batch: jnp.array, params: jnp.array):
        def layer(x, param):
            p = param
            y = jnp.dot(x, p)
            return y + x, None

        x, _ = jax.lax.scan(layer, batch, params)
        return x

    param_spec = no_shard
    batch_grad_spec = no_shard
    microbatch_spec = NamedSharding(mesh, P(None, "dp", *(None,) * (1)))

    params = jax.device_put(model_params, param_spec)

    def loss_fn(params, batch):
        out = fwd(batch, params)
        loss = jnp.mean(out)
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

    per_device_parallelism = (
        batch_size // args.dp // grad_acc_steps
    )  # compute per-device batch size
    with mesh:
        train_step_dp = jax.jit(
            partial(
                train_step,
                grad_fn=grad_fn,
                per_device_parallelism=per_device_parallelism,
                dp_axis_size=args.dp,
                batch_spec=microbatch_spec,
            ),
            in_shardings=(param_spec, batch_sharding),
            out_shardings=(param_spec, no_shard),
        )

        init_batch = jax.numpy.ones(shape=(batch_size, d_model))
        batch = jax.device_put(init_batch, batch_sharding)
        grads, metrics = train_step_dp(params, batch)

        start = time()
        with tqdm(total=num_iter) as pbar:
            for i in range(num_iter):
                with maybe_profiler:
                    batch = jax.numpy.ones(shape=(batch_size, d_model))
                    batch = jax.device_put(batch, batch_sharding)
                    grads, metrics = train_step_dp(params, batch)
                    grads[0].block_until_ready()
                    pbar.update()

        print(metrics)
        total_time = time() - start

        print(
            f"Global BS {batch_size} - accum steps {grad_acc_steps} - Num Executions {num_iter}"
        )
        print(f"Mesh Layout (dp,mp): {(args.dp,args.mp)}")
        print(f"Total Time: {total_time:.4f}s")
