import argparse
from functools import partial
from time import time

import jax
import numpy as np
import optax
from jax.sharding import Mesh
from omegaconf import OmegaConf
from tqdm import tqdm
from jax.sharding import PartitionSpec as P
from src.models.GPT import model_getter
from src.partitioning.partition import (
    create_opt_spec,
    set_partitions_rules,
    _get_partition_rules_tp,
)
import jax.numpy as jnp
from typing import Any
from jax.experimental.shard_map import shard_map
from jax.experimental.pjit import pjit
from jax.lax import with_sharding_constraint
import contextlib


def parse():
    parser = argparse.ArgumentParser(
        description="pjit/jit training emulator & benchmarker"
    )
    parser.add_argument("--emulation", default=False, action="store_true")
    parser.add_argument("--iter", default=10, type=int)
    parser.add_argument("--dp", default=4, type=int)
    parser.add_argument("--mp", default=2, type=int)
    args = parser.parse_args()
    return args


def train_step(
    params: Any,
    batch: jnp.array,
    accum_steps: int = 8,
    model: Any = None,
):
    """
    Computes loss/grads for a single batch of data, optionally with gradient accumulation
    """
    _, context = batch.shape

    # reshape to add a microbatch dimension
    batch = batch.reshape(accum_steps, -1, context)
    params = to_bf16(params)

    def loss_fn(params, batch):
        _, loss = model.apply(
            {"params": params["params"]},
            x=batch,
            labels=batch,
            train=False,
        )
        return jnp.mean(loss)

    grad_fn = jax.value_and_grad(loss_fn)

    # accumulate gradients
    def cumul_minibatch_step(carry, x_y):
        cumul_loss, cumul_grads = carry
        minibatch = x_y
        loss, grads = grad_fn(params, minibatch)
        cumul_grads = jax.tree_map(jnp.add, cumul_grads, grads)
        return (cumul_loss + loss, cumul_grads), None

    grad_init = to_bf16(jax.tree_util.tree_map(jnp.zeros_like, params))

    with jax.named_scope("scanned_microbatch"):
        (loss, grads), _ = jax.lax.scan(
            cumul_minibatch_step, init=(jnp.zeros(()), grad_init), xs=batch
        )

    with jax.named_scope("gradient_all_reduce"):
        grads = jax.lax.pmean(grads, axis_name="dp")

    loss, grads = jax.tree_util.tree_map(lambda x: x / accum_steps, (loss, grads))

    metrics = {
        "train/loss": loss,
        "train/ppl": jnp.exp(loss),
    }

    return grads, metrics


def update_opt_state(
    params: Any, grads: Any, opt_state: Any, optimizer: Any, tp_spec: Any
):
    # updates the optimizer state and params
    params = with_sharding_constraint(params, tp_spec)
    grads = with_sharding_constraint(grads, tp_spec)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


# Emulating 8 TPU cores
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == "__main__":

    rng = jax.random.PRNGKey(23)
    args = parse()

    # per-device shapes can be printed with shard_map
    # this allows us to investigate model sharding issues

    if args.emulation:
        print("Emulating 8 TPU cores")
        GRAD_ACCUM_STEPS = 8
        BATCH_SIZE = 256
        CTX_LEN = 32
        NUM_PASSES = args.iter
        MODEL_SIZE = "test_no_tp"

        def to_bf16(t):
            return t

        maybe_profiler = contextlib.nullcontext()

    else:
        GRAD_ACCUM_STEPS = 32
        BATCH_SIZE = 128  # assuming 4 hosts and a total BS of 0.5M tokens
        CTX_LEN = 1024
        NUM_PASSES = args.iter
        MODEL_SIZE = "base"

        # only works on TPU
        def to_bf16(t):
            return jax.tree_map(
                lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t
            )

        maybe_profiler = contextlib.nullcontext()

    # Setting up device mesh
    mesh = Mesh(np.array(jax.devices()).reshape(args.dp, args.mp), ("dp", "mp"))

    # indicates batch dim is split across dp axis
    batch_spec = P("dp", None)
    no_shard = P(None)

    # # Setting up model + param spec
    model = model_getter(MODEL_SIZE, return_cfg=False)

    batch_tok = jax.random.randint(rng, shape=(1, CTX_LEN), maxval=50257, minval=0)

    # init is a bit odd now since the transformer fwd pass contains collective operations now
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(no_shard, no_shard),
        out_specs=no_shard,
        check_rep=False,
    )
    def init(rng, init_batch):
        return model.init(rng, init_batch, None, False)

    params = jax.jit(init)(rng, (jnp.ones((1, CTX_LEN), dtype=jnp.int32)))

    param_shape = jax.tree_map(
        lambda x: x.size, params
    )  # we literally just do this to get keys

    mask = jax.tree_map(
        lambda x: x.ndim != 1 and x.shape != (model.block_size, model.embedding_dim),
        params,
    )

    # not scale invariant!
    tx = optax.chain(
        optax.clip(1.0),
        optax.lion(
            learning_rate=0.001,
            weight_decay=0.1,
            mask=mask,
        ),
    )

    if args.mp > 1:
        param_spec = set_partitions_rules(
            param_shape, mesh, _get_partition_rules_tp, axis_name="mp"
        )

        # jax.debug.visualize_array_sharding(params['params']['wte']['embedding'])
        params = shard_map(lambda x: x, mesh, in_specs=no_shard, out_specs=param_spec)(
            params
        )
        # jax.debug.visualize_array_sharding(params['params']['wte']['embedding'])

        grad_spec = param_spec

    else:
        param_spec = no_shard
        grad_spec = param_spec

    opt_state_shapes = jax.eval_shape(tx.init, params)
    opt_state_spec = create_opt_spec(param_spec, opt_state_shapes)

    with mesh:
        opt_state = pjit(
            tx.init,
            in_axis_resources=(param_spec,),
            out_axis_resources=opt_state_spec,
        )(params)

    configs = OmegaConf.load("conf/model_config.yaml")
    model_info = configs[MODEL_SIZE]

    L, H, Q, T = (
        model_info.N,
        model_info.num_head,
        model_info.embedding_dim // model_info.num_head,
        CTX_LEN,
    )

    train_step_tp = jax.jit(
        shard_map(
            partial(train_step, model=model, accum_steps=GRAD_ACCUM_STEPS),
            in_specs=(param_spec, batch_spec),
            out_specs=(grad_spec, P(None)),
            mesh=mesh,
            check_rep=False,
        )
    )

    with mesh:
        update_opt_step_tp = pjit(
            partial(update_opt_state, optimizer=tx, tp_spec=grad_spec),
            in_axis_resources=(param_spec, grad_spec, opt_state_spec),
            out_axis_resources=(param_spec, opt_state_spec),
            donate_argnums=0,
        )

    rng, dropout_rng = jax.random.split(rng, 2)

    init_batch = jax.numpy.ones(shape=(BATCH_SIZE, CTX_LEN), dtype=jax.numpy.int32)

    grads, metrics = train_step_tp(params, init_batch)

    with mesh:
        params, opt_state = update_opt_step_tp(params, grads, opt_state)

    start = time()

    for i in tqdm(range(NUM_PASSES)):
        with maybe_profiler:
            dropout_rng, rng = jax.random.split(dropout_rng)

            batch = jax.numpy.ones(shape=(BATCH_SIZE, CTX_LEN), dtype=jax.numpy.int32)
            grads, metrics = train_step_tp(params, batch)
            with mesh:
                params, opt_state = update_opt_step_tp(params, grads, opt_state)

    jnp.zeros((10, 10)).block_until_ready()
    total_time = time() - start
    print(metrics)

    print(
        f"TP Step - Global BS {BATCH_SIZE} - accum steps {GRAD_ACCUM_STEPS} - Num Executions {NUM_PASSES}"
    )
    print(f"Mesh Layout (dp,mp): {(args.dp,args.mp)}")
    print(f"Model Size: {MODEL_SIZE}")
    print(f"Total Time: {total_time:.4f}s")

    param_count = sum(p for p in jax.tree_util.tree_leaves(param_shape))

    flops_per_token = 6 * param_count + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * BATCH_SIZE

    total_flops = flops_per_iter * NUM_PASSES
    v2_flops = 180e12

    effective_tflops = total_flops / (total_time)

    mfu = effective_tflops / v2_flops

    print(f"Param Count: {param_count}")
    # from https://github.com/kingoflolz/mesh-transformer-jax/blob/4c15ee74a8ce5d4bf2aee2462638c1b33c8288a8/tpuv38_example.py
    print(f"Effective TFLOPS: {total_flops / (total_time)/1e12:.06}")
    print(f"MFU: {100*mfu:.06}")
