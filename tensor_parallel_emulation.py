import argparse
import functools
from functools import partial
from time import time

import jax
import numpy as np
import optax
from jax.sharding import Mesh
from omegaconf import OmegaConf
from tqdm import tqdm
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding, NamedSharding
from jax.sharding import PartitionSpec as P
from src.models.GPT import model_getter
from src.partitioning.partition import create_opt_spec, set_partitions_tp
from src.training.training_utils import initialized
import jax.numpy as jnp 
from typing import Any
from jax.lax import with_sharding_constraint




def parse():
    parser = argparse.ArgumentParser(description="pjit/jit training emulator & benchmarker")
    parser.add_argument("--emulation", default=False, action="store_true")
    parser.add_argument("--iter", default=10, type=int)
    args = parser.parse_args()
    return args

def train_step(
    params: Any,
    batch: jnp.array,
    rng_key: jax.random.PRNGKey = None,
    accum_steps: int = 8,
    model: Any = None,
    model_mp_spec: Any = None
):
    """
    Computes loss/grads for a single batch of data.
    """
    
    # this now has to be modified to only take a batch of shape (BS,CTX) and pull data of size (MB)
    batch_size,context = batch.shape

    params = with_sharding_constraint(params, model_mp_spec)

    def get_minibatch(grad_idx):
        minibatch = jax.lax.dynamic_slice(batch, (grad_idx,0), slice_sizes = (batch_size//accum_steps, context))
        return minibatch

    def loss_fn(params, batch):
        _, loss = model.apply(
            {"params": params["params"]},
            x=batch,
            labels=batch,
            train=True,
            rngs={"dropout": rng_key},
        )
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

    # accumulate gradients
    def cumul_minibatch_step(carry, _):
        grad_idx, cumul_loss, cumul_grads = carry
        minibatch = get_minibatch(grad_idx) if grad_idx is not None else batch
        loss, grads = grad_fn(to_bf16(params), minibatch)
        cumul_loss, cumul_grads = jax.tree_util.tree_map(
            jnp.add, (cumul_loss, cumul_grads), (loss, grads)
        )
        return (grad_idx + minibatch.shape[0], loss, grads), None 
    
    (idx,loss,grads), _ = jax.lax.scan(cumul_minibatch_step, init = (0, 0.0, to_bf16(jax.tree_util.tree_map(jnp.zeros_like, params))), xs = None, length = accum_steps)

    grads = with_sharding_constraint(grads, model_mp_spec)

    loss, grads = jax.tree_util.tree_map(lambda x: x / accum_steps, (loss, grads))

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
        GRAD_ACCUM_STEPS = 8
        BATCH_SIZE = 128
        CTX_LEN = 32
        NUM_PASSES = args.iter
        MODEL_SIZE = "test"

        def to_bf16(t):
            return t

    else:
        GRAD_ACCUM_STEPS = 8
        BATCH_SIZE = 512
        CTX_LEN = 1024
        NUM_PASSES = args.iter
        MODEL_SIZE = "base"

        # only works on TPU
        def to_bf16(t):
            return jax.tree_map(
                lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t
            )

    # Setting up device mesh
    # following new jax array tutorial

    # 4-way dp / 2 way tp
    
    # named sharding is easier to follow along with 
    mesh = Mesh(np.array(jax.devices()).reshape(4,2), ('dp','mp'))

    # indicates batch dim is split across dp axis
    batch_sharding = jax.sharding.NamedSharding(mesh, P('dp', None))
    
    # # Setting up model + param spec
    model = model_getter(MODEL_SIZE, return_cfg=False)

    batch_tok = jax.random.randint(rng, shape=(1, CTX_LEN), maxval=50257, minval=0)
    params = initialized(rng, model, input_shape=(1, model.block_size))
    param_shape = jax.eval_shape(model.init, rng, batch_tok)

    param_spec = set_partitions_tp(param_shape, mesh)

    configs = OmegaConf.load("conf/model_config.yaml")
    model_info = configs[MODEL_SIZE]

    L, H, Q, T = (
        model_info.N,
        model_info.num_head,
        model_info.embedding_dim // model_info.num_head,
        CTX_LEN,
    )

    # shard model across desired TP axes
    params = jax.device_put(params, param_spec)

    with mesh:
        train_step_dp = jax.jit(
            partial(train_step, model=model, accum_steps=GRAD_ACCUM_STEPS, model_mp_spec = param_spec),
        )
        rng, dropout_rng = jax.random.split(rng, 2)

        init_batch = jax.numpy.ones(shape=(BATCH_SIZE, CTX_LEN), dtype=jax.numpy.int32)

        batch = jax.device_put(init_batch, batch_sharding)
        
        grads, metrics = train_step_dp(params, batch, dropout_rng)

        start = time()
        
        for i in tqdm(range(NUM_PASSES)):

            dropout_rng, rng = jax.random.split(dropout_rng)

            batch = jax.numpy.ones(shape=(BATCH_SIZE, CTX_LEN), dtype=jax.numpy.int32)
            batch = jax.device_put(batch, batch_sharding)
            grads, metrics = train_step_dp(params, batch, dropout_rng)

            params = jax.tree_map(lambda x,y : x - 0.01*y, params, grads)
            print(metrics)
        total_time = time() - start

        print(
            f"TP Step - Global BS {BATCH_SIZE} - accum steps {GRAD_ACCUM_STEPS} - Num Executions {NUM_PASSES}"
        )
        print(f"Mesh Layout (dp,mp): (4,2)")
        print(f"Model Size: {MODEL_SIZE}")
        print(f"Total Time: {total_time:.4f}s")
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(param_shape))

        flops_per_token = 6 * param_count + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * BATCH_SIZE

        total_flops = flops_per_iter * NUM_PASSES
        v2_flops = 45e12 

        effective_tflops = total_flops / (total_time)

        mfu = effective_tflops / v2_flops

        print(f"Param Count: {param_count}")
        # from https://github.com/kingoflolz/mesh-transformer-jax/blob/4c15ee74a8ce5d4bf2aee2462638c1b33c8288a8/tpuv38_example.py
        print(f"Effective TFLOPS: {total_flops / (total_time)/1e12:.06}")
        print(f"MFU: {100*mfu:.06}")