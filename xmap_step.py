
import numpy as np 
from jax.sharding import Mesh
import jax
from src.models.GPT import model_getter
from src.training.training_utils import initialized
import optax 
from src.partitioning.xmap_train_functions import train_step, eval_step, to_bf16
from src.partitioning.xmap_train_functions import update_opt_state
from src.partitioning.partition import set_partitions_zero, create_opt_spec
from functools import partial
import functools
from tqdm import tqdm 
from time import time 
from jax.experimental.maps import xmap
import argparse
from jax.experimental.pjit import pjit
from omegaconf import OmegaConf


def parse():
    parser = argparse.ArgumentParser(description="xmap training emulator & benchmarker")

    parser.add_argument("--emulation",default=False,
        action="store_true")
    
    parser.add_argument("--iter",default=10, type = int)
    
    args = parser.parse_args()
    return args
# Emulating 8 TPU cores
import os 
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#First step is to get regular dp working in xmap
if __name__ == '__main__':

    args = parse()

    if args.emulation:
        print("Emulating 8 TPU cores")
        GRAD_ACCUM_STEPS = 8
        BATCH_SIZE = 128
        CTX_LEN = 32
        NUM_PASSES = args.iter
        MODEL_SIZE = 'smol' 

    else:
        GRAD_ACCUM_STEPS = 8
        BATCH_SIZE = 512
        CTX_LEN = 1024
        NUM_PASSES = args.iter
        MODEL_SIZE = 'base' 
   
    # Setting up device mesh
    devices = np.asarray(jax.devices())
    mesh = Mesh(devices, ["dp"])

    # Setting up model + param spec
    model = model_getter(MODEL_SIZE, return_cfg=False)
    rng = jax.random.PRNGKey(23)
    
    configs = OmegaConf.load("conf/model_config.yaml")

    model_info = configs[MODEL_SIZE]

    L, H, Q, T = model_info.N, model_info.num_head, model_info.embedding_dim//model_info.num_head, CTX_LEN


    batch_tok = jax.random.randint(rng, shape=(1, CTX_LEN), maxval=50257, minval=0)
    params = initialized(rng, model, input_shape=(1, model.block_size))


    param_shape = jax.eval_shape(model.init, rng, batch_tok)

    # Create optax optimizer
    mask = jax.tree_map(
            lambda x: x.ndim != 1 and x.shape != (model.block_size, model.embedding_dim),
            params,
        )

    tx = optax.chain(
        optax.clip(1.0),
        optax.adamw(
            learning_rate=6e-4,
            weight_decay=0.01,
            mask=mask,
            b2=0.95,
        ),
    )

    # axis_list_params = jax.tree_map(lambda x: [...], params)

    in_axes =(
        [...], 
        ['batch', ...], 
        [...], 
    )
    out_axes = (
        [...],
        [...]
    )

    # standard data parallel training step with xmap!
    train_step_xmap = xmap(
            partial(train_step, model = model, accum_steps = GRAD_ACCUM_STEPS),
            in_axes=in_axes,
            out_axes=out_axes,
            axis_resources={"batch": "dp"}
        )
    
    eval_axes = (
        [...], 
        ['batch', ...], 
    )
    eval_step_xmap = xmap(
        partial(eval_step, model = model), 
        in_axes=eval_axes,
        out_axes=[...],
        axis_resources={"batch": "dp"}
    )

    # optimizer state update with pjit
    opt_state_shapes = jax.eval_shape(tx.init, params) # length 2 tuple

    grad_param_spec = set_partitions_zero(param_shape)
    opt_state_spec = create_opt_spec(grad_param_spec, opt_state_shapes)

    with mesh:

        rng, dropout_rng = jax.random.split(rng, 2)
    
        params = jax.device_get(params)

        opt_state = pjit(
            tx.init,
            in_axis_resources=None,
            out_axis_resources=opt_state_spec
        )(params)
        
        update_opt_state_pjit = pjit(
            functools.partial(update_opt_state, optimizer = tx, grad_spec = grad_param_spec),
            in_axis_resources=(grad_param_spec, opt_state_spec, grad_param_spec),
            out_axis_resources=(None,opt_state_spec),
        )

        print("Optimizer State Sharded Sucessfully")

        init_batch = jax.numpy.ones(shape=(BATCH_SIZE, CTX_LEN), dtype=jax.numpy.int32)


        batch = init_batch.reshape(
            GRAD_ACCUM_STEPS,
            init_batch.shape[0] // GRAD_ACCUM_STEPS,
            CTX_LEN,
        ).transpose(1, 0, 2)
        batch = batch.reshape(jax.local_device_count(), 
                              init_batch.shape[0] // (jax.local_device_count()* GRAD_ACCUM_STEPS), 
                              GRAD_ACCUM_STEPS, CTX_LEN)


        print(f"Host batch shape (device, bs, accum, ctx): {batch.shape}")

        grad_shard = pjit(lambda x:x, in_axis_resources=None, out_axis_resources=grad_param_spec)

        # 1 step to compile
        grads, metrics = train_step_xmap(params, batch, dropout_rng)

        start = time()
        for i in tqdm(range(NUM_PASSES)):

            dropout_rng, rng = jax.random.split(dropout_rng)

            # batch = jax.random.randint(key = dropout_rng, shape=(BATCH_SIZE, CTX_LEN), maxval=256, minval=0)
            batch = jax.numpy.ones(shape=(BATCH_SIZE, CTX_LEN), dtype = jax.numpy.int32)

            batch = batch.reshape(
                GRAD_ACCUM_STEPS,
                init_batch.shape[0] // GRAD_ACCUM_STEPS,
                CTX_LEN,
            ).transpose(1, 0, 2)
            batch = batch.reshape(jax.local_device_count(), init_batch.shape[0] // (jax.local_device_count()* GRAD_ACCUM_STEPS), GRAD_ACCUM_STEPS, CTX_LEN)

            grads, metrics = train_step_xmap(params, batch, dropout_rng)

            # shard to respective devices
            grads = grad_shard(grads)
            params = grad_shard(params)


            params,opt_state = update_opt_state_pjit(grads, opt_state, params)

            del grads 

        total_time = time() - start

        print(
            f"ZeRO Step - Global BS {BATCH_SIZE} - accum steps {GRAD_ACCUM_STEPS} - Num Executions {NUM_PASSES}"
        )
        print(f"Mesh Layout (dp): (8)")
        print(f"Model Size: {MODEL_SIZE}")
        print(f"Total Time: {total_time:.4f}s")
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(param_shape))

        flops_per_token = 6*param_count + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * BATCH_SIZE
        
        total_flops = flops_per_iter*NUM_PASSES
        v2_flops = 180e12 # from https://paperswithcode.com/paper/performance-and-power-evaluation-of-ai/review/

        effective_tflops = total_flops / (total_time)

        mfu = effective_tflops/v2_flops

        print(f"Param Count: {param_count}")
        # from https://github.com/kingoflolz/mesh-transformer-jax/blob/4c15ee74a8ce5d4bf2aee2462638c1b33c8288a8/tpuv38_example.py
        print(f"Effective TFLOPS: {total_flops / (total_time)/1e12:.06}")
        print(f"MFU: {100*mfu:.06}")
