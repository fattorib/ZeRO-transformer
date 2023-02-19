""" 
Hybrid train step with pmapped train step and pjitted optimizer step
"""

import numpy as np 
import jax
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
from src.models.GPT import model_getter
from src.partitioning.partition import set_partitions_zero, create_opt_spec
from src.training.training_utils import initialized
import optax 
from src.partitioning.pjit_train_functions import update_opt_state, train_step
import functools
from tqdm import tqdm 
from time import time 
from flax.training.common_utils import shard, shard_prng_key
import flax


# Explicitly disable new jax array API
import jax
jax.config.update('jax_array', False)


# # Emulating 8 TPU cores
# import os 
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':

    # GRAD_ACCUM_STEPS = 8
    # BATCH_SIZE = 128
    # CTX_LEN = 32
    # NUM_PASSES = 100
    # MODEL_SIZE = 'test' 

    GRAD_ACCUM_STEPS = 32
    BATCH_SIZE = 512
    CTX_LEN = 512
    NUM_PASSES = 20
    MODEL_SIZE = 'base' 

    # Setting up device mesh
    devices = np.asarray(jax.devices())
    mesh = Mesh(devices, ["dp"])

    model = model_getter(MODEL_SIZE, return_cfg=False)
    rng = jax.random.PRNGKey(23)

    batch_tok = jax.random.randint(rng, shape=(1, CTX_LEN), maxval=256, minval=0)

    param_shape = jax.eval_shape(model.init, rng, batch_tok)
    param_spec = None # model params are replicated along DP axis
    params = initialized(rng, model, input_shape=(1, model.block_size))

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

    opt_state_shapes = jax.eval_shape(tx.init, param_shape)
    grad_param_spec = set_partitions_zero(param_shape)
    opt_state_spec = create_opt_spec(grad_param_spec, opt_state_shapes)

    params = flax.jax_utils.replicate(params, devices=jax.local_devices())

    with mesh:

        rng, dropout_rng = jax.random.split(rng, 2)

        train_step_pmap = jax.pmap(
            train_step,
            axis_name="batch",
            static_broadcasted_argnums=(3, 4)
        )

        update_opt_state_pjit = pjit(
            functools.partial(update_opt_state),
            in_axis_resources=(grad_param_spec, opt_state_spec, grad_param_spec),
            out_axis_resources=(grad_param_spec,opt_state_spec),
            static_argnums=(3,)
        )


        # Create actual optimizer state 
        params = jax.device_get(params)

        opt_state = pjit(
            tx.init,
            in_axis_resources=param_spec,
            out_axis_resources=opt_state_spec
        )(params)

        print("Optimizer State Sharded Sucessfully")

        init_batch = jax.numpy.ones(shape=(BATCH_SIZE, CTX_LEN), dtype=jax.numpy.int32)

        batch = init_batch.reshape(
            GRAD_ACCUM_STEPS,
            init_batch.shape[0] // GRAD_ACCUM_STEPS,
            CTX_LEN,
        ).transpose(1, 0, 2)
        
        batch = shard(batch)

        rng_sharded = shard_prng_key(dropout_rng)

        grads, metrics = train_step_pmap(params, batch, rng_sharded,GRAD_ACCUM_STEPS, model)

        params = pjit(
                lambda x:x, 
                in_axis_resources=None, 
                out_axis_resources = grad_param_spec
            )(params)

        grads = pjit(
                lambda x:x, 
                in_axis_resources=None, 
                out_axis_resources = grad_param_spec
            )(grads)

        params, opt_state = update_opt_state_pjit(grads, opt_state, params, tx)
        
        del grads 

        params = pjit(
                lambda x:x, 
                in_axis_resources=(grad_param_spec,), 
                out_axis_resources = None
            )(params)


        times_grads = []
        times_update_opt = []
        for i in tqdm(range(NUM_PASSES)):

            dropout_rng, rng = jax.random.split(dropout_rng)

            batch = jax.numpy.ones(shape=(BATCH_SIZE, CTX_LEN), dtype=jax.numpy.int32)

            batch = batch.reshape(
                GRAD_ACCUM_STEPS,
                init_batch.shape[0] // GRAD_ACCUM_STEPS,
                CTX_LEN,
            ).transpose(1, 0, 2)
            
            batch = shard(batch)
            rng_sharded = shard_prng_key(dropout_rng)

            t0 = time() 
            grads, metrics = train_step_pmap(params, batch, rng_sharded,GRAD_ACCUM_STEPS, model)
            
            # shard params and grads

            grads = pjit(
                lambda x:x, 
                in_axis_resources=None, 
                out_axis_resources = grad_param_spec
            )(grads)

            params = pjit(
                lambda x:x, 
                in_axis_resources=None, 
                out_axis_resources = grad_param_spec
            )(params)

            times_grads.append(time() - t0)

            t0 = time()
            params, opt_state = update_opt_state_pjit(grads, opt_state, params, tx)
            times_update_opt.append(time() - t0)

            del grads 

            params = pjit(
                lambda x:x, 
                in_axis_resources=(grad_param_spec,), 
                out_axis_resources = None
            )(params)

            print(metrics["Train LM Loss"][0])

        
        print(
            f"ZeRO Step - Global BS {BATCH_SIZE} - accum steps {GRAD_ACCUM_STEPS} - Num Executions {NUM_PASSES}"
        )
        print(f"Mesh Layout (dp): (8)")
        print(f"Model Size: {MODEL_SIZE}")
        print(f"Mean Grad Time {np.mean(times_grads):.4f} Seconds")
        print(f"Mean Opt Time {np.mean(times_update_opt):.4f} Seconds")
        print()