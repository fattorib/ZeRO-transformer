
import numpy as np 
from jax.sharding import Mesh, PartitionSpec
import jax
from jax.experimental.pjit import pjit
from src.models.GPT import model_getter
from src.partitioning.partition import set_partitions_zero, create_opt_spec
from src.training.training_utils import initialized
import optax 
from src.partitioning.pjit_train_functions import train_step
from src.partitioning.pjit_train_functions import update_opt_state
import functools
from tqdm import tqdm 
from time import time 
from jax.experimental.shard_map import shard_map

# Explicitly disable new jax array API
# import jax
# jax.config.update('jax_array', False)

# # Emulating 8 TPU cores
import os 
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':

    # CONSTANTS
    # GRAD_ACCUM_STEPS = 32
    # BATCH_SIZE = 512
    # CTX_LEN = 512
    # NUM_PASSES = 20
    # MODEL_SIZE = 'base' 

    GRAD_ACCUM_STEPS = 8
    BATCH_SIZE = 128
    CTX_LEN = 32
    NUM_PASSES = 20
    MODEL_SIZE = 'smol' 

    # Setting up device mesh
    devices = np.asarray(jax.devices())
    mesh = Mesh(devices, ["dp"])


    # Setting up model + param spec
    model = model_getter(MODEL_SIZE, return_cfg=False)
    rng = jax.random.PRNGKey(23)


    batch_tok = jax.random.randint(rng, shape=(1, CTX_LEN), maxval=50257, minval=0)

    param_shape = jax.eval_shape(model.init, rng, batch_tok)
    param_spec = PartitionSpec() # model params are replicated along DP axis
    params = initialized(rng, model, input_shape=(1, model.block_size))


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

    # Create optimizer state spec
    opt_state_shapes = jax.eval_shape(tx.init, param_shape)
    grad_param_spec = set_partitions_zero(param_shape)
    opt_state_spec = create_opt_spec(grad_param_spec, opt_state_shapes)

    train_step_shmap = shard_map(
            functools.partial(train_step, model = model, accum_steps = GRAD_ACCUM_STEPS),
            mesh=mesh,
            in_specs=(param_spec,PartitionSpec('dp'),param_spec),
            out_specs=(grad_param_spec,PartitionSpec()),
        )

    with mesh:

        rng, dropout_rng = jax.random.split(rng, 2)

        update_opt_state_pjit = pjit(
            functools.partial(update_opt_state),
            in_axis_resources=(grad_param_spec, opt_state_spec, param_spec),
            out_axis_resources=(param_spec,opt_state_spec),
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

        print(f"Host batch shape (bs, accum, ctx): {batch.shape}")

        # 1 step to compile
        grads, metrics = train_step_shmap(params, batch, dropout_rng)
        params, opt_state = update_opt_state_pjit(grads, opt_state, params, tx)

        times_grads = []
        times_update_opt = []
        for i in tqdm(range(NUM_PASSES)):

            dropout_rng, rng = jax.random.split(dropout_rng)

            batch = jax.numpy.ones(shape=(BATCH_SIZE, CTX_LEN), maxval=256, minval=0)

            batch = batch.reshape(
                GRAD_ACCUM_STEPS,
                init_batch.shape[0] // GRAD_ACCUM_STEPS,
                CTX_LEN,
            ).transpose(1, 0, 2)

            t0 = time() 
            grads, metrics = train_step_shmap(params, batch, rng,GRAD_ACCUM_STEPS, model)
            times_grads.append(time() - t0)

            # t0 = time()
            # params, opt_state = update_opt_state_pjit(grads, opt_state, params, tx)
            # times_update_opt.append(time() - t0)

        print(
            f"ZeRO Step - Global BS {BATCH_SIZE} - accum steps {GRAD_ACCUM_STEPS} - Num Executions {NUM_PASSES}"
        )
        print(f"Mesh Layout (dp): (8)")
        print(f"Model Size: {MODEL_SIZE}")
        print(f"Mean Grad Time {np.mean(times_grads):.4f} Seconds")
        print(f"Mean Opt Time {np.mean(times_update_opt):.4f} Seconds")
        print()