import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from jax.experimental import PartitionSpec
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit, with_sharding_constraint

from main import train_step
from src.models.GPT import model_getter
from src.training.training_utils import TrainState, get_optimizer
from src.utils.partitioning import create_opt_spec, set_partitions
from time import time
from tqdm import tqdm 


if __name__ == '__main__':
    # CONSTANTS
    GRAD_ACCUM_STEPS = 32
    BATCH_SIZE = 512
    CTX_LEN = 1024


    # Benchmarking Constants
    NUM_PASSES = 50

    # Setting up device mesh
    mesh_shape = (1, 8)
    devices = np.asarray(jax.devices()).reshape(*mesh_shape)
    mesh = Mesh(devices, ("dp", "mp"))

    # Setting up model + param spec
    model = model_getter("test", return_cfg=False)
    rng = jax.random.PRNGKey(23)
    batch_tok = jax.random.randint(rng, shape=(1, CTX_LEN), maxval=50257, minval=0)
    param_shape = jax.eval_shape(model.init, rng, batch_tok)
    param_spec = set_partitions(param_shape)

    # Setting up optimizer + opt spec
    tx = get_optimizer(3e-4, 0.01, model, param_shape)

    def init_state(params):

        return TrainState.create(
            apply_fn=model.apply,
            tx=tx,
            params=params,
        )

    opt_state_shapes = jax.eval_shape(tx.init, param_shape)
    opt_state_spec = create_opt_spec(param_spec, opt_state_shapes)

    # Setting up state spec
    state_spec = TrainState(
        params=param_spec,
        opt_state=opt_state_spec,
        tx=tx,
        step=None,
        apply_fn=model.apply,
    )

    def train_step(
        state: Any, batch: jnp.array, rng_key: jax.random.PRNGKey = None, param_spec: Any = None, grad_accum_steps: int = None
    ):
        """Train on a single Gradient-Accumulation batch
        This means that the batch will be size (local_bs*grad_accum, ctx) instead of (local_bs, ctx)

        """
        def get_minibatch(batch, grad_idx):
            return jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False, axis=1),
                batch,
            )
        
        def loss_fn(params, batch):
            _, loss = state.apply_fn(
                {"params": params["params"]},
                x=batch,
                labels=batch,
                train=True,
                rngs={"dropout": rng_key},
            )
            return loss 

        grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

        def loss_and_grad(grad_idx):
            minibatch = (
                    get_minibatch(batch, grad_idx) if grad_idx is not None else batch
                )
            minibatch = with_sharding_constraint(minibatch, PartitionSpec('dp'))

            loss, grads = grad_fn(state.params, minibatch)

            grads = with_sharding_constraint(grads, param_spec)

            return loss, grads 

        init_minibatch = (
            0.0, 
            with_sharding_constraint(
                        jax.tree_util.tree_map(jnp.zeros_like, state.params), param_spec
            )
        )

        # accumulate gradients
        def cumul_minibatch_step(grad_idx, cumul_loss_grad):
            cumul_loss, cumul_grads= cumul_loss_grad
            loss, grads = loss_and_grad(grad_idx)
            cumul_loss, cumul_grads = jax.tree_util.tree_map(
                jnp.add, (cumul_loss, cumul_grads), (loss, grads)
            )
            cumul_grads = with_sharding_constraint(cumul_grads, param_spec)
            return cumul_loss, cumul_grads

        loss, grads = jax.lax.fori_loop(
                    0,
                    grad_accum_steps,
                    cumul_minibatch_step,
                    init_minibatch,
                )
        grads = with_sharding_constraint(grads, param_spec)
        # sum -> mean
        loss, grads = jax.tree_util.tree_map(
            lambda x: x / grad_accum_steps, (loss, grads)
        )

        grads = with_sharding_constraint(grads, param_spec)

        # only update train_state at the end of a single full batch 
        new_state = state.apply_gradients(
            grads=grads,
        )

        metrics = {
            "Train LM Loss": loss,
            "Train LM PPL": jnp.exp(loss),
        }

        return new_state, metrics



    with mesh:
        train_step_pjit = pjit(
            functools.partial(train_step, param_spec=param_spec, grad_accum_steps = GRAD_ACCUM_STEPS),
            in_axis_resources=(state_spec, PartitionSpec("dp"), None),
            out_axis_resources=(state_spec, None),
        )

        rng, dropout_rng = jax.random.split(rng, 2)
        init_batch = jax.numpy.ones(shape=(BATCH_SIZE, CTX_LEN), dtype=jax.numpy.int32)

        # shard params across mesh
        sharded_params = pjit(
            functools.partial(model.init, train=False),
            in_axis_resources=(None, None),
            out_axis_resources=(param_spec),
        )(rng, init_batch)

        state = pjit(
            init_state,
            in_axis_resources=(param_spec,),
            out_axis_resources=(state_spec),
        )(sharded_params)

        print("State Sharded Sucessfully")

        batch = jax.tree_util.tree_map(
                            lambda x: x.reshape((GRAD_ACCUM_STEPS,)+ (x.shape[0]//GRAD_ACCUM_STEPS,) + x.shape[1:]),
                            init_batch,
                        ).transpose(1, 0, 2)
        
        times = []
        for _ in tqdm(range(NUM_PASSES)):
            rng, batch_rng = jax.random.split(rng, 2)
            
            # Create a test batch of data
            test_batch = jax.random.randint(shape=(BATCH_SIZE, CTX_LEN), dtype=jax.numpy.int32)
            test_batch = jax.tree_util.tree_map(
                            lambda x: x.reshape((GRAD_ACCUM_STEPS,)+ (x.shape[0]//GRAD_ACCUM_STEPS,) + x.shape[1:]),
                            test_batch,
                        ).transpose(1, 0, 2)

            t0 = time()
            state, metrics = train_step_pjit(state, batch, dropout_rng)
            times.append(time() - t0)


    print(
        f"Optimized Pmap Step - Global BS {BATCH_SIZE} - accum steps {GRAD_ACCUM_STEPS} - Num Executions {NUM_PASSES}"
    )
    print(f"Mean Batch Time {np.mean(times):.4f} Seconds")
    print()