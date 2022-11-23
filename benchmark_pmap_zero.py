"""
Runs without OOM but is:
    - Extremely slow
    - Brittle (forgetting to delete grads anywhere will cause OOM)
"""

import argparse
from time import time
from typing import Any

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.jax_utils import replicate
from flax.training.common_utils import shard, shard_prng_key
from typing import Callable, Union
import flax.linen as nn

from src.models.GPT import model_getter
from src.training.training_utils import initialized
from flax.training import train_state
from tqdm import tqdm 

def parse():
    parser = argparse.ArgumentParser(description="Pjit benchmarking code")

    parser.add_argument("--grad-accum", default=32, type=int)

    parser.add_argument("--batch-size", default=512, type=int)

    parser.add_argument("--ctx", default=512, type=int)

    args = parser.parse_args()
    return args

def create_zero_train_state(
    rng: jax.random.PRNGKey,
    learning_rate_fn: Union[float, Callable],
    weight_decay: float,
    model: nn.Module,
):
    """
    Initializes parameters and returns a _simplified_ TrainState without an optimizer.

    Returns TrainState, opt_state, GradientTransformation
    
    """
    params = initialized(rng, model, input_shape=(1, model.block_size))

    # This mask turns off weight decay for bias terms, LN terms and position embeddings
    mask = jax.tree_map(
        lambda x: x.ndim != 1 and x.shape != (model.block_size, model.embedding_dim),
        params,
    )

    tx = optax.chain(
        optax.clip(1.0),
        optax.adamw(
            learning_rate=learning_rate_fn,
            weight_decay=weight_decay,
            mask=mask,
            b2=0.95,
        ),
    )

    opt_state = tx.init(params)
    state = train_state.TrainState(
        apply_fn=model.apply,
        params=params,
        step = 0, 
        opt_state=None, 
        tx = None
    )
    return state, tx, opt_state

@partial(jax.pmap, axis_name= "batch", static_broadcasted_argnums=(3,4))
def train_step(
    params: Any,
    batch: jnp.array,
    rng_key: jax.random.PRNGKey = None,
    accum_steps: int = 8,
    model: Any = None
):
    """
    Computes loss/grads for a single batch of data, pmeans across all devices to sync grads
    and returns loss/grads
    """

    def get_minibatch(batch, grad_idx):
        return jax.tree_util.tree_map(
            lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False, axis=1),
            batch,
        )

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

    def loss_and_grad(grad_idx):
        minibatch = get_minibatch(batch, grad_idx) if grad_idx is not None else batch

        loss, grads = grad_fn(params, minibatch)

        return loss, grads


    init_minibatch = (0.0, jax.tree_util.tree_map(jnp.zeros_like, params))

    # accumulate gradients
    def cumul_minibatch_step(grad_idx, cumul_loss_grad):
        cumul_loss, cumul_grads = cumul_loss_grad
        loss, grads = loss_and_grad(grad_idx)
        cumul_loss, cumul_grads = jax.tree_util.tree_map(
            jnp.add, (cumul_loss, cumul_grads), (loss, grads)
        )

        return cumul_loss, cumul_grads

    # this logic could probably be movied into cumul_minibatch_step,
    loss, grads = jax.lax.fori_loop(
        0,
        accum_steps,
        cumul_minibatch_step,
        init_minibatch,
    )

    loss, grads = jax.tree_util.tree_map(lambda x: x / accum_steps, (loss, grads))

    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")

    metrics = {
        "Train LM Loss": loss,
        "Train LM PPL": jnp.exp(loss),
    }

    return loss, grads, metrics

@partial(jax.pmap, axis_name= "shard", devices = jax.local_devices())
def slice_grads(grads, device_index):
    grad_slice = jax.tree_util.tree_map(lambda x: x[device_index, ...], grads)
    return grad_slice

@partial(jax.pmap, axis_name= "shard", static_broadcasted_argnums = (3,), devices = jax.local_devices())
def update_sharded_state(grads: Any, 
        optimizer_state: Any,
        params: Any, 
        optimizer: Any,
        device_index: Any
        ):
    """
    Updates the sharded optimizer state
    """


    # These two lines update the specific shard of state/parameters sitting on device 'i'
    updates, new_opt_state = optimizer.update(grads, optimizer_state, params)
    new_param_slice = optax.apply_updates(params, updates)

    new_params = jax.lax.all_gather(new_param_slice, axis_name = 'shard')
    return new_params, new_opt_state


def partition_shard(xs, local_device_count, devices):
    """
    Partitions optimizer state by splitting the first dimension of buffers across local devices
    """
    return jax.tree_util.tree_map(
      lambda x: x.reshape((local_device_count, -1) + x.shape[1:]) if x.ndim > 0 else jax.device_put_replicated(x, devices), xs)


@partial(jax.pmap, devices = jax.local_devices())
def split_sharded_device_array(arr, device_index):
    """
    Pmappable way to get shards of param/grad pytrees

    By default, anything that is not a pmapped function applied to a ShardedDeviceArray will 
    force a gather to the 0'th device and return a DeviceArray. 

    See: https://github.com/google/jax/issues/2535


    """
    local_device_count = 8
    return jax.tree_util.tree_map(lambda x: x.reshape(local_device_count, -1, x.shape[-1])[device_index, ...] if x.ndim >= 2 else x.reshape(local_device_count,-1)[device_index, ...],arr)

@partial(jax.pmap, devices = jax.local_devices())
def deshard(xs):
    """
    Pmappable way to get reshape a sharded device array containing param replicas

    By default, anything that is not a pmapped function applied to a ShardedDeviceArray will 
    force a gather to the 0'th device and return a DeviceArray. 

    See: https://github.com/google/jax/issues/2535

    """
    return jax.tree_util.tree_map(lambda x: x.reshape((-1,x.shape[-1])) if x.ndim > 2 else x.reshape(-1), xs)
    

if __name__ == "__main__":
    

    args = parse()

    # CONSTANTS
    GRADIENT_ACCUMULATION_STEPS = args.grad_accum
    GLOBAL_BATCH_SIZE = args.batch_size
    SEQ_LEN = args.ctx
    MODEL_SIZE = "base"
    NUM_PASSES = 25

    model = model_getter(MODEL_SIZE, return_cfg=False, dtype = jnp.bfloat16)
    local_device_count = jax.local_device_count()

    # State Creation, etc
    init_rng = jax.random.PRNGKey(0)

    state, optimizer, opt_state = create_zero_train_state(init_rng,
        3e-4,
        weight_decay=0.1,
        model=model,)
    
    batch = jax.random.randint(
            init_rng, (GLOBAL_BATCH_SIZE, SEQ_LEN), maxval=50257, minval=0
        )
    batch = batch.reshape(
        GRADIENT_ACCUMULATION_STEPS,
        GLOBAL_BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS,
        SEQ_LEN,
    ).transpose(1, 0, 2)

    batch = shard(batch)
    rng_sharded = shard_prng_key(init_rng)

    # optimizer state is completely partitioned across devices, needs to be done pre replication of state
    # once this is passed through our pmapped function, all arrays here are ShardedDeviceArrays sitting on all local_devices
    opt_state = partition_shard(opt_state, local_device_count, jax.local_devices()) 
    opt_state = jax.pmap(lambda x: x)(opt_state) # shard opt state to free up memory

    # replicate state across devices
    params = state.params
    del state

    # replicate state across devices
    params = replicate(params)

    rng, _ = jax.random.split(init_rng)

    # compute loss, grads, can add accumulation here
    loss, grads, metrics = train_step(
        params,
        batch,
        rng_sharded, 
        GRADIENT_ACCUMULATION_STEPS, 
        model
    ) 
    
    
    grads = split_sharded_device_array(grads, jax.numpy.arange(jax.device_count())) 
    params = split_sharded_device_array(params, jax.numpy.arange(jax.device_count()))
        
    
    # update sharded state
    params, opt_state = update_sharded_state(grads,
        opt_state,
        grads,
        optimizer, 
        device_index = jax.numpy.arange(jax.device_count())
        )

    params = deshard(params) #reshape params

    del grads # manually free grad mem since scope exists outside of train_step function

    times = []
    
    for _ in tqdm(range(NUM_PASSES)):
        rng, batch_rng = jax.random.split(rng, 2)
        test_batch = jax.random.randint(
            batch_rng, (GLOBAL_BATCH_SIZE, SEQ_LEN), maxval=50257, minval=0
        )
        test_batch = test_batch.reshape(
            GRADIENT_ACCUMULATION_STEPS,
            GLOBAL_BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS,
            SEQ_LEN,
        ).transpose(1, 0, 2)

        test_batch = shard(test_batch)
        rng_sharded = shard_prng_key(rng)

        t0 = time()
        loss, grads, metrics = train_step(
            params,
            batch,
            rng_sharded, 
            GRADIENT_ACCUMULATION_STEPS, 
            model
        )
        
        grads = split_sharded_device_array(grads, jax.numpy.arange(jax.device_count())) 
        params = split_sharded_device_array(params, jax.numpy.arange(jax.device_count()))
        
        # update sharded state
        params, opt_state = update_sharded_state(grads,
            opt_state,
            params,
            optimizer, 
            device_index = jax.numpy.arange(jax.device_count())
        )

        del grads

        params = deshard(params) # reshape params

        times.append(time() - t0)

    print(
    f"Optimized Pmap Step - Global BS {GLOBAL_BATCH_SIZE} - accum steps {GRADIENT_ACCUMULATION_STEPS} - Num Executions {NUM_PASSES}"
    )
    print(f"Mean Batch Time {np.mean(times):.4f} Seconds")
    print()