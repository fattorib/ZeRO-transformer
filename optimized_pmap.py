""" 

Pmap code with gradient accumulation communication @ end of batch

"""

from typing import Any 
import jax.numpy as jnp 
import jax 
from functools import partial

@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3,))
def train_step(
    state: Any,
    batch: jnp.array,
    rng_key: jax.random.PRNGKey = None,
    accum_steps: int = 8
):
    """
    Train on a single batch
    This means that the batch will be size (local_bs*grad_accum, ctx) instead of (local_bs, ctx)

    Designed to perform gradient and loss communication once per _batch_ instead of once per mini batch 
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
        minibatch = get_minibatch(batch, grad_idx) if grad_idx is not None else batch

        loss, grads = grad_fn(state.params, minibatch)

        return loss, grads

    init_minibatch = (
        0.0,
        jax.tree_util.tree_map(jnp.zeros_like, state.params)
    )

    # accumulate gradients
    def cumul_minibatch_step(grad_idx, cumul_loss_grad):
        cumul_loss, cumul_grads = cumul_loss_grad
        loss, grads = loss_and_grad(grad_idx)
        cumul_loss, cumul_grads = jax.tree_util.tree_map(
            jnp.add, (cumul_loss, cumul_grads), (loss, grads)
        )

        return cumul_loss, cumul_grads

    loss, grads = jax.lax.fori_loop(
        0,
        accum_steps,
        cumul_minibatch_step,
        init_minibatch,
    )

    loss, grads = jax.tree_util.tree_map(
        lambda x: x / accum_steps, (loss, grads)
    )

    loss = jax.lax.pmean(loss, axis_name='batch')
    grads = jax.lax.pmean(grads, axis_name='batch')

    # only update train_state at the end of a single full batch
    new_state = state.apply_gradients(
        grads=grads,
    )

    metrics = {
        "Train LM Loss": loss,
        "Train LM PPL": jnp.exp(loss),
    }

    return new_state, metrics

@partial(jax.pmap, axis_name="batch")
def naive_train_step(state: Any, batch: jnp.array, rng_key: jax.random.PRNGKey = None):
    """Train on a single micro batch of data """

    def loss_fn(params):
        _, loss = state.apply_fn(
            {"params": params["params"]},
            x=batch,
            labels=batch,
            train=True,
            rngs={"dropout": rng_key},
        )
        return loss

    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)
    # NOTE: compute all-reduce mean for gradients and loss
    # Ex: If we have 8 devices, each device takes the gradients from the other 7 and averages them all together
    # that way, all device replicas have the same gradients and optimization step can occur in parallel
    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")

    new_state = state.apply_gradients(
        grads=grads,
    )

    metrics = {
        "Train LM Loss": loss,
        "Train LM PPL": jnp.exp(loss),
    }


    return new_state, metrics
