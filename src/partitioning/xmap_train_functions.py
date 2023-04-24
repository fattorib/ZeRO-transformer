"""  
Move these to a separate function temporarily
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax.lax import with_sharding_constraint


def to_bf16(t):
    return jax.tree_map(
        lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t
    )


def to_f32(t):
    return jax.tree_map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t
    )


# we xmap this
def train_step(
    params: Any,
    batch: jnp.array,
    rng_key: jax.random.PRNGKey = None,
    accum_steps: int = 8,
    model: Any = None,
):
    """
    Computes loss/grads for a single batch of data, pmeans across all devices/hosts to sync grads
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
        loss, grads = grad_fn(to_bf16(params), minibatch)

        return loss, grads

    init_minibatch = (0.0, to_bf16(jax.tree_util.tree_map(jnp.zeros_like, params)))

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

    loss, grads = jax.tree_util.tree_map(lambda x: x / accum_steps, (loss, grads))

    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")

    metrics = {
        "train/loss": loss,
        "train/ppl": jnp.exp(loss),
    }

    return grads, metrics


def eval_step(
    params: Any,
    batch: jnp.array,
    model: Any,
):
    _, loss = model.apply(
        {"params": params["params"]}, x=batch, labels=batch, train=False
    )

    loss = jax.lax.pmean(loss, axis_name="batch")

    metrics = {"validation/loss": loss, "validation/ppl": jnp.exp(loss)}

    return metrics


def update_opt_state(
    grads: Any, optimizer_state: Any, params: Any, optimizer: Any, grad_spec: Any
):
    """
    Updates the sharded optimizer state and parameters. Expects grads, optimizer_state, and params
    to have the same partition specs
    """

    params = with_sharding_constraint(params, grad_spec)
    grads = with_sharding_constraint(grads, grad_spec)
    updates, new_opt_state = optimizer.update(grads, optimizer_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state
