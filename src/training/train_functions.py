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
        loss, grads = grad_fn(to_bf16(params), minibatch)
        cumul_grads = jax.tree_map(jnp.add, cumul_grads, grads)
        return (cumul_loss + loss, cumul_grads), None

    grad_init = to_bf16(jax.tree_util.tree_map(jnp.zeros_like, params))

    with jax.named_scope("scanned_microbatch"):
        (loss, grads), _ = jax.lax.scan(
            cumul_minibatch_step, init=(jnp.zeros(()), grad_init), xs=batch
        )

    with jax.named_scope("gradient_all_reduce"):
        grads = jax.lax.pmean(grads, axis_name="dp")
        loss = jax.lax.pmean(loss, axis_name="dp")

    loss, grads = jax.tree_util.tree_map(lambda x: x / accum_steps, (loss, grads))

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
    params = to_bf16(params)
    _, loss = model.apply(
        {"params": params["params"]}, x=batch, labels=batch, train=False
    )

    loss = jax.lax.pmean(loss, axis_name="dp")

    metrics = {"validation/loss": loss, "validation/ppl": jnp.exp(loss)}

    return metrics


def update_opt_state(
    params: Any, grads: Any, opt_state: Any, optimizer: Any, tp_spec: Any
):
    # updates the optimizer state and params
    params = with_sharding_constraint(params, tp_spec)
    grads = with_sharding_constraint(grads, tp_spec)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state
