"""  
Move these to a separate function temporarily
"""


from typing import Any
from jax.experimental.pjit import with_sharding_constraint
from jax.experimental import PartitionSpec


import jax
import jax.numpy as jnp
import optax

def train_step(
    params: Any,
    batch: jnp.array,
    rng_key: jax.random.PRNGKey = None,
    accum_steps: int = 8,
    model: Any = None,
    grad_param_spec: Any = None
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

    loss, grads = jax.lax.fori_loop(
        0,
        accum_steps,
        cumul_minibatch_step,
        init_minibatch,
    )

    loss, grads = jax.tree_util.tree_map(lambda x: x / accum_steps, (loss, grads))

    metrics = {
        "Train LM Loss": loss,
        "Train LM PPL": jnp.exp(loss),
    }
    grads = with_sharding_constraint(grads, grad_param_spec) 

    return grads, metrics


def update_opt_state(
    grads: Any,
    optimizer_state: Any,
    params: Any,
    optimizer: Any,
    grad_param_spec: Any, 
    opt_state_spec: Any, 

):
    """
    Updates the sharded optimizer state
    """
    optimizer_state = with_sharding_constraint(optimizer_state, opt_state_spec)
    grads = with_sharding_constraint(grads, grad_param_spec) 

    updates, new_opt_state = optimizer.update(grads, optimizer_state, params)

    new_params = optax.apply_updates(params, updates)
    optimizer_state = with_sharding_constraint(optimizer_state, opt_state_spec)

    return new_params, new_opt_state