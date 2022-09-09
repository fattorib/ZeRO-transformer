""" 
Helper methods used during training setup. 
"""
import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.training import train_state
import flax.linen as nn
from typing import Union, Callable


def initialized(key: random.PRNGKey, model: nn.Module):
    """Initializes param dict for a model
    Args:
        key (_type_): _description_
        image_size (_type_): _description_
        model (_type_): _description_
    Returns:
        _type_: _description_
    """
    rng_init, batch_init = jax.random.split(key, num=2)

    init_batch = random.randint(
        batch_init,
        shape=(1, model.block_size),
        maxval=model.vocab_size,
        minval=0,
    )

    @jax.jit
    def init(rng, init_batch):
        return model.init(rng, init_batch, None, False)

    variables = init(rng=rng_init, init_batch=init_batch)
    return variables


def create_train_state(
    rng: random.PRNGKey,
    learning_rate_fn: Union[float, Callable],
    weight_decay: float,
    model: nn.Module,
    grad_accum_steps: int,
):
    """Creates initial `TrainState` for model."""
    params = initialized(rng, model)

    # This mask turns off weight decay for bias terms, LN terms and position embeddings
    mask = jax.tree_map(
        lambda x: x.ndim != 1
        and x.shape != (model.block_size, model.embedding_dim),
        params,
    )

    tx = optax.adamw(
        learning_rate=learning_rate_fn,
        weight_decay=weight_decay,
        mask=mask,
        b2=0.95,
    )

    if grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=grad_accum_steps)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    return state
