""" 
Helper methods used during training setup. 
"""
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random


def initialized(key: random.PRNGKey, model: nn.Module, input_shape: Tuple[int, int]):
    """Initializes param dict for a model
    Args:
        key (_type_): _description_
        image_size (_type_): _description_
        model (_type_): _description_
    Returns:
        _type_: _description_
    """

    init_batch = jnp.ones((input_shape), dtype=jnp.int32)

    def init(rng, init_batch):
        return model.init(rng, init_batch, None, False)

    jit_apply = jax.jit(init, backend="cpu")
    variables = jit_apply(rng=key, init_batch=init_batch)
    return variables


def compute_tokens_seen(absolute_step, max_context):

    return absolute_step * max_context
