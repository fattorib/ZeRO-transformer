""" 
Helper methods used during training setup. 
"""
from typing import Callable, List, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import random


def to_precision(t, dtype: jnp.dtype):
    return jax.tree_map(lambda x: x.astype(dtype) if x.dtype == dtype else x, t)


def initialized(key: random.PRNGKey, model: nn.Module, dtype: jnp.dtype):
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

    def init(rng, init_batch):
        return model.init(rng, init_batch, None, False)

    jit_apply = jax.jit(init, backend="cpu")
    variables = jit_apply(rng=rng_init, init_batch=init_batch)
    variables = to_precision(variables, dtype)
    return variables


def create_train_state(
    rng: random.PRNGKey,
    learning_rate_fn: Union[float, Callable],
    weight_decay: float,
    model: nn.Module,
    grad_accum_steps: int,
    dtype: jnp.dtype = jnp.float32,
):
    """Creates initial `TrainState` for model."""
    params = initialized(rng, model, dtype)

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

    if grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=grad_accum_steps)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    return state


def step_to_seq_len(
    current_step: int, stages: List, max_steps: int, max_context=1024
) -> int:
    """
    Returns the sequence length at a specific training step (if using staged sequence training)

    Example::
        >>> step_to_seq_len(400,[256, 512], 1200, 1024)
        256

        >>> step_to_seq_len(800, [256, 512], 1200, 1024)
        512

        >>> step_to_seq_len(1300, [256, 512], 1200, 1024)
        1024

    """
    steps_per_stage = max_steps // len(stages)

    remainder = current_step % steps_per_stage
    stage_idx = (current_step - remainder) // steps_per_stage

    if stage_idx > (len(stages) - 1):
        return max_context
    else:
        return stages[stage_idx]


def compute_tokens_seen(
    current_step: int, stages: List, max_steps: int, max_context=1024
) -> int:
    """Compute the number of tokens seen by the model. Requires scaling by batch size after

    Example::
        >>> compute_tokens_seen(400,[256, 512], 1200, 1024)
        102400

        >>> compute_tokens_seen(800, [256, 512], 1200, 1024)
        256000

        >>> compute_tokens_seen(1300, [256, 512], 1200, 1024)
        563200
    """

    steps_per_stage = max_steps // len(stages)

    # get how many steps we have passed in the current stage
    remainder = current_step % steps_per_stage

    stage_idx = (current_step - remainder) // steps_per_stage
    if stage_idx > 0:
        tokens_seen = 0
        for stage in range(0, stage_idx):
            if stage < len(stages):
                tokens_seen += stages[stage] * steps_per_stage

        if stage_idx >= len(stages):
            remaining_steps = current_step - max_steps
            tokens_seen += max_context * remaining_steps
        else:
            tokens_seen += stages[stage_idx] * remainder
        return tokens_seen
    else:
        # end is scaled by BS - This is constant
        return stages[0] * remainder
