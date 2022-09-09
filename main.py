import numpy as np
from jax import random
from typing import Any
import jax.numpy as jnp 

# Flax imports
import jax
import optax
import flax 

from src.models.GPT import model_getter
from src.training.training_utils import create_train_state
from functools import partial

# Logging/Config Stuffs
import argparse
import logging
from omegaconf import OmegaConf

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def parse():
    parser = argparse.ArgumentParser(description="Transformer Training")

    parser.add_argument("--cfg", default="conf/config.yaml", type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse()
    cfg = OmegaConf.load(args.cfg)

    num_devices = jax.device_count()
    
    # create state, etc
    state = flax.jax_utils.replicate(state)







@partial(jax.pmap, axis_name='batch')
def train_step(state: Any, batch: jnp.array, rng_key: random.PRNGKey):
    """Train for a single step."""

    def loss_fn(params):
        _, loss = state.apply_fn(
            {"params": params["params"]},
            x = batch,
            labels = batch,
            train=True,
            rngs={"dropout": rng_key},
        )

        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)

    # compute all-reduce mean for gradients and loss
    # Ex: If we have 8 devices, each device takes the gradients from the other 7 and averages them all together
    # that way, all device replicas have the same gradients and optimization step can occur in parallel
    loss = jax.lax.pmean(loss, axis_name='batch')
    grads = jax.lax.pmean(grads, axis_name='batch')

    state = state.apply_gradients(
        grads=grads,
    )

    metrics = {
        'LM Loss': loss, 
        'LM PPL': jnp.exp(loss)
    }

    return state, metrics
