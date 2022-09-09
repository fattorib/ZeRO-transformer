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

    parser.add_argument(
        "--model-cfg", default="conf/model_config.yaml", type=str
    )

    args = parser.parse_args()
    return args


def shard_batch(xs):
    local_device_count = jax.local_device_count()
    return xs.reshape((local_device_count, -1) + xs.shape[1:])


def main():
    args = parse()
    cfg = OmegaConf.load(args.cfg)

    # getting system information
    num_devices = jax.device_count()
    platform = jax.local_devices()[0].platform

    model = model_getter(cfg.model.size, config_path=args.model_cfg)

    learning_rate_fn = optax.warmup_cosine_decay_schedule(
        init_value=0,
        peak_value=cfg.training.peak_learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        decay_steps=cfg.training.decay_steps,
        end_value=cfg.training.end_learning_rate,
    )

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(
        init_rng,
        learning_rate_fn,
        weight_decay=cfg.training.weight_decay,
        model=model,
        grad_accum_steps=cfg.training.gradient_accumulation_steps,
    )

    # create state, etc
    state = flax.jax_utils.replicate(state)

    local_batch_size = cfg.training.batch_size // jax.device_count()


@partial(jax.pmap, axis_name="batch")
def train_step(state: Any, batch: jnp.array, rng_key: random.PRNGKey):
    """Train on a single batch"""

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

    # compute all-reduce mean for gradients and loss
    # Ex: If we have 8 devices, each device takes the gradients from the other 7 and averages them all together
    # that way, all device replicas have the same gradients and optimization step can occur in parallel
    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")

    state = state.apply_gradients(
        grads=grads,
    )

    metrics = {"LM Loss": loss, "LM PPL": jnp.exp(loss)}

    return state, metrics


@partial(jax.pmap, axis_name="batch")
def eval_step(state: Any, batch: jnp.array):
    """Evaluate on a single batch"""

    _, loss = state.apply_fn(
        {"params": state.params["params"]},
        x=batch,
        labels=batch,
        train=False,
    )
    loss = jax.lax.pmean(loss, axis_name="batch")

    metrics = {"LM Loss": loss, "LM PPL": jnp.exp(loss)}

    return metrics
