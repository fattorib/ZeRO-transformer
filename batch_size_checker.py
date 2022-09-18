"""
Since we use Sequence Length warmup, the TPU/GPU memory is never maxed out during the warmup steps (well, hopefully not).
This script just checks for the largest useable batch size @ the maximum context
"""

import argparse
import logging
import random as pyrandom
import time
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import webdataset as wds
from flax.training import checkpoints
from flax.training.common_utils import shard, shard_prng_key
from jax import random
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.models.GPT import model_getter
from src.training.training_utils import create_train_state, step_to_seq_len
from src.utils.configs import flatten_dict
from src.utils.dataloader import numpy_collate
from src.utils.losses import kl_div_loss

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def parse():
    parser = argparse.ArgumentParser(description="Transformer Training")

    parser.add_argument("--cfg", default="conf/config.yaml", type=str)

    parser.add_argument("--model-cfg", default="conf/model_config.yaml", type=str)

    parser.add_argument("--batch-size", type = int)

    args = parser.parse_args()
    return args


def main():
    args = parse()
    cfg = OmegaConf.load(args.cfg)

    # getting system information
    num_devices = jax.device_count()
    platform = jax.local_devices()[0].platform

    model, model_config = model_getter(
        cfg.model.size, config_path=args.model_cfg, return_cfg=True
    )

    learning_rate_fn = optax.warmup_cosine_decay_schedule(
        init_value=0,
        peak_value=cfg.training.peak_learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        decay_steps=cfg.training.decay_steps,
        end_value=cfg.training.end_learning_rate,
    )

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    if cfg.training.precision == "fp16":
        model_dtype = jnp.float16
    elif cfg.training.precision == "bf16":
        model_dtype = jnp.bfloat16
    else:
        model_dtype = jnp.float32
    
    state = create_train_state(
        init_rng,
        learning_rate_fn,
        weight_decay=cfg.training.weight_decay,
        model=model,
        grad_accum_steps=cfg.training.gradient_accumulation_steps,
        dtype=model_dtype,
    )

    if jax.process_index() == 0:
        logger.debug(f"Host setup with {num_devices} devices.")
        logger.debug(f"Using platform: {platform} with precision {model_dtype}")
        if cfg.training.staged_sequences is not None:
            logger.debug(
                f"Running sequence length warmup for {cfg.training.staged_warmup_steps} total steps with stages: {cfg.training.staged_sequences}"
            )
    
    # replicating state across devices
    state = flax.jax_utils.replicate(state)

    if jax.process_index() == 0:
            logger.debug(f"Attempting run with global batch size {args.batch_size}")

    local_batch_size = args.batch_size // jax.device_count()

    save_to_bucket = False
    if platform == "tpu":
        if cfg.data.bucket_path is not None:
            # use GCP
            from google.cloud import storage
            from google.cloud.exceptions import NotFound

            client = storage.Client()
            save_to_bucket = True

            # will this work?
            train_shards = open(cfg.data.index_path_train).read().splitlines()
            validation_shards = open(cfg.data.index_path_validation).read().splitlines()

    else:
        train_shards = cfg.data.train_shard_urls
        validation_shards = cfg.data.validation_shard_urls

    def preprocess(batch):
        x = batch["input_id.pth"][: cfg.data.max_context]
        return jnp.array(x.long(), dtype=jnp.int32)

    train_dataset = wds.DataPipeline(
        wds.SimpleShardList(train_shards),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(1e5, initial=1e5, rng=pyrandom.Random(23)),
        wds.decode(),
        wds.map(preprocess),
    )

    validation_dataset = wds.DataPipeline(
        wds.SimpleShardList(validation_shards),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(1e5, initial=1e5, rng=pyrandom.Random(23)),
        wds.decode(),
        wds.map(preprocess),
    )

    tl = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=numpy_collate,
    )

    vl = DataLoader(
        dataset=validation_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=numpy_collate,
    )

    step_to_seq = lambda x: cfg.data.max_context

    for i, text in enumerate(tqdm(tl, disable=not jax.process_index() == 0, total = 100)):

        seq_len = step_to_seq(i)
        text = text[:, :seq_len]

        # sharding batch
        sharded_batch = shard(text)

        state, metrics = train_step(
                state,
                sharded_batch,
                # sharded_rng,
            )
        
        if (i+1) % 100 == 0:
            break 

@partial(jax.pmap, axis_name="batch")
def train_step(state: Any, batch: jnp.array, rng_key: random.PRNGKey = None):
    """Train on a single batch"""

    def loss_fn(params):
        _, loss = state.apply_fn(
            {"params": params["params"]},
            x=batch,
            labels=batch,
            train=False,
            # rngs={"dropout": rng_key},
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

    metrics = {"Train LM Loss": loss, "Train LM PPL": jnp.exp(loss)}

    return state, metrics


if __name__ == "__main__":
    main()
