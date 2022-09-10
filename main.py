import argparse
import logging
import random
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
from src.training.training_utils import create_train_state
from src.utils.dataloader import numpy_collate

logging.basicConfig(level=logging.INFO)


def parse():
    parser = argparse.ArgumentParser(description="Transformer Training")

    parser.add_argument("--cfg", default="conf/config.yaml", type=str)

    parser.add_argument("--model-cfg", default="conf/model_config.yaml", type=str)

    args = parser.parse_args()
    return args


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


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

    if cfg.training.precision == "fp16":
        model_dtype = jnp.float16
    elif cfg.training.precision == "bf16":
        model_dtype = jnp.bfloat16
    else:
        model_dtype = jnp.float32

    # Helper function for converting dtypes across model
    def to_precision(t, dtype: jnp.dtype):
        return jax.tree_map(
            lambda x: x.astype(dtype) if x.dtype == jnp.float32 else x, t
        )

    logging.info(f"Host setup with {num_devices} devices.")
    logging.info(f"Using platform: {platform} with precision {model_dtype}")

    state.params = to_precision(state.params)

    # replicating state across devices
    state = flax.jax_utils.replicate(state)

    local_batch_size = cfg.training.batch_size // jax.device_count()

    if jax.process_index() == 0:
        id = wandb.util.generate_id()
        wandb.init(id=id, resume="allow", project="LJX")
        wandb.config.steps = cfg.training.total_steps
        wandb.config.batch_size = cfg.training.batch_size

        # Hyperparam Setup
        wandb.config.weight_decay = cfg.training.weight_decay
        wandb.config.warmup = cfg.training.warmup_steps
        wandb.config.accum_steps = cfg.training.gradient_accumulation_steps
        wandb.config.seq_len = model.block_size
        wandb.config.model = cfg.model.size

        # Model setup
        wandb.config.corpus = cfg.data.corpus

    train_shards = cfg.data.train_shard_urls
    validation_shards = cfg.data.validation_shard_urls

    def preprocess(batch):
        x = batch["input_id.pth"][: cfg.data.max_context]
        return jnp.array(x.long(), dtype=jnp.int32)

    train_dataset = wds.DataPipeline(
        wds.SimpleShardList(train_shards),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(1e4, initial=1e4, rng=random.Random(23)),
        wds.decode(),
        wds.map(preprocess),
    )

    validation_dataset = wds.DataPipeline(
        wds.SimpleShardList(validation_shards),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(1e4, initial=1e4, rng=random.Random(23)),
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

    running_metrics = []

    # I mean, we should eventually wrap this in an epoch loop
    for i, text in enumerate(tqdm(tl, disable=not jax.process_index() == 0)):

        # sharding batch/keys for dataparallel
        sharded_batch = shard(text)
        rng, batch_rng = random.split(rng)
        sharded_rng = shard_prng_key(batch_rng)

        state, metrics = train_step(
            state,
            sharded_batch,
            sharded_rng,
        )
        running_metrics.append(metrics)

        if (i) % cfg.training.gradient_accumulation_steps == 0:
            # we've completed a full batch of data, log the metrics

            train_metrics_np = {
                k: np.mean([metrics[k] for metrics in running_metrics])
                for k in running_metrics[0]
            }

            running_metrics = []

            validation_metrics = []

            if (i) % cfg.training.evaluation_frequency == 0:
                for val_it, val_text in enumerate(
                    tqdm(vl, disable=not jax.process_index() == 0)
                ):
                    if val_it < cfg.training.maximum_evaluation_steps:
                        sharded_batch = shard(val_text)
                        metrics = eval_step(state, sharded_batch)
                        validation_metrics.append(metrics)
                    else:
                        break

                validation_metrics_np = {
                    k: np.mean([metrics[k] for metrics in validation_metrics])
                    for k in validation_metrics[0]
                }

                if jax.process_index() == 0:
                    wandb.log(train_metrics_np.update(validation_metrics_np))

            else:
                if jax.process_index() == 0:
                    wandb.log(train_metrics_np)


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

    metrics = {"Train LM Loss": loss, "Train LM PPL": jnp.exp(loss)}

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

    metrics = {"Validation LM Loss": loss, "Validation LM PPL": jnp.exp(loss)}

    return metrics


if __name__ == "__main__":
    main()
