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
import torch
import webdataset as wds
from flax.training import checkpoints
from flax.training.common_utils import shard
from jax import random
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.models.GPT import model_getter
from src.training.training_utils import (compute_tokens_seen,
                                         create_train_state, step_to_seq_len)
from src.utils.configs import flatten_dict
from src.utils.dataloader import numpy_collate

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def parse():
    parser = argparse.ArgumentParser(description="Transformer Training")

    parser.add_argument("--cfg", default="conf/config.yaml", type=str)

    parser.add_argument("--model-cfg", default="conf/model_config.yaml", type=str)

    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    return args


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=2, overwrite=True)


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def main():
    args = parse()
    cfg = OmegaConf.load(args.cfg)

    # getting system information
    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    num_host = num_devices // num_local_devices
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
        logger.debug(f"VM setup with {num_devices} devices.")
        logger.debug(f"Host setup with {num_local_devices} devices.")
        logger.debug(f"Using platform: {platform} with precision {model_dtype}")
        if len(cfg.training.staged_sequences) > 0:
            logger.debug(
                f"Running sequence length warmup for {cfg.training.staged_warmup_steps} total steps with stages: {cfg.training.staged_sequences}"
            )

    save_to_bucket = False
    if platform == "tpu":
        if cfg.data.bucket_path is not None:
            # use GCP
            from google.cloud import storage
            from google.cloud.exceptions import NotFound

            client = storage.Client()
            save_to_bucket = True

            train_shards = open(cfg.data.index_path_train).read().splitlines()
            validation_shards = open(cfg.data.index_path_validation).read().splitlines()

    else:
        train_shards = cfg.data.train_shard_urls
        validation_shards = cfg.data.validation_shard_urls

    resume_step = None
    if args.resume:
        if save_to_bucket:
            state = restore_checkpoint(
                state,
                workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}",
            )
        else:
            state = restore_checkpoint(state, workdir=cfg.data.checkpoint_directory)

        if jax.process_index() == 0:
            logger.debug(f"Resuming training from step {int(state.step)}")

        # resume step is ga_steps*global steps
        resume_step = int(state.step)

    else:
        # clear bucket here
        client = storage.Client()
        if jax.process_index() == 0:
            bucket = storage.Bucket(client, f"{cfg.data.bucket_path}")
            blobs = bucket.list_blobs(prefix=f"{cfg.data.checkpoint_directory}")
            for blob in blobs:
                blob.delete()

    # replicating state across devices
    state = flax.jax_utils.replicate(state)

    local_batch_size = cfg.training.batch_size // jax.local_device_count()

    # This is computed in terms of absolute steps
    if len(cfg.training.staged_sequences) > 0:
        total_tokens = num_host * (
            cfg.training.batch_size
            * cfg.training.gradient_accumulation_steps
            * compute_tokens_seen(
                cfg.training.total_steps,
                stages=cfg.training.staged_sequences,
                max_steps=cfg.training.staged_warmup_steps,
                max_context=cfg.data.max_context,
            )
        )
    else:
        total_tokens = num_host * (
            cfg.training.batch_size
            * cfg.training.gradient_accumulation_steps
            * cfg.data.max_context
            * cfg.training.total_steps
        )

    if jax.process_index() == 0:
        id = wandb.util.generate_id()
        wandb.init(id=id, resume="allow", project="LJX")
        flat_dict = flatten_dict(cfg)

        for key in model_config.keys():
            flat_dict[f"model.{key}"] = model_config[key]

        flat_dict["training.local_batch_size"] = local_batch_size
        flat_dict["runtime"] = platform
        flat_dict["Total Training Tokens"] = total_tokens / 1e9
        flat_dict["Total Devices"] = num_devices
        wandb.config.update(flat_dict)

    def preprocess(batch):
        x = batch["input_id.pth"][: cfg.data.max_context]
        if type(x) == torch.tensor:
            return jnp.array(x.long(), dtype=jnp.int32)
        else:
            return jnp.array(x, dtype=jnp.int32)

    from itertools import islice

    def split_by_jax_process(src):
        host_id, num_process = (
            jax.process_index(),
            num_host,
        )
        if num_process > 1:
            for s in islice(src, host_id, None, num_process):
                yield s
        else:
            for s in src:
                yield s

    train_dataset = wds.DataPipeline(
        wds.SimpleShardList(train_shards),
        split_by_jax_process,
        wds.tarfile_to_samples(),
        wds.shuffle(1e6, initial=1e6, rng=pyrandom.Random(23)),
        wds.decode(),
        wds.map(preprocess),
    ).repeat(nepochs=cfg.training.max_epochs)

    validation_dataset = wds.DataPipeline(
        wds.SimpleShardList(validation_shards),
        split_by_jax_process,
        wds.tarfile_to_samples(),
        wds.shuffle(1e6, initial=1e6, rng=pyrandom.Random(23)),
        wds.decode(),
        wds.map(preprocess),
    )

    tl = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=numpy_collate,
        drop_last=True,
    )

    vl = DataLoader(
        dataset=validation_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=numpy_collate,
        drop_last=True,
    )

    running_metrics = []

    if len(cfg.training.staged_sequences) > 0:

        step_to_seq = partial(
            step_to_seq_len,
            stages=cfg.training.staged_sequences,
            max_steps=cfg.training.gradient_accumulation_steps
            * cfg.training.staged_warmup_steps,
            max_context=cfg.data.max_context,
        )

    else:
        step_to_seq = lambda x: cfg.data.max_context

    epoch = 0
    for i, text in enumerate(tqdm(tl, disable=not jax.process_index() == 0)):

        if (i // cfg.training.gradient_accumulation_steps) > cfg.data.full_steps_in_batch:
            epoch += 1

        # TODO: This may be problematic when crossing epoch boundary.
        if (i // cfg.training.gradient_accumulation_steps) + (
            epoch * cfg.data.full_steps_in_batch
        ) > cfg.training.total_steps:
            if jax.process_index() == 0:
                logger.debug(f"Training has completed.")

            return True

        if resume_step != None and i <= resume_step and epoch == 0:
            continue

        seq_len = step_to_seq(
            i
            + (
                epoch
                * cfg.data.full_steps_in_batch
                * cfg.training.gradient_accumulation_steps
            )
        )

        text = text[:, :seq_len]

        # sharding batch
        sharded_batch = shard(text)

        t0 = time.time()

        state, metrics = train_step(
            state,
            sharded_batch,
        )

        metrics["Train Batch Time"] = time.time() - t0
        metrics["Train Sequence Length"] = seq_len

        running_metrics.append(metrics)

        if (i) % cfg.training.gradient_accumulation_steps == 0:
            # we've completed a full batch of data, log the metrics

            train_metrics_np = {
                k: np.mean([metrics[k] for metrics in running_metrics])
                for k in running_metrics[0]
            }

            running_metrics = []

            validation_metrics = []

            if (i) % (
                cfg.training.evaluation_frequency
                * cfg.training.gradient_accumulation_steps
            ) == 0:
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
                    train_metrics_np.update(validation_metrics_np)
                    train_metrics_np["Train Step Time"] = (
                        cfg.training.gradient_accumulation_steps
                        * train_metrics_np["Train Batch Time"]
                    )

                    absolute_step = (i // cfg.training.gradient_accumulation_steps) + (
                        epoch * cfg.data.full_steps_in_batch
                    )
                    if len(cfg.training.staged_sequences) > 0:
                        train_metrics_np["Tokens Seen (B)"] = (
                            num_host
                            * (
                                cfg.training.batch_size
                                * cfg.training.gradient_accumulation_steps
                                * compute_tokens_seen(
                                    absolute_step,
                                    stages=cfg.training.staged_sequences,
                                    max_steps=cfg.training.staged_warmup_steps,
                                    max_context=cfg.data.max_context,
                                )
                            )
                            / 1e9
                        )
                    else:
                        train_metrics_np["Tokens Seen (B)"] = (
                            num_host
                            * (
                                cfg.training.batch_size
                                * cfg.training.gradient_accumulation_steps
                                * absolute_step
                                * cfg.data.max_context
                            )
                            / 1e9
                        )

                    train_metrics_np.pop("Train Batch Time")
                    wandb.log(train_metrics_np)

                    if save_to_bucket:
                        save_checkpoint(
                            state,
                            workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}",
                        )
                    else:
                        save_checkpoint(state, workdir=cfg.data.checkpoint_directory)

            else:
                if jax.process_index() == 0:
                    train_metrics_np["Train Step Time"] = (
                        cfg.training.gradient_accumulation_steps
                        * train_metrics_np["Train Batch Time"]
                    )

                    absolute_step = (i // cfg.training.gradient_accumulation_steps) + (
                        epoch * cfg.data.full_steps_in_batch
                    )
                    if len(cfg.training.staged_sequences) > 0:
                        train_metrics_np["Tokens Seen (B)"] = (
                            num_host
                            * (
                                cfg.training.batch_size
                                * cfg.training.gradient_accumulation_steps
                                * compute_tokens_seen(
                                    absolute_step,
                                    stages=cfg.training.staged_sequences,
                                    max_steps=cfg.training.staged_warmup_steps,
                                    max_context=cfg.data.max_context,
                                )
                            )
                            / 1e9
                        )
                    else:
                        train_metrics_np["Tokens Seen (B)"] = (
                            num_host
                            * (
                                cfg.training.batch_size
                                * cfg.training.gradient_accumulation_steps
                                * absolute_step
                                * cfg.data.max_context
                            )
                            / 1e9
                        )

                    train_metrics_np.pop("Train Batch Time")
                    wandb.log(train_metrics_np)


@partial(jax.pmap, axis_name="batch")
def train_step(state: Any, batch: jnp.array, rng_key: random.PRNGKey = None):
    """Train on a single batch"""

    def loss_fn(params):
        _, loss = state.apply_fn(
            {"params": params["params"]},
            x=batch,
            labels=batch,
            train=False,
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
    try:
        main()
    except Exception as e:
        print(f"Error encountered: {e}")
