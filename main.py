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
from flax.training.common_utils import shard, shard_prng_key
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.models.GPT import model_getter
from src.training.training_utils import compute_tokens_seen, create_train_state
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
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=5, overwrite=True)


def restore_checkpoint(state, workdir, prefix):
    return checkpoints.restore_checkpoint(workdir, state, prefix=prefix)


def main():
    args = parse()
    cfg = OmegaConf.load(args.cfg)

    # getting system information
    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    num_host = num_devices // num_local_devices
    platform = jax.local_devices()[0].platform

    assert (
        cfg.device.mp_devices == 1
    ), f"This train script only supports data parallel training through pmap."

    if cfg.training.precision == "fp16":
        model_dtype = jnp.float16
    elif cfg.training.precision == "bf16":
        model_dtype = jnp.bfloat16
    else:
        model_dtype = jnp.float32

    # setting up GCP bucket/client info if training on TPU
    save_to_bucket = False
    client = None
    bucket_path = None
    if platform == "tpu":
        if cfg.data.bucket_path is not None:
            # use GCP
            from google.cloud import storage
            from google.cloud.exceptions import NotFound

            client = storage.Client()
            save_to_bucket = True
            bucket_path = cfg.data.bucket_path
            train_shards = open(cfg.data.index_path_train).read().splitlines()
            validation_shards = open(cfg.data.index_path_validation).read().splitlines()

    else:
        train_shards = cfg.data.train_shard_urls
        validation_shards = cfg.data.validation_shard_urls

    model, model_config = model_getter(
        cfg.model.size, config_path=args.model_cfg, return_cfg=True, dtype=model_dtype
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

    resume_step = 0

    if cfg.device.mp_devices == 1:
        state = create_train_state(
            init_rng,
            learning_rate_fn,
            weight_decay=cfg.training.weight_decay,
            model=model,
        )

        if args.resume:
            if save_to_bucket:
                state = restore_checkpoint(
                    state,
                    workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}",
                    prefix="checkpoint_",
                )
            else:
                state = restore_checkpoint(state, workdir=cfg.data.checkpoint_directory)

            if jax.process_index() == 0:
                logger.debug(f"Resuming training from step {int(state.step)}")

            resume_step = int(state.step)

    else:
        pass

    if jax.process_index() == 0:
        logger.debug(f"VM setup with {num_devices} devices.")
        logger.debug(f"Host setup with {num_local_devices} devices.")
        logger.debug(f"Using platform: {platform} with precision {model_dtype}")

        if cfg.device.mp_devices == 1:
            logger.debug(
                f"Performing data parallel training only. Model and train state will be replicated across all devices"
            )

        else:
            logger.debug(
                f"Performing DP and MP training with grid shape {(cfg.device.dp_devices, cfg.device.mp_devices)}"
            )

        if len(cfg.training.staged_sequences) > 0:
            logger.debug(
                f"Running sequence length warmup for {cfg.training.staged_warmup_steps} total steps with stages: {cfg.training.staged_sequences}"
            )

    if not args.resume:
        if cfg.data.bucket_path is not None:
            # clear bucket
            client = storage.Client()
            if jax.process_index() == 0:
                bucket = storage.Bucket(client, f"{cfg.data.bucket_path}")
                blobs = bucket.list_blobs(prefix=f"{cfg.data.checkpoint_directory}")
                for blob in blobs:
                    blob.delete()

    local_batch_size = cfg.training.batch_size // (
        jax.local_device_count() // cfg.device.mp_devices
    )

    total_tokens = num_host * (
        cfg.training.batch_size
        * compute_tokens_seen(
            cfg.training.total_steps,
            stages=cfg.training.staged_sequences,
            max_steps=cfg.training.staged_warmup_steps,
            max_context=cfg.data.max_context,
        )
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
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(1e6, initial=1e6, rng=pyrandom.Random(23)),
        wds.decode(handler=wds.warn_and_continue),
        wds.map(preprocess),
    ).repeat(nepochs=cfg.training.max_epochs)

    validation_dataset = wds.DataPipeline(
        wds.SimpleShardList(validation_shards),
        split_by_jax_process,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(1e6, initial=1e6, rng=pyrandom.Random(23)),
        wds.decode(handler=wds.warn_and_continue),
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
        batch_size=cfg.training.batch_size // 8,
        collate_fn=numpy_collate,
        drop_last=True,
    )

    running_metrics = []

    step_to_seq = lambda x: cfg.training.train_context

    state = flax.jax_utils.replicate(state)

    for i, text in enumerate(tqdm(tl, disable=not jax.process_index() == 0)):

        if (i + resume_step) > cfg.training.total_steps:
            if jax.process_index() == 0:
                logger.debug(f"Training has completed.")

            return True

        if resume_step > 0 and (i <= resume_step % 24558):
            # since we repeat epochs, just iterate partially through repeated ds
            continue

        rng, dropout_rng = jax.random.split(rng, 2)

        seq_len = step_to_seq(i + resume_step)

        if seq_len < cfg.data.max_context: 
            text = text.reshape(-1, seq_len)

        # we add a 'grad_accum' batch dimension which we then iterate through in train_step
        text = text.reshape(
            cfg.training.gradient_accumulation_steps,
            text.shape[0] // cfg.training.gradient_accumulation_steps,
            seq_len,
        ).transpose(1, 0, 2)

        text = shard(text)

        rng_sharded = shard_prng_key(dropout_rng)

        t0 = time.time()

        state, metrics = train_step(
            state, text, rng_sharded, cfg.training.gradient_accumulation_steps
        )

        metrics["Train Batch Time"] = time.time() - t0
        metrics["Train Sequence Length"] = seq_len

        running_metrics.append(metrics)

        train_metrics_np = {
            k: np.mean([metrics[k] for metrics in running_metrics])
            for k in running_metrics[0]
        }

        running_metrics = []
        validation_metrics = []

        absolute_step = i + resume_step

        train_metrics_np["Tokens Seen (B)"] = (
            num_host
            * (
                cfg.training.batch_size
                * compute_tokens_seen(
                    absolute_step,
                    stages=cfg.training.staged_sequences,
                    max_steps=cfg.training.staged_warmup_steps,
                    max_context=cfg.data.max_context,
                )
            )
            / 1e9
        )

        if (i) % (cfg.training.evaluation_frequency) == 0:
            for val_it, val_text in enumerate(
                tqdm(vl, disable=not jax.process_index() == 0)
            ):
                val_text = val_text[:, :seq_len]
                val_text = shard(val_text)

                if val_it < cfg.training.maximum_evaluation_steps:
                    metrics = eval_step(state, val_text)
                    validation_metrics.append(metrics)
                else:
                    break

            validation_metrics_np = {
                k: np.mean([metrics[k] for metrics in validation_metrics])
                for k in validation_metrics[0]
            }

            if jax.process_index() == 0:
                train_metrics_np.update(validation_metrics_np)
                train_metrics_np.pop("Train Batch Time")
                wandb.log(train_metrics_np)

                if cfg.device.mp_devices > 1:
                    device_state = jax.device_get(
                        state
                    )  # pull a copy of the sharded state to CPU and save
                    save_checkpoint(
                        device_state,
                        workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}",
                    )

                else:
                    if save_to_bucket:
                        save_checkpoint(
                            state,
                            workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}",
                        )
                    else:
                        save_checkpoint(state, workdir=cfg.data.checkpoint_directory)

        else:
            if jax.process_index() == 0:
                train_metrics_np["Train Step Time"] = train_metrics_np[
                    "Train Batch Time"
                ]
                train_metrics_np.pop("Train Batch Time")
                wandb.log(train_metrics_np)


@partial(
    jax.pmap, axis_name="batch", static_broadcasted_argnums=(3,)
)
def train_step(
    state: Any,
    batch: jnp.array,
    rng_key: jax.random.PRNGKey = None,
    accum_steps: int = 8,
):
    """
    Train on a single batch
    This means that the batch will be size (local_bs*grad_accum, ctx) instead of (local_bs, ctx)

    Designed to perform gradient and loss communication once per _batch_ instead of once per mini batch
    """

    def get_minibatch(batch, grad_idx):
        return jax.tree_util.tree_map(
            lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False, axis=1),
            batch,
        )

    def loss_fn(params, batch):
        _, loss = state.apply_fn(
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

        loss, grads = grad_fn(state.params, minibatch)

        return loss, grads

    init_minibatch = (0.0, jax.tree_util.tree_map(jnp.zeros_like, state.params))

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

    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")

    # only update train_state at the end of a single full batch
    new_state = state.apply_gradients(
        grads=grads,
    )

    metrics = {
        "Train LM Loss": loss,
        "Train LM PPL": jnp.exp(loss),
    }

    return new_state, metrics


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
    # try:
    # main()
    # except Exception as e:
    # print(f"Error encountered: {e}")
    main()
