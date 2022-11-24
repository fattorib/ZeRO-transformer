import argparse
import logging
import random as pyrandom
import time
from functools import partial
from typing import Any, Callable, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import webdataset as wds
from flax.training import checkpoints, train_state
from flax.training.common_utils import shard, shard_prng_key
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.models.GPT import model_getter
from src.training.training_utils import compute_tokens_seen, initialized
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


def save_checkpoint_params(params, step, workdir):
    """
    Checkpoints params
    """
    if jax.process_index() == 0:
        params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))
        checkpoints.save_checkpoint(
            workdir, params, step, keep=5, overwrite=True, prefix="params_"
        )


def save_checkpoint_optimizer(opt_state, step, workdir):
    """
    Checkpoints sharded optimizer state
    """
    if jax.process_index() == 0:
        opt_state = jax.device_get(opt_state)
        checkpoints.save_checkpoint(
            workdir, opt_state, step, keep=5, overwrite=True, prefix="optimizer_"
        )


def restore_param_checkpoint(workdir):
    params = checkpoints.restore_checkpoint(workdir, target=None, prefix="params_")

    return params


def restore_opt_checkpoint(workdir):
    opt_state = checkpoints.restore_checkpoint(
        workdir, target=None, prefix="optimizer_"
    )

    return opt_state


def partition_shard(xs, local_device_count, devices):
    """
    Partitions optimizer state by splitting the first dimension of buffers across local devices
    """
    return jax.tree_util.tree_map(
        lambda x: x.reshape((local_device_count, -1) + x.shape[1:])
        if x.ndim > 0
        else jax.device_put_replicated(x, devices),
        xs,
    )


@partial(jax.pmap, devices=jax.local_devices())
def split_sharded_device_array(arr, device_index):
    """
    Pmappable way to get shards of param/grad pytrees

    By default, anything that is not a pmapped function applied to a ShardedDeviceArray will
    force a gather to the 0'th device and return a DeviceArray.

    See: https://github.com/google/jax/issues/2535


    """
    local_device_count = 8
    return jax.tree_util.tree_map(
        lambda x: x.reshape(local_device_count, -1, x.shape[-1])[device_index, ...]
        if x.ndim >= 2
        else x.reshape(local_device_count, -1)[device_index, ...],
        arr,
    )


@partial(jax.pmap, devices=jax.local_devices())
def deshard(xs):
    """
    Pmappable way to get reshape a sharded device array containing param replicas

    By default, anything that is not a pmapped function applied to a ShardedDeviceArray will
    force a gather to the 0'th device and return a DeviceArray.

    See: https://github.com/google/jax/issues/2535

    """
    return jax.tree_util.tree_map(
        lambda x: x.reshape((-1, x.shape[-1])) if x.ndim > 2 else x.reshape(-1), xs
    )


def create_zero_train_state(
    rng: jax.random.PRNGKey,
    learning_rate_fn: Union[float, Callable],
    weight_decay: float,
    model: nn.Module,
):
    """
    Initializes parameters and returns a _simplified_ TrainState without an optimizer.

    Returns TrainState, opt_state, GradientTransformation

    """
    params = initialized(rng, model, input_shape=(1, model.block_size))

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
            mu_dtype=jnp.float32,  # keep fp32 optimizer states
        ),
    )

    opt_state = tx.init(params)
    state = train_state.TrainState(
        apply_fn=model.apply, params=params, step=0, opt_state=None, tx=None
    )
    return state, tx, opt_state


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

    state, optimizer, opt_state = create_zero_train_state(
        init_rng,
        learning_rate_fn,
        weight_decay=0.1,
        model=model,
    )
    params = state.params
    del state

    if args.resume:
        del params

        if save_to_bucket:
            opt_state = restore_opt_checkpoint(
                workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/optimizer"
            )
            params = restore_param_checkpoint(
                workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/params"
            )

            # Hacky way to strip training step from checkpoint names
            client = storage.Client()
            blob_list = []
            bucket = storage.Bucket(client, f"{cfg.data.bucket_path}")
            blobs = bucket.list_blobs(
                prefix=f"{cfg.data.checkpoint_directory}/optimizer"
            )
            for blob in blobs:
                blob_list.append(blob.name)

            all_steps = [b.split("_")[-1] for b in blobs]

            resume_step = int(max(all_steps))

        else:
            raise NotImplementedError(
                "Checkpointing not currently implemented for GPU/CPU"
            )

        if jax.process_index() == 0:
            logger.debug(f"Resuming training from step {resume_step}")

    opt_state = partition_shard(
        opt_state, jax.local_device_count(), jax.local_devices()
    )
    opt_state = jax.pmap(lambda x: x, devices=jax.local_devices())(
        opt_state
    )  # shard opt state to free up memory

    if jax.process_index() == 0:
        logger.debug(f"VM setup with {num_devices} devices.")
        logger.debug(f"Host setup with {num_local_devices} devices.")
        logger.debug(f"Using platform: {platform} with precision {model_dtype}")

        if cfg.device.mp_devices == 1:
            logger.debug(
                f"Performing data parallel training only. Model and train state will be replicated across all devices"
            )

    if not args.resume:
        if cfg.data.bucket_path is not None:
            # clear bucket
            client = storage.Client()
            if jax.process_index() == 0:
                bucket = storage.Bucket(client, f"{cfg.data.bucket_path}")
                blobs = bucket.list_blobs(
                    prefix=f"{cfg.data.checkpoint_directory}/optimizer"
                )
                for blob in blobs:
                    blob.delete()

                blobs = bucket.list_blobs(
                    prefix=f"{cfg.data.checkpoint_directory}/params"
                )
                for blob in blobs:
                    blob.delete()

    local_batch_size = cfg.training.batch_size // (
        jax.local_device_count() // cfg.device.mp_devices
    )

    total_tokens = num_host * (
        cfg.training.batch_size
        * compute_tokens_seen(
            cfg.training.total_steps,
            max_context=cfg.data.max_context,
        )
    )

    if jax.process_index() == 0:
        id = wandb.util.generate_id()
        wandb.init(id=id, resume="allow", project=cfg.data.wandb_project)
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
        wds.shuffle(1e7, initial=1e7, rng=pyrandom.Random(23 + resume_step)),
        wds.decode(handler=wds.warn_and_continue),
        wds.map(preprocess),
    ).repeat(nepochs=cfg.training.max_epochs)

    validation_dataset = wds.DataPipeline(
        wds.SimpleShardList(validation_shards),
        split_by_jax_process,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(1e6, initial=1e6, rng=pyrandom.Random(23 + resume_step)),
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
        batch_size=cfg.training.batch_size,
        collate_fn=numpy_collate,
        drop_last=True,
    )

    running_metrics = []

    if cfg.training.warmup_train_context < cfg.data.max_context:
        step_to_seq = (
            lambda x: cfg.training.warmup_train_context
            if x < cfg.training.staged_warmup_steps
            else cfg.training.max_context
        )
    else:
        step_to_seq = lambda x: cfg.data.max_context

    accum_steps = (
        lambda x: 2
        if x < cfg.training.staged_warmup_steps
        else cfg.training.gradient_accumulation_steps
    )

    params = flax.jax_utils.replicate(params)

    rng = jax.random.fold_in(rng, resume_step)  # fold in resume step to create new rng

    # quick way to track global step count when resuming a run
    new_steps = 0

    for i, text in enumerate(tqdm(tl, disable=not jax.process_index() == 0)):

        if (resume_step + new_steps) > cfg.training.total_steps:
            if jax.process_index() == 0:
                logger.debug(f"Training has completed.")

            return True

        rng, dropout_rng = jax.random.split(rng, 2)

        seq_len = step_to_seq(resume_step + new_steps)

        gradient_accumulation_steps = accum_steps(resume_step + new_steps)

        if seq_len < cfg.data.max_context:
            text = text.reshape(-1, seq_len)

        # we add a 'grad_accum' batch dimension which we then iterate through in train_step
        text = text.reshape(
            gradient_accumulation_steps,
            text.shape[0] // gradient_accumulation_steps,
            seq_len,
        ).transpose(1, 0, 2)

        text = shard(text)

        rng_sharded = shard_prng_key(dropout_rng)

        t0 = time.time()

        grads, metrics = train_step(
            params, text, rng_sharded, gradient_accumulation_steps, model
        )

        grads = split_sharded_device_array(
            grads, jax.numpy.arange(jax.local_device_count())
        )
        params = split_sharded_device_array(
            params, jax.numpy.arange(jax.local_device_count())
        )

        params, opt_state = update_sharded_state(
            grads,
            opt_state,
            params,
            optimizer,
        )

        del grads  # manually free grad mem

        params = deshard(params)  # reshape params

        metrics["Train Batch Time"] = time.time() - t0
        metrics["Train Sequence Length"] = seq_len
        metrics["Learning Rate"] = learning_rate_fn(resume_step + new_steps)

        running_metrics.append(metrics)

        train_metrics_np = {
            k: np.mean([metrics[k] for metrics in running_metrics])
            for k in running_metrics[0]
        }

        running_metrics = []
        validation_metrics = []

        absolute_step = resume_step + new_steps

        train_metrics_np["Tokens Seen (B)"] = (
            num_host
            * (
                cfg.training.batch_size
                * compute_tokens_seen(
                    absolute_step,
                    max_context=cfg.data.max_context,
                )
            )
            / 1e9
        )

        new_steps += 1

        if (i) % (cfg.training.evaluation_frequency) == 0:
            for val_it, val_text in enumerate(
                tqdm(vl, disable=not jax.process_index() == 0)
            ):
                val_text = shard(val_text)

                if val_it < cfg.training.maximum_evaluation_steps:
                    metrics = eval_step(params, model, val_text)
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

                if save_to_bucket:
                    save_checkpoint_params(
                        params,
                        absolute_step,
                        workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/params",
                    )
                    save_checkpoint_optimizer(
                        opt_state,
                        absolute_step,
                        workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/optimizer",
                    )

                else:
                    raise NotImplementedError(
                        "Checkpointing not currently implemented for GPU/CPU"
                    )

        else:
            if jax.process_index() == 0:
                train_metrics_np["Train Step Time"] = train_metrics_np[
                    "Train Batch Time"
                ]
                train_metrics_np.pop("Train Batch Time")
                wandb.log(train_metrics_np)


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4))
def train_step(
    params: Any,
    batch: jnp.array,
    rng_key: jax.random.PRNGKey = None,
    accum_steps: int = 8,
    model: Any = None,
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

    # this logic could probably be movied into cumul_minibatch_step,
    loss, grads = jax.lax.fori_loop(
        0,
        accum_steps,
        cumul_minibatch_step,
        init_minibatch,
    )

    loss, grads = jax.tree_util.tree_map(lambda x: x / accum_steps, (loss, grads))

    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")

    metrics = {
        "Train LM Loss": loss,
        "Train LM PPL": jnp.exp(loss),
    }

    return grads, metrics


@partial(jax.pmap, axis_name="shard", devices=jax.local_devices())
def slice_grads(grads, device_index):
    grad_slice = jax.tree_util.tree_map(lambda x: x[device_index, ...], grads)
    return grad_slice


@partial(
    jax.pmap,
    axis_name="shard",
    devices=jax.local_devices(),
    static_broadcasted_argnums=(3,),
)
def update_sharded_state(
    grads: Any,
    optimizer_state: Any,
    params: Any,
    optimizer: Any,
):
    """
    Updates the sharded optimizer state
    """

    # These two lines update the specific shard of state/parameters sitting on device 'i'
    updates, new_opt_state = optimizer.update(grads, optimizer_state, params)
    new_params = optax.apply_updates(params, updates)

    new_params = jax.lax.all_gather(new_params, axis_name="shard")
    return new_params, new_opt_state


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(1,))
def eval_step(params: Any, model: Any, batch: jnp.array):
    """Evaluate on a single batch"""

    _, loss = model.apply(
        {"params": params["params"]}, x=batch, labels=batch, train=False
    )

    loss = jax.lax.pmean(loss, axis_name="batch")

    metrics = {"Validation LM Loss": loss, "Validation LM PPL": jnp.exp(loss)}

    return metrics


if __name__ == "__main__":
    main()
