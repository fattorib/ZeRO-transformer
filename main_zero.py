import argparse
import gc
import logging
import random as pyrandom
from functools import partial
from typing import Any, Callable, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import webdataset as wds
from flax.serialization import msgpack_restore
from flax.training import checkpoints, train_state
from jax.experimental.maps import xmap
from jax.experimental.pjit import pjit
from jax.sharding import Mesh
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.models.GPT import model_getter
from src.partitioning.partition import create_opt_spec, set_partitions_zero
from src.partitioning.xmap_train_functions import (eval_step, train_step,
                                                   update_opt_state)
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


def save_checkpoint_params(params: Any, step: int, workdir: str) -> None:
    """
    Save a copy of params.

    TODO: Add async manager to do this in a background process
    """
    if jax.process_index() == 0:
        params = jax.device_get(params)
        faux_state = train_state.TrainState(
            step=step, apply_fn=None, params=params, tx=None, opt_state=None
        )
        checkpoints.save_checkpoint(
            workdir, faux_state, step, keep=3, overwrite=True, prefix="params_"
        )


def save_checkpoint_optimizer(opt_state: Any, step: int, workdir: str) -> None:
    """
    Function to gather and save the sharded optimizer state.

    TODO: Add async manager to do this in a background process
    """
    if jax.process_index() == 0:
        # print(type(opt_state))
        # def grab_shards(tree):
        #     return jax.experimental.multihost_utils.process_allgather(tree)

        # opt_state = grab_shards(opt_state)
        opt_state = jax.device_get(opt_state)

        faux_state = train_state.TrainState(
            step=step, apply_fn=None, params=None, tx=None, opt_state=opt_state
        )
        checkpoints.save_checkpoint(
            workdir, faux_state, step, keep=3, overwrite=True, prefix="optimizer_"
        )


def restore_param_checkpoint(workdir: str) -> Any:
    """
    Restores the most recent parameter dict
    """
    params = checkpoints.restore_checkpoint(workdir, target=None, prefix="params_")

    return flax.core.freeze(params["params"])


def restore_opt_checkpoint(workdir: str) -> Tuple[Any, int]:
    """
    Function to restore optimizer state from a sequence of serialized Flax
    state dicts. By default, restoring a flax state dict to an optax state
    doesn't work so we manually recreate the optimizer state and return it.
    """
    opt_state_restored = checkpoints.restore_checkpoint(
        workdir, target=None, prefix="optimizer_"
    )

    mu_pytree = jax.tree_util.tree_map(
        lambda x: jnp.array(x), opt_state_restored["opt_state"]["1"]["0"]["mu"]
    )

    nu_pytree = jax.tree_util.tree_map(
        lambda x: jnp.array(x), opt_state_restored["opt_state"]["1"]["0"]["nu"]
    )

    count_pytree = jax.tree_util.tree_map(
        lambda x: jnp.array(x), opt_state_restored["opt_state"]["1"]["0"]["count"]
    )

    restoredadamstate = optax.ScaleByAdamState(
        count_pytree, flax.core.FrozenDict(mu_pytree), flax.core.FrozenDict(nu_pytree)
    )
    restored_state = (
        optax.EmptyState(),
        (
            restoredadamstate,
            optax.MaskedState(inner_state=optax.EmptyState()),
            optax.ScaleByScheduleState(count=jnp.array(opt_state_restored["step"])),
        ),
    )

    return restored_state, opt_state_restored["step"]


def create_zero_train_state(
    rng: jax.random.PRNGKey,
    learning_rate_fn: Union[float, Callable],
    weight_decay: float,
    model: nn.Module,
) -> Tuple[train_state.TrainState, Any, optax.GradientTransformation]:
    """
    Initializes model parameters, optimizer state and returns a simplified flax
    TrainState object.
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
        ),
    )

    init_batch = jnp.ones((1, model.block_size), dtype=jnp.int32)
    param_shape = jax.eval_shape(model.init, rng, init_batch)

    return params, param_shape, tx


def main():
    args = parse()
    cfg = OmegaConf.load(args.cfg)

    # getting system information
    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    num_host = num_devices // num_local_devices
    platform = jax.local_devices()[0].platform

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
            train_shards = open(cfg.data.index_path_train).read().splitlines()
            validation_shards = open(cfg.data.index_path_validation).read().splitlines()

    else:
        raise NotImplementedError("Training not currently supported on GPU.")

    model, model_config = model_getter(
        cfg.model.size, config_path=args.model_cfg, return_cfg=True, dtype=jnp.float32
    )

    learning_rate_fn = optax.warmup_cosine_decay_schedule(
        init_value=0,
        peak_value=cfg.training.peak_learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        decay_steps=cfg.training.total_steps,
        end_value=cfg.training.end_learning_rate,
    )

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    resume_step = 0

    params, param_shape, tx = create_zero_train_state(
        init_rng,
        learning_rate_fn,
        weight_decay=cfg.training.weight_decay,
        model=model,
    )

    devices = np.asarray(jax.devices())
    mesh = Mesh(devices, ("dp",))

    # axis_list_params = jax.tree_map(lambda x: [...], params)
    axis_list_params = [...]

    in_axes = (
        axis_list_params,
        ["batch", ...],
        [...],
    )
    out_axes = (axis_list_params, [...])

    # standard data parallel training step with xmap!
    train_step_xmap = xmap(
        partial(
            train_step,
            model=model,
            accum_steps=cfg.training.gradient_accumulation_steps,
        ),
        in_axes=in_axes,
        out_axes=out_axes,
        axis_resources={"batch": "dp"},
    )

    eval_axes = (
        axis_list_params,
        ["batch", ...],
    )
    eval_step_xmap = xmap(
        partial(eval_step, model=model),
        in_axes=eval_axes,
        out_axes=[...],
        axis_resources={"batch": "dp"},
    )

    opt_state_shapes = jax.eval_shape(tx.init, params)

    grad_param_spec = set_partitions_zero(param_shape)
    opt_state_spec = create_opt_spec(grad_param_spec, opt_state_shapes)

    if (cfg.model.warm_init) and not (args.resume):
        # only start from warm init params @ beginning of training run 
        del params

        if save_to_bucket:
            opt_state, step = restore_opt_checkpoint(
                workdir=f"gs://{cfg.data.bucket_path}/{cfg.model.warm_init_dir}/optimizer"
            )
            opt_state = jax.device_get(opt_state)  # copy to CPU

            params = restore_param_checkpoint(
                workdir=f"gs://{cfg.data.bucket_path}/{cfg.model.warm_init_dir}/params"
            )
            params = jax.device_get(params)  # copy to CPU

        else:
            raise NotImplementedError(
                "Checkpointing not currently implemented for GPU."
            )

        if jax.process_index() == 0:
            logger.debug(f"Warm starting training for pretrained checkpoint.")

    if args.resume:
        del params

        if save_to_bucket:
            opt_state, step = restore_opt_checkpoint(
                workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/optimizer"
            )
            opt_state = jax.device_get(opt_state)  # copy to CPU

            params = restore_param_checkpoint(
                workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/params"
            )
            params = jax.device_get(params)  # copy to CPU

            resume_step = int(step)

        else:
            raise NotImplementedError(
                "Checkpointing not currently implemented for GPU."
            )

        if jax.process_index() == 0:
            logger.debug(f"Resuming training from step {resume_step}")

    params = jax.device_get(params)  # copy params to VM CPU

    if jax.process_index() == 0:
        logger.debug(f"VM setup with {num_devices} devices.")
        logger.debug(f"Host setup with {num_local_devices} devices.")
        logger.debug(f"Using platform: {platform}.")

        logger.debug(
            f"Performing data parallel training. Model parameters are replicated across all devices. Optimizer state is sharded across {num_devices} devices"
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

    local_batch_size = cfg.training.batch_size // (jax.local_device_count())

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

    if cfg.training.train_context < cfg.data.max_context:
        seq_len = cfg.training.train_context
    else:
        seq_len = cfg.data.max_context

    accum_steps = cfg.training.gradient_accumulation_steps

    rng = jax.random.fold_in(rng, resume_step)  # fold in resume step to create new rng

    # quick way to track global step count when resuming a run
    new_steps = 0

    iterator_resume_step = int(resume_step % cfg.data.steps_per_epoch)
    with mesh:

        params = jax.device_get(params)

        if args.resume:
            opt_state = pjit(
                lambda x: x, in_axis_resources=None, out_axis_resources=opt_state_spec
            )(opt_state)

        else:
            opt_state = pjit(
                tx.init, in_axis_resources=None, out_axis_resources=opt_state_spec
            )(params)

        update_opt_state_pjit = pjit(
            partial(update_opt_state, optimizer=tx, grad_spec=grad_param_spec),
            in_shardings=(grad_param_spec, opt_state_spec, grad_param_spec),
            out_shardings=(None, opt_state_spec),
        )

        grad_shard = pjit(
            lambda x: x, in_axis_resources=None, out_axis_resources=grad_param_spec
        )

        for i, text in enumerate(tqdm(tl, disable=not jax.process_index() == 0)):

            if (resume_step + new_steps) > cfg.training.total_steps:
                if jax.process_index() == 0:
                    logger.debug(f"Training has completed.")

                return True

            if i < iterator_resume_step:
                continue

            rng, dropout_rng = jax.random.split(rng, 2)

            gradient_accumulation_steps = accum_steps

            if seq_len < cfg.data.max_context:
                text = text.reshape(-1, seq_len)

            # we add a 'grad_accum' batch dimension which we then iterate through in train_step
            text = text.reshape(
                gradient_accumulation_steps,
                text.shape[0] // gradient_accumulation_steps,
                seq_len,
            ).transpose(1, 0, 2)
            text = text.reshape(
                jax.device_count(),
                cfg.training.batch_size
                * (cfg.data.max_context // cfg.training.train_context)
                // (jax.device_count() * gradient_accumulation_steps),
                gradient_accumulation_steps,
                seq_len,
            )  # (8, 4, 2, 2048) -> (32, 1, 2, 2048)

            grads, metrics = train_step_xmap(params, text, dropout_rng)

            grads = grad_shard(grads)
            params = grad_shard(params)

            params, opt_state = update_opt_state_pjit(grads, opt_state, params)

            del grads  # manually free grad mem

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
                    val_text = val_text.reshape(-1, seq_len)
                    val_text = val_text.reshape(
                        jax.device_count(),
                        val_text.shape[0] // (jax.device_count()),
                        seq_len,
                    )
                    if val_it < cfg.training.maximum_evaluation_steps:
                        metrics = eval_step_xmap(params, val_text)
                        validation_metrics.append(metrics)
                    else:
                        break

                validation_metrics_np = {
                    k: np.mean([metrics[k] for metrics in validation_metrics])
                    for k in validation_metrics[0]
                }

                def grab_shards(tree):
                    return jax.experimental.multihost_utils.process_allgather(tree)

                opt_state_cpu = grab_shards(opt_state)

                if jax.process_index() == 0:

                    train_metrics_np.update(validation_metrics_np)
                    wandb.log(train_metrics_np)

                    if save_to_bucket:
                        save_checkpoint_params(
                            params,
                            absolute_step,
                            workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/params",
                        )

                        save_checkpoint_optimizer(
                            opt_state_cpu,
                            absolute_step,
                            workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/optimizer",
                        )

                    else:
                        raise NotImplementedError(
                            "Checkpointing not currently implemented for GPU/CPU"
                        )

                    # TODO: Call gc.collect here?
                    # del opt_state_cpu
                    # gc.collect()

            else:
                if jax.process_index() == 0:
                    wandb.log(train_metrics_np)


if __name__ == "__main__":
    main()
