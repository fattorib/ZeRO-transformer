import argparse
import io
import logging
import random as pyrandom
import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import webdataset as wds
from flax.training import checkpoints
from jax import random
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit, with_sharding_constraint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.models.GPT import model_getter
from src.training.training_utils import (TrainState, compute_tokens_seen,
                                         create_train_state, get_optimizer)
from src.utils.configs import flatten_dict
from src.utils.dataloader import numpy_collate
from src.utils.partitioning import (create_opt_spec, set_partitions,
                                    setup_dp_mesh, setup_mp_mesh)

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


def save_checkpoint(state, workdir, bucket_path=None, client=None):
    # if jax.process_index() == 0:
    #     step = int(state.step)
    #     checkpoints.save_checkpoint(workdir, state, step, keep=3, overwrite=True)
    return None


def restore_checkpoint(state, workdir, prefix):
    # return checkpoints.restore_checkpoint(workdir, state, prefix=prefix)
    return None


def main():
    args = parse()
    cfg = OmegaConf.load(args.cfg)

    # getting system information
    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    num_host = num_devices // num_local_devices
    platform = jax.local_devices()[0].platform

    assert (
        num_devices // (cfg.device.dp_devices * cfg.device.mp_devices) == 1
    ), f"Incorrect mesh shape specified for {num_devices} devices with mesh shape {(cfg.device.dp_devices,cfg.device.mp_devices)}. Check your device configs"

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

    if cfg.device.mp_devices == 1:
        mesh = setup_dp_mesh()

    else:
        mesh = setup_mp_mesh(cfg)

    resume_step = None

    if cfg.device.mp_devices == 1:
        state = create_train_state(
            init_rng,
            learning_rate_fn,
            weight_decay=cfg.training.weight_decay,
            model=model,
            grad_accum_steps=cfg.training.gradient_accumulation_steps,
        )
        param_spec = None

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

            # resume step is ga_steps*global steps
            resume_step = int(state.step)

    else:
        # TODO: Most of this code can probably be moved out to partitioning.py

        # use jax.eval_shape to get pytree with empty params and correct shapes
        # saves us having to do an actual model forward pass / any actual computation
        batch_tok = jnp.ones(shape=(1, 512), dtype=jnp.int32)
        param_shape = jax.eval_shape(model.init, init_rng, batch_tok)
        param_spec = set_partitions(param_shape)

        # creating optimizer
        tx = get_optimizer(
            learning_rate_fn,
            weight_decay=cfg.training.weight_decay,
            model=model,
            grad_accum_steps=cfg.training.gradient_accumulation_steps,
            param_shape=param_shape,
        )

        # get optimizer state spec
        opt_state_shapes = jax.eval_shape(tx.init, param_shape)
        opt_state_spec = create_opt_spec(param_spec, opt_state_shapes)

        # create TrainState spec
        state_spec = TrainState(
            params=param_spec,
            opt_state=opt_state_spec,
            tx=tx,
            step=None,
            apply_fn=model.apply,
        )

        def init_state(params):
            return TrainState.create(
                apply_fn=model.apply,
                tx=tx,
                params=params,
            )

        # pjit-able way to restore sharded state from a non-sharded state
        # using lambda x: x doesn't work, pjit complains about opt_state being different (even though it isnt!)
        def restore_state(params, step, opt_state):
            return TrainState(
                params=params,
                opt_state=opt_state,
                step=step,
                tx=tx,
                apply_fn=model.apply,
            )

        if args.resume:

            # Just make a valid copy of the trainstate to read into
            state = create_train_state(
                init_rng,
                learning_rate_fn,
                weight_decay=cfg.training.weight_decay,
                model=model,
                grad_accum_steps=cfg.training.gradient_accumulation_steps,
            )

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

            # resume step is ga_steps*global steps
            resume_step = int(state.step)

            with mesh:
                state = pjit(
                    restore_state,
                    in_axis_resources=(None, None, None),
                    out_axis_resources=(state_spec),
                )(state.params, state.step, state.opt_state)

        else:
            with mesh:
                init_batch = jax.numpy.ones(shape=(1, 512), dtype=jax.numpy.int32)

                # shard params across mesh
                sharded_params = pjit(
                    partial(model.init, train=False),
                    in_axis_resources=(None, None),
                    out_axis_resources=(param_spec),
                )(rng, init_batch)

                # shard state across mesh
                state = pjit(
                    init_state,
                    in_axis_resources=(param_spec,),
                    out_axis_resources=(state_spec),
                )(sharded_params)

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

    # if not args.resume:
    #     if cfg.data.bucket_path is not None:
    #         # clear bucket
    #         client = storage.Client()
    #         if jax.process_index() == 0:
    #             bucket = storage.Bucket(client, f"{cfg.data.bucket_path}")
    #             blobs = bucket.list_blobs(prefix=f"{cfg.data.checkpoint_directory}")
    #             for blob in blobs:
    #                 blob.delete()

    local_batch_size = cfg.training.batch_size // (
        jax.local_device_count() // cfg.device.mp_devices
    )

    # This is computed in terms of absolute steps
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
        handler=wds.warn_and_continue,
    ).repeat(nepochs=cfg.training.max_epochs)

    validation_dataset = wds.DataPipeline(
        wds.SimpleShardList(validation_shards),
        split_by_jax_process,
        wds.tarfile_to_samples(),
        wds.shuffle(1e6, initial=1e6, rng=pyrandom.Random(23)),
        wds.decode(),
        wds.map(preprocess),
        handler=wds.warn_and_continue,
    )

    tl = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=numpy_collate,
        drop_last=True,
    )

    vl = DataLoader(
        dataset=validation_dataset,
        batch_size=cfg.training.batch_size//8,
        collate_fn=numpy_collate,
        drop_last=True,
    )

    running_metrics = []

    step_to_seq = lambda x: 512

    if cfg.device.mp_devices == 1:
        with mesh:
            pjit_train_step = pjit(
                train_step,
                in_axis_resources=(None, PartitionSpec("dp"), None, None),
                out_axis_resources=None,
            )

            pjit_eval_step = pjit(
                eval_step,
                in_axis_resources=(None, PartitionSpec("dp")),
                out_axis_resources=None,
            )

    else:
        with mesh:
            pjit_train_step = pjit(
                partial(train_step, param_spec=param_spec),
                in_axis_resources=(state_spec, PartitionSpec("dp"), None),
                out_axis_resources=(state_spec, None),
                donate_argnums=(0,),
            )

            pjit_eval_step = pjit(
                eval_step,
                in_axis_resources=(state_spec, PartitionSpec("dp")),
                out_axis_resources=None,
            )

    with mesh:

        for i, text in enumerate(tqdm(tl, disable=not jax.process_index() == 0)):

            if (
                i // cfg.training.gradient_accumulation_steps
            ) > cfg.training.total_steps:
                if jax.process_index() == 0:
                    logger.debug(f"Training has completed.")

                return True

            if resume_step != None and i <= resume_step:
                continue

            rng, dropout_rng = jax.random.split(rng, 2)

            seq_len = step_to_seq(i)

            text = text.reshape(-1, seq_len)

            # we add a 'grad_accum' batch dimension
            # in our case, we have BS (64, 512) -> (grad_accum_cnt, bs, 512)
            text = text.reshape(8, text.shape[0]//8, seq_len)
        
            t0 = time.time()

            state, metrics = pjit_train_step(
                state,
                text,
                dropout_rng,
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

                absolute_step = i // cfg.training.gradient_accumulation_steps

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

                if (i) % (
                    cfg.training.evaluation_frequency
                    * cfg.training.gradient_accumulation_steps
                ) == 0:
                    for val_it, val_text in enumerate(
                        tqdm(vl, disable=not jax.process_index() == 0)
                    ):
                        val_text = val_text[:,:512]
                        if val_it < cfg.training.maximum_evaluation_steps:
                            metrics = pjit_eval_step(state, val_text)
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
                            # save_checkpoint(
                            #     device_state,
                            #     workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}",
                            # )

                        else:
                            # if save_to_bucket:
                            #     save_checkpoint(
                            #         state,
                            #         workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}",
                            #     )
                            # else:
                            #     save_checkpoint(
                            #         state, workdir=cfg.data.checkpoint_directory
                            #     )
                            pass 

                else:
                    if jax.process_index() == 0:
                        train_metrics_np["Train Step Time"] = (
                            cfg.training.gradient_accumulation_steps
                            * train_metrics_np["Train Batch Time"]
                        )
                        train_metrics_np.pop("Train Batch Time")
                        wandb.log(train_metrics_np)


def train_step(
    state: Any, batch: jnp.array, rng_key: jax.random.PRNGKey = None, param_spec: Any = None
):
    """Train on a single Gradient-Accumulation batch
    This means that the batch will be size (local_bs*grad_accum, ctx) instead of (local_bs, ctx)

    """

    # effective BS is (128,512)
    # loop through this in groups of 16

    def get_minibatch(batch, grad_idx):
        return jax.tree_util.tree_map(
            lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False),
            batch,
        )
    
    def loss_fn(params, batch, rng_key):
        _, loss = state.apply_fn(
            {"params": params["params"]},
            x=batch,
            labels=batch,
            train=True,
            rngs={"dropout": rng_key},
        )
        return loss 

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

    def loss_and_grad(grad_idx, dropout_rng): 
        minibatch = (
                get_minibatch(batch, grad_idx) if grad_idx is not None else batch
            )
        minibatch = with_sharding_constraint(minibatch, PartitionSpec('dp'))

        rng_new, rng_drop = jax.random.split(dropout_rng)

        loss, grads = grad_fn(state.params, minibatch, rng_drop)

        grads = with_sharding_constraint(grads, param_spec)

        return loss, grads, rng_new

    init_minibatch = (
        0.0, 
        with_sharding_constraint(
                    jax.tree_util.tree_map(jnp.zeros_like, state.params), param_spec
        ), 
        rng_key
    )

    # accumulate gradients
    def cumul_minibatch_step(grad_idx, cumul_loss_grad):
        cumul_loss, cumul_grads, dropout_rng = cumul_loss_grad
        loss, grads = loss_and_grad(grad_idx, dropout_rng)
        cumul_loss, cumul_grads = jax.tree_util.tree_map(
            jnp.add, (cumul_loss, cumul_grads), (loss, grads)
        )
        cumul_grads = with_sharding_constraint(cumul_grads, param_spec)
        return cumul_loss, cumul_grads, dropout_rng

    loss, grads, dropout_rng = jax.lax.fori_loop(
                0,
                8, #TODO: Unhardcode these
                cumul_minibatch_step,
                init_minibatch,
            )
    grads = with_sharding_constraint(grads, param_spec)
    # sum -> mean
    loss, grads = jax.tree_util.tree_map(
        lambda x: x / 8, (loss, grads)
    )

    grads = with_sharding_constraint(grads, param_spec)

    # only update train_state at the end of a single full batch 
    new_state = state.apply_gradients(
        grads=grads,
    )

    metrics = {
        "Train LM Loss": loss,
        "Train LM PPL": jnp.exp(loss),
    }

    return new_state, metrics


def eval_step(state: Any, batch: jnp.array):
    """Evaluate on a single batch"""

    _, loss = state.apply_fn(
        {"params": state.params["params"]},
        x=batch,
        labels=batch,
        train=False,
    )

    metrics = {"Validation LM Loss": loss, "Validation LM PPL": jnp.exp(loss)}

    return metrics


if __name__ == "__main__":
    # try:
    # main()
    # except Exception as e:
    # print(f"Error encountered: {e}")
    main()
