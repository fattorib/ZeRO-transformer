import argparse
import logging
import random as pyrandom
from functools import partial
from time import time
from typing import Any, Callable, Tuple, Union

import flax 
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import webdataset as wds
from flax.training import checkpoints, train_state
from jax.experimental.pjit import pjit
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.models.GPT import Transformer, model_getter
from src.partitioning.partition import create_opt_spec
from src.utils.configs import flatten_dict
from src.utils.dataloader import numpy_collate
from src.training.train_functions import eval_step, train_step, update_opt_state

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
            workdir, faux_state, step, keep=5, overwrite=True, prefix="params_"
        )

def save_checkpoint_optimizer(opt_state: Any, step: int, workdir: str) -> None:
    """
    Save a copy of opt_state.

    TODO: Add async manager to do this in a background process
    """
    if jax.process_index() == 0:
        opt_state = jax.device_get(opt_state)
        faux_state = train_state.TrainState(
            step=step, apply_fn=None, params=None, tx=None, opt_state=opt_state
        )
        checkpoints.save_checkpoint(
            workdir, faux_state, step, keep=5, overwrite=True, prefix="opt_"
        )


def restore_checkpoint_params(
    workdir: str, param_spec: Any
) -> Tuple[Any, Any, int]:
    """
    Restores the most recent parameter dict
    """
    restored = checkpoints.restore_checkpoint(workdir, target=None, prefix="params_")
    with jax.default_device(jax.devices("cpu")[0]):
        params = flax.core.freeze(restored["params"])    
        params = jax.tree_map(lambda x, y: nn.Partitioned(value = jnp.array(y['value']), names = x, mesh = None),param_spec, params)

        return params, restored["step"]

def restore_checkpoint_opt(opt_spec: Any, workdir: str
) -> Tuple[Any, Any, int]:
    """
    Restores the most recent opt state dict.
    """
    restored = checkpoints.restore_checkpoint(workdir, target=None, prefix="opt_")

    with jax.default_device(jax.devices("cpu")[0]):
        mu_pytree = jax.tree_map(
            lambda x: jnp.array(x), restored["opt_state"]["1"]["0"]["mu"]
        )
        mu_pytree = jax.tree_map(lambda x, y: nn.Partitioned(value = jnp.array(y['value']), names = x, mesh = None),opt_spec[1][0].mu, flax.core.freeze(mu_pytree))                                                                                                                                           

        count_pytree = jax.tree_map(
            lambda x: jnp.array(x), restored["opt_state"]["1"]["0"]["count"]
        )

        restoredlionstate = optax.ScaleByLionState(
            count_pytree, flax.core.FrozenDict(mu_pytree)
        )


        opt_state = (
            optax.EmptyState(),
            (
                restoredlionstate,
                optax.MaskedState(inner_state=optax.EmptyState()),
                optax.ScaleByScheduleState(count=jnp.array(restored["step"])),
            ),
        )
        return opt_state

def create_train_state(
    rng: jax.random.PRNGKey,
    learning_rate_fn: Union[float, Callable],
    weight_decay: float,
    model: nn.Module,
) -> Tuple[Any, optax.GradientTransformation]:
    """
    Gets the abstract model shapes and sets up the optimizer.
    """

    batch = jax.random.randint(rng, shape=(1, model.block_size), maxval=50257, minval=0)
    param_abstract = jax.eval_shape(model.init, rng, batch)

    # This mask turns off weight decay for bias terms, LN terms and position embeddings
    mask = jax.tree_map(
        lambda x: x.ndim != 1 and x.shape != (model.block_size, model.embedding_dim),
        param_abstract,
    )

    tx = optax.chain(
        optax.clip(1.0),
        optax.lion(
            learning_rate=learning_rate_fn,
            weight_decay=weight_decay,
            mask=mask,
            b1 = 0.95,
            b2 = 0.98
        ),
    )

    return param_abstract, tx


def main():
    args = parse()
    cfg = OmegaConf.load(args.cfg)

    # getting system information
    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    num_host = num_devices // num_local_devices
    platform = jax.local_devices()[0].platform

    mesh = Mesh(
        np.array(jax.devices()).reshape(cfg.training.dp, cfg.training.mp), ("dp", "mp")
    )

    if jax.process_index() == 0:
        logger.debug(f"VM setup with {num_devices} devices.")
        logger.debug(f"Host setup with {num_local_devices} devices.")
        logger.debug(f"Using platform: {platform}.")
        logger.debug(f"Mesh Shape (dp,mp): {(mesh.shape['dp'], mesh.shape['mp'])}.")

    # setting up GCP bucket/client info if training on TPU
    save_to_bucket = False
    client = None
    if platform == "tpu":
        if cfg.data.bucket_path is not None:
            # use GCP
            from google.cloud import storage

            client = storage.Client()
            save_to_bucket = True
            train_shards = open(cfg.data.index_path_train).read().splitlines()
            validation_shards = open(cfg.data.index_path_validation).read().splitlines()

    else:
        raise NotImplementedError("Training not currently supported on GPU.")

    model_full, model_config = model_getter(
        cfg.model.size, config_path=args.model_cfg, return_cfg=True, dtype=jnp.float32
    )

    def compute_tokens_seen(absolute_step, max_context):
        return absolute_step * max_context
    
    
    # set up sharded config and model too
    model_config["num_shard"] = int(cfg.training.mp)
    model_config["tp_comms"] = True if mesh.shape["mp"] > 1 else False
    model_shard = Transformer(**model_config)

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

    param_abstract, tx = create_train_state(
        init_rng,
        learning_rate_fn,
        weight_decay=cfg.training.weight_decay,
        model=model_full,
    )

    # Setup partition specs

    param_spec = nn.get_partition_spec(param_abstract)
    grad_spec = param_spec
    batch_spec = P("dp", None)

    
    # Setup params and optimizer states
    with mesh:
        # do actual layer init wrapping with pjit
        if not args.resume:
            batch = jnp.ones((1, model_full.block_size), dtype=jnp.int32)
            params = pjit(model_full.init, out_axis_resources=param_spec)(rng, batch)

        opt_state_shapes = jax.eval_shape(tx.init, param_abstract)

        opt_state_spec = create_opt_spec(param_spec, opt_state_shapes)

        if not args.resume:
            opt_state = pjit(
                tx.init,
                in_axis_resources=(param_spec,),
                out_axis_resources=opt_state_spec,
            )(params)

    if jax.process_index() == 0:
        logger.debug(f"Params and Optimizer state compiled and sharded")
        
    train_step_tp = jax.jit(
        shard_map(
            partial(
                train_step,
                model=model_shard,
                accum_steps=cfg.training.gradient_accumulation_steps,
            ),
            in_specs=(param_spec, batch_spec),
            out_specs=(grad_spec, P(None)),
            mesh=mesh,
            check_rep=False,
        )
    )

    eval_step_tp = jax.jit(
        shard_map(
            partial(eval_step, model=model_shard),
            in_specs=(param_spec, batch_spec),
            out_specs=(P(None)),
            mesh=mesh,
            check_rep=False,
        )
    )

    with mesh:
        update_opt_step_tp = pjit(
            partial(update_opt_state, optimizer=tx, tp_spec=grad_spec),
            in_axis_resources=(param_spec, grad_spec, opt_state_spec),
            out_axis_resources=(param_spec, opt_state_spec),
            donate_argnums=0,
        )

    if args.resume:

        if save_to_bucket:

            params, step = restore_checkpoint_params(
                workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/params",
                param_spec = param_spec,
                
            )
            resume_step = int(step)

            opt_state = restore_checkpoint_opt(
                opt_state_spec,
                workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/opt",
            ) 
            
            import gc
            gc.collect()
            with mesh:
                params = pjit(lambda x:x, in_axis_resources= (param_spec,), out_axis_resources=param_spec, donate_argnums=0)(params)
                opt_state = pjit(lambda x:x, in_axis_resources= (opt_state_spec,), out_axis_resources=opt_state_spec, donate_argnums=0)(opt_state)


        else:
            raise NotImplementedError(
                "Checkpointing not currently implemented for GPU."
            )

        if jax.process_index() == 0:
            logger.debug(f"Resuming training from step {resume_step}")

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

    # TODO: Update
    local_batch_size = cfg.training.batch_size // (cfg.training.dp)

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
        flat_dict["mesh (mp)"] = mesh.shape["mp"]
        flat_dict["mesh (dp)"] = mesh.shape["dp"]
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
        wds.shuffle(1e5, initial=1e5, rng=pyrandom.Random(23)),
        wds.decode(handler=wds.warn_and_continue),
        wds.map(preprocess),
    ).repeat(nepochs=cfg.training.max_epochs)

    validation_dataset = wds.DataPipeline(
        wds.SimpleShardList(validation_shards),
        split_by_jax_process,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(1e5, initial=1e5, rng=pyrandom.Random(23)),
        wds.decode(handler=wds.warn_and_continue),
        wds.map(preprocess),
    )

    tl = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=numpy_collate,
        drop_last=True,
        num_workers=0,
    )

    vl = DataLoader(
        dataset=validation_dataset,
        batch_size=cfg.training.batch_size // 4,
        collate_fn=numpy_collate,
        drop_last=True,
        num_workers=0,
    )

    running_metrics = []

    if cfg.training.train_context < cfg.data.max_context:
        seq_len = cfg.training.train_context
    else:
        seq_len = cfg.data.max_context


    # quick way to track global step count when resuming a run
    new_steps = 0

    # iterator_resume_step = int(resume_step % cfg.data.steps_per_epoch)
    iterator_resume_step = 0
    for i, batch in enumerate(tqdm(tl, disable=not jax.process_index() == 0)):

        if (resume_step + new_steps) > cfg.training.total_steps:
            if jax.process_index() == 0:
                logger.debug("Training has completed.")

            return True

        if i < iterator_resume_step:
            continue

        if seq_len < cfg.data.max_context:
            batch = batch.reshape(-1, seq_len)
        t0 = time()
        grads, metrics = train_step_tp(params, batch)

        with mesh:
            params, opt_state = update_opt_step_tp(params, grads, opt_state)
        metrics["train/seq_len"] = seq_len
        metrics["train/lr"] = learning_rate_fn(resume_step + new_steps)
        t1 = time()
        metrics["train/step_time"] = t1 - t0

        running_metrics.append(metrics)

        train_metrics_np = {
            k: np.mean([metrics[k] for metrics in running_metrics])
            for k in running_metrics[0]
        }

        running_metrics = []
        validation_metrics = []

        absolute_step = resume_step + new_steps

        train_metrics_np["train/tokens"] = (
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
            for val_it, val_batch in enumerate(
                tqdm(vl, disable=not jax.process_index() == 0)
            ):
                val_batch = val_batch.reshape(-1, seq_len)
                if val_it < cfg.training.maximum_evaluation_steps:
                    metrics = eval_step_tp(params, val_batch)
                    validation_metrics.append(metrics)
                else:
                    break

            validation_metrics_np = {
                k: np.mean([metrics[k] for metrics in validation_metrics])
                for k in validation_metrics[0]
            }

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
                        opt_state,
                        absolute_step,
                        workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/opt",
                    )
            
            jax.experimental.multihost_utils.sync_global_devices("checkpoint_barr")

        else:
            if jax.process_index() == 0:
                wandb.log(train_metrics_np)


if __name__ == "__main__":
    main()
