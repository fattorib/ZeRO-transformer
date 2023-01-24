import argparse
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
from flax.training import checkpoints
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state
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


class TrainState(train_state.TrainState):
    dynamic_scale: dynamic_scale_lib.DynamicScale


def save_checkpoint_params(
    params: Any, opt_state: Any, dynamic_scale: Any, step: int, workdir: str
) -> None:
    """
    Save a copy of params.
    """
    if jax.process_index() == 0:
        params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))
        opt_state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], opt_state))
        dynamic_scale = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], dynamic_scale))

        faux_state = TrainState(
            step=step,
            apply_fn=None,
            params=params,
            tx=None,
            opt_state=opt_state,
            dynamic_scale=dynamic_scale,
        )
        checkpoints.save_checkpoint(
            workdir, faux_state, step, keep=5, overwrite=True, prefix="state_"
        )


def restore_state(workdir: str) -> Any:
    """
    Restores the most trainstate parameter dict
    """

    state = checkpoints.restore_checkpoint(workdir, target=None, prefix="state_")

    mu_pytree = jax.tree_util.tree_map(
        lambda x: jnp.array(x), state["opt_state"]["1"]["0"]["mu"]
    )

    nu_pytree = jax.tree_util.tree_map(
        lambda x: jnp.array(x), state["opt_state"]["1"]["0"]["nu"]
    )

    count_pytree = jax.tree_util.tree_map(
        lambda x: jnp.array(x), state["opt_state"]["1"]["0"]["count"]
    )

    restoredadamstate = optax.ScaleByAdamState(
        count_pytree, flax.core.FrozenDict(mu_pytree), flax.core.FrozenDict(nu_pytree)
    )
    restored_state = (
        optax.EmptyState(),
        (
            restoredadamstate,
            optax.MaskedState(inner_state=optax.EmptyState()),
            optax.ScaleByScheduleState(count=jnp.array(state["step"])),
        ),
    )

    return (
        flax.core.freeze(state["params"]),
        state["dynamic_scale"],
        restored_state,
        state["step"],
    )


def create_train_state(
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

    train_shards = open(cfg.data.shard_path_train).read().splitlines()
    validation_shards = open(cfg.data.shard_path_validation).read().splitlines()

    model, model_config = model_getter(
        cfg.model.size, config_path=args.model_cfg, return_cfg=True, dtype=jnp.float16
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

    state, optimizer, opt_state = create_train_state(
        init_rng,
        learning_rate_fn,
        weight_decay=0.1,
        model=model,
    )
    params = state.params
    dynamic_scale = dynamic_scale_lib.DynamicScale()

    del state

    if args.resume:
        del params
        del opt_state

        params, dynamic_scale_dict, opt_state, step = restore_state(
            workdir=f"checkpoints/solu"
        )

        resume_step = int(step)

        if jax.process_index() == 0:
            logger.debug(f"Resuming training from step {resume_step}")
        dynamic_scale = dynamic_scale_lib.DynamicScale(fin_steps=dynamic_scale_dict['fin_steps'], scale=dynamic_scale_dict['scale'])
    params = jax.device_get(params)  # copy params to CPU

    opt_state = jax.device_get(opt_state)  # copy opt_state to CPU

    if jax.process_index() == 0:
        logger.debug(f"Training setup with {num_devices} devices.")
        logger.debug(f"Host setup with {num_local_devices} devices.")
        logger.debug(f"Using platform: {platform}.")

    local_batch_size = cfg.training.batch_size // (jax.local_device_count())

    total_tokens = num_host * (
        cfg.training.batch_size
        * compute_tokens_seen(
            cfg.training.total_steps,
            max_context=cfg.data.max_context,
        )
    )

    if jax.process_index() == 0:
        if resume_step > 0:
            id = cfg.data.wandb_id
            wandb.init(id=id, resume="allow", project=cfg.data.wandb_project)
        else:
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
        wds.shuffle(1e4, initial=1e4, rng=pyrandom.Random(23 + resume_step)),
        wds.decode(handler=wds.warn_and_continue),
        wds.map(preprocess),
    ).repeat(nepochs=cfg.training.max_epochs)

    validation_dataset = wds.DataPipeline(
        wds.SimpleShardList(validation_shards),
        split_by_jax_process,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(1e4, initial=1e4, rng=pyrandom.Random(23 + resume_step)),
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

    accum_steps = lambda x: cfg.training.gradient_accumulation_steps

    params = flax.jax_utils.replicate(params, devices=jax.local_devices())
    opt_state = flax.jax_utils.replicate(opt_state, devices=jax.local_devices())
    dynamic_scale = flax.jax_utils.replicate(dynamic_scale, devices=jax.local_devices())

    

    rng = jax.random.fold_in(rng, resume_step)  # fold in resume step to create new rng

    # quick way to track global step count when resuming a run
    new_steps = 0

    for i, text in enumerate(tqdm(tl, disable=not jax.process_index() == 0)):

        if (resume_step + new_steps) > cfg.training.total_steps:
            if jax.process_index() == 0:
                logger.debug(f"Training has completed.")

            return True

        if (i < int(cfg.data.resume_step)) and (resume_step > 0):
            continue

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

        grads, metrics, dynamic_scale, is_fin = train_step(
            params, text, rng_sharded, gradient_accumulation_steps, model, dynamic_scale
        )

        params, opt_state = update_state(grads, opt_state, params, optimizer, is_fin)

        

        metrics["Train Sequence Length"] = seq_len
        metrics["Learning Rate"] = learning_rate_fn(resume_step + new_steps)

        metrics["Scaler"] = dynamic_scale.scale

        metrics["l2_norm/embedding/param"] = jax.tree_util.tree_map(lambda x : jnp.sqrt(jnp.linalg.norm(x)), params["params"]["wte"]["embedding"])
        metrics["l2_norm/embedding/second_moment_sqrt"] = jax.tree_util.tree_map(lambda x : jnp.sqrt(jnp.linalg.norm(x)), opt_state[1][0].mu['params']["wte"]["embedding"])
        metrics["l2_norm/embedding/first_moment"] = jax.tree_util.tree_map(lambda x : jnp.linalg.norm(x), opt_state[1][0].mu['params']["wte"]["embedding"])
        metrics["l2_norm/embedding/gradient"] = jax.tree_util.tree_map(lambda x : jnp.linalg.norm(x), grads['params']["wte"]["embedding"])

        del grads  # manually free grad mem

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
                val_text = val_text[:, : cfg.training.warmup_train_context]
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
                wandb.log(train_metrics_np)

                save_checkpoint_params(
                    params,
                    opt_state,
                    dynamic_scale,
                    absolute_step,
                    workdir=f"checkpoints/solu",
                )

        else:
            if jax.process_index() == 0:
                wandb.log(train_metrics_np)


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4))
def train_step(
    params: Any,
    batch: jnp.array,
    rng_key: jax.random.PRNGKey = None,
    accum_steps: int = 8,
    model: Any = None,
    dynamic_scale: Any = None,
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

    grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=False, axis_name="batch")

    def loss_and_grad(grad_idx):
        minibatch = get_minibatch(batch, grad_idx) if grad_idx is not None else batch

        dynamic_scale, is_fin, loss, grads = grad_fn(params, minibatch)

        return dynamic_scale, is_fin, loss, grads

    init_minibatch = (dynamic_scale, True, 0.0, jax.tree_util.tree_map(jnp.zeros_like, params))

    def cumul_minibatch_step(grad_idx, cumul_loss_grad):
        _, cumul_is_fin, cumul_loss, cumul_grads = cumul_loss_grad
        dynamic_scale, is_fin, loss, grads = loss_and_grad(grad_idx)

        cumul_loss, cumul_grads = jax.tree_util.tree_map(
            jnp.add, (cumul_loss, cumul_grads), (loss, grads)
        )

        cumul_is_fin = jax.tree_util.tree_map(jnp.logical_and, cumul_is_fin, is_fin)

        return dynamic_scale, cumul_is_fin, cumul_loss, cumul_grads

    dynamic_scale, is_fin, loss, grads = jax.lax.fori_loop(
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

    return grads, metrics, dynamic_scale, is_fin


@partial(
    jax.pmap,
    axis_name="batch",
    devices=jax.local_devices(),
    static_broadcasted_argnums=(3,),
)
def update_state(
    grads: Any, optimizer_state: Any, params: Any, optimizer: Any, is_fin: Any
):
    """
    Updates the sharded optimizer state
    """

    # These two lines update the specific shard of state/parameters sitting on device 'i'
    updates, new_opt_state = optimizer.update(grads, optimizer_state, params)
    new_params = optax.apply_updates(params, updates)

    new_opt_state = jax.tree_util.tree_map(
        partial(jnp.where, is_fin), new_opt_state, optimizer_state
    )

    new_params = jax.tree_util.tree_map(partial(jnp.where, is_fin), new_params, params)

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
