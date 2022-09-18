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
from jax import random
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.models.GPT import model_getter
from src.training.training_utils import create_train_state, step_to_seq_len, compute_tokens_seen
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
    return checkpoints.restore_checkpoint(workdir, state, step = 368001)


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

    if cfg.distillation.distill:
        from flax.training import train_state
        from transformers import FlaxAutoModelForCausalLM

        teacher = FlaxAutoModelForCausalLM.from_pretrained("gpt2", dtype=model_dtype)

        # Yes, I'm aware this is silly. Without spending too much time this is a pretty quick fix, sgd doesn't consume extra params
        teacher_state = train_state.TrainState.create(
            apply_fn=teacher.__call__,
            params=teacher.params,
            tx=optax.sgd(
                learning_rate=0,
            ),
        )
        temperature = cfg.distillation.temperature
        alpha = cfg.distillation.alpha

        teacher_state = flax.jax_utils.replicate(teacher_state)

    if jax.process_index() == 0:
        logger.debug(f"Host setup with {num_devices} devices.")
        logger.debug(f"Using platform: {platform} with precision {model_dtype}")
        if cfg.training.staged_sequences is not None:
            logger.debug(
                f"Running sequence length warmup for {cfg.training.staged_warmup_steps} total steps with stages: {cfg.training.staged_sequences}"
            )

    resume_step = None

    if args.resume:
        # TODO: Get wandb ID for run too
        if platform == "tpu":
            restore_checkpoint(
                state,
                workdir=f"gs://{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}",
            )
        else:
            restore_checkpoint(state, workdir=cfg.data.checkpoint_directory)


        if jax.process_index() == 0:
            logger.debug(f"Resuming training from step {int(state.step)}")

        # resume step is ga_steps*global steps
        resume_step = int(state.step)

    # replicating state across devices
    state = flax.jax_utils.replicate(state)

    local_batch_size = cfg.training.batch_size // jax.device_count()

    # This is computed in terms of absolute steps
    total_tokens = cfg.training.batch_size*cfg.training.gradient_accumulation_steps*compute_tokens_seen(cfg.training.total_steps, stages = cfg.training.staged_sequences, max_steps = cfg.training.staged_warmup_steps,
            max_context=cfg.data.max_context)


    if jax.process_index() == 0:
        id = wandb.util.generate_id()
        wandb.init(id=id, resume="allow", project="LJX")
        flat_dict = flatten_dict(cfg)

        for key in model_config.keys():
            flat_dict[f"model.{key}"] = model_config[key]

        flat_dict["training.local_batch_size"] = local_batch_size
        flat_dict["runtime"] = platform
        flat_dict["Total Training Tokens"] = total_tokens/1e9
        wandb.config.update(flat_dict)

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

    def preprocess(batch):
        x = batch["input_id.pth"][: cfg.data.max_context]
        if type(x) == torch.tensor:
            return jnp.array(x.long(), dtype=jnp.int32)
        else:
            return jnp.array(x, dtype=jnp.int32)

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

    running_metrics = []

    if cfg.training.staged_sequences is not None:

        step_to_seq = partial(
            step_to_seq_len,
            stages=cfg.training.staged_sequences,
            max_steps=cfg.training.gradient_accumulation_steps
            * cfg.training.staged_warmup_steps,
            max_context=cfg.data.max_context,
        )

    else:
        step_to_seq = lambda x: cfg.data.max_context

    # I mean, we should eventually wrap this in an epoch loop
    for i, text in enumerate(tqdm(tl, disable=not jax.process_index() == 0)):

        if i > cfg.training.total_steps * cfg.training.gradient_accumulation_steps:
            # End training
            break

        if resume_step != None and i <= resume_step:
            continue

        seq_len = step_to_seq(i)
        text = text[:, :seq_len]

        # sharding batch
        sharded_batch = shard(text)

        t0 = time.time()

        if cfg.distillation.distill:
            state, metrics = distillation_train_step(
                state, teacher_state, sharded_batch, temperature, alpha
            )
        else:
            state, metrics = train_step(
                state,
                sharded_batch,
            )
        metrics["Train Batch Time"] = time.time() - t0
        metrics["Train Sequence Length"] = seq_len
        metrics["Tokens Seen (B)"] = (i/cfg.training.gradient_accumulation_steps)/total_tokens

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


# static_broadcasted_argnums is used for constants (@compile time) that aren't broadcasted
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4))
def distillation_train_step(
    state: Any,
    teacher_state: Any,
    batch: jnp.array,
    temperature: float,
    alpha: float,
    rng_key: random.PRNGKey = None,
):
    """Train on a single batch"""

    def loss_fn(params, teacher):
        student_logits, loss = state.apply_fn(
            {"params": params["params"]},
            x=batch,
            labels=batch,
            train=False,
        )

        teacher_logits = teacher_state.apply_fn(
            batch, params=teacher, train=False
        ).logits

        student_logits = student_logits / temperature

        teacher_logits = teacher_logits / temperature

        kd_loss = (temperature ** 2) * kl_div_loss(teacher_logits, student_logits)

        total_loss = (kd_loss * alpha) + (loss * (1 - alpha))

        return total_loss, (kd_loss, loss)

    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, (kd_loss, ce_loss)), grads = grad_fn(state.params, teacher_state.params)

    # compute all-reduce mean for gradients and loss
    # Ex: If we have 8 devices, each device takes the gradients from the other 7 and averages them all together
    # that way, all device replicas have the same gradients and optimization step can occur in parallel
    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")

    state = state.apply_gradients(
        grads=grads,
    )

    kd_loss = jax.lax.pmean(kd_loss, axis_name="batch")
    ce_loss = jax.lax.pmean(ce_loss, axis_name="batch")

    metrics = {
        "Train Total Loss": loss,
        "Train KD Loss": kd_loss,
        "Train LM Loss": ce_loss,
        "Train LM PPL": jnp.exp(ce_loss),
    }

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
