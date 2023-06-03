""" 
Replication of GPT2 transformers in Flax
"""
from typing import Any, Tuple, Union

import flax.linen as nn
import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
from omegaconf import OmegaConf

from src.models.layers import CausalAttention, MLPBlock
from src.utils.losses import cross_entropy_loss
from src.models.replicated_utils import g_psum, f_psum
from jax.sharding import PartitionSpec as P


class TransformerBlock(nn.Module):
    """One full transformer block"""

    embedding_dim: int
    num_head: int
    block_size: int
    residual_dropout: float = 0.0
    N: int = None
    dtype: Any = jnp.float32
    alibi_attn: bool = True


    tp_comms: bool = False
    num_shard: int = 1

    @nn.compact
    def __call__(
        self,
        x: jnp.array,
        train: bool = False,
    ) -> jnp.array:

        attn_out = CausalAttention(
            self.embedding_dim,
            self.num_head,
            self.block_size,
            self.residual_dropout,
            self.N,
            self.alibi_attn,
            self.dtype,
            tp_comms=self.tp_comms,
            num_shard=self.num_shard
        )(nn.LayerNorm(dtype=self.dtype, use_bias=False, scale_init=nn.with_partitioning(jax.nn.initializers.ones, P(None)))(x), train)
        x = x + attn_out
        x = x + MLPBlock(
            self.embedding_dim,
            dropout=self.residual_dropout,
            N=self.N,
            dtype=self.dtype,
            tp_comms=self.tp_comms,
            num_shard=self.num_shard
        )(nn.LayerNorm(dtype=self.dtype, use_bias=False, scale_init=nn.with_partitioning(jax.nn.initializers.ones, P(None)))(x), train)
        return x


class Transformer(nn.Module):
    """
    Full transformer module
    """

    embedding_dim: int
    vocab_size: int
    num_head: int
    block_size: int
    dropout: float = 0.0
    N: int = None
    dtype: Any = jnp.float32
    alibi_attn: bool = False


    tp_comms: bool = False
    num_shard: int = 1

    @nn.compact
    def __call__(
        self,
        x: jnp.array,
        labels: jnp.array = None,
        train: bool = False,
    ) -> Union[jnp.array, Tuple[jnp.array, jnp.array]]:
        
        embed = nn.Embed(
            name="wte",
            num_embeddings=self.vocab_size//self.num_shard,
            features=self.embedding_dim,
            embedding_init=nn.with_partitioning(initializers.normal(stddev=0.02), P("mp", None)),
            dtype=self.dtype,
        )
        
        out = embed(x)

        if self.tp_comms:
            out = g_psum(out)

        for _ in range(self.N):
            out = nn.checkpoint(TransformerBlock)(
                self.embedding_dim,
                self.num_head,
                self.block_size,
                self.dropout,
                self.N,
                self.dtype,
                self.alibi_attn,
                self.tp_comms,
                self.num_shard
            )(out, train)

        out = nn.LayerNorm(dtype=self.dtype, use_bias=False, scale_init=nn.with_partitioning(jax.nn.initializers.ones, P(None)))(out)

        logits = embed.attend(out)
        
        if self.tp_comms:
            # inefficient all-gather for debugging 
            # logits = jax.lax.all_gather(logits, axis_name='mp')
            # logits = jnp.concatenate(logits, axis = -1)
            if labels is None:
                logits = jax.lax.all_gather(logits, axis_name='mp')
                logits = jnp.concatenate(logits, axis = -1)
                return logits

            else:
                # each mp shard computes local loss and then we all-gather these to reduce 
                # total comm volume
                # loss calculation from mesh-transformer-jax: 
                # https://github.com/kingoflolz/mesh-transformer-jax/blob/master/mesh_transformer/layers.py#L569
                labels = labels[..., 1:]
                logits = logits[..., :-1, :]

                dim_per_shard = self.vocab_size//self.num_shard
                shard_start_index = jax.lax.axis_index('mp') * dim_per_shard
                global_max = jax.lax.pmax(jax.lax.stop_gradient(logits.max(-1, keepdims=True)), "mp")
                logits -= jax.lax.stop_gradient(global_max)

                gt_onehot = jax.nn.one_hot(labels - shard_start_index, dim_per_shard)
                predicted_logits = jnp.sum(jnp.multiply(gt_onehot, logits), axis=-1)
                predicted_logits = g_psum(predicted_logits)

                exp_logits = jnp.exp(logits.astype(jnp.float32))

                sum_exp_logits = exp_logits.sum(axis=-1)
                sum_exp_logits = g_psum(sum_exp_logits)

                loss = jnp.log(sum_exp_logits) - predicted_logits

                return logits,loss 

        else:
            if labels is None:
                return logits
            else:
                labels_shifted = labels[..., 1:]
                logits_shifted = logits[..., :-1, :]

                oh_labels_shifted = jax.nn.one_hot(
                    labels_shifted, num_classes=self.vocab_size
                )

                # label is of shape (16, 31, 256) and logits is of shape (4, 16, 31, 64)
                loss = cross_entropy_loss(oh_labels_shifted, logits_shifted)

                return logits, loss


def model_getter(
    model_size,
    config_path="conf/model_config.yaml",
    return_cfg=False,
    dtype=jnp.float32,
) -> nn.Module:
    """Loads model configuration from YAML files
    and returns models

    Args:
        model_size (str): model name
            This is checked against all top-level model names in the
            YAML file (defaults to 'conf/model_config.yaml')
    """

    configs = OmegaConf.load(config_path)
    assert model_size in list(configs.keys()), "Invalid model name provided"
    assert dtype in [jnp.float16, jnp.bfloat16, jnp.float32], "Invalid dtype provided"
    if return_cfg:
        return Transformer(**configs[model_size], dtype=dtype), configs[model_size]
    else:
        return Transformer(**configs[model_size], dtype=dtype)
