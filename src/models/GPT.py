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
from einops import rearrange


class TransformerBlock(nn.Module):
    """One full transformer block"""

    embedding_dim: int
    num_head: int
    block_size: int
    residual_dropout: float = 0.0
    N: int = None
    dtype: Any = jnp.float32
    alibi_attn: bool = True

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
        )(nn.LayerNorm(dtype=self.dtype, use_bias=False)(x), train)
        x = x + attn_out
        x = x + MLPBlock(
            self.embedding_dim,
            dropout=self.residual_dropout,
            N=self.N,
            dtype=self.dtype,
        )(nn.LayerNorm(dtype=self.dtype, use_bias=False)(x), train)
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

    @nn.compact
    def __call__(
        self,
        x: jnp.array,
        labels: jnp.array = None,
        train: bool = False,
    ) -> Union[jnp.array, Tuple[jnp.array, jnp.array]]:
        T = x.shape[-1]
        embed = nn.Embed(
            name="wte",
            num_embeddings=self.vocab_size,
            features=self.embedding_dim,
            embedding_init=initializers.normal(stddev=0.02),
            dtype=self.dtype,
        )

        wte = embed(x)

        out = wte

        for _ in range(self.N):
            out = TransformerBlock(
                self.embedding_dim,
                self.num_head,
                self.block_size,
                self.dropout,
                self.N,
                self.dtype,
                self.alibi_attn,
            )(out, train)

        out = nn.LayerNorm(dtype=self.dtype, use_bias=False)(out)

        logits = embed.attend(out)

        if labels is None:
            return logits
        else:
            labels_shifted = labels[..., 1:]
            logits_shifted = logits[..., :-1, :]

            oh_labels_shifted = jax.nn.one_hot(
                labels_shifted, num_classes=self.vocab_size
            )
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
