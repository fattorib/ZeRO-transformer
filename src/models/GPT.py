""" 
Replication of GPT2 transformers in Flax
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Any
import jax.nn.initializers as initializers
from omegaconf import OmegaConf


class MLPBlock(nn.Module):
    """Standard MLP Block"""

    embedding_dim: int
    dimension_multiplier: int = 4
    dropout: float = 0.0
    N: int = None

    # TODO: How does mixed precision work?
    @nn.compact
    def __call__(self, x: jnp.array, train: bool) -> jnp.array:
        dropout = partial(
            nn.Dropout, rate=self.dropout, deterministic=not train
        )
        x = nn.Dense(
            features=self.dimension_multiplier * self.embedding_dim,
            name="fc_in",
            kernel_init=initializers.normal(stddev=0.02),
            bias_init=initializers.zeros,
        )(x)
        x = nn.gelu(x)
        out = nn.Dense(
            features=self.embedding_dim,
            name="fc_residual",
            kernel_init=initializers.normal(stddev=0.02 / jnp.sqrt(self.N)),
            bias_init=initializers.zeros,
        )(x)
        return dropout()(out)


class CausalAttention(nn.Module):
    """Standard causal multi-headed attention"""

    embedding_dim: int
    num_head: int
    block_size: int
    dropout: float = 0.0
    N: int = None

    dtype = None

    @nn.compact
    def __call__(self, x: jnp.array, train: bool) -> jnp.array:
        dropout = partial(
            nn.Dropout, rate=self.dropout, deterministic=not train
        )

        B, T, C = x.shape[:3]

        # Shape is (B, nh, T, h_dim)
        key = (
            nn.Dense(
                name="key_proj",
                features=self.embedding_dim,
                kernel_init=initializers.normal(stddev=0.02),
                bias_init=initializers.zeros,
            )(x)
            .reshape(B, T, self.num_head, self.embedding_dim // self.num_head)
            .transpose(0, 2, 1, 3)
        )

        # Shape is (B, nh, T, h_dim)
        value = (
            nn.Dense(
                name="value_proj",
                features=self.embedding_dim,
                kernel_init=initializers.normal(stddev=0.02),
                bias_init=initializers.zeros,
            )(x)
            .reshape(B, T, self.num_head, self.embedding_dim // self.num_head)
            .transpose(0, 2, 1, 3)
        )

        # Shape is (B, nh, T, h_dim)
        query = (
            nn.Dense(
                name="query_proj",
                features=self.embedding_dim,
                kernel_init=initializers.normal(stddev=0.02),
                bias_init=initializers.zeros,
            )(x)
            .reshape(B, T, self.num_head, self.embedding_dim // self.num_head)
            .transpose(0, 2, 1, 3)
        )

        # get raw attention scores
        attn_full = (query @ key.transpose(0, 1, 3, 2)) / jnp.sqrt(
            key.shape[-1]
        )  # Shape is (B, nh, T, T)

        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.int8)).reshape(1, 1, T, T)
        masked_attn = jnp.where(mask, attn_full, jnp.finfo(self.dtype).min)

        attn_scores = nn.softmax(masked_attn, axis=-1)
        attn_out = (attn_scores @ value).transpose(
            0, 2, 1, 3
        )  # Shape is (B, T, nh, h_dim)

        attn_out = attn_out.reshape(B, T, C)
        out = nn.Dense(
            name="residual_out",
            features=self.embedding_dim,
            kernel_init=jax.nn.initializers.normal(
                stddev=0.02 / jnp.sqrt(self.N)
            ),
            bias_init=initializers.zeros,
        )(attn_out)

        return dropout()(out)


class TransformerBlock(nn.Module):
    """One full transformer block"""

    embedding_dim: int
    num_head: int
    block_size: int
    residual_dropout: float = 0.0
    N: int = None
    dtype: Any = None
    fused_residuals: bool = False

    @nn.compact
    def __call__(self, x: jnp.array, train: bool = False) -> jnp.array:

        if self.fused_residuals:
            norm = nn.LayerNorm()
            return CausalAttention(
                self.embedding_dim,
                self.num_head,
                self.block_size,
                self.residual_dropout,
                self.N,
            )(norm(x), train) + MLPBlock(
                self.embedding_dim, dropout=self.residual_dropout, N=self.N
            )(
                norm(x), train
            )
        else:
            x = x + CausalAttention(
                self.embedding_dim,
                self.num_head,
                self.block_size,
                self.residual_dropout,
                self.N,
            )(nn.LayerNorm()(x), train)
            x = x + MLPBlock(
                self.embedding_dim, dropout=self.residual_dropout, N=self.N
            )(nn.LayerNorm()(x), train)
            return x


class Transformer(nn.Module):
    """
    Full transformer
    """

    embedding_dim: int
    vocab_size: int
    num_head: int
    block_size: int
    dropout: float = 0.0
    N: int = None
    dtype: Any = None
    fused_residuals: bool = False

    @nn.compact
    def __call__(self, x: jnp.array, train: bool = False) -> jnp.array:

        B, T = x.shape[0:2]

        embed = nn.Embed(
            name="wte",
            num_embeddings=self.vocab_size,
            features=self.embedding_dim,
            embedding_init=initializers.normal(stddev=0.02),
        )

        wte = embed(x)

        wpe = nn.Embed(
            name="wpe",
            num_embeddings=self.block_size,
            features=self.embedding_dim,
            embedding_init=initializers.normal(stddev=0.02),
        )(jnp.ones((B, T), dtype=jnp.uint8))

        x = wte + wpe

        for _ in range(self.N):

            x = TransformerBlock(
                self.embedding_dim,
                self.num_head,
                self.block_size,
                self.dropout,
                self.N,
                self.fused_residuals,
            )(x, train)

        x = nn.LayerNorm()(x)

        logits = embed.attend(x)

        return logits


def model_getter(model_size, config_path="conf/model_config.yaml") -> nn.Module:
    """Loads model configuration from YAML files
    and returns models

    Args:
        model_size (str): model name. This is checked against all top-level model names in the YAML file (defaults to 'conf/model_config.yaml')
    """

    configs = OmegaConf.load(config_path)
    assert model_size in list(configs.keys()), "Invalid model name provided"

    return Transformer(**configs.model_size)
