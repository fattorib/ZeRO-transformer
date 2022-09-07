""" 
Replication of GPT2 transformers in Flax
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial


class MLPBlock(nn.Module):

    embedding_dim: int
    dimension_multiplier: int = 4
    dropout: float = 0.0
    N: int = None

    # Scale residual weights by 1/sqrt(N) where N is number of layers
    residual_init = jax.nn.initializers.normal(stddev=0.02 / jnp.sqrt(N))
    dense_init = jax.nn.initializers.normal(stddev=0.02)
    bias_init = jax.nn.initializers.zeros()

    # TODO: How does mixed precision work?

    @nn.compact
    def __call__(self, x: jnp.array, train: bool) -> jnp.array:
        dropout = partial(nn.Dropout, rate=self.dropout, deterministic=not train)
        x = nn.Dense(
            features=self.dimension_multiplier * self.embedding_dim,
            name="fc_in",
            kernel_init=self.dense_init,
            bias=self.bias_init,
        )(x)
        x = nn.gelu(x)
        out = nn.Dense(
            features=self.embedding_dim,
            name="fc_residual",
            kernel_init=self.residual_init,
            bias=self.bias_init,
        )(x)
        return dropout(out)


class CausalAttention(nn.Module):

    embedding_dim: int
    num_head: int
    block_size: int
    dropout: float = 0.0
    N: int = None

    # Scale residual weights by 1/sqrt(N) where N is number of layers
    residual_init = jax.nn.initializers.normal(stddev=0.02 / jnp.sqrt(N))
    dense_init = jax.nn.initializers.normal(stddev=0.02)
    bias_init = jax.nn.initializers.zeros()

    dtype = None

    @nn.compact
    def __call__(self, x: jnp.array, train: bool) -> jnp.array:
        dropout = partial(nn.Dropout, rate=self.dropout, deterministic=not train)

        B, T, C = x.size(0), x.size(1), x.size(2)

        key = (
            nn.Dense(
                name="key_proj",
                features=self.embedding_dim,
                kernel_init=self.dense_init,
                bias=self.bias_init,
            )(x)
            .reshape(B, T, self.num_head, self.embedding_dim // self.num_head)
            .transpose(1, 2)
        )  # Shape is (B, nh, T, h_dim)

        value = (
            nn.Dense(
                name="value_proj",
                features=self.embedding_dim,
                kernel_init=self.dense_init,
                bias=self.bias_init,
            )(x)
            .reshape(B, T, self.num_head, self.embedding_dim // self.num_head)
            .transpose(1, 2)
        )  # Shape is (B, nh, T, h_dim)

        query = (
            nn.Dense(
                name="query_proj",
                features=self.embedding_dim,
                kernel_init=self.dense_init,
                bias=self.bias_init,
            )(x)
            .reshape(B, T, self.num_head, self.embedding_dim // self.num_head)
            .transpose(1, 2)
        )  # Shape is (B, nh, T, h_dim)

        # get raw attention scores
        attn_full = (query @ key.transpose(-2, -1)) / jnp.sqrt(
            key.size(-1)
        )  # Shape is (B, nh, T, T)

        mask = jnp.tril(jnp.ones(T, T, dtype=jnp.int8)).view(1, 1, T, T)
        masked_attn = jnp.where(mask, attn_full, big_neg=jnp.finfo(self.dtype).min)

        attn_scores = nn.softmax(masked_attn, axis=-1)
        attn_out = (attn_scores @ value).tranpose(1, 2)  # Shape is (B, T, nh, h_dim)

        attn_out = attn_out.reshape(B, T, C)
        out = nn.Dense(
            name="residual_out",
            features=self.embedding_dim,
            kernel_init=self.residual_init,
            bias=self.bias_init,
        )(attn_out)

        return dropout(out)
