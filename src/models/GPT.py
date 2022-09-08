""" 
Replication of GPT2 transformers in Flax
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Any, Callable
import jax.nn.initializers as initializers


class MLPBlock(nn.Module):
    """Standard MLP Block"""

    embedding_dim: int
    dimension_multiplier: int = 4
    dropout: float = 0.0
    N: int = None

    # TODO: How does mixed precision work?
    @nn.compact
    def __call__(self, x: jnp.array, train: bool) -> jnp.array:
        dropout = partial(nn.Dropout, rate=self.dropout, deterministic=not train)
        x = nn.Dense(
            features=self.dimension_multiplier * self.embedding_dim,
            name="fc_in",
            kernel_init=initializers.normal(stddev=0.02),
            bias_init=initializers.zeros
        )(x)
        x = nn.gelu(x)
        out = nn.Dense(
            features=self.embedding_dim,
            name="fc_residual",
            kernel_init=initializers.normal(stddev=0.02 / jnp.sqrt(self.N)),
            bias_init=initializers.zeros
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
        dropout = partial(nn.Dropout, rate=self.dropout, deterministic=not train)

        B, T, C = x.shape[:3]

        # Shape is (B, nh, T, h_dim)
        key = nn.Dense(
                name="key_proj",
                features=self.embedding_dim,
                kernel_init=initializers.normal(stddev=0.02),
                bias_init=initializers.zeros
            )(x).reshape(B, T, self.num_head, self.embedding_dim // self.num_head).transpose(0,2, 1,3)
          
        # Shape is (B, nh, T, h_dim)
        value = nn.Dense(
                name="value_proj",
                features=self.embedding_dim,
                kernel_init=initializers.normal(stddev=0.02),
                bias_init=initializers.zeros
            )(x).reshape(B, T, self.num_head, self.embedding_dim // self.num_head).transpose(0,2, 1,3)

        # Shape is (B, nh, T, h_dim)
        query = nn.Dense(
                name="query_proj",
                features=self.embedding_dim,
                kernel_init=initializers.normal(stddev=0.02),
                bias_init=initializers.zeros
            )(x).reshape(B, T, self.num_head, self.embedding_dim // self.num_head).transpose(0,2, 1,3)

        # get raw attention scores
        attn_full = (query @ key.transpose(0,1, 3, 2)) / jnp.sqrt(
            key.shape[-1]
        )  # Shape is (B, nh, T, T)

        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.int8)).reshape(1, 1, T, T)
        masked_attn = jnp.where(mask, attn_full, jnp.finfo(self.dtype).min)

        attn_scores = nn.softmax(masked_attn, axis=-1)
        attn_out = (attn_scores @ value).transpose(0,2, 1, 3)  # Shape is (B, T, nh, h_dim)

        attn_out = attn_out.reshape(B, T, C)
        out = nn.Dense(
            name="residual_out",
            features=self.embedding_dim,
            kernel_init=jax.nn.initializers.normal(stddev=0.02 / jnp.sqrt(self.N)),
            bias_init=initializers.zeros
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

    def setup(self):
        self.attn = CausalAttention(
            self.embedding_dim,
            self.num_head,
            self.block_size,
            self.residual_dropout,
            self.N,
        )
        self.mlp = MLPBlock(self.embedding_dim, dropout=self.residual_dropout, N=self.N)
        self.ln1 = nn.LayerNorm()

        if not self.fused_residuals:
            self.ln2 = nn.LayerNorm()

    def __call__(self, x: jnp.array, train: bool = False) -> jnp.array:

        if self.fused_residuals:
            return self.attn(self.ln1(x), train) + self.mlp(self.ln1(x), train)
        else:
            x = x + self.attn(self.ln1(x), train)
            x = x + self.mlp(self.ln2(x), train)
            return x


# class Transformer(nn.Module):
#     """
#     Full transformer
#     """

#     embedding_dim: int
#     vocab_size: int
#     num_head: int
#     block_size: int
#     residual_dropout: float = 0.0
#     N: int = None
#     dtype: Any = None
#     fused_residuals: bool = False

#     dense_init: Callable = jax.nn.initializers.normal(stddev=0.02)
#     bias_init: Callable = jax.nn.initializers.zeros

#     @nn.compact
#     def __call__(self, x: jnp.array, train: bool = False) -> jnp.array:

#         B, T = x.size(0), x.size(1)

#         wte = nn.Embed(
#             name="wte",
#             num_embeddings=self.vocab_size,
#             features=self.embedding_dim,
#             embedding_init=self.dense_init,
#         )(x)
#         wpe = nn.Embed(
#             name="wpe",
#             num_embeddings=self.block_size,
#             features=self.embedding_dim,
#             embedding_init=self.dense_init,
#         )(jnp.ones(B, T))

#         x = wte + wpe

#         for _ in range(self.N):

#             x = TransformerBlock(
#                 self.embedding_dim,
#                 self.num_head,
#                 self.block_size,
#                 self.residual_dropout,
#                 self.N,
#                 self.fused_residuals,
#             )(x, train)

#         x = nn.LayerNorm(x)

#         logits = wte.attend(x)

#         return logits
