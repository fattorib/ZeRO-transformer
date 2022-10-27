""" 
Holds experimental modules
"""
from functools import partial
from typing import Any

import flax.linen as nn
import jax.nn.initializers as initializers
import jax.numpy as jnp
from einops import rearrange
from jax import lax


class SGU(nn.Module):
    """Static SGU module"""

    block_size: int
    embedding_dim: int
    kernel_size: int

    def setup(self):
        k = jnp.triu(
            jnp.tril(
                jnp.ones((self.block_size, self.block_size)), -1 * self.kernel_size
            )
        )
        k = k / (k.cumsum(-2) + 1)
        self.kernel = k

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:

        _, T, _ = x.shape[:3]

        x = lax.stop_gradient(x)

        x = rearrange(x, "b n d -> b d n")
        x = x @ (self.kernel.T[:T, :T])
        x = rearrange(x, "b d n -> b n d")

        return x


class MLPBoom(nn.Module):
    """'Boom' layer as described in
    Single Headed Attention RNN: Stop Thinking With Your Head
    <https://arxiv.org/abs/1911.11423>
    """

    embedding_dim: int
    dimension_multiplier: int = 4
    dropout: float = 0.0
    N: int = None

    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.array, train: bool) -> jnp.array:
        dropout = partial(nn.Dropout, rate=self.dropout, deterministic=not train)
        x = nn.Dense(
            features=self.dimension_multiplier * self.embedding_dim,
            name="fc_in",
            kernel_init=initializers.normal(stddev=0.02 / jnp.sqrt(self.embedding_dim)),
            bias_init=initializers.zeros,
            dtype=self.dtype,
        )(x)
        x = nn.gelu(x)

        x = jnp.array(
            jnp.split(x, indices_or_sections=self.dimension_multiplier, axis=-1)
        ).transpose(1, 2, 3, 0)

        out = jnp.mean(x, axis=-1)

        self.sow("intermediates", "boom_out", x)
        return dropout()(out)
