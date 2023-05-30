import math
from functools import partial
from typing import Any, List

import flax.linen as nn
import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
from einops import rearrange
from src.models.replicated_utils import f_psum, g_psum


def get_slopes(n: int) -> List:
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def create_mask(seq_len_k, slopes):

    a = -jnp.tril(
        jnp.tile(jnp.arange(seq_len_k).reshape(seq_len_k, 1), (1, seq_len_k))
        + jnp.arange(0, -seq_len_k, step=-1)
    )

    a = a * (slopes.reshape(slopes.shape[0], 1, 1))

    alibi_mask = a[:, seq_len_k - 1, :].reshape(a.shape[0], 1, a.shape[2])

    return alibi_mask


class MLPBlock(nn.Module):
    """Standard MLP Block"""

    embedding_dim: int
    dimension_multiplier: int = 4
    dropout: float = 0.0
    N: int = None
    tp_comms: bool = True

    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.array, train: bool) -> jnp.array:
        dropout = partial(nn.Dropout, rate=self.dropout, deterministic=not train)

        if self.tp_comms:
            x = f_psum(x)

        x = nn.Dense(
            features=self.dimension_multiplier * self.embedding_dim,
            name="fc_in",
            kernel_init=initializers.normal(stddev=0.02),
            bias_init=initializers.zeros,
            dtype=self.dtype,
            use_bias=False,
        )(x)
        x = nn.gelu(x)
        out = nn.Dense(
            features=self.embedding_dim,
            name="fc_residual",
            kernel_init=initializers.normal(stddev=(0.02 / jnp.sqrt(2 * self.N))),
            bias_init=initializers.zeros,
            dtype=self.dtype,
            use_bias=False,
        )(x)
        if self.tp_comms:
            
            out = g_psum(out)
        return dropout()(out)


class CausalAttention(nn.Module):
    """Standard causal multi-headed attention

    Supports:
    - ALiBi attention biasing from
    `Train Short, Test Long: Attention with Linear Biases Enables Input
    Length Extrapolation <https://ofir.io/train_short_test_long.pdf>`

    """

    embedding_dim: int
    num_head: int
    block_size: int
    dropout: float = 0.0
    N: int = None
    alibi_attn: bool = False
    dtype: Any = jnp.float32
    tp_comms: bool = True

    def setup(self):
        self.slopes = jnp.array(get_slopes(self.num_head))
        self.mask = jnp.tril(
            jnp.ones((self.block_size, self.block_size), dtype=jnp.int8)
        ).reshape(1, 1, self.block_size, self.block_size)
        self.alibi_mask = create_mask(self.block_size, self.slopes)

    @nn.compact
    def __call__(
        self,
        x: jnp.array,
        train: bool,
    ) -> jnp.array:
        dropout = partial(nn.Dropout, rate=self.dropout, deterministic=not train)
        T, C = x.shape[-2:]

        if self.tp_comms:
            x = f_psum(x)

        key = nn.Dense(
            name="key_proj",
            features=self.embedding_dim,
            kernel_init=initializers.normal(stddev=0.02),
            bias_init=initializers.zeros,
            dtype=self.dtype,
            use_bias=False,
        )(x)

        value = nn.Dense(
            name="value_proj",
            features=self.embedding_dim,
            kernel_init=initializers.normal(stddev=0.02),
            bias_init=initializers.zeros,
            dtype=self.dtype,
            use_bias=False,
        )(x)

        query = nn.Dense(
            name="query_proj",
            features=self.embedding_dim,
            kernel_init=initializers.normal(stddev=0.02),
            bias_init=initializers.zeros,
            dtype=self.dtype,
            use_bias=False,
        )(x)

        key = rearrange(
            key, "b t (nh hd) -> b nh hd t", nh=self.num_head, hd=C // self.num_head
        )
        query = rearrange(
            query, "b t (nh hd) -> b nh t hd", nh=self.num_head, hd=C // self.num_head
        )
        value = rearrange(
            value, "b t (nh hd) -> b nh t hd", nh=self.num_head, hd=C // self.num_head
        )

        # get raw attention scores
        attn_full = (query @ key) / jnp.sqrt(key.shape[-1])  # Shape is (B, nh, sq, sk)

        if self.alibi_attn:
            # NOTE: We are fixing the ALiBi mask since this is for training, 
            # during inference or is seq_len changes this will cause issues
            attn_full = attn_full + self.alibi_mask

        masked_attn = jnp.where(
            self.mask, attn_full.astype(jnp.float32), jnp.finfo(jnp.float32).min
        )

        attn_scores = nn.softmax(masked_attn, axis=-1)
        attn_scores = dropout()(attn_scores)
        attn_out = attn_scores @ value
        attn_out = rearrange(attn_out, "b n t h -> b t (n h)")

        out = nn.Dense(
            name="residual_out",
            features=self.embedding_dim,
            kernel_init=jax.nn.initializers.normal(
                stddev=(0.02 / jnp.sqrt(2 * self.N))
            ),
            bias_init=initializers.zeros,
            dtype=self.dtype,
            use_bias=False,
        )(attn_out)

        if self.tp_comms:
            out = g_psum(out)

        return dropout()(out)
