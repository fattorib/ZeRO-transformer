import math
from functools import partial
from typing import Any, List, Tuple

import flax.linen as nn
import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
from flax.linen import partitioning as nn_partitioning
from jax.experimental import PartitionSpec


def shard_noop(x, spec):
    return nn_partitioning.with_sharding_constraint(x, spec)


def get_slopes(n: int) -> List:
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

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

    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.array, train: bool) -> jnp.array:
        dropout = partial(nn.Dropout, rate=self.dropout, deterministic=not train)
        x = nn.Dense(
            features=self.dimension_multiplier * self.embedding_dim,
            name="fc_in",
            kernel_init=initializers.normal(stddev=0.02),
            bias_init=initializers.zeros,
            dtype=self.dtype,
        )(x)
        x = nn.gelu(x)
        out = nn.Dense(
            features=self.embedding_dim,
            name="fc_residual",
            kernel_init=initializers.normal(stddev=(0.02 / jnp.sqrt(2 * self.N))),
            bias_init=initializers.zeros,
            dtype=self.dtype,
        )(x)
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

    def setup(self):
        self.slopes = jnp.array(get_slopes(self.num_head))

        self.alibi_mask = create_mask(self.block_size, self.slopes)

    @nn.compact
    def __call__(
        self,
        x: jnp.array,
        train: bool,
    ) -> jnp.array:
        dropout = partial(nn.Dropout, rate=self.dropout, deterministic=not train)
        B, T, C = x.shape[:3]

        # Shape is (B, nh, T, h_dim)
        key = (
            nn.Dense(
                name="key_proj",
                features=self.embedding_dim,
                kernel_init=initializers.normal(stddev=0.02),
                bias_init=initializers.zeros,
                dtype=self.dtype,
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
                dtype=self.dtype,
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
                dtype=self.dtype,
            )(x)
            .reshape(B, T, self.num_head, self.embedding_dim // self.num_head)
            .transpose(0, 2, 1, 3)
        )

        # get raw attention scores
        attn_full = (query @ key.transpose(0, 1, 3, 2)) / jnp.sqrt(
            key.shape[-1]
        )  # Shape is (B, nh, sq, sk)

        if self.alibi_attn:
            # NOTE: We are fixing the ALiBi mask since this is for training, during inference or is seq_len changes this will cause issues
            attn_full = attn_full + self.alibi_mask[:, :T, :T]

        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.int8)).reshape(1, 1, T, T)

        masked_attn = jnp.where(mask, attn_full, jnp.finfo(self.dtype).min)

        attn_scores = nn.softmax(masked_attn, axis=-1)
        attn_scores = dropout()(attn_scores)
        attn_out = (attn_scores @ value).transpose(
            0, 2, 1, 3
        )  # Shape is (B, T, nh, h_dim)

        attn_out = attn_out.reshape(B, T, C)
        out = nn.Dense(
            name="residual_out",
            features=self.embedding_dim,
            kernel_init=jax.nn.initializers.normal(
                stddev=(0.02 / jnp.sqrt(2 * self.N))
            ),
            bias_init=initializers.zeros,
            dtype=self.dtype,
        )(attn_out)

        return dropout()(out)


# TODO: Work in progress code, not currently used in src/models/GPT.py
class ShardedCausalAttention(nn.Module):
    """Standard causal multi-headed attention modified to support sharding

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
    model_shards: int = None

    def setup(self):
        self.slopes = jnp.array(get_slopes(self.num_head))

        self.alibi_mask = create_mask(self.block_size, self.slopes)
        self.mask = jnp.tril(
            jnp.ones((self.block_size, self.block_size), dtype=jnp.int8)
        ).reshape(1, 1, self.block_size, self.block_size)

        assert self.embedding_dim % self.num_head == 0
        assert self.embedding_dim % self.model_shards == 0

        self.heads_per_shard = self.num_head // self.model_shards
        self.dim_per_shard = self.embedding_dim // self.model_shards

    @nn.compact
    def __call__(
        self,
        x: jnp.array,
        train: bool,
    ) -> jnp.array:
        raise NotImplementedError
    #     dropout = partial(nn.Dropout, rate=self.dropout, deterministic=not train)
    #     seq_len = x.shape[1]

    #     # Shape is (bs, seq_len, d_embed)
    #     key = nn.Dense(
    #         name="key_proj",
    #         features=self.embedding_dim,
    #         kernel_init=initializers.normal(stddev=0.02),
    #         use_bias=False,
    #         dtype=self.dtype,
    #     )(x)

    #     # Shape is (bs, seq_len, d_embed)
    #     value = nn.Dense(
    #         name="value_proj",
    #         features=self.embedding_dim,
    #         kernel_init=initializers.normal(stddev=0.02),
    #         use_bias=False,
    #         dtype=self.dtype,
    #     )(x)

    #     # Shape is (bs, seq_len, d_embed)
    #     query = nn.Dense(
    #         name="query_proj",
    #         features=self.embedding_dim,
    #         kernel_init=initializers.normal(stddev=0.02),
    #         use_bias=False,
    #         dtype=self.dtype,
    #     )(x)

    #     query = shard_noop(
    #         query, PartitionSpec("dp", None, "mp")
    #     )  # ensure sharding along batch dimension for dp and head dimension for mp
    #     key = shard_noop(
    #         key, PartitionSpec("dp", None, "mp")
    #     )  # ensure sharding along batch dimension for dp and head dimension for mp
    #     value = shard_noop(
    #         value, PartitionSpec("dp", None, "mp")
    #     )  # ensure sharding along batch dimension for dp and head dimension for mp

    #     query = jnp.reshape(query, query.shape[:-1] + (self.model_shards, -1))
    #     key = jnp.reshape(key, key.shape[:-1] + (self.model_shards, -1))
    #     value = jnp.reshape(value, value.shape[:-1] + (self.model_shards, -1))

    #     query = shard_noop(query, PartitionSpec("dp", None, "mp", None))
    #     key = shard_noop(key, PartitionSpec("dp", None, "mp", None))
    #     value = shard_noop(value, PartitionSpec("dp", None, "mp", None))

    #     attn_full = jnp.einsum(
    #         "bthd,bThd->bhtT", query, key
    #     )  # Shape is (bs, h_per_shard, seq_len, seq_len)

    #     attn_full = shard_noop(attn_full, PartitionSpec("dp", "mp", None, None))

    #     attn_full = attn_full / key.shape[-1]

    #     if self.alibi_attn:
    #         attn_full = attn_full + self.alibi_mask[:, :seq_len, :seq_len]

    #     masked_attn = jnp.where(self.mask, attn_full, jnp.finfo(self.dtype).min)

    #     attn_scores = nn.softmax(masked_attn, axis=-1)
    #     attn_scores = dropout()(attn_scores)
    #     attn_out = jnp.einsum("bhtT,bThd->bthd", attn_scores, value)
    #     attn_out = shard_noop(attn_out, PartitionSpec("dp", None, "mp", None))
    #     attn_out = attn_out.reshape(
    #         attn_out.shape[:2] + (self.model_shards, self.heads_per_shard, -1)
    #     )  # Shape is (bs, seq_len, model_shards, h_per_shard, h_dim)
    #     attn_out = shard_noop(attn_out, PartitionSpec("dp", None, "mp", None, None))

    #     attn_out = attn_out.reshape(
    #         attn_out.shape[:2] + (self.model_shards, -1)
    #     )  # # Shape is (bs, seq_len, model_shards, d_embed)

    #     attn_out = shard_noop(attn_out, PartitionSpec("dp", None, "mp", None))

    #     out = nn.Dense(
    #         name="residual_out",
    #         features=self.embedding_dim,
    #         kernel_init=jax.nn.initializers.normal(
    #             stddev=(0.02 / jnp.sqrt(2 * self.N))
    #         ),
    #         use_bias=False,
    #         dtype=self.dtype,
    #     )(attn_out)

    #     return dropout()(out)
