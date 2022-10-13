""" 
Replication of GPT2 transformers in Flax
"""
import math
from functools import partial
from typing import Any, List, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
from einops import rearrange
from omegaconf import OmegaConf

from src.utils.losses import cross_entropy_loss


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


class MLPBlock(nn.Module):
    """Standard MLP Block"""

    embedding_dim: int
    dimension_multiplier: int = 4
    dropout: float = 0.0
    N: int = None

    @nn.compact
    def __call__(self, x: jnp.array, train: bool) -> jnp.array:
        dropout = partial(nn.Dropout, rate=self.dropout, deterministic=not train)
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
            kernel_init=initializers.normal(stddev=(0.02 / jnp.sqrt(2 * self.N))),
            bias_init=initializers.zeros,
        )(x)
        self.sow("intermediates", "mlp_out", x)
        return dropout()(out)


class CausalAttention(nn.Module):
    """Standard causal multi-headed attention

    Supports ALiBi attention biasing from
    `Train Short, Test Long: Attention with Linear Biases Enables Input
    Length Extrapolation <https://ofir.io/train_short_test_long.pdf>`

    """

    embedding_dim: int
    num_head: int
    block_size: int
    dropout: float = 0.0
    N: int = None
    alibi_attn: bool = False

    dtype = None

    def setup(self):
        self.slopes = jnp.array(get_slopes(self.num_head))

    @nn.compact
    def __call__(
        self,
        x: jnp.array,
        train: bool,
        alibi_mask: jnp.array = None,
        use_cache: bool = False,
        layer_past: Tuple[jnp.array, jnp.array] = None,
        pad_mask: jnp.array = None,
    ) -> Tuple[jnp.array, jnp.array]:
        dropout = partial(nn.Dropout, rate=self.dropout, deterministic=not train)
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

        present = None
        if use_cache:
            if layer_past is not None:
                past_keys, past_values = layer_past  # (1, nh, T, h_dim)
                # get shape here, we only keep the past block_size values so lax.scan is happy that we are passing stuff with a fixed size over
                key = jnp.concatenate((past_keys, key), axis=-2)[
                    :, :, -self.block_size :, :
                ]
                value = jnp.concatenate((past_values, value), axis=-2)[
                    :, :, -self.block_size :, :
                ]

            present = jnp.stack((key, value))

        # get raw attention scores
        attn_full = (query @ key.transpose(0, 1, 3, 2)) / jnp.sqrt(
            key.shape[-1]
        )  # Shape is (B, nh, sq, sk)

        if self.alibi_attn:

            seq_len_k, seq_len_q = key.shape[-2], query.shape[-2]

            if alibi_mask is None:

                a = -jnp.tril(
                    jnp.tile(
                        jnp.arange(seq_len_k).reshape(seq_len_k, 1), (1, seq_len_k)
                    )
                    + jnp.arange(0, -seq_len_k, step=-1)
                )

                a = a * (self.slopes.reshape(self.slopes.shape[0], 1, 1))

                alibi_mask = a[:, seq_len_k - 1, :].reshape(a.shape[0], 1, a.shape[2])

                attn_full = attn_full + alibi_mask

        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.int8)).reshape(1, 1, T, T)
        if pad_mask is None:
            masked_attn = jnp.where(mask, attn_full, jnp.finfo(self.dtype).min)

        else:
            combined_mask = mask * pad_mask
            masked_attn = jnp.where(combined_mask, attn_full, jnp.finfo(self.dtype).min)

        attn_scores = nn.softmax(masked_attn, axis=-1)
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
        )(attn_out)

        self.sow("intermediates", "attn_out", out)

        return dropout()(out), present


class TransformerBlock(nn.Module):
    """One full transformer block"""

    embedding_dim: int
    num_head: int
    block_size: int
    residual_dropout: float = 0.0
    N: int = None
    dtype: Any = jnp.float32
    fused_residuals: bool = False
    alibi_attn: bool = False
    use_static_sgu: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.array,
        train: bool = False,
        use_cache: bool = False,
        layer_past: Tuple[jnp.array, jnp.array] = None,
        pad_mask: jnp.array = None,
    ) -> Tuple[jnp.array, jnp.array]:

        if self.fused_residuals:
            norm = nn.LayerNorm()
            attn_out = CausalAttention(
                self.embedding_dim,
                self.num_head,
                self.block_size,
                self.residual_dropout,
                self.N,
                self.alibi_attn,
            )(norm(x), train, None, use_cache, layer_past, pad_mask)
            return (
                x
                + attn_out[0]
                + MLPBlock(self.embedding_dim, dropout=self.residual_dropout, N=self.N)(
                    norm(x), train
                ),
                attn_out[1],
            )
        else:
            attn_out = CausalAttention(
                self.embedding_dim,
                self.num_head,
                self.block_size,
                self.residual_dropout,
                self.N,
                self.alibi_attn,
            )(nn.LayerNorm()(x), train, use_cache, layer_past, pad_mask)
            x = x + attn_out[0]
            x = x + MLPBlock(
                self.embedding_dim, dropout=self.residual_dropout, N=self.N
            )(nn.LayerNorm()(x), train)
            return x, attn_out[1]


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
    fused_residuals: bool = False
    alibi_attn: bool = False

    def generate(
        self,
        variables: flax.core.frozen_dict.FrozenDict,
        context: jnp.array,
        max_length: int = 30,
        temperature: float = 1.0,
        sample: bool = False,
        sample_rng: jax.random.PRNGKey = None,
    ) -> jnp.array:
        """Performs basic text generation. Supports temperature-based sampling

        Args:
            context (list): tokenized text to continue
            max_length (int): The maximum length of tokens to generate (sum of context tokens + *generated tokens*)
            sample (bool): Boolean whether to sample from logits distribution
            model (flax.linen.Module): Flax module structure
            variables (flax.core.frozen_dict.FrozenDict): Serialized model variables
            sample_rng (jax.random.PRNGKey): RNG key used if sampling from distribution

        Returns:
            jnp.array: Generated text, must be detokenized
        """

        context = jnp.array(context, dtype=jnp.int32)

        x = context.reshape(1, -1)

        num_generation_steps = max_length - x.shape[1]

        if x.shape[1] > self.block_size:
            x_cond = x[:, -self.block_size :]
        else:
            x_cond = x

        layer_past = None
        for _ in range(num_generation_steps):

            if sample_rng is not None:
                sample_rng, rng = jax.random.split(sample_rng, 2)
            logits, layer_past = self.apply(
                variables, x_cond, use_cache=True, past_states=layer_past
            )
            logits = logits[:, -1, :] / temperature

            probs = jax.nn.softmax(logits, axis=-1)

            if not sample:
                x_cond = jnp.array(jnp.argmax(probs), dtype=jnp.int32).reshape(1, -1)
                x = jnp.concatenate((x, x_cond), axis=1)

            else:
                assert (
                    sample_rng is not None
                ), "Must provide rng key when sampling from logit distribution"
                sample = jax.random.categorical(rng, logits=logits)
                x_cond = jnp.array(sample, dtype=jnp.int32).reshape(1, -1)
                x = jnp.concatenate((x, x_cond), axis=1)

        return jnp.squeeze(x)

    @nn.compact
    def __call__(
        self,
        x: jnp.array,
        labels: jnp.array = None,
        train: bool = False,
        use_cache: bool = False,
        past_states: Tuple[jnp.array, jnp.array] = None,
        pad_mask: jnp.array = None,
    ) -> Union[jnp.array, Tuple[jnp.array, jnp.array]]:

        B, T = x.shape[0:2]

        embed = nn.Embed(
            name="wte",
            num_embeddings=self.vocab_size,
            features=self.embedding_dim,
            embedding_init=initializers.normal(stddev=0.02),
        )

        wte = embed(x)

        out = wte
        if not self.alibi_attn:
            wpe = nn.Embed(
                name="wpe",
                num_embeddings=self.block_size,
                features=self.embedding_dim,
                embedding_init=initializers.normal(stddev=0.02),
            )(jnp.ones((B, T), dtype=jnp.uint8))

            out += wpe

        present_states = []
        if not use_cache:
            past_states = [None] * self.N

        if past_states is None:
            past_states = [None] * self.N

        for i, past_state in zip(range(self.N), past_states):

            out, layer_past = TransformerBlock(
                self.embedding_dim,
                self.num_head,
                self.block_size,
                self.dropout,
                self.N,
                self.fused_residuals,
                self.alibi_attn,
            )(out, train, use_cache, past_state, pad_mask)

            present_states.append(layer_past)

        out = nn.LayerNorm()(out)

        logits = embed.attend(out)

        if labels is None:
            if use_cache:
                return logits, present_states
            else:
                return logits
        else:
            labels_shifted = labels[..., 1:].reshape(-1)
            logits_shifted = logits[..., :-1, :].reshape(-1, logits.shape[-1])

            oh_labels_shifted = jax.nn.one_hot(
                labels_shifted, num_classes=self.vocab_size
            )
            loss = cross_entropy_loss(oh_labels_shifted, logits_shifted)

            return logits, loss


def model_getter(
    model_size, config_path="conf/model_config.yaml", return_cfg=False
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

    if return_cfg:
        return Transformer(**configs[model_size]), configs[model_size]
    else:
        return Transformer(**configs[model_size])
