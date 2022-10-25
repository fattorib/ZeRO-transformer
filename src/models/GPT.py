""" 
Replication of GPT2 transformers in Flax
"""
from typing import Any, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
from omegaconf import OmegaConf

from src.models.layers import TransformerBlock
from src.utils.losses import cross_entropy_loss


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
    qk_norm: bool = False

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
            dtype=self.dtype,
        )

        wte = embed(x)

        out = wte
        if not self.alibi_attn:
            wpe = nn.Embed(
                name="wpe",
                num_embeddings=self.block_size,
                features=self.embedding_dim,
                embedding_init=initializers.normal(stddev=0.02),
                dtype=self.dtype,
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
                self.dtype,
                self.fused_residuals,
                self.alibi_attn,
                self.qk_norm,
            )(out, train, use_cache, past_state, pad_mask)

            present_states.append(layer_past)

        out = nn.LayerNorm(dtype=self.dtype)(out)

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
