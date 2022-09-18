""" 
Unittests for all experimental features: QK trick, SGU splitting, etc
"""
import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random

from src.models.GPT import (CausalAttention, MLPBlock, Transformer,
                            TransformerBlock)
from src.utils.losses import cross_entropy_loss


class TestTransformerBlock(unittest.TestCase):
    def setUp(self) -> None:
        self.init_rng, self.rng = random.split(random.PRNGKey(0))
        self.block_size = 512

    def tearDown(self) -> None:
        pass

    def test_block_fwd_sgu(self):

        block = TransformerBlock(
            embedding_dim=128,
            num_head=8,
            block_size=512,
            residual_dropout=0.1,
            N=6,
            dtype=None,
            fused_residuals=True,
            use_static_sgu=True,
        )
        batch_cts = random.normal(self.rng, shape=(1, 512, 128))
        params = block.init(self.init_rng, batch_cts, False)

        out = block.apply(
            {"params": params["params"]},
            batch_cts,
            train=True,
            rngs={"dropout": self.rng},
        )
        self.assertEqual(out.shape, batch_cts.shape)

        block = TransformerBlock(
            embedding_dim=128,
            num_head=8,
            block_size=512,
            residual_dropout=0.1,
            N=6,
            dtype=None,
            fused_residuals=True,
            alibi_attn=True,
            use_static_sgu=True,
        )
        batch_cts = random.normal(self.rng, shape=(1, 512, 128))
        params = block.init(self.init_rng, batch_cts, False)

        out = block.apply(
            {"params": params["params"]},
            batch_cts,
            train=True,
            rngs={"dropout": self.rng},
        )
        self.assertEqual(out.shape, batch_cts.shape)


class TestGPT(unittest.TestCase):
    def setUp(self) -> None:
        self.init_rng, self.rng = random.split(random.PRNGKey(0))
        self.vocab_size = 256
        self.block_size = 512

    def tearDown(self) -> None:
        pass

    def test_gpt_fwd_static_sgu(self):
        # Ensure the QK trick shapes are correct

        block = Transformer(
            embedding_dim=128,
            vocab_size=self.vocab_size,
            num_head=8,
            block_size=512,
            dropout=0.1,
            N=6,
            dtype=None,
            fused_residuals=True,
            head_qk_trick=False,
            use_static_sgu=True,
        )
        batch_tok = random.randint(self.rng, shape=(1, 512), maxval=256, minval=0)
        params = block.init(self.init_rng, batch_tok, None, False)

        out = block.apply(
            {"params": params["params"]},
            batch_tok,
            train=True,
            rngs={"dropout": self.rng},
        )
        self.assertEqual((1, self.block_size, self.vocab_size), out.shape)

        block = Transformer(
            embedding_dim=128,
            vocab_size=self.vocab_size,
            num_head=8,
            block_size=512,
            dropout=0.1,
            N=6,
            dtype=None,
            fused_residuals=True,
            alibi_attn=True,
            use_static_sgu=True,
        )
        batch_tok = random.randint(self.rng, shape=(1, 512), maxval=256, minval=0)
        params = block.init(self.init_rng, batch_tok, None, False)

        out = block.apply(
            {"params": params["params"]},
            batch_tok,
            train=True,
            rngs={"dropout": self.rng},
        )
        self.assertEqual((1, self.block_size, self.vocab_size), out.shape)

    def test_gpt_fwd_qk_trick(self):
        # Ensure the QK trick shapes are correct

        block = Transformer(
            embedding_dim=128,
            vocab_size=self.vocab_size,
            num_head=8,
            block_size=512,
            dropout=0.1,
            N=6,
            dtype=None,
            fused_residuals=False,
            head_qk_trick=True,
        )
        batch_tok = random.randint(self.rng, shape=(1, 512), maxval=256, minval=0)
        params = block.init(self.init_rng, batch_tok, None, False)

        out = block.apply(
            {"params": params["params"]},
            batch_tok,
            train=True,
            rngs={"dropout": self.rng},
        )
        self.assertEqual((1, self.block_size, self.vocab_size), out.shape)

        block = Transformer(
            embedding_dim=128,
            vocab_size=self.vocab_size,
            num_head=8,
            block_size=512,
            dropout=0.1,
            N=6,
            dtype=None,
            fused_residuals=False,
            alibi_attn=True,
            head_qk_trick=True,
        )
        batch_tok = random.randint(self.rng, shape=(1, 512), maxval=256, minval=0)
        params = block.init(self.init_rng, batch_tok, None, False)

        out = block.apply(
            {"params": params["params"]},
            batch_tok,
            train=True,
            rngs={"dropout": self.rng},
        )
        self.assertEqual((1, self.block_size, self.vocab_size), out.shape)