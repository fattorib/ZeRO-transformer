import unittest

import jax
import jax.numpy as jnp
import jax.random as random

from src.models.GPT import Transformer, TransformerBlock
from src.models.layers import CausalAttention, MLPBlock
from src.utils.losses import cross_entropy_loss


class TestMLP(unittest.TestCase):
    def setUp(self) -> None:
        self.init_rng, self.rng = random.split(random.PRNGKey(0))
        self.block_size = 512

    def tearDown(self) -> None:
        pass

    def test_MLP_create(self):

        mlp = MLPBlock(embedding_dim=128, dimension_multiplier=4, dropout=0.1, N=10)
        batch_cts = random.normal(self.rng, shape=(1, 512, 128))
        params = mlp.init(self.init_rng, batch_cts, False)

    def test_MLP_fwd(self):
        mlp = MLPBlock(embedding_dim=128, dimension_multiplier=4, dropout=0.1, N=6)
        batch_cts = random.normal(self.rng, shape=(1, 512, 128))
        params = mlp.init(self.init_rng, batch_cts, False)

        out = mlp.apply(
            {"params": params["params"]},
            batch_cts,
            train=True,
            rngs={"dropout": self.rng},
        )

        out_nodrop = mlp.apply(
            {"params": params["params"]},
            batch_cts,
            train=False,
            rngs={"dropout": self.rng},
        )
        self.assertEqual(
            (out.shape, out_nodrop.shape), (batch_cts.shape, batch_cts.shape)
        )


class TestAttn(unittest.TestCase):
    def setUp(self) -> None:
        self.init_rng, self.rng = random.split(random.PRNGKey(0))
        self.block_size = 512

    def tearDown(self) -> None:
        pass

    def test_attn_create(self):

        attn = CausalAttention(
            embedding_dim=128, num_head=8, block_size=512, dropout=0.1, N=6
        )
        batch_cts = random.normal(self.rng, shape=(1, 512, 128))
        params = attn.init(self.init_rng, batch_cts, False)

    def test_attn_fwd(self):
        attn = CausalAttention(
            embedding_dim=128, num_head=8, block_size=512, dropout=0.1, N=6
        )
        batch_cts = random.normal(self.rng, shape=(1, 512, 128))
        params = attn.init(self.init_rng, batch_cts, False)

        out = attn.apply(
            {"params": params["params"]},
            batch_cts,
            train=True,
            rngs={"dropout": self.rng},
        )

        out_nodrop = attn.apply(
            {"params": params["params"]},
            batch_cts,
            train=False,
            rngs={"dropout": self.rng},
        )
        self.assertEqual(
            (out.shape, out_nodrop.shape), (batch_cts.shape, batch_cts.shape)
        )

    def test_attn_fwd_ALiBi(self):
        attn = CausalAttention(
            embedding_dim=128,
            num_head=8,
            block_size=512,
            dropout=0.1,
            N=6,
            alibi_attn=True,
        )
        batch_cts = random.normal(self.rng, shape=(1, 512, 128))
        params = attn.init(self.init_rng, batch_cts, False)

        out = attn.apply(
            {"params": params["params"]},
            batch_cts,
            train=True,
            rngs={"dropout": self.rng},
        )

        out_nodrop = attn.apply(
            {"params": params["params"]},
            batch_cts,
            train=False,
            rngs={"dropout": self.rng},
        )
        self.assertEqual(
            (out.shape, out_nodrop.shape), (batch_cts.shape, batch_cts.shape)
        )


class TestTransformerBlock(unittest.TestCase):
    def setUp(self) -> None:
        self.init_rng, self.rng = random.split(random.PRNGKey(0))
        self.block_size = 512

    def tearDown(self) -> None:
        pass

    def test_block_create_standard(self):

        block = TransformerBlock(
            embedding_dim=128,
            num_head=8,
            block_size=512,
            residual_dropout=0.1,
            N=6,
            dtype=None,
        )
        batch_cts = random.normal(self.rng, shape=(1, 512, 128))
        params = block.init(self.init_rng, batch_cts, False)

    def test_block_fwd_standard(self):

        block = TransformerBlock(
            embedding_dim=128,
            num_head=8,
            block_size=512,
            residual_dropout=0.1,
            N=6,
            dtype=None,
            alibi_attn=True,
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


    def test_gpt_create_standard(self):

        block = Transformer(
            embedding_dim=128,
            vocab_size=256,
            num_head=8,
            block_size=512,
            dropout=0.1,
            N=6,
            dtype=None,
        )
        batch_tok = random.randint(self.rng, shape=(1, 512), maxval=256, minval=0)
        params = block.init(self.init_rng, batch_tok, None, False)

    def test_gpt_fwd_standard(self):

        block = Transformer(
            embedding_dim=128,
            vocab_size=self.vocab_size,
            num_head=8,
            block_size=512,
            dropout=0.1,
            N=6,
            dtype=None,
            alibi_attn=True,
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

    def test_gpt_fwd_fp16(self):

        block = Transformer(
            embedding_dim=128,
            vocab_size=self.vocab_size,
            num_head=8,
            block_size=512,
            dropout=0.1,
            N=6,
            dtype=jnp.float16,
            alibi_attn=True,
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

    def test_gpt_loss_standard(self):

        block = Transformer(
            embedding_dim=128,
            vocab_size=self.vocab_size,
            num_head=8,
            block_size=512,
            dropout=0.1,
            N=6,
            dtype=None,
            alibi_attn=True,
        )
        batch_tok = random.randint(self.rng, shape=(1, 512), maxval=256, minval=0)
        params = block.init(self.init_rng, batch_tok, None, False)

        logits, loss = block.apply(
            {"params": params["params"]},
            x=batch_tok,
            labels=batch_tok,
            train=True,
            rngs={"dropout": self.rng},
        )

        labels_shifted = batch_tok[..., 1:].reshape(-1)
        logits_shifted = logits[..., :-1, :].reshape(-1, logits.shape[-1])

        oh_labels_shifted = jax.nn.one_hot(labels_shifted, num_classes=self.vocab_size)

        loss_external = cross_entropy_loss(oh_labels_shifted, logits_shifted)

        self.assertEqual(loss, loss_external)
