import unittest
from chex import assert_equal 
import flax.linen as nn 
import jax.numpy as jnp 
import jax.random as random
from src.models.GPT import CausalAttention, MLPBlock, TransformerBlock, Transformer

class TestMLP(unittest.TestCase):

    def setUp(self) -> None:
        self.init_rng, self.rng = random.split(random.PRNGKey(0)) 
        self.block_size = 512

    def tearDown(self) -> None:
        pass 

    def test_MLP_create(self):

        mlp = MLPBlock(embedding_dim=128, dimension_multiplier=4, dropout=0.1, N = 10)
        batch_cts = random.normal(self.rng, shape = (1, 512, 128))
        params = mlp.init(self.init_rng, batch_cts, False)

    def test_MLP_fwd(self):
        mlp = MLPBlock(embedding_dim=128, dimension_multiplier=4, dropout=0.1, N = 6)
        batch_cts = random.normal(self.rng, shape = (1, 512, 128))
        params = mlp.init(self.init_rng, batch_cts, False)

        out = mlp.apply({'params':params['params']},batch_cts, train = True, rngs = {"dropout": self.rng})
        assert_equal(out.shape, batch_cts.shape)

        out = mlp.apply({'params':params['params']},batch_cts, train = False, rngs = {"dropout": self.rng})
        assert_equal(out.shape, batch_cts.shape)

class TestAttn(unittest.TestCase):

    def setUp(self) -> None:
        self.init_rng, self.rng = random.split(random.PRNGKey(0)) 
        self.block_size = 512

    def tearDown(self) -> None:
        pass 

    def test_attn_create(self):

        attn = CausalAttention(embedding_dim=128, num_head = 8, block_size = 512, dropout=0.1, N = 6)
        batch_cts = random.normal(self.rng, shape = (1, 512, 128))
        params = attn.init(self.init_rng, batch_cts, False)

    def test_attn_fwd(self):
        attn = CausalAttention(embedding_dim=128, num_head = 8, block_size = 512, dropout=0.1, N = 6)
        batch_cts = random.normal(self.rng, shape = (1, 512, 128))
        params = attn.init(self.init_rng, batch_cts, False)

        out = attn.apply({'params':params['params']},batch_cts, train = True, rngs = {"dropout": self.rng})
        assert_equal(out.shape, batch_cts.shape)

        out = attn.apply({'params':params['params']},batch_cts, train = False, rngs = {"dropout": self.rng})
        assert_equal(out.shape, batch_cts.shape)


class TestTransformerBlock(unittest.TestCase):

    def setUp(self) -> None:
        self.init_rng, self.rng = random.split(random.PRNGKey(0)) 
        self.block_size = 512

    def tearDown(self) -> None:
        pass

    def test_block_create_fused(self):

        block = TransformerBlock(embedding_dim=128, num_head = 8, block_size = 512, residual_dropout=0.1, N = 6, dtype = None, fused_residuals=True)
        batch_cts = random.normal(self.rng, shape = (1, 512, 128))
        params = block.init(self.init_rng, batch_cts, False)

        
    def test_block_fwd_fused(self):

        block = TransformerBlock(embedding_dim=128, num_head = 8, block_size = 512, residual_dropout=0.1, N = 6, dtype = None, fused_residuals=True)
        batch_cts = random.normal(self.rng, shape = (1, 512, 128))
        params = block.init(self.init_rng, batch_cts, False)
        
        out = block.apply({'params':params['params']},batch_cts, train = True, rngs = {"dropout": self.rng})
        assert_equal(out.shape, batch_cts.shape)

    def test_block_create_standard(self):

        block = TransformerBlock(embedding_dim=128, num_head = 8, block_size = 512, residual_dropout=0.1, N = 6, dtype = None, fused_residuals=False)
        batch_cts = random.normal(self.rng, shape = (1, 512, 128))
        params = block.init(self.init_rng, batch_cts, False)

    def test_block_fwd_standard(self):

        block = TransformerBlock(embedding_dim=128, num_head = 8, block_size = 512, residual_dropout=0.1, N = 6, dtype = None, fused_residuals=False)
        batch_cts = random.normal(self.rng, shape = (1, 512, 128))
        params = block.init(self.init_rng, batch_cts, False)

        out = block.apply({'params':params['params']},batch_cts, train = True, rngs = {"dropout": self.rng})
        assert_equal(out.shape, batch_cts.shape)


class TestGPT(unittest.TestCase):

    def setUp(self) -> None:
        self.init_rng, self.rng = random.split(random.PRNGKey(0)) 
        self.vocab_size = 256
        self.block_size = 512

    def tearDown(self) -> None:
        pass

    def test_gpt_create_fused(self):

        block = Transformer(embedding_dim=128, vocab_size = 256, num_head = 8, block_size = 512, residual_dropout=0.1, N = 6, dtype = None, fused_residuals=True)
        batch_tok = random.randint(self.rng, shape = (1, 512),maxval=256, minval = 0)
        params = block.init(self.init_rng, batch_tok, False)
    
    def test_gpt_fwd_fused(self):

        block = Transformer(embedding_dim=128, vocab_size = self.vocab_size, num_head = 8, block_size = 512, residual_dropout=0.1, N = 6, dtype = None, fused_residuals=True)
        batch_tok = random.randint(self.rng, shape = (1, 512),maxval=256, minval = 0)
        params = block.init(self.init_rng, batch_tok, False)

        out = block.apply({'params':params['params']},batch_tok, train = True, rngs = {"dropout": self.rng})
        assert_equal((1,self.block_size,self.vocab_size), out.shape)

    def test_gpt_create_standard(self):

        block = Transformer(embedding_dim=128, vocab_size = 256, num_head = 8, block_size = 512, residual_dropout=0.1, N = 6, dtype = None, fused_residuals=False)
        batch_tok = random.randint(self.rng, shape = (1, 512),maxval=256, minval = 0)
        params = block.init(self.init_rng, batch_tok, False)
    
    def test_gpt_fwd_standard(self):

        block = Transformer(embedding_dim=128, vocab_size = self.vocab_size, num_head = 8, block_size = 512, residual_dropout=0.1, N = 6, dtype = None, fused_residuals=False)
        batch_tok = random.randint(self.rng, shape = (1, 512),maxval=256, minval = 0)
        params = block.init(self.init_rng, batch_tok, False)

        out = block.apply({'params':params['params']},batch_tok, train = True, rngs = {"dropout": self.rng})
        assert_equal((1,self.block_size,self.vocab_size), out.shape)
