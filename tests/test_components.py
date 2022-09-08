import unittest
from chex import assert_equal 
import flax.linen as nn 
import jax.numpy as jnp 
import jax.random as random 
from src.models.GPT import CausalAttention, MLPBlock

class TestMLP(unittest.TestCase):

    def setUp(self) -> None:
        self.init_rng, self.rng = random.split(random.PRNGKey(0)) 

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

    def tearDown(self) -> None:
        pass 

    def test_attn_create(self):

        attn = CausalAttention(embedding_dim=128, num_head = 8, block_size = 64, dropout=0.1, N = 6)
        batch_cts = random.normal(self.rng, shape = (1, 512, 128))
        params = attn.init(self.init_rng, batch_cts, False)

    def test_attn_fwd(self):
        attn = CausalAttention(embedding_dim=128, num_head = 8, block_size = 64, dropout=0.1, N = 6)
        batch_cts = random.normal(self.rng, shape = (1, 512, 128))
        params = attn.init(self.init_rng, batch_cts, False)

        out = attn.apply({'params':params['params']},batch_cts, train = True, rngs = {"dropout": self.rng})
        assert_equal(out.shape, batch_cts.shape)

        out = attn.apply({'params':params['params']},batch_cts, train = False, rngs = {"dropout": self.rng})
        assert_equal(out.shape, batch_cts.shape)

