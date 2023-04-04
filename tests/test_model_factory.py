import unittest

import jax.numpy as jnp
import jax.random as random
from omegaconf import OmegaConf

from src.models.GPT import model_getter


class TestLoadModels(unittest.TestCase):
    def setUp(self) -> None:
        self.init_rng, self.rng = random.split(random.PRNGKey(0))
        self.block_size = 512
        self.config = OmegaConf.load("conf/model_config.yaml")

    def tearDown(self) -> None:
        pass

    def test_call_valid(self) -> None:

        model_size = "test"
        model = model_getter(model_size, dtype=jnp.float16)

        self.assertEqual(model.N, self.config[model_size].N)

    def test_call_invalid(self) -> None:

        model_size = "Humongous"
        self.assertRaises(AssertionError, model_getter, model_size)

        model_size = "test"
        self.assertRaises(AssertionError, model_getter, model_size, dtype=jnp.float64)
