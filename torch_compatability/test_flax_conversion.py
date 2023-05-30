""" 
Tests to ensure converting a model from flax to torch retains
the proper values
"""

import os

import jax
import numpy as np
import pytest
from flax.core import unfreeze
from flax.serialization import msgpack_restore, msgpack_serialize

from src.models.GPT import model_getter as jax_model_getter
from torch_compatability.flax_to_pytorch import (
    create_transformer_block_mapping,
    flatten,
    match_and_save,
)
from torch_compatability.GPT2 import model_getter as torch_model_getter

os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.fixture
def test_jax_model():
    # create a tiny 2L jax model and return its pytree
    model = jax_model_getter("test_no_tp")
    rng = jax.random.PRNGKey(0)
    init_batch = jax.random.randint(rng, shape=(1, 32), maxval=256, minval=0)

    params = model.init(rng, init_batch, None, False)

    param_bytes = msgpack_serialize(unfreeze(params))

    with open("checkpoints/test.msgpack", "wb") as f:
        f.write(param_bytes)

    return "checkpoints/test.msgpack"


def test_match_conversion(test_jax_model):

    torch_model = torch_model_getter("test")

    jax_pytree_path = test_jax_model

    match_and_save(torch_model, jax_pytree_path, "checkpoints/test.pth")

    with open(jax_pytree_path, "rb") as f:
        jax_pytree = msgpack_restore(f.read())

    torch_model = torch_model_getter("test", model_checkpoint="checkpoints/test.pth")

    for block_idx in range(torch_model.N):
        block_mapping_dict = create_transformer_block_mapping(block_idx)

        flattened_block = dict(
            flatten(jax_pytree["params"][f"TransformerBlock_{block_idx}"])
        )

        for key, value in block_mapping_dict.items():
            jax_pytree_val = np.array(flattened_block[key])
            torch_param_val = torch_model.state_dict()[value].detach().numpy()

            if jax_pytree_val.ndim > 1:
                assert np.allclose(
                    np.transpose(jax_pytree_val, (1, 0)), torch_param_val
                )
            else:
                assert np.allclose(jax_pytree_val, torch_param_val)
