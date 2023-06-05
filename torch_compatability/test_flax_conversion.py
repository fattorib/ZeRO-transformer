""" 
Tests to ensure converting a model from flax to torch retains
the proper values
"""

import os

import jax
import numpy as np
import pytest
from flax.training import train_state, checkpoints

from flax.serialization import msgpack_serialize

from src.models.GPT import model_getter as jax_model_getter
from torch_compatability.flax_to_pytorch import (
    create_transformer_block_mapping, flatten, match_and_save)
from torch_compatability.GPT2 import model_getter as torch_model_getter

os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.fixture
def test_jax_model_noremat():
    # create a tiny 2L jax model, saves its checkpoint and returns the path
    model = jax_model_getter("test")
    rng = jax.random.PRNGKey(0)
    init_batch = jax.random.randint(rng, shape=(1, 32), maxval=256, minval=0)

    params = model.init(rng, init_batch, None, False)

    faux_state = train_state.TrainState(
            step=0, apply_fn=None, params=params, tx=None, opt_state=None
        )
    checkpoints.save_checkpoint(
        "checkpoints/test", faux_state, 0, keep=1, overwrite=True, prefix="params_"
    )

    return "checkpoints/test"


def test_match_conversion_noremat(test_jax_model_noremat):

    torch_model = torch_model_getter("test")

    jax_pytree_dir = test_jax_model_noremat
    
    # read flax checkpoint and optionally serialize to msgpack
    restored = checkpoints.restore_checkpoint(jax_pytree_dir, target=None, prefix="params_")
    param_bytes = msgpack_serialize(restored['params'])
    jax_pytree = restored['params']
    
    with open("checkpoints/test/params.msgpack", "wb") as f:
        f.write(param_bytes)

    match_and_save(torch_model, "checkpoints/test/params.msgpack", "checkpoints/test.pth", False, False)

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
            
    # checks on top-level weights too

    assert np.allclose(
                    torch_model.state_dict()["norm.weight"], np.array(jax_pytree["params"]["LayerNorm_0"]["scale"]["value"])
                )
    
    assert np.allclose( torch_model.state_dict()["lm_head.weight"], np.transpose(np.array(jax_pytree["params"]["logits_untied"]["kernel"]["value"])[
            : torch_model.vocab_size,
        ], (1, 0)))

    assert np.allclose(
                    torch_model.state_dict()["wte.weight"], np.array(jax_pytree["params"]["wte"]["kernel"]["value"])[: torch_model.vocab_size])


