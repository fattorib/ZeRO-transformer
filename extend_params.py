"""
Utility function to extend model parameters as outlined in 
Section G3.3. from `Scaling Language Models: Methods, Analysis
& Insights from Training Gopher`
    <https://arxiv.org/abs/2112.11446>
"""
import os

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax.serialization import msgpack_restore, msgpack_serialize

from src.models.GPT import model_getter

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def extend_params(target_pytree, params_pytree):
    """
    Extends params from a trained model pytree to a new pytree that duplicates
    params in groups of 2
    """
    param_keys = params_pytree["params"].keys()

    # handle LN/Token Embeddings
    target_pytree["params"]["LayerNorm_0"] = params_pytree["params"]["LayerNorm_0"]
    target_pytree["params"]["wte"] = params_pytree["params"]["wte"]

    out = params_pytree["params"].pop("LayerNorm_0")
    del out

    out = params_pytree["params"].pop("wte")
    del out

    block_mapping = create_mapping(18)

    for key in list(param_keys):

        if ("LayerNorm" in key) or ("wte" in key):
            pass

        else:
            data = params_pytree["params"].pop(key)
            orig_block_idx = int(key.split("_")[-1])
            blocks_to_map = block_mapping[orig_block_idx]
            for i in blocks_to_map:
                target_pytree["params"][f"TransformerBlock_{i}"] = data

    return target_pytree


def create_mapping(original_blocks):
    block_mapping = {}

    for i, block in enumerate(range(original_blocks)):
        block_mapping[block] = [block + i, block + i + 1]

    return block_mapping


if __name__ == "__main__":

    # load in params pytree
    with open("checkpoints/LAWA.msgpack", "rb") as f:
        params_pytree = msgpack_restore(f.read())

    model = model_getter(
        "XXL", return_cfg=False
    )  # init model to draw empty pytree from

    # create empty target pytree
    rng = jax.random.PRNGKey(0)
    batch_tok = jax.random.randint(rng, shape=(1, 1024), maxval=50257, minval=0)
    shape_pytree = jax.eval_shape(model.init, rng, batch_tok)

    empty_pytree = jax.tree_util.tree_map(lambda x: jnp.empty(x.shape), shape_pytree)

    extended_pytree = extend_params(unfreeze(empty_pytree), unfreeze(params_pytree))
    del params_pytree

    with open("checkpoints/warmstart_params.msgpack", "wb") as f:
        f.write(msgpack_serialize(unfreeze(extended_pytree), in_place=True))
