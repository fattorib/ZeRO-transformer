"""
Utility function to extend model parameters as outlined in 
Section G3.3. from `Scaling Language Models: Methods, Analysis
& Insights from Training Gopher`
    <https://arxiv.org/abs/2112.11446>
"""
import os

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

    block_mapping = create_mapping()

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


def create_mapping():
    num_layers_old = 18
    block_mapping = {i: [i + i, i + 1 + i] for i in range(num_layers_old)}
    return block_mapping
