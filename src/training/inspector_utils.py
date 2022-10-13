from typing import List

import flax
import jax
import jax.numpy as jnp
import numpy as np


def to_list(arr: jnp.array) -> List:
    """
    Helper function for conversion of jnp.array to a list
    """
    return list(np.array(arr, dtype=np.float32).squeeze())


def get_intermediates(intermediates) -> List:
    """
    Pulls out intermediate values from PyTree and reshapes for histogram
    """
    reshaped = jax.tree_util.tree_map(
        lambda xs: jnp.reshape(xs, (1, -1)), intermediates
    )
    features = {}
    for key in reshaped["intermediates"].keys():
        for activation in reshaped["intermediates"][key].keys():
            if "CausalAttention" in activation:
                act = reshaped["intermediates"][key][activation]["attn_out"]
            else:
                act = reshaped["intermediates"][key][activation]["mlp_out"]
            features[f"{key}_{activation}"] = list(np.array(act).squeeze())
    return features


def get_weights_gradients(params) -> List:
    weight_params = flax.traverse_util.ModelParamTraversal(
        lambda path, _: ("bias" not in path and "scale" not in path)
    )
    weight_params = weight_params.iterate(params)
    weights = []
    for p in weight_params:
        if p.ndim > 1:
            weights.append(to_list(p.reshape(1, -1)))
    return weights
