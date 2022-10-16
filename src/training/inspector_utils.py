from typing import List

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from scipy.linalg import svdvals
from sklearn.decomposition import PCA


# Utility functions for working with jax arrays and pytrees
def to_list(arr: jnp.array) -> List:
    """
    Helper function for conversion of jnp.array to a list
    """
    return list(np.array(arr, dtype=np.float32).squeeze())


def pytree_devices(pytree: FrozenDict) -> FrozenDict:
    """
    Get device for all arrays in a FrozenDict
    """
    device_pytree = jax.tree_util.tree_map(lambda xs: xs.device(), pytree)
    return device_pytree


def pytree_to_cpu(pytree: FrozenDict) -> FrozenDict:
    """
    Send all arrays in a FrozenDict to host device
    """
    cpu_pytree = jax.tree_util.tree_map(lambda xs: jax.device_get(xs), pytree)
    return cpu_pytree


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


def get_num_components_pca(params, explained_variance=0.95) -> int:
    embedding_weight = np.array(params["params"]["wte"]["embedding"]).squeeze()
    pca_cls = PCA(n_components=explained_variance)
    _ = pca_cls.fit_transform(embedding_weight)
    return pca_cls.components_.shape[0]


def get_embedding_spectrum(params) -> int:
    embedding_weight = np.array(params["params"]["wte"]["embedding"]).squeeze()
    out = svdvals(embedding_weight)
    out = out[out > 0].shape
    return out[0]


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
