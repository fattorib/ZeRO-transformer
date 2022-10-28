""" 
LAtest Weight Averaging (LAWA) module in Flax
    <https://twitter.com/jeankaddour/status/1578174175738404864>

"""
from flax.training import checkpoints
import os 
from typing import Any
from flax.core.frozen_dict import FrozenDict
import jax 

def pytree_to_cpu(pytree: FrozenDict) -> FrozenDict:
    """
    Send all arrays in a FrozenDict to host device
    """
    cpu_pytree = jax.tree_util.tree_map(lambda xs: jax.device_get(xs), pytree)
    return cpu_pytree

def sum_params(pytree: FrozenDict, pytree_new: FrozenDict) -> FrozenDict:
    """ 
    Sum params from 2 pytrees
    """

    summed_pytree = jax.tree_util.tree_map(lambda x,y: x+y, pytree, pytree_new)

    return summed_pytree

def average_params(pytree: FrozenDict, k: int = 5) -> FrozenDict:
    """ 
    Compute average among summed pytree params
    """

    mean_pytree = jax.tree_util.tree_map(lambda xs: xs/k, pytree)

    return mean_pytree

def create_lawa_pytree(checkpoint_dir: str = 'checkpoints', k: int = 5, state: Any = None ):
    all_ckpt = os.listdir(checkpoint_dir)
    ckpt_steps = [int(f.split('checkpoint_')[-1]) for f in all_ckpt]

    params_cpu_base = pytree_to_cpu(state.params['params'])

    for ckpt in ckpt_steps:
        state_new = checkpoints.restore_checkpoint(checkpoint_dir, step = ckpt, target = state)

        new_params = pytree_to_cpu(state_new.params['params'])
        params_cpu_base = sum_params(params_cpu_base, new_params)

    return average_params(params_cpu_base)
        



