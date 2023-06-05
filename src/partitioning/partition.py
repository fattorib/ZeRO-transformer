""" 
Partitioning Rules for use with pjit operators. 
"""


import jax
import optax
from flax.core import FrozenDict
from jax.sharding import PartitionSpec

def create_opt_spec(param_spec, opt_state_shapes):
    """
    Create optimizer PartitionSpec frozendict to match params PartitionSpec
    """

    # used to assign PartitionSpecs to Optax optimizer states
    # we need this special function since the optimizer state includes extra items beyond just 2 FrozenDicts for the buffers
    def get_opt_spec(x):
        if isinstance(
            x, (FrozenDict,)
        ):  # if we get first/second moment buffers, clone PSpec of the params
            return param_spec
        return (
            PartitionSpec()
        )  # else, PSpec of None (this is to be copied across all devices) (stuff like GA step, skip_step, etc)

    opt_state_spec = jax.tree_util.tree_map(
        get_opt_spec,
        opt_state_shapes,
        is_leaf=lambda x: isinstance(
            x,
            (
                FrozenDict,
                optax.EmptyState,
            ),
        ),
    )

    return opt_state_spec
