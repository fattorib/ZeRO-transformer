""" 
Partitioning Rules for use with pjit operators. 

Modified from: https://github.com/borisdayma/dalle-mini/blob/main/src/dalle_mini/model/partitions.py
"""

import re

import jax
import numpy as np
import optax
from flax.core import FrozenDict
from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import Mesh, PartitionSpec
from jax.sharding import NamedSharding
from typing import Callable


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def _replacement_rules(rules):
    def replace(key, val):
        for rule, replacement in rules:
            if _match(rule, key):
                return replacement
        return val

    return replace

def _get_partition_rules_dp(mesh: Mesh):
    """
    Follows Megatron-LM partition rules from

    `Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism`
    by Shoeybi et al.
    <https://arxiv.org/abs/1909.08053>

    """
    return [
        (("wte", "embedding"), NamedSharding(mesh, PartitionSpec("dp", None, None))),
        (("wpe", "embedding"), NamedSharding(mesh,PartitionSpec("dp", None, None))),
        # attention
        (("(query_proj|key_proj|value_proj)", "kernel"), NamedSharding(mesh,PartitionSpec("dp", None, None))),
        (("residual_out", "kernel"), NamedSharding(mesh,PartitionSpec("dp", None, None))),
        (("(query_proj|key_proj|value_proj)", "bias"), NamedSharding(mesh,PartitionSpec("dp", None))),
        (("residual_out", "bias"), NamedSharding(mesh,PartitionSpec("dp", None))),
        # MLP
        (("fc_in", "kernel"), NamedSharding(mesh,PartitionSpec("dp", None, None))),
        (("fc_residual", "kernel"), NamedSharding(mesh,PartitionSpec("dp", None, None))),
        (("fc_in", "bias"), NamedSharding(mesh,PartitionSpec("dp", None))),
        (("fc_residual", "bias"), NamedSharding(mesh,PartitionSpec("dp", None))),
        # layer norms
        (
            (
                "LayerNorm_0",
                "(bias|scale)",
            ),
            NamedSharding(mesh,PartitionSpec("dp", None)),
        ),
        # layer norms
        (
            (
                "LayerNorm_1",
                "(bias|scale)",
            ),
            NamedSharding(mesh,PartitionSpec("dp", None)),
        ),
    ]

def _get_partition_rules_tp(mesh: Mesh):
    """
    Follows Megatron-LM partition rules from

    `Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism`
    by Shoeybi et al.
    <https://arxiv.org/abs/1909.08053>

    """
    return [
        (("wte", "embedding"), NamedSharding(mesh, PartitionSpec("mp", None))),
        (("wpe", "embedding"), NamedSharding(mesh,PartitionSpec("mp", None))),
        # attention
        (("(query_proj|key_proj|value_proj)", "kernel"), NamedSharding(mesh,PartitionSpec(None, "mp"))),
        (("residual_out", "kernel"), NamedSharding(mesh,PartitionSpec("mp", None))),
        (("(query_proj|key_proj|value_proj)", "bias"), NamedSharding(mesh,PartitionSpec(None))),
        (("residual_out", "bias"), NamedSharding(mesh,PartitionSpec("None"))),
        # MLP
        (("fc_in", "kernel"), NamedSharding(mesh,PartitionSpec(None, "mp"))),
        (("fc_residual", "kernel"), NamedSharding(mesh,PartitionSpec("mp", None))),
        (("fc_in", "bias"), NamedSharding(mesh,PartitionSpec(None))),
        (("fc_residual", "bias"), NamedSharding(mesh,PartitionSpec(None))),
        # layer norms
        (
            (
                "LayerNorm_0",
                "(bias|scale)",
            ),
            NamedSharding(mesh,PartitionSpec(None)),
        ),
        # layer norms
        (
            (
                "LayerNorm_1",
                "(bias|scale)",
            ),
            NamedSharding(mesh,PartitionSpec(None)),
        ),
    ]

def _get_partition_rules_tp_dp(mesh: Mesh):
    """
    Follows Megatron-LM partition rules from

    `Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism`
    by Shoeybi et al.
    <https://arxiv.org/abs/1909.08053>

    """
    return [
        (("wte", "embedding"), NamedSharding(mesh, PartitionSpec("dp","mp", None))),
        (("wpe", "embedding"), NamedSharding(mesh,PartitionSpec("dp","mp", None))),
        # attention
        (("(query_proj|key_proj|value_proj)", "kernel"), NamedSharding(mesh,PartitionSpec("dp",None, "mp"))),
        (("residual_out", "kernel"), NamedSharding(mesh,PartitionSpec("dp","mp", None))),
        (("(query_proj|key_proj|value_proj)", "bias"), NamedSharding(mesh,PartitionSpec("dp",None))),
        (("residual_out", "bias"), NamedSharding(mesh,PartitionSpec("dp","mp"))),
        # MLP
        (("fc_in", "kernel"), NamedSharding(mesh,PartitionSpec("dp",None, "mp"))),
        (("fc_residual", "kernel"), NamedSharding(mesh,PartitionSpec("dp","mp", None))),
        (("fc_in", "bias"), NamedSharding(mesh,PartitionSpec("dp",None))),
        (("fc_residual", "bias"), NamedSharding(mesh,PartitionSpec("dp",None))),
        # layer norms
        (
            (
                "LayerNorm_0",
                "(bias|scale)",
            ),
            NamedSharding(mesh,PartitionSpec("dp",None)),
        ),
        # layer norms
        (
            (
                "LayerNorm_1",
                "(bias|scale)",
            ),
            NamedSharding(mesh,PartitionSpec("dp",None)),
        ),
    ]

def set_partitions_rules(in_dict, mesh: Mesh, rules_func: Callable):
    """
    Takes a FrozenDict and returns the associated PartitionSpec rule
    for all groups of parameters
    """

    rules = rules_func(mesh)
    replace = _replacement_rules(rules)

    _unmatched = object()
    initd = {
        k: _unmatched for k in flatten_dict(in_dict)
    }  # replaces all values in the dict with _object()

    result = {
        k: replace(k, v) for k, v in initd.items()
    }  # replaces all values in the initd dict with defined PartitionSpec rules

    assert (
        _unmatched not in result.values()
    ), f"Incomplete partition spec! All parameters must have a partitioning rule."
    return freeze(unflatten_dict(result))


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
        return None  # else, PSpec of None (this is to be copied across all devices) (stuff like GA step, skip_step, etc)

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
