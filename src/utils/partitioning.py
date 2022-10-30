""" 
Partitioning Rules for use with pjit operators. 

Modified from: https://github.com/borisdayma/dalle-mini/blob/main/src/dalle_mini/model/partitions.py
"""

import re

import jax
import optax
from flax.core import FrozenDict
from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental import PartitionSpec


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


def _get_partition_rules():
    """
    Follows Megatron-LM partition rules from

    `Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism`
    by Shoeybi et al.
    <https://arxiv.org/abs/1909.08053>

    Question: How are biases handled?
    """
    return [
        (("wte", "embedding"), PartitionSpec("mp", None)),
        (("wpe", "embedding"), PartitionSpec("mp", None)),
        # attention
        (("(query_proj|key_proj|value_proj)", "kernel"), PartitionSpec(None, "mp")),
        (("residual_out", "kernel"), PartitionSpec("mp", None)),
        # TODO: What is done about biases?
        (("(query_proj|key_proj|value_proj)", "bias"), None),
        (("residual_out", "bias"), None),
        # MLP
        (("fc_in", "kernel"), PartitionSpec(None, "mp")),
        (("fc_residual", "kernel"), PartitionSpec("mp", None)),
        # TODO: What is done about biases?
        (("fc_in", "bias"), None),
        (("fc_residual", "bias"), None),
        # layer norms
        (
            (
                "LayerNorm_0",
                "(bias|scale)",
            ),
            None,
        ),
    ]


def set_partitions(in_dict):
    """
    Takes a FrozenDict and returns the associated PartitionSpec rule
    for all groups of parameters
    """

    rules = _get_partition_rules()
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
    ), "Incomplete partition spec! All parameters must have a partitioning rule"
    return freeze(unflatten_dict(result))


def create_opt_spec(param_spec, param_shape):
    """
    Create optimizer PartitionSpec frozendict to match params PartitionSpec
    """
    opt_state_spec = {}
    for k, p in param_spec.items():
        opt_state_spec[k] = jax.tree_util.tree_map(
            lambda p: p,
            param_shape[k],
            # return None spec for empty elements
            is_leaf=lambda x: isinstance(x, (FrozenDict, optax.EmptyState)),
        )

    return freeze(opt_state_spec)
