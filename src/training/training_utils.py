""" 
Helper methods used during training setup. 
"""
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

def compute_tokens_seen(absolute_step, max_context):
    return absolute_step * max_context
