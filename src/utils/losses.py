"""Loss functions

"""
from jax import jit, vmap
import jax.numpy as jnp 
from jax.nn import one_hot
import flax.linen as nn 

@jit
def cross_entropy_loss(labels: jnp.array, logits: jnp.array) -> jnp.array:
    """Standard Cross Entropy Loss function

    Args:
        labels (jnp.array): Array of one-hot encoded labels
        logits (jnp.array): Array of model logits

    Returns:
        jnp.array: Loss
    """

    return -jnp.mean(jnp.sum(labels * nn.log_softmax(logits, axis=-1), axis=-1))

