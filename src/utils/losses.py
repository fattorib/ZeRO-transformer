"""
Loss function
"""
import jax
import jax.numpy as jnp


def cross_entropy_loss(labels: jnp.array, logits: jnp.array) -> jnp.array:
    """Standard Cross Entropy Loss function

    Args:
        labels (jnp.array): Array of one-hot encoded labels
        logits (jnp.array): Array of model logits

    Returns:
        jnp.array: Loss
    """

    return -jnp.sum(
        labels * jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1), axis=-1
    )
