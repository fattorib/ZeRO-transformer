"""Loss functions

"""
import flax.linen as nn
import jax.numpy as jnp
from jax import jit, vmap
from jax.nn import one_hot


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


@jit
def kl_div_loss(teacher_logits: jnp.array, student_logits: jnp.array) -> jnp.array:
    """KL Divergence function for knowledge distillation

    Args:
        teacher_logits (jnp.array): Array of teacher logits. Assumes we have already scaled by T
        student_logits (jnp.array): Array of model logits

    Returns:
        jnp.array: Loss
    """
    targets = nn.softmax(teacher_logits, axis=-1)
    return -jnp.sum(
        targets * (jnp.log(targets) - nn.log_softmax(student_logits, axis=-1)), axis=-1
    )
