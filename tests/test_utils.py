""" 
Tests for any extra utility functions 
"""

import jax.numpy as jnp
import pytest

from src.utils.losses import cross_entropy_loss


@pytest.mark.parametrize(
    "labels, logits",
    [
        (
            jnp.array([[1]], dtype=jnp.int32),
            jnp.array([[-0.5, 1.2, 2.4, -3.2]], dtype=jnp.float32),
        ),
        (
            jnp.array([[3]], dtype=jnp.int32),
            jnp.array([[1.2, 2.4, -0.5, -3.2]], dtype=jnp.float16),
        ),
    ],
)
def test_loss_dtype(labels, logits):
    """Ensure loss is computed in single precision
    even if we have half-precision inputs.

    Softmaxes cannot be computed in lower than single precision.
    """

    out = cross_entropy_loss(labels, logits)

    assert out.dtype == jnp.float32


@pytest.mark.parametrize(
    "labels, logits, expected",
    [
        (
            jnp.array([[1]], dtype=jnp.int32),
            jnp.array([[-0.5, 1.2, 2.4, -3.2]], dtype=jnp.float32),
            jnp.array(10.929689),
        ),
        (
            jnp.array([[3]], dtype=jnp.int32),
            jnp.array([[1.2, 2.4, -0.5, -3.2]], dtype=jnp.float16),
            jnp.array(32.788956),
        ),
    ],
)
def test_loss_expected(labels, logits, expected):
    """
    Ensure loss returns correct values
    """
    out = cross_entropy_loss(labels, logits)

    assert jnp.allclose(out, expected)
