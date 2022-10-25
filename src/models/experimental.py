""" 
Holds experimental modules
"""
import flax.linen as nn
import jax.lax as lax
import jax.numpy as jnp
from einops import rearrange


class SGU(nn.Module):
    """Static SGU module"""

    block_size: int
    embedding_dim: int
    kernel_size: int

    def setup(self):
        k = jnp.triu(
            jnp.tril(
                jnp.ones((self.block_size, self.block_size)), -1 * self.kernel_size
            )
        )
        k = k / (k.cumsum(-2) + 1)
        self.kernel = k

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:

        _, T, _ = x.shape[:3]

        x = lax.stop_gradient(x)

        x = rearrange(x, "b n d -> b d n")
        x = x @ (self.kernel.T[:T, :T])
        x = rearrange(x, "b d n -> b n d")

        return x
