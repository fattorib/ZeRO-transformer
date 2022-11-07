import math
from typing import List

import jax.numpy as jnp


def get_slopes(n: int) -> List:
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def create_mask(seq_len_k, slopes):

    a = -jnp.tril(
        jnp.tile(jnp.arange(seq_len_k).reshape(seq_len_k, 1), (1, seq_len_k))
        + jnp.arange(0, -seq_len_k, step=-1)
    )

    a = a * (slopes.reshape(slopes.shape[0], 1, 1))

    alibi_mask = a[:, seq_len_k - 1, :].reshape(a.shape[0], 1, a.shape[2])

    return alibi_mask


out = create_mask(seq_len_k=1024, slopes=jnp.array(get_slopes(n=16)))

print(out.shape)
