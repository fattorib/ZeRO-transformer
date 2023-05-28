""" 
From https://github.com/kingoflolz/mesh-transformer-jax/blob/master/mesh_transformer/util.py#L99
"""
import jax


@jax.custom_vjp
def f_psum(x):
    return x


def f_psum_fwd(x):
    return f_psum(x), None


def f_psum_bwd(_, g):
    return (jax.lax.psum(g, "mp"),)


f_psum.defvjp(f_psum_fwd, f_psum_bwd)


# identity in forward pass, pmean in backward
@jax.custom_vjp
def f_pmean(x):
    return x


def f_pmean_fwd(x):
    return f_psum(x), None


def f_pmean_bwd(_, g):
    return (jax.lax.pmean(g, "mp"),)


f_pmean.defvjp(f_pmean_fwd, f_pmean_bwd)


# psum in forward pass, identity in backward
@jax.custom_vjp
def g_psum(x):
    return jax.lax.psum(x, "mp")


def g_psum_fwd(x):
    return g_psum(x), None


def g_psum_bwd(_, g):
    return (g,)


g_psum.defvjp(g_psum_fwd, g_psum_bwd)
