import jax

# The total number of TPU cores in the Pod
device_count = jax.device_count()

# The number of TPU cores attached to this host
local_device_count = jax.local_device_count()

# The psum is performed over all mapped devices across the Pod
xs = jax.numpy.ones(jax.local_device_count())
r = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)

# Print from a single host to avoid duplicated output
if jax.process_index() == 0:
    print('global device count:', jax.device_count())
    print('local device count:', jax.local_device_count())
    print('pmap result:', r)
