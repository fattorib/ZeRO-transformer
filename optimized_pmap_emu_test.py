import functools
import os
from typing import Any

import jax

from src.models.GPT import model_getter
from src.training.training_utils import create_train_state
from src.utils.partitioning import set_partitions
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import replicate
from optimized_pmap import train_step

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':

    model = model_getter("small", return_cfg=False)

    rng = jax.random.PRNGKey(23)
    batch_tok = jax.random.randint(rng, shape=(1, 32), maxval=50257, minval=0)
    param_shape = jax.eval_shape(model.init, rng, batch_tok)
    param_spec = set_partitions(param_shape)

    state = create_train_state(
                rng,
                3e-4,
                weight_decay=0.1,
                model=model,
            )

    state = replicate(state)

    global_bs = 128
    ga_steps = 8 
    seq_len = 32

    # giving correct init loss, need to try this out in some v2 benchmarks tomorrow!

    test_batch = jax.random.randint(rng, (global_bs, seq_len), maxval=50257, minval=0)

    # shape is ga_steps, local_bs, ctx
    test_batch = test_batch.reshape(ga_steps, global_bs//ga_steps, seq_len).transpose(1,0,2)

    print(test_batch.shape)

    test_batch = shard(test_batch)
    print(test_batch.shape)

    rng_sharded = shard_prng_key(rng)

    state, metrics = train_step(state, test_batch, rng_sharded, ga_steps)

    print(metrics)


    """"
    
    Usual pmap behaviour 

        Global BS of 128  
        4 devices 
        8 accum steps

        batch: (128, ctx)

        this is split up into 8 _separate_ batches

        micro_batch: (16, ctx)
        -> Shard -> (device_cnt,4, ctx)
        Each of the 4 devices sees a batch of size 4 -> Each devices see (4,ctx)

        Once we have looped through 8 of these, update gradients    

    "Optimized" pmap behaviour

        Global BS of 128 
        4 devices 
        8 accum steps

        batch: (128, ctx)
        -> reshape into (accum_steps, 128/accum_steps, ctx) : (8,16,ctx)

        
        -> *Shard batches, splits first axis. Since we don't want to split GA, transpose dims 0 and 1*

        -> reorder (16,8, ctx)

        -> proper shard (4,4,8,ctx)

        (device_cnt,4,8,ctx) -> Each device sees a batch with shape (4,8,ctx) -> From here, we loop through batches of size (4,ctx)

    """