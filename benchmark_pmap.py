from time import time 
import jax 
import jax.numpy as jnp 
from src.models.GPT import model_getter
from src.training.training_utils import create_train_state
from optimized_pmap import naive_train_step, train_step
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import replicate
import numpy as np 
from tqdm import tqdm 

# the quantity GLOBAL_BATCH_SIZE must be divisible by 8 (or num local devices)

GLOBAL_BATCH_SIZE = 512
GRADIENT_ACCUMULATION_STEPS = 32
SEQ_LEN = 512
NUM_PASSES = 100

def main_optimized():
    # base model is ~125M params
    model = model_getter("base", return_cfg=False)

    # State Creation, etc 
    init_rng = jax.random.PRNGKey(0)
    state = create_train_state(
                    init_rng,
                    3e-4,
                    weight_decay=0.1,
                    model=model,
                )
    
    #replicate state across devices 
    state = replicate(state)

    # compile optimized train_step by doing single fwd pass 
    rng, batch_rng = jax.random.split(init_rng, 2)
    test_batch = jax.random.randint(batch_rng, (GLOBAL_BATCH_SIZE, SEQ_LEN), maxval=50257, minval=0)
    # reshape for proper idx with pmap
    test_batch = test_batch.reshape(GRADIENT_ACCUMULATION_STEPS, GLOBAL_BATCH_SIZE//GRADIENT_ACCUMULATION_STEPS, SEQ_LEN).transpose(1,0,2)
    # shard first dimension across devices
    test_batch = shard(test_batch)
    rng_sharded = shard_prng_key(rng)
    train_step(state, test_batch, rng_sharded, GRADIENT_ACCUMULATION_STEPS)

    times = []
    for _ in tqdm(range(NUM_PASSES)):
        rng, batch_rng = jax.random.split(rng, 2)
        test_batch = jax.random.randint(batch_rng, (GLOBAL_BATCH_SIZE, SEQ_LEN), maxval=50257, minval=0)
        test_batch = test_batch.reshape(GRADIENT_ACCUMULATION_STEPS, GLOBAL_BATCH_SIZE//GRADIENT_ACCUMULATION_STEPS, SEQ_LEN).transpose(1,0,2)
        test_batch = shard(test_batch)
        rng_sharded = shard_prng_key(rng)
        t0 = time()
        train_step(state, test_batch, rng_sharded, GRADIENT_ACCUMULATION_STEPS)
        times.append(time() - t0)
    
    print(f"Optimized Pmap Step - Global BS {GLOBAL_BATCH_SIZE} - accum steps {GRADIENT_ACCUMULATION_STEPS} - Num Executions {NUM_PASSES}")
    print(f"Mean Batch Time {np.mean(times)}")
    print()


def main_naive():
    # base model is ~125M params
    model = model_getter("base", return_cfg=False)

    # State Creation, etc 
    init_rng = jax.random.PRNGKey(0)
    state = create_train_state(
                    init_rng,
                    3e-4,
                    weight_decay=0.1,
                    model=model,
                )
    
    #replicate state across devices 
    state = replicate(state)

    # compile optimized train_step by doing single fwd pass 
    rng, batch_rng = jax.random.split(init_rng, 2)
    test_batch = jax.random.randint(batch_rng, (GLOBAL_BATCH_SIZE//GRADIENT_ACCUMULATION_STEPS, SEQ_LEN), maxval=50257, minval=0)
    # shard first dimension across devices
    test_batch = shard(test_batch)
    rng_sharded = shard_prng_key(rng)
    naive_train_step(state, test_batch, rng_sharded)

    times = []
    for _ in tqdm(range(NUM_PASSES)):
        rng, batch_rng = jax.random.split(rng, 2)
        single_batch_times = []
        for _ in range(GRADIENT_ACCUMULATION_STEPS):
            test_batch = jax.random.randint(batch_rng, (GLOBAL_BATCH_SIZE//GRADIENT_ACCUMULATION_STEPS, SEQ_LEN), maxval=50257, minval=0)
            test_batch = shard(test_batch)
            rng_sharded = shard_prng_key(rng)
            t0 = time()
            train_step(state, test_batch, rng_sharded, GRADIENT_ACCUMULATION_STEPS)
            single_batch_times.append(time() - t0)
        times.append(sum(single_batch_times))
    
    print(f"Naive Pmap Step - Global BS {GLOBAL_BATCH_SIZE} - accum steps {GRADIENT_ACCUMULATION_STEPS} - Num Executions {NUM_PASSES}")
    print(f"Mean Batch Time {np.mean(times)}")




if __name__ == '__main__':
    main_optimized()

    main_naive()

    """
    V2-8 Benchmarks



    """