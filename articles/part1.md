# pmap to pjit - Part 1: Data Parallel with pmap

Benjamin Fattori 

## Introduction

Jax is a great package that blah blah blah but there is minimal documentation on pjitting code. Your options are:

- Reading repos 
- Reading more theoretical articles
- Reading HF code 

The purpose of these posts is twofold. First, they serve as a place to summarize everything I've learned about ```pmap```/```pjit``` these past few months and second, they serve as a guide for others wishing to use ```pjit``` with flax. 

## Background

We modify code for training causal language models. The code and scrips follow the general structure of training models in flax. To summarize, the general structure for training models with flax is:

1. Define our model in flax and initialize its parameters with ```model.init```:
```python 

# define model 
model = GPT2(**config)

# needed to initialize params
rng = jax.random.PRNGKey(0)
init_batch = jnp.ones((input_shape), dtype=jnp.int32)

# initialize parameters
params = model.init(rng, init_batch, None, False)
```

2. Create a ```TrainState``` object to hold model parameters, optimizer buffers and training step count. Optimizer is handled through [Optax](https://github.com/deepmind/optax). The ```TrainState``` object holds all the state and it is what is passed around and updated:
```python
# create your optimizer
tx = optax.adamw(
    learning_rate=learning_rate_fn,
    weight_decay=0.1,
    mask=mask, # mask for LN/bias params
    b2=0.95,  
)

# create new TrainState object 
state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
```

3. Write a ```train_step``` function which takes in a single batch from our dataset and performs gradient descent to update the model parameters:
```python
def train_step(
    state: Any, batch: jnp.array, rng_key: random.PRNGKey = None,
):
    """Train on a single batch"""

    def loss_fn(params):
        _, loss = state.apply_fn(
            {"params": params["params"]},
            x=batch,
            labels=batch,
            train=True,
            rngs={"dropout": rng_key},
        )

        return loss

    # grad_fn returns (grad of loss wrt params, loss)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

    # get loss and gradients
    loss, grads = grad_fn(state.params)

    # apply gradients to update trainstate
    new_state = state.apply_gradients(
        grads=grads,
    )

    # log any important metrics
    metrics = {
        "Train LM Loss": loss,
        "Train LM PPL": jnp.exp(loss),
    }

    return new_state, metrics
```

## 1. Pmap

The code we wrote above can only be executed on a single device (ex: 1 GPU or 1 core of TPU). Even if we have multiple devices available, we haven't done anything to 'tell' Jax how all the devices should interact. As a first step towards parallelization, we can use ```pmap```! ```pmap``` is designed for running SPMD (Single Program Multiple Data) programs and works by replicating the program across multiple devices (ex: running the same train step across all cores in a TPU).

In our case, a common approch is to split the batch of data across all available cores (replicating the ```TrainState```) on each core. Then, each core processes its own batch, computes the local gradients and then performs an all-reduce (via ```jax.lax.pmean```) to synchronize gradients across model replicas. From here, the optimization step, ```state.apply_gradients``` can be executed in parallel as all replicas have the same gradients. 

We can update our ```train_step`` code from above by making the following additions:

- Wrapping the function in ```pmap```
- Adding an all-reduce operation for the loss and the gradients

Our new function will look like:

```python
@partial(jax.pmap, axis_name="batch")
def train_step(
    state: Any, batch: jnp.array, rng_key: random.PRNGKey = None,
):
    """Train on a single batch"""

    def loss_fn(params):
        _, loss = state.apply_fn(
            {"params": params["params"]},
            x=batch,
            labels=batch,
            train=True,
            rngs={"dropout": rng_key},
        )

        return loss

    # grad_fn returns (grad of loss wrt params, loss)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

    # get loss and gradients
    loss, grads = grad_fn(state.params)

    # compute all-reduce to synchronize gradients and loss
    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")


    # apply gradients to update trainstate
    new_state = state.apply_gradients(
        grads=grads,
    )

    # log any important metrics
    metrics = {
        "Train LM Loss": loss,
        "Train LM PPL": jnp.exp(loss),
    }

    return new_state, metrics
```

Note the addition of the ```axis_name``` variable in ```pmap``` and ```pmean```. This is needed to tell ```pmap``` which axis the function is to be replicated along. In this case, we are replicating it along the batch dimension. 

Are we ready to start doing Data Parallel training? Not yet! There are a few more small changes we need to make. First, even though our function is ready to take in data from multiple devices, all the data (state, rng, batch) are all sitting on the 0th device. Flax has some helpful utility functions to do this for us:

- To replicate the ```TrainState```, we can use ```flax.jax_utils.replicate```

- To split the batch across devices, we can use ```flax.training.common_utils.shard```. **Note**: this function effectively adds a device axis and then reshapes the batch. As such, the number of elements in the batch must be divisible by the number of *local* devices. For example, if we are training on an 8-core TPU, then a batch size of 8 will be reshaped such that each core of the TPU gets a single element of the batch. 

- We can use ```flax.training.common_utils.shard_prng_key``` to shard the rng key across all *local* devices too. 

With these modifications, we are ready to start training! In full, our setup and training loop will now look like this:

```python 

# replicate state across all local devices
state = flax.jax_utils.replicate(state)

for batch in dataloader: # loop through batch 
    batch_sharded = flax.training.common_utils.shard(batch) # shard batch across devices 

    rng, dropout_rng = jax.random.split(rng, 2)

    dropout_rng = flax.training.common_utils.shard_prng_key(dropout_rng) # shard rng 

    state, metrics = train_step(state, batch, dropout_rng) # train
```

Up to now, we have only written this code for a single host. On a TPU pod, extending this is trivial since each host executes the same code. All cores of a TPU will communicate only when an all-reduce such as ```pmean``` is called. Outside of that, each host will operate independently as the flax utils used above will only shard and replicate data within the local devices the host has control over (on a TPU, for example, 1 host controls 8 TPU pods). 

