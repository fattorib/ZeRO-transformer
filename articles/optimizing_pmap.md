# Optimizing Pmapped code for gradient accumulation

## 0 Introduction 

As models get larger, a common practice during training is to perform **gradient accumulation**. where a large batch of data is split into many smaller batches and the gradients of the model are 'accumulated' over the course of all *microbatches* in the larger batch. 

In [flax](https://github.com/google/flax), a training loop for training a language model with ```pmap```looks like this:

```python 

for i, batch in enumerate(dataloader): #iterate over data batches 

    # Batch of 512 samples

    # shard batch over all processes (ex: cores of TPU)
    batch = shard(batch) 

    # state consists of model parameters + optax optimizer state holding grad buffers, etc 
    state, metrics = train_step(
                state,
                batch,
                rng
            )
```

with the ```train_step``` function:

```python 

@partial(jax.pmap, axis_name="batch", donate_argnums = (0,))
def train_step(state: Any, batch: jnp.array, rng_key: jax.random.PRNGKey = None):

    def loss_fn(params):
        _, loss = state.apply_fn(
            {"params": params["params"]},
            x=batch,
            labels=batch,
            train=True,
            rngs={"dropout": rng_key},
        )
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)

    loss = jax.lax.pmean(loss, axis_name="batch") # all-reduce across devices
    grads = jax.lax.pmean(grads, axis_name="batch") # all-reduce across devices

    new_state = state.apply_gradients(
        grads=grads,
    )

    metrics = {
        "Train LM Loss": loss,
        "Train LM PPL": jnp.exp(loss),
    }

    return new_state, metrics

```

## 1 Naive Implementation

If we wanted to adjust this code for gradient accumulation, we could make the following changes:

- Decrease the batch size in our dataloader by a factor of ```grad_accum_steps```
- Wrap the optimizer in an ```optax.MultiSteps``` transform:
```python 
tx = optax.MultiSteps(
        tx,
        every_k_schedule=grad_accum_steps,
    )
```

With this changes in place, we don't have to modify our ```train_step``` logic at all! Calling ```state.apply_gradients``` simply accumulates the new gradients to a running buffer of gradients that is only updated after ```grad_accum``` gradients have been accumulated. 

## 2 Optimized Implementation

The lack of changes to our ```train_step``` logic mean that even on batches where the optimizer params are not updated, we are still performing an all-reduce of the microbatch gradients across devices. This sync point every microbatch is unnecessary and significantly reduces the training throughput (benchmarks below).

Instead, if we were able to modify our code to accumulate each the sharded gradients locally and only perform an all-reduce once per batch, we would be able to reduce the number of device-wide communications by ```grad_accum``` steps since we are only calling ```jax.lax.pmean``` once per batch instead of once per microbatch. 

To do this, however, requires changing both the batching logic and the ```train_step``` code. 

### 2.1 Updated Batching Logic 

We keep the batch size set at the full batch size and add the following logic to modify a batch after:

```python 

batch = batch.reshape(
    grad_accum_steps, batch_size//grad_accum_steps, -1
) 
batch = batch.transpose(1,0,2) 

```

The line
```
batch = batch.reshape(
    grad_accum_steps, batch_size//grad_accum_steps, -1
) 
```
reshapes the batch to add a "gradient accumulation" axis. 

The next line 
```
batch = batch.transpose(1,0,2)
```
is used to ensure batches are properly sharded across devices. By default ```shard``` in flax reshapes the *first* axis of the tensor to split it across devices. We want to split the batch axis across processes, not the gradient accumulation axis. 

As an example on a TPU with 8 devices, the shape tracking would look like:

```python 
batch_size = 512
grad_accum_steps = 32
seq_len = 2048

batch = next(iter(dataloader)) #(512, 2048) <-> (bs, ctx)

batch = batch.reshape(
    grad_accum_steps, batch_size//grad_accum_steps, -1
) # (32, 16, 2048) <-> (accum_steps, micro_bs, ctx)

batch = batch.transpose(1,0,2) #(16, 32, 2048)

batch = shard(batch) # (8, 2, 32, 2048) < - > (devices, local_bs, accum_steps, ctx)
```

Passing this in into a pmapped function means that every device gets a batch of size ```(2,32,2048)```

## 2.2 Updated train logic

*Based on pjitted code with the same concept from Boris Dayma's [DALL·E Mini](https://github.com/borisdayma/dalle-mini)*

If we tried to apply our current ```train_step``` function to this batch of data we would run into issues. The original train step code expects a batch with 2 dimensions but we are passing it a batch with 3 dimensions! We modify the train_step to iterate over the 2nd axis of data (reminder, this was the 'gradient accumulation' axis):

```python 

@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3,), donate_argnums = (0,))
def train_step(
    state: Any,
    batch: jnp.array,
    rng_key: jax.random.PRNGKey = None,
    accum_steps: int = 8,
):

    def get_minibatch(batch, grad_idx):
        return jax.tree_util.tree_map(
            lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False, axis=1),
            batch,
        )

    def loss_fn(params, batch):
        _, loss = state.apply_fn(
            {"params": params["params"]},
            x=batch,
            labels=batch,
            train=True,
            rngs={"dropout": rng_key},
        )
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

    def loss_and_grad(grad_idx):
        minibatch = get_minibatch(batch, grad_idx) if grad_idx is not None else batch

        loss, grads = grad_fn(state.params, minibatch)

        return loss, grads

    init_minibatch = (0.0, jax.tree_util.tree_map(jnp.zeros_like, state.params))

    # accumulate gradients
    def cumul_minibatch_step(grad_idx, cumul_loss_grad):
        cumul_loss, cumul_grads = cumul_loss_grad
        loss, grads = loss_and_grad(grad_idx)
        cumul_loss, cumul_grads = jax.tree_util.tree_map(
            jnp.add, (cumul_loss, cumul_grads), (loss, grads)
        )

        return cumul_loss, cumul_grads

    loss, grads = jax.lax.fori_loop(
        0,
        accum_steps,
        cumul_minibatch_step,
        init_minibatch,
    )

    loss, grads = jax.tree_util.tree_map(lambda x: x / accum_steps, (loss, grads))

    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")

    # only update train_state at the end of a single full batch
    new_state = state.apply_gradients(
        grads=grads,
    )

    metrics = {
        "Train LM Loss": loss,
        "Train LM PPL": jnp.exp(loss),
    }

    return new_state, metrics
```

This function is a lot longer! Let's break down the additions:

```python
def get_minibatch(batch, grad_idx):
        return jax.tree_util.tree_map(
            lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False, axis=1),
            batch,
        )
```

This code uses ```jax.lax.dynamic_index_in_dim``` to pull out the training batch at index = ```grad_idx```. The ```axis=1``` argument is used to tell jax we are iterating over the 2nd axis. In our above example, calling ```get_minibatch(batch, 0)``` would return a batch of data with size ```(2,2048)```.

```python 
def loss_and_grad(grad_idx):
    minibatch = get_minibatch(batch, grad_idx) if grad_idx is not None else batch

    loss, grads = grad_fn(state.params, minibatch)

    return loss, grads
```

Computes the loss for a specific index of data, we will apply this function in a loop over our gradient accumulation axis. 


```python
init_minibatch = (0.0, jax.tree_util.tree_map(jnp.zeros_like, state.params))

# accumulate gradients
def cumul_minibatch_step(grad_idx, cumul_loss_grad):
    cumul_loss, cumul_grads = cumul_loss_grad
    loss, grads = loss_and_grad(grad_idx)
    cumul_loss, cumul_grads = jax.tree_util.tree_map(
        jnp.add, (cumul_loss, cumul_grads), (loss, grads)
    )

    return cumul_loss, cumul_grads
```

This takes in a tuple of cumulative loss and gradients, computes the loss and gradient at the current index and adds it to the respective cumulative values. 


```python 
loss, grads = jax.lax.fori_loop(
        0,
        accum_steps,
        cumul_minibatch_step,
        init_minibatch,
    )
```

A jax-optimized for loop which applies ```cumul_minibatch_step``` over all microbatches and returns the *summed* losses and gradients. 

```
loss, grads = jax.tree_util.tree_map(lambda x: x / accum_steps, (loss, grads))
```
Scales the gradient and loss by the total accumulation steps. 

With these modifications, a smart reordering to function logics means that have reduced the total number of collective operations by a factor of ```grad_accum_steps```!

## 3 Benchmarks

More benchmarks to come (pending TPU availability)

#### TPU V2-8 

125M Param Causal Transformer with sequence length of 512:

| Batch Size | Accumulation Steps | Naive Speed (s) | Optimized Speed (s) | Speedup |
|------------|--------------------|-----------------|---------------------|---------|
| 512        | 32                 | 3.4460          | 2.0715              | 39%     |
| 512        | 64                 | 5.1833          | 2.3332              | 55%     |
