# ZeRO Optimizer Sharding with xmap and pjit 

TL;DR I improve upon my earlier codebase by implementing ZeRO-1 optimizer sharding using a combination of ```xmap``` and ```pjit```. The resultant method is more performant and scales better across TPUs. I use this method to train a 1.3B parameter decoder-only transformer on 200B tokens.

## Background

For the past 4 or 5 months, I have been spending time learning more about jax and all easy-to-use parallelism APIs it offers. The first large project I completed was an implementation of ZeRO-1 (CITE) optimizer sharding using the ```pmap``` operator. Using this, I was able to train a 1.1B parameter decoder-only transformer on a TPU v3-32, by using 8-way optimizer state sharding on each of the TPU hosts. 

While this project was successful, using ```pmap``` requires a lot of manual array handling to ensure that everything passed into a pmapped function can be distributed across the local devices. In my code, this resulted in multiple helper functions solely for sharding/unsharding arrays. In retrospect, these functions somewhat of a performance bottleneck and were (XYZ). In addition, without some complicated, and error-prone, communication code, extending the optimizer sharding across hosts would be very difficult. (Add footnote exclaiming that Gopher was trained solely using pmap)

## ZeRO Optimizer Sharding Overview 

## Going forward - pjit everything? 

I recently got access to some more TPU compute and wanted to extend my previous code to address some of the pain points mentioned above. Originally, I had planned to only use pjit to accomplish this. In theory, this is quite easy to do, we specify the ```PartitionSpec``` for the optimizer pytree, duplicate that for the gradient pytree, ensure the batches are split across the same axis and then we're good to go!

In practice, however, I found that naively using pjit this way resulted in significant throughput decreases. I suspect that the pjit-compilied function was forcing a gradient all-reduce every accumulation step, instead of once every batch, decreasing the training throughput by roughly a factor of ```gradient_accumulation_steps```. However, I have been unable to alleviate this issue even after adding in shard annotations with ```with_sharding_constraint```. I suspect that I still have a mistake in my code somewhere. I'm currently going over how [jaxformer](https://github.com/salesforce/jaxformer) implements their gradient accumulation code and am hopeful following how [they do](https://github.com/salesforce/jaxformer/blob/main/jaxformer/models/decoder/inter/model.py#L214) it will work.

## xmap to the rescue!

Thankfully, there is a second jax parallelism API that is intended to shard arrays across multiple hosts: ```xmap```. The main between ```xmp``` and ```pjit``` that we are interested in is that ```xmap``` compiled code still requires the user to specify when collective operations (ex: ```pmean```, ```psum```, ```all_gather```) occur, unlike ```pjit``` which will include these whenever the compiler deems necessary. From this perspective, ```xmap``` can be seen as a generalization of ```pmap```, which also requires that users specify when they want collective operations to be applied. 

One nice upgrade that ```xmap``` has over ```pmap``` is its reliance on named axes. By specifying a list of named axes to xmap, we can control how inputs and outputs to an xmapped function are sharded as well as how and when operations such as ```pmean``` are applied. Most importantly, xmapped functions compose properly with pjitted functions, something I was not able to get working with pmap. 

## Putting it all together: xmap -> pjit

Because xmap and pjit arrays are of the same type, it is possible to pass the output from an xmapped function into a pjitted function. The communication pattern for the gradient accumulation (```train_step```) code is quite simple: we iterate over ```gradient_accumulation_steps``` microbatches and have every TPU core compute its own local set of gradients, once that is completed, we use a single ```pmean``` to synchronize gradients across all TPU cores and we're done!

From here, we can take these output gradients and shard them to match the ```PartitionSpec``` of the optimizer states. Remember from above, we've distributed these across all available TPU cores (the easiest way to do this is to just use ```ParitionSpec('dp', None)``` or ```ParitionSpec('dp')``` to shard the first axis of all weights across the data parallel axis, in this case denoted by ```'dp```. In the end, the resulting code is quite compact:

```python 
in_axes =(
    [...], 
    ['batch', ...], 
    [...], 
    )

out_axes = (
    [...],
    [...]
)
# compute gradients with standard data-parallel
grads, metrics = xmap(
    partial(train_step, model = model, accum_steps = GRAD_ACCUM_STEPS),
    in_axes=in_axes,
    out_axes=out_axes,
    axis_resources={"batch": "dp"}
)(params, batch, dropout_rng)   

grads = pjit(lambda x:x, in_axis_resources=None, out_axis_resources=grad_param_spec)(grads)
params = pjit(lambda x:x, in_axis_resources=None, out_axis_resources=grad_param_spec)(params)

# each dp process updates their own copy of the optimizer state before updating params and 
# performing an all-gather to sync params
new_params,new_opt_state = pjit(
    functools.partial(update_opt_state, optimizer = tx, grad_spec = grad_param_spec),
    in_axis_resources=(grad_param_spec, opt_state_spec, grad_param_spec),
    out_axis_resources=(None,opt_state_spec),
)(grads, opt_state, params)
```

In comparison to my original code, this is a lot more simple. Specifying proper in/out axis resources ensures I don't need to write any of pytree reshaping code which can eat up device memory and cause slowdowns. The only manual sharding we do is the pjitted identity functions to shard the gradients and paramaters to the correct processes for the optimizer update. (improve paragraph here)

## Training a 1.3B Parameter Decoder-only Transformer

(Training Log!)

Main points to get across:

- Model size/shape
- Training dataset, using repeated tokens to save time/$$
- Hardware, TPU v3-32, etc etc 
- Downstream benchmarks, performance on tasks, etc etc 
- 