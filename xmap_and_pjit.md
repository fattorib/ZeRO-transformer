# ZeRO Optimizer Sharding with xmap and pjit 

TL;DR I improve upon my earlier codebase by implementing ZeRO-1 optimizer sharding using a combination of ```xmap``` and ```pjit```. The resultant method is more performant and scales better across multiple TPU hosts, achieving 67% MFU on a TPU v3-32. I use this method to train a 1.3B parameter decoder-only transformer on 200B tokens.

## Background

For the past 4 or 5 months, I have been spending time learning more about jax and the parallelism APIs it offers. The first large project I completed was an implementation of [ZeRO-1](https://arxiv.org/abs/1910.02054) optimizer sharding using the ```pmap``` operator. Using this, I was able to train a 1.1B parameter decoder-only transformer on a TPU v3-32, by using 8-way optimizer state sharding on each of the TPU hosts. 

While this project was successful, using ```pmap``` requires a lot of manual array handling to ensure that everything passed into a pmapped function can be distributed across the local devices. In my code, this resulted in multiple helper functions solely for sharding/unsharding arrays. In retrospect, these functions were more of a performance bottleneck than I had originally expected. In addition, without some complicated, and error-prone, communication code, extending the optimizer sharding across hosts would be very difficult. (Add footnote exclaiming that Gopher was trained solely using pmap)

## ZeRO Optimizer Sharding Overview 

ZeRO, introduced in the paper "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" by Rajbhandari et al stands for Zero Redundancy Optimizer. The authors of the paper propose a modification to dataparallel training that reduces memory usage my splitting the buffers of the model's optimizer across all the data parallel ranks. By implementing custom communications, they are able to retain the high throughput of data parallel training while training larger models than were previously possible. The original paper introduces multiple levels of ZeRO: ZeRO-1 shards the optimizer states, ZeRO-2 also shards the gradients as well, and finally, ZeRO-3 shards the model parameters themselves across data parallel ranks. 


## Going forward - pjit everything? 

I recently got access to some more TPU compute and wanted to extend my previous code to address some of the pain points mentioned above. Originally, I had planned to only use pjit to accomplish this. In theory, this is quite easy to do, we specify the ```PartitionSpec``` for the optimizer pytree, duplicate that for the gradient pytree, ensure the batches are split across the same axis and then we're good to go!

In practice, however, I found that naively using pjit this way resulted in significant throughput decreases. I suspect that the pjit-compilied function was forcing a gradient all-reduce every accumulation step, instead of once every batch, decreasing the training throughput by roughly a factor of ```gradient_accumulation_steps```. However, I have been unable to alleviate this issue even after adding in shard annotations with ```with_sharding_constraint```. I suspect that I still have a mistake in my code somewhere. I'm currently going over how [jaxformer](https://github.com/salesforce/jaxformer) implements their gradient accumulation code and am hopeful following how [they do](https://github.com/salesforce/jaxformer/blob/main/jaxformer/models/decoder/inter/model.py#L214) it will work.

## xmap to the rescue!

Thankfully, there is a second jax parallelism API that is intended to shard arrays across multiple hosts: ```xmap```. The main between ```xmp``` and ```pjit``` that we are interested in is that ```xmap``` compiled code still requires the user to specify when collective operations (ex: ```pmean```, ```psum```, ```all_gather```) occur, unlike ```pjit``` which will include these whenever the compiler deems necessary. From this perspective, ```xmap``` can be seen as a generalization of ```pmap```, which also requires that users specify when they want collective operations to be applied. 

One nice upgrade that ```xmap``` has over ```pmap``` is its reliance on named axes. By specifying a list of named axes to xmap, we can control how inputs and outputs to an xmapped function are sharded as well as how and when operations such as ```pmean``` are applied. Most importantly, xmapped functions compose properly with pjitted functions, something I was not able to get working with pmap. 

## Putting it all together: xmap & pjit

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

In comparison to my original code, this is much more simple. Specifying proper in/out axis resources ensures I don't need to write any pytree reshaping code which I empirically found eats up device HBM and causes slowdowns. The only manual sharding we do is the pjitted identity functions to shard the gradients and paramaters to the correct processes for the optimizer update, which is quick.

## Performance Benchmarks

On a TPU v3-32, training a 1.3B parameter model with BF16 mixed-precision, a sequence length of 1024 tokens and a global batch size of 512, the code processes one batch every 1.65 seconds. This converts to 2.638 PFLOP/s or 67% Model FLOPs Utilization (MFU). (TPU Max TFLOPs calculated by multiplying [peak compute per chip](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_v3) by 32.) 

## Training a 1.3B Parameter Decoder-only Transformer

To prove out that my method works, I trained a 1.3B parameter language model on 200B tokens from The Pile(CITE). Naively, the model requires more than ~17 GiB to store the params and optimizer states alone, which is already more than the 16.0 GiB of memory each TPU V3 core has access to. 

I selected The Pile as my training dataset as it is a large and high-quality corpus of text that is easily available. To save on storage and processing costs, I only trained on a small subset of the pile, specifically ```00.json.zst``` to ```03.jsonl.zst```, from [here](https://the-eye.eu/public/AI/pile/train/). The total pile was shuffled prior to being split into the files I downloaded, so we get a representative sample of the data. The text was tokenized using the Byte-Level [GPTNeoX tokenizer](https://huggingface.co/docs/transformers/model_doc/gpt_neox#transformers.GPTNeoXTokenizerFast). Sequences were tokenized and an end-of-text token was appended to the end of documents. The total dataset consists of approximately 30B tokens. The pile validation file [here](https://the-eye.eu/public/AI/pile/) was used for the validation set. 


### Model

| Hyperparameter      | Value |
|---------------------|-------|
| n_parameters        | 1.3B  |
| n_layers            | 24    |
| d_model             | 2048  |
| d_ff                | 8192  |
| num_head            | 16    |
| d_head              | 128   |
| vocab_size          | 50304 |
| Positional Encoding | ALiBi |
| n_ctx               | 1024  |


### Training

The model was trained for 200B tokens with a batch size of ~0.5M tokens with the following hyperparameters:

| Hyperparameter       | Value        |
|----------------------|--------------|
| Batch Size           | 0.5M Tokens  |
| Peak Learning Rate   | 2.0e-4       |
| Warmup Steps         | 2000         |
| Residual Dropout     | 0.1          |
| Attention Dropout    | 0.1          |
| Embedding Dropout    | 0.0          |
| Precision            | bfloat16     |
| Weight Decay         | 0.1          |
| Optimizer            | AdamW        |
| Schedule             | Cosine to 10%|

### Benchmarks



Main points to get across:

- Model size/shape
- Training dataset, using repeated tokens to save time/$$
- Hardware, TPU v3-32, etc etc 
- Downstream benchmarks, performance on tasks, etc etc 
- 

## Acknowledgements

TPU Development and training supported with Cloud TPUs from Google's TPU Research Cloud (TRC). Thank you to the excellent TRC team for granting me access to upgraded TPU VMs and for the extensions I received while working on this project!

