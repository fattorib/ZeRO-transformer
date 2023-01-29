# Table of Contents

1. [ZeRO Optimizer Sharding with jax.pmap](#zero-optimizer-sharding-with-jax.pmap)

2. [Configuration Setup](#configuration-setup)

3. [Training](#training)

4. [Trained Models](#trained-models)

5. [How This Works](#how-this-works)

6. [Acknowledgements](#acknowledgements)

7. [TODO](#todo)

# ZeRO Optimizer Sharding with jax.pmap

JAX codebase demonstrating an application of [ZeRO](https://arxiv.org/abs/1910.02054)-style optimizer sharding using only ```jax.pmap```. This codebase was used to train a 1.1B parameter transformer model on a TPU v3-32, something that would not be possible with standard data parallel training[^1].

# Configuration Setup

## Model Config

Add your model config to ```conf/model_config.yaml```:

```yaml
model_name:
  embedding_dim: 
  vocab_size: 
  num_head: 
  block_size: # maximum context length 
  dropout: 
  N: 
  alibi_attn: # bool for using ALiBi attention 
```

## Training Config

All other configuration is handled in ```conf/config.yaml```:

```yaml
training:
  max_epochs: # maximum epochs to train for
  batch_size: # per-host batch size.
  peak_learning_rate: # maximum learning rate, follows cosine decay
  end_learning_rate: # minimum learning rate
  warmup_steps: # warmup steps for learning rate
  total_steps: # total steps to train for
  weight_decay: # weight decay
  gradient_accumulation_steps: # gradient accumulation steps 
  evaluation_frequency: # interval to evaluate on validation set
  maximum_evaluation_steps: # maximum evaluation steps
  staged_warmup_steps: # if using sequence length warmup, number of steps @ shorter CTX
  warmup_train_context: # if using sequence length warmup, defines shorter CTX

model:
  size: # model size. Must be defined in model_config.yaml
  warm_start: # bool to determine whether to start from a pretrained ckpt
  model_path: # if warm starting, path of saved pretrained ckpt

data:
  corpus: # metadata for logging corpus name
  max_context: # maximum context of processed samples
  train_samples: # metadata, number of processed samples in train corpus
  checkpoint_directory: # ckpt dir 
  bucket_path: # GCP bucket dir
  index_path_train: # path to .index files with list of train data urls
  index_path_validation: # path to .index files with list of val data urls
  wandb_project: # weights and biases project name
  resume_step: # if resuming training, step to resume from
```

## Training

This assumes you have your data setup on a GCP bucket and .index files created for your datasets. In addition, you must have a running TPU VM instance and be connected to it.

```bash
python main_zero.py
```

If resuming a run, pass the ```--resume``` flag to your script.

## Trained Models

TODO

## How This Works

### Code Overview

The canonical flax ```train_step``` (ex: [here](https://github.com/google/flax/blob/main/examples/imagenet/train.py#L106)) is split into two separate pmapped functions. One to compute the loss and gradients as normal and the other to update and sync the sharded optimizer state.

The main training logic is then as follows:

```python
...
# compute grads and loss
grads, metrics = train_step(
            params, batch, model
        )

# reshape gradients and params to match the sharding of the optimizer state
grads = split_sharded_device_array(
            grads, jax.numpy.arange(jax.local_device_count())
        )
params = split_sharded_device_array(
            params, jax.numpy.arange(jax.local_device_count())
        )

# update sharded optimizer state and params
params, opt_state = update_sharded_state(
            grads,
            opt_state,
            params,
            optimizer,
        )

# reshape params for next forward pass
params = deshard(params)  
```

In more detail:

- ```train_step``` is the regular pmapped train_step you would expect. We compute loss/grads on each TPU and then use ```jax.lax.pmean``` to synchronize them before returning.

- ```split_sharded_device_array``` is a pmapped function that reshapes all params to add an extra "device" dimension and then selects the relevant "slice" of parameters for that device. We do this on both params and gradients to ensure that the optimizer update does not require multiple communication steps.

- ```update_sharded_state``` is a pmapped function that takes the parameter and gradient slices and performs the optimizer update for the specific slice. After updating, we use ```jax.lax.all_gather``` to synchronize the updated slices of all params.

- ```deshard``` simply reshapes the new params so they are ready for the next forward pass.

To set the partitioned optimizer states up, we use ```partition_shard``` which works like ```split_sharded_device_array``` by adding an extra "device" dimension. To avoid OOM, we compute this on the CPU (where we have plenty of memory) before sending the param slices to their respective devices through a pmapped identity function:

```python
...
opt_state = jax.device_get(opt_state)  # copy opt_state to VM CPU
opt_state = partition_shard(
        opt_state,
        jax.local_device_count(),
        jax.local_devices(),
    )
opt_state = jax.pmap(lambda x: x, devices=jax.local_devices())(
    opt_state
)  # shard opt state to free up memory
```

### Limitations

- **Extending to Multi-host**: While the above code works and is relatively performant, extending the optimizer partitioning *across hosts* would be more complicated.```pmap``` can be run in multihost environments (such as TPU pods), but the pmapped functions themselves only communicate with other hosts through collective operations (```pmean```, ```psum```, etc). Outside of these operations, they only see local copies of the functions and data. See [here](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html) for more detail.

- **More Complicated Partitioning**: The first ZeRO paper extends their sharding strategy beyond just the optimizer state to include partitioning of gradients and model params across the data parallel processes. This is certainly possible to do with ```pmap```, but it would be much easier to accomplish with something like ```pjit``` or ```xmap```.

## TPU Setup

```bash
git clone https://github.com/fattorib/transformer.git
cd transformer 
bash prepareTPUVM.sh
```

**Note** Due to the jax.Array [changes](https://jax.readthedocs.io/en/latest/jax_array_migration.html#jax-array-migration) in Jax 0.4.1, JAX pre-0.4.1 must be used. JAX 0.3.25 is sufficient and the pmapped code has been tested up to this version.

# Acknowledgements

TPU Development and training supported with Cloud TPUs from Google's TPU Research Cloud (TRC)

# TODO

- [] Upload trained models to public storage
- [] Add text section for ```app.py```

[^1]: Naively, a 1.1B param model requires ~16.4 GiB to store the params and optimizer states alone, which is already more than the 16.0 GiB of memory each TPU V3 core has access to. The total memory requirements during training are even higher due to the need for storing activations + misc. temporary buffers.
