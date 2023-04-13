# Table of Contents

1. [ZeRO Optimizer Sharding in Jax](#zero-optimizer-sharding-in-jax)
2. [Configuration Setup](#configuration-setup)
3. [Training](#training)
4. [Trained Models](#trained-models)
5. [Acknowledgements](#acknowledgements)

# ZeRO Optimizer Sharding in Jax

JAX codebase demonstrating an application of [ZeRO](https://arxiv.org/abs/1910.02054)-style optimizer sharding using a combination of ```xmap``` and ```pjit```. This codebase was used to train a 1.3B parameter transformer model on a TPU v3-32, something that would not be possible with standard data parallel training. I have a full post detailing my work which you can read [here](TODO).

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
  alibi_attn: # boolean for using ALiBi attention 
```

## Training Config

All other configuration is handled in ```conf/config.yaml```.

## Training

This assumes you have your data setup on a GCP bucket and .index files created for your datasets:

```bash
python main_zero.py
```

If resuming a run, pass the ```--resume``` flag to your script.

## Trained Models

TODO

## Tests

Tests are written in a combination of ```unittest``` and ```pytest```. All tests can be run with:
```bash
pytest
```
from the base directory.

## TPU Setup

```bash
git clone https://github.com/fattorib/transformer.git
cd transformer 
bash prepareTPUVM.sh
```

# Acknowledgements

TPU Development and training supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/). Thank you to the excellent TRC team for granting me access to upgraded TPU VMs and for the extensions I received while working on this project! 
