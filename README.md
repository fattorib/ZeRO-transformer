# Transformer - JAX

JAX codebase demonstrating an application of [ZeRO](https://arxiv.org/abs/1910.02054)-style optimizer sharding with ```pmap```. This codebase was used to train a 1.1B parameter transformer model on a TPU v3-32, something that would not be possible with standard data parallel training. 

# Todos: 
1. 1.1B training log
2. Finish readme install instructions, etc 
3. Write up how everything works 

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
Pretty much all other configuration is handled in ```conf/config.yaml```:

```yaml
TODO
```


## Training 

```pmap```:

```bash 
python main_zero.py
```


## TPU Setup

```bash
git clone https://github.com/fattorib/transformer.git
cd transformer 
bash prepareTPUVM.sh
```

# Acknowledgements
TPU Development and training supported with Cloud TPUs from Google's TPU Research Cloud (TRC)


# Testing
```bash 
make test
```

