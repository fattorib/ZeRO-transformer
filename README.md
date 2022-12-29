# Transformer - JAX

JAX codebase demonstrating an application of [ZeRO](https://arxiv.org/abs/1910.02054)-style optimizer sharding with ```pmap```. This codebase was used to train a 1.1B parameter transformer model on a TPU v3-32, something that would not be possible with standard data parallel training. 

# Todos:
1. Finish 1.1B training 
2. 1.1B training log
3. Finish readme install instructions, etc 
4. Write up how everything works 
5. Drop benchmark/unused training scripts

# Configuration Setup

## Models

Add your model config to ```conf/model_config.yaml```:

```yaml
model_name:
  embedding_dim: 
  vocab_size: 
  num_head: 
  block_size: # maximum context length 
  dropout: 
  N: 
  fused_residuals: # bool for fusing attn + mlp
  alibi_attn: # bool for using ALiBi attention 
```

## Training 
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

