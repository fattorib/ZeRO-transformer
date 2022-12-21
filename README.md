# Transformer - JAX

JAX codebase demonstrating an application of [ZeRO](https://arxiv.org/abs/1910.02054)-style optimizer sharding with ```pmap```. This codebase was used to train a 760M parameter transformer model on a v3-32, something that would not be possible with standard data parallel training. 

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
training:
  max_epochs: 
  batch_size: 
  peak_learning_rate: 
  warmup_steps: 
  decay_steps: 
  total_steps: 
  end_learning_rate: 
  weight_decay: 
  gradient_accumulation_steps: 
  evaluation_frequency: 
  maximum_evaluation_steps: 
  precision: 'bf16' # can be ['fp32', 'fp16', 'bf16']
  staged_warmup_steps: # number of steps to perform at a shortened context
  train_context: # context used during sequence length warmup

model:
  size: 

data:
  corpus: 
  train_shard_urls:
  validation_shard_urls: 
  max_context: 
  full_steps_in_batch: 
  checkpoint_directory: 
  bucket_path:
  index_path_train: 
  index_path_validation: 
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
python -m pytest
```

