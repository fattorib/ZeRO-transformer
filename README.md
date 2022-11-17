# Transformer - JAX

JAX codebase for distributed training of language models in flax using ```pmap``` or ```pjit```. This repo is currently a work in progress. The end goal is a well-documented and efficient training script for Transformers.

# Todos
- Clean up current code
- Extend ```main_pjit.py```
- Optimized GPT model class for better sharding 
- Write everything up in ```articles```

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

device: # dimensions for ('dp', 'mp')
  dp_devices: 
  mp_devices: 
```


## Training 

```pmap```:

```bash 
python main.py
```

```pjit``` (unoptimized):

```bash 
python main_pjit.py 
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

