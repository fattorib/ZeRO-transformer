# ZeRO Optimizer Sharding with jax.pmap

JAX codebase demonstrating an application of [ZeRO](https://arxiv.org/abs/1910.02054)-style optimizer sharding using only ```jax.pmap```. This codebase was used to train a 1.1B parameter transformer model on a TPU v3-32, something that would not be possible with standard data parallel training.

# Todos

- [ ] Finish readme install instructions, etc
- [ ] Write up how everything works

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

## TPU Setup

```bash
git clone https://github.com/fattorib/transformer.git
cd transformer 
bash prepareTPUVM.sh
```

# Acknowledgements

TPU Development and training supported with Cloud TPUs from Google's TPU Research Cloud (TRC)
