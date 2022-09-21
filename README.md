# Transformer - JAX

Experimental codebase for TPU/GPU training of GPT-style transformers in JAX. A JAX successor to [Little-GPT](https://github.com/fattorib/Little-GPT).

## Setup

### GPU:

```bash
git clone https://github.com/fattorib/transformer.git
cd transformer 
# download your data and unpack it 
pip install --upgrade pip
pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements
```

### TPU:

```bash
git clone https://github.com/fattorib/transformer.git
cd transformer 
bash prepareTPUVM.sh
```

### Dataset

The dataset scripts here assume your dataset has been processed into a collection of tar archives that can be read with [webdataset](https://github.com/webdataset/webdataset). Once that is done change the required paths in ```conf/config.yaml```:

```yaml
data:
    corpus: "openwebtext" #for logging corpus name in Weights and Biases
    train_shard_urls: "data/processed/books_train-{000000..000013}.tar.gz" # required if not using TPUs. This is the path where you unpacked your dataset from above
    validation_shard_urls: "data/processed/bookcorpus_val-{000000..000002}.tar.gz" # required if not using TPUs. This is the path where you unpacked your dataset from above
    max_context: 1024 # maximum sequence length in tokens
    full_steps_in_batch: 24558 # Number of individual steps required to complete an epoch
    workers: 1 # number of workers for PyTorch dataloader 
    checkpoint_directory: "checkpoints"
    bucket_path: "bfattoribooks2" #bucket path if training with TPUs
    index_path_train: "data/index/openwebtext.train.index" # list of all shards + GCP urls
    index_path_validation: "data/index/openwebtext.val.index" # list of all shards + GCP urls
```

# Training 

TODO 

# TODOS:
1. Ability to port weights from [Little-GPT](https://github.com/fattorib/Little-GPT)


# Acknowledgements
TPU Development and training supported with Cloud TPUs from Google's TPU Research Cloud (TRC)


# Testing

```bash 
python -m pytest
```