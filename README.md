# transformer

little-GPT replication in JAX

TPU Development and training supported with Cloud TPUs from Google's TPU Research Cloud (TRC)


TODOS:
1. Windowed attention 
2. Ability to port weights (from GPT-354)

The dataset scripts here assume your dataset has been processed into a collection of tar archives that can be read with [webdataset](https://github.com/webdataset/webdataset). Once that is done change the required paths in ```conf/config.yaml```:

```yaml
data:
    corpus: "openwebtext" #for logging corpus name in Weights and Biases
    train_shard_urls: "data/processed/books_train-{000000..000013}.tar.gz" # required if not using TPUs
    validation_shard_urls: "data/processed/bookcorpus_val-{000000..000002}.tar.gz" # required if not using TPUs
    max_context: 1024 # maximum sequence length in tokens
    workers: 1 # number of workers for PyTorch dataloader 
    checkpoint_directory: "checkpoints"
    bucket_path: "bfattoribooks2" #bucket path if training with TPUs
    index_path_train: "data/index/openwebtext.train.index" # list of all shards + GCP urls
    index_path_validation: "data/index/openwebtext.val.index" # list of all shards + GCP urls
```





# Testing

```bash 
python -m pytest
```