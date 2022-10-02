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

# Supported Features

## ALiBi

ALiBi (Attention with Linear Biases) modifies the attention matrix by adding in a position-dependant bias matrix before the softmax layer. Empirically, this has been shown to allow GPT-style transformer models to *retain* their performance on sequences longer than what they were trained on (usually you will see minimal PPL improvements when increasing the context length, this is still better than other relative position embeddings and it is quite fast). In your ```model_config``` file, set ```alibi_attn``` to ```True```. **Enabled by default**

## Channel-Split SGU

*I'm unable to find the paper that introduces this idea but I know it exists* Channel split SGU takes each token embedding and splits the total embedding dimensions in two. Half of the embeddings are attented to normally with MHA and the other half of the embeddings are fed through a fixed causal SGU with a relatively small kernel. In theory, the short-range mixing from the SGU kernel should allow the attention part of the embedding to attend to longer-range dependancies as the short range dependances are handled automatically through the SGU. In your ```model_config``` file, set ```use_static_sgu``` to ```True```. **Disabled by default**

## Head-QK Trick
See [here](https://github.com/BlinkDL/RWKV-LM#the-head-qk-trick-learning-to-copy-and-avoid-tokens) for an explanation. In your ```model_config``` file, set ```head_qk_trick``` to ```True```. **Disabled by default**

## Parallel Residual Blocks

Introduced in [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax). Training and inference throughput increase at the cost of minor performance degradation. In your ```model_config``` file, set ```fused_residuals``` to ```True```. **Enabled by default**

# Experiments:

See ```writeup.md```

# Training 

# TODOS:
1. Ability to port weights from [Little-GPT](https://github.com/fattorib/Little-GPT)
2. Caching previously generated states for faster sequence decoding

# Acknowledgements
TPU Development and training supported with Cloud TPUs from Google's TPU Research Cloud (TRC)

# Text Generation

For now, we only support greedy decoding, and temperature-based sampling. Generation is very slow as I haven't implemented caching of past states yet. To sample from a trained model:

```python 
# load model and state
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
sample_text = "This is a sample text sentence that"
tokenized_text = tokenizer.encode(sample_text)
generation_rng = jax.random.PRNGKey(23)
out = model.generate(
    state_params,
    tokenized_text,
    max_length=100,
    temperature=0.7,
    sample=True,
    sample_rng=generation_rng,
)
tokenized_text_out = tokenizer.decode(out)
```

# Testing
```bash 
python -m pytest
```