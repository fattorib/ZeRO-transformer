# Transformer - JAX

JAX codebase for distributed training of language models in flax using ```pmap``` or ```pjit```. This repo is currently a work in progress. The end goal is a well-documented and efficient training script for Transformers.

# Todos
- Clean up current code
- Extend ```main_pjit.py```
- Optimized GPT model class for better sharding 
- Write everything up in ```articles```

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

