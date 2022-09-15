# transformer

little-GPT replication in JAX

TPU Development and training supported with Cloud TPUs from Google's TPU Research Cloud (TRC)


TODOS:
1. Windowed attention 
2. Ability to port weights (from GPT-354)
3. Logging other information (wandb id, etc for easy restore)

Completed:

Transformer


TODOS: 
- Does LR scheduling match grad accum steps?

# Testing

```bash 
python -m pytest
```