import jax

from src.models.GPT import model_getter

model = model_getter("large", return_cfg=False)


rng = jax.random.PRNGKey(23)
batch_tok = jax.random.randint(rng, shape=(1, 512), maxval=50257, minval=0)
param_shape = jax.eval_shape(model.init, rng, batch_tok)

print(f"{sum(p.size for p in jax.tree_leaves(param_shape))/1e6:.2f}")
