import jax
from flax.core.frozen_dict import unfreeze
from flax.serialization import msgpack_serialize
from flax.training import checkpoints


def flatten(p, label=None):
    if isinstance(p, dict):
        for k, v in p.items():
            yield from flatten(v, k if label is None else f"{label}.{k}")
    else:
        yield (label, p)


def params_from_trainstate(state, out_path):
    """
    Helper function to extract the raw pytree params from a trainstate. Extracted
    state is saved to msgpack for easy restoration

    Saved file can be restored with

    ```
    with open(out_path, 'rb') as f:
        pytree = msgpack_restore(f.read())
    ```

    """

    params = state["params"]
    del state
    param_bytes = msgpack_serialize(unfreeze(params))

    with open(out_path, "wb") as f:
        f.write(param_bytes)


ckpt_path = [
    f"checkpoints_checkpoint_{i}" for i in [106503, 107003, 107503, 108003, 108503]
]

for ckpt in ckpt_path:

    state = checkpoints.restore_checkpoint(
        ckpt_dir="checkpoints",
        target=None,
        prefix=ckpt,
    )
    idx = ckpt.split("_")[-1]

    params_from_trainstate(state, out_path=f"checkpoints/model_params_{idx}.msgpack")
