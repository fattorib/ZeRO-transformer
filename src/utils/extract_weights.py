import jax
from flax.core.frozen_dict import unfreeze
from flax.serialization import msgpack_restore, msgpack_serialize
from flax.training import checkpoints

from src.models.GPT import model_getter
from src.training.training_utils import create_train_state

state = checkpoints.restore_checkpoint(
    ckpt_dir="checkpoints",
    target=None,
    prefix="temporary_checkpoint_109674",
)


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


params_from_trainstate(state, out_path="checkpoints/model_params_only.msgpack")
