import argparse
import os 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
from flax.core.frozen_dict import unfreeze
from flax.serialization import msgpack_serialize
from flax.training import checkpoints

def parse():
    parser = argparse.ArgumentParser(description="Convert Flax to msgpack")

    parser.add_argument("--ckpt-dir", type=str)  # path of flax msgpack object
    parser.add_argument("--prefix", type=str)  # prefix of saved checkpoint

    args = parser.parse_args()
    return args

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

def main():

    args = parse()

    state = checkpoints.restore_checkpoint(
        ckpt_dir=args.ckpt_dir,
        target=None,
        prefix=args.prefix,
    )
    idx = args.prefix.split("_")[-1]
    params_from_trainstate(state, out_path=f"{args.ckpt_dir}/model_params_{idx}.msgpack")


if __name__ == '__main__':

    main()

