"""
Convert trained model bytes to a PyTorch statedict. We could keep everything in 
flax, but I already have all the PyTorch inference code done and this will also 
make it a bit easier to push to HF Hub if I want to do that.
"""

import argparse

from torch_compatability.flax_to_pytorch import match_and_save
from torch_compatability.GPT2 import model_getter


def parse():
    parser = argparse.ArgumentParser(description="Convert Flax to PyTorch")

    parser.add_argument("--model-name", type=str)
    parser.add_argument("--flax-path", type=str)  # path of flax msgpack object
    parser.add_argument("--torch-path", type=str)  # path to save torch statedict to
    parser.add_argument("--vocab-size", type=int, default=50304)  # model vocab size
    parser.add_argument("--seq-len", type=int, default=1024)  # maximum model context

    args = parser.parse_args()
    return args


def main():

    args = parse()
    model = model_getter(
        model_name=args.model_name, vocab_size=args.vocab_size, num_ctx=args.seq_len
    )

    match_and_save(model, args.flax_path, args.torch_path)


if __name__ == "__main__":
    main()
