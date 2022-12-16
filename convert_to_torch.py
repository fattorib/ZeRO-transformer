"""
Convert trained model bytes to a PyTorch statedict. We could keep everything in 
flax, but I already have all the PyTorch inference code done and this will also 
make it a bit easier to push to HF Hub if I want to do that.
"""

from torch_compatability.flax_to_pytorch import match_and_save
from torch_compatability.GPT2 import model_getter
import argparse


def parse():
    parser = argparse.ArgumentParser(description="Convert Flax to PyTorch")

    parser.add_argument(
        "--model-name",
        type = str
    )
    parser.add_argument( # path of flax msgpack object 
        "--flax-path",
        type = str
    )
    parser.add_argument( # path to save torch statedict to 
        "--torch-path",
        type = str
    )

    parser.add_argument( # model vocab size
        "--vocab-sze",
        type = int,
        default= 50304
    )

    parser.add_argument( # maximum model context
        "--seq-len",
        type = int,
        default= 1024
    )

    args = parser.parse_args()
    return 

def main():

    args = parse()

    model = model_getter(model_name=args.model_name, vocab_size = args.vocab_size, num_ctx=args.seq_len)

    match_and_save(model, args.flax_path, args.torch_path)