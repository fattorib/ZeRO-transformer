import numpy as np
import torch
from flax.serialization import msgpack_restore


def create_transformer_block_mapping(block_idx: int):
    """
    Creates required flax -> PyTorch mapping for a specific transformer block
    """

    torch_flax_dict = {
        "CausalAttention_0.key_proj.kernel": f"blocks.{block_idx}.attn.key.weight",
        "CausalAttention_0.value_proj.kernel": f"blocks.{block_idx}.attn.value.weight",
        "CausalAttention_0.query_proj.kernel": f"blocks.{block_idx}.attn.query.weight",
        "CausalAttention_0.key_proj.bias": f"blocks.{block_idx}.attn.key.bias",
        "CausalAttention_0.value_proj.bias": f"blocks.{block_idx}.attn.value.bias",
        "CausalAttention_0.query_proj.bias": f"blocks.{block_idx}.attn.query.bias",
        "CausalAttention_0.residual_out.kernel": f"blocks.{block_idx}.attn.fc_resid.weight",
        "CausalAttention_0.residual_out.bias": f"blocks.{block_idx}.attn.fc_resid.bias",
        "MLPBlock_0.fc_in.kernel": f"blocks.{block_idx}.mlp.fc1.weight",
        "MLPBlock_0.fc_in.bias": f"blocks.{block_idx}.mlp.fc1.bias",
        "MLPBlock_0.fc_residual.kernel": f"blocks.{block_idx}.mlp.fc_resid.weight",
        "MLPBlock_0.fc_residual.bias": f"blocks.{block_idx}.mlp.fc_resid.bias",
        "LayerNorm_0.scale": f"blocks.{block_idx}.ln1.weight",
        "LayerNorm_0.bias": f"blocks.{block_idx}.ln1.bias",
        "LayerNorm_1.scale": f"blocks.{block_idx}.ln2.weight",
        "LayerNorm_1.bias": f"blocks.{block_idx}.ln2.bias",
    }

    return torch_flax_dict

def flatten(p, label=None):
    if isinstance(p, dict):
        for k, v in p.items():
            yield from flatten(v, k if label is None else f"{label}.{k}")
    else:
        yield (label, p)


def match_transformer_block(pytree, state_dict, block_idx):
    """
    Performs matching at an individual block level.

    Takes in params['params'] and performs the mapping for
    the required block
    """

    block_mappings = create_transformer_block_mapping(block_idx)

    flattened_block = dict(flatten(pytree["params"][f"TransformerBlock_{block_idx}"]))

    for key, value in flattened_block.items():

        pytorch_block_key = block_mappings[key]

        if value.ndim > 1:
            # Not an LN or bias parameter, tranpose the weight
            value = np.transpose(value, (1, 0))
        state_dict[pytorch_block_key] = torch.from_numpy(np.array(value))

    return state_dict


def match_and_save(model: torch.nn.Module, include_bias: bool, flax_save_path: str, out_save_path: str):
    """
    Top-level function which performs matching for all blocks, including wte/LN

    Example Use
    ```
    >>> model = model_getter("flax-distill", vocab_size=50304, num_ctx=2048)
    >>> match_and_save(model,
            "checkpoints/small.msgpack",
            "checkpoints/torch_small.pth")
    ```
    """

    with open(flax_save_path, "rb") as f:
        pytree = msgpack_restore(f.read())

    state_dict = model.state_dict()

    for block_idx in range(model.N):
        state_dict = match_transformer_block(pytree, state_dict, block_idx)

    # Manually set top-layer weights
    state_dict["norm.weight"] = torch.from_numpy(
        np.array(pytree["params"]["LayerNorm_0"]["scale"])
    )

    if include_bias:
        state_dict["norm.bias"] = torch.from_numpy(
            np.array(pytree["params"]["LayerNorm_0"]["bias"])
        )
    state_dict["wte.weight"] = torch.from_numpy(
        np.array(pytree["params"]["wte"]["embedding"])[
            : model.vocab_size,
        ]
    )
    state_dict["lm_head.weight"] = torch.from_numpy(
        np.array(pytree["params"]["wte"]["embedding"])[
            : model.vocab_size,
        ]
    )

    if model.head_qk_trick:
        state_dict["head_q.weight"] = np.transpose(torch.from_numpy(
        np.array(pytree["params"]["head_q"]["kernel"])
        ), (1, 0))

        state_dict["head_k.weight"] = np.transpose(torch.from_numpy(
        np.array(pytree["params"]["head_k"]["kernel"])
        ), (1, 0))

    model.load_state_dict(state_dict)

    torch.save(model.state_dict(), out_save_path)
