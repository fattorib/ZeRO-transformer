"""
Tests for PyTorch models
"""
import pytest
import torch

from torch_compatability.GPT2 import (GPT2, ALiBi, GPT2Block, MLPBlock,
                                      model_getter)


@pytest.fixture
def model_config():
    return {"ctx": 32, "embed_dim": 128, "n_head": 8, "n_layer": 2, "vocab_size": 64}


def test_mlpblock(model_config):
    # Ensure we can fwd MLPBlock
    cfg = model_config
    layer = MLPBlock(cfg["embed_dim"], 4 * cfg["embed_dim"], 0.0, 4)
    batch = torch.rand((1, cfg["ctx"], cfg["embed_dim"]))

    out = layer(batch)

    assert out.shape == batch.shape


def test_alibi_attn_nocache(model_config):
    # Ensure we can fwd ALiBi without cache
    cfg = model_config
    layer = ALiBi(
        embedding_dim=cfg["embed_dim"],
        num_head=cfg["n_head"],
        block_size=cfg["ctx"],
        resid_dropout=0.0,
        num_layers=cfg["n_layer"],
    )
    batch = torch.rand((1, cfg["ctx"], cfg["embed_dim"]))
    out = layer(batch)

    assert out[0].shape == batch.shape
    assert out[1] is None


def test_alibi_attn_cache(model_config):
    # Ensure we can fwd ALiBi with k/v cache
    cfg = model_config
    cache_size = 4

    # create a small cache of random vectors
    k_prev = torch.rand(
        (1, cfg["n_head"], cache_size, cfg["embed_dim"] // cfg["n_head"])
    )
    v_prev = torch.rand(
        (1, cfg["n_head"], cache_size, cfg["embed_dim"] // cfg["n_head"])
    )
    layer_past = torch.stack((k_prev, v_prev))

    layer = ALiBi(
        embedding_dim=cfg["embed_dim"],
        num_head=cfg["n_head"],
        block_size=cfg["ctx"],
        resid_dropout=0.0,
        num_layers=cfg["n_layer"],
    )
    batch = torch.rand((1, 1, cfg["embed_dim"]))

    out = layer(batch, use_cache=True, layer_past=layer_past)
    assert out[0].shape == batch.shape

    present_cache = out[1]
    assert present_cache.shape == (
        2,
        1,
        cfg["n_head"],
        cache_size + 1,
        cfg["embed_dim"] // cfg["n_head"],
    )

    # Ensure we can pass in the cached vectors again
    batch = torch.rand((1, 1, cfg["embed_dim"]))
    out = layer(batch, use_cache=True, layer_past=present_cache)

    present_cache = out[1]
    assert present_cache.shape == (
        2,
        1,
        cfg["n_head"],
        cache_size + 2,
        cfg["embed_dim"] // cfg["n_head"],
    )


def test_transformerblock_nocache(model_config):
    # Ensure we can fwd TransformerBlock without cache
    cfg = model_config

    layer = GPT2Block(
        embedding_dim=cfg["embed_dim"],
        num_head=cfg["n_head"],
        block_size=cfg["ctx"],
        resid_dropout=0.0,
        num_layers=cfg["n_layer"],
    )
    batch = torch.rand((1, cfg["ctx"], cfg["embed_dim"]))
    out = layer(batch)

    assert out[0].shape == batch.shape
    assert out[1] is None


def test_transformerblock_cache(model_config):
    # Ensure we can fwd TransformerBlock with k/v cache
    cfg = model_config
    cache_size = 4

    layer = GPT2Block(
        embedding_dim=cfg["embed_dim"],
        num_head=cfg["n_head"],
        block_size=cfg["ctx"],
        resid_dropout=0.0,
        num_layers=cfg["n_layer"],
    )

    # create a small cache of random vectors
    k_prev = torch.rand(
        (1, cfg["n_head"], cache_size, cfg["embed_dim"] // cfg["n_head"])
    )
    v_prev = torch.rand(
        (1, cfg["n_head"], cache_size, cfg["embed_dim"] // cfg["n_head"])
    )
    layer_past = torch.stack((k_prev, v_prev))

    batch = torch.rand((1, 1, cfg["embed_dim"]))
    out = layer(batch, use_cache=True, layer_past=layer_past)
    assert out[0].shape == batch.shape

    present_cache = out[1]
    assert present_cache.shape == (
        2,
        1,
        cfg["n_head"],
        cache_size + 1,
        cfg["embed_dim"] // cfg["n_head"],
    )

    # Ensure we can pass in the cached vectors again
    batch = torch.rand((1, 1, cfg["embed_dim"]))
    out = layer(batch, use_cache=True, layer_past=present_cache)

    present_cache = out[1]
    assert present_cache.shape == (
        2,
        1,
        cfg["n_head"],
        cache_size + 2,
        cfg["embed_dim"] // cfg["n_head"],
    )


def test_gpt2(model_config):
    cfg = model_config
    model = GPT2(
        embedding_dim=cfg["embed_dim"],
        num_head=cfg["n_head"],
        num_ctx=cfg["ctx"],
        vocab_size=cfg["vocab_size"],
        N=cfg["n_layer"],
        use_alibi=True,
    )

    # test with no labels
    batch = torch.ones((1, cfg["ctx"]), dtype=torch.int32).long()
    out = model(batch)
    assert out.shape == batch.shape + (cfg["vocab_size"],)

    # test with labels
    out_tuple = model(batch, batch)
    assert out_tuple[1].shape == ()
    assert out_tuple[0].shape == batch.shape + (cfg["vocab_size"],)


def test_model_getter():
    # Ensure we can call valid model and use it
    model = model_getter("flax-test", vocab_size=256, num_ctx=32)
    batch = torch.ones((1, 32), dtype=torch.int32).long()
    with torch.no_grad():
        logits, loss = model(batch, batch)


def test_model_getter_errors():

    with pytest.raises(AssertionError):
        model_getter("flax-XXL", vocab_size=1024, num_ctx=1024)

    with pytest.raises(AssertionError):
        model_getter("flax-test", vocab_size=-1, num_ctx=32)

    with pytest.raises(AssertionError):
        model_getter("flax-test", vocab_size=1024, num_ctx=-1)
