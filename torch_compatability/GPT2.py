import copy
import math
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

"""
Module class for GPT2. Follows paper specifications wherever possible.
Certain sections of code are copied from the following repos:
    https://github.com/karpathy/minGPT
    https://github.com/EleutherAI/gpt-neox
    https://github.com/ofirpress/attention_with_linear_biases
"""

# GPT2-Style weight initialization (scaling residual layers by 1/sqrt(N))
def _weights_init(m, num_layers):
    if isinstance(m, (nn.Linear)):
        m.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.Linear) and m.bias is None:
            m.weight.data.normal_(mean=0.0, std=0.02)

    if isinstance(m, (nn.Embedding)):
        m.weight.data.normal_(mean=0.0, std=0.02)

    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

    for name, p in m.named_parameters():
        if "fc_resid" in name and "weight" in name:
            p.data.normal_(mean=0.0, std=(0.02 / math.sqrt(2 * num_layers)))


def _embedding_init(m):
    if isinstance(m, (nn.Embedding)):
        m.weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(m, nn.Linear) and m.bias is None:
        m.weight.data.normal_(mean=0.0, std=0.02)


class MLPBlock(nn.Module):
    def __init__(self, dim1: int, dim2: int, p: float, num_layers: int) -> None:
        """An MLP block.

        Args:
            dim1 (int): Input dimension
            dim2 (int): Output dimension
            p (float): Dropout probability
            num_layers (int): Number of total module layers. Used for weight initialization

        """
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.p = p
        self.num_layers = num_layers

        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(self.dim1, self.dim2)
        self.fc_resid = nn.Linear(self.dim2, self.dim1)
        self.dropout = nn.Dropout(p=self.p)

        init_function_partial = partial(
            _weights_init, **{"num_layers": self.num_layers}
        )

        self.apply(init_function_partial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc_resid(x)
        return self.dropout(x)


class ALiBi(nn.Module):
    """
    Self-attention module with ALiBi as described in paper
    `From Train Short, Test Long: Attention with Linear Biases Enables Input
    Length Extrapolation <https://ofir.io/train_short_test_long.pdf>`

    Source code modified from
    <https://github.com/ofirpress/attention_with_linear_biases> and
    <https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/positional_embeddings.py>

    """

    def __init__(
        self,
        embedding_dim: int,
        num_head: int,
        block_size: int,
        resid_dropout: float,
        num_layers: int,
    ):
        super().__init__()
        assert embedding_dim % num_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        # regularization
        self.resid_drop = nn.Dropout(resid_dropout)
        # output projection
        self.fc_resid = nn.Linear(embedding_dim, embedding_dim)

        self.alibi_cache = None
        self.cached_ctx = None

        self.n_head = num_head
        self.num_layers = num_layers

        self.register_buffer("slopes", torch.Tensor(self.get_slopes(self.n_head)))
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.uint8)).view(
                1, 1, block_size, block_size
            ),
        )

        init_function_partial = partial(
            _weights_init, **{"num_layers": self.num_layers}
        )

        self.apply(init_function_partial)

    def get_slopes(self, n: int) -> List:
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self.get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        layer_past: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()

        k, q, v = self.key(x), self.query(x), self.value(x)
        k = rearrange(
            k, "b t (nh hd) -> b t nh hd", nh=self.n_head, hd=C // self.n_head
        )
        q = rearrange(
            q, "b t (nh hd) -> b t nh hd", nh=self.n_head, hd=C // self.n_head
        )
        v = rearrange(
            v, "b t (nh hd) -> b t nh hd", nh=self.n_head, hd=C // self.n_head
        )

        k = rearrange(k, "b t n c -> b n t c")
        v = rearrange(v, "b t n c -> b n t c")
        q = rearrange(q, "b t n c -> b n t c")

        present = None
        if use_cache:
            if layer_past is not None:
                past_keys, past_values = layer_past
                k = torch.cat((past_keys, k), dim=-2)
                v = torch.cat((past_values, v), dim=-2)

            present = torch.stack((k, v))

        # Need to grab these
        seq_len_k, seq_len_q = k.size(-2), q.size(-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        # Creation of ALiBi distance matrix -> Computed on first forward pass
        # and stored. If CTX changes, we update this
        if self.cached_ctx != seq_len_k:

            # Update Buffer mask
            self.mask = (
                torch.tril(torch.ones(seq_len_k, seq_len_k, dtype=torch.uint8))
                .view(1, 1, seq_len_k, seq_len_k)
                .to(x.device)
            )

            # Create ALiBi distance matrix
            a = -torch.tril(
                torch.arange(seq_len_k).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1)
            )
            a = a.to(x.device).to(x.dtype)

            self.alibi_cache = a * self.slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_ctx = seq_len_k
            self.alibi_cache = self.alibi_cache.masked_fill(
                self.mask[:, :, :T, :T] == 0, float("-inf")
            )

        if seq_len_k != seq_len_q:
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            # Update Buffer mask
            self.mask = (
                torch.tril(torch.ones(seq_len_k, seq_len_k, dtype=torch.uint8))
                .view(1, 1, seq_len_k, seq_len_k)
                .to(x.device)
            )

            # Create ALiBi distance matrix
            a = -torch.tril(
                torch.arange(seq_len_k).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1)
            )

            a = a.to(x.device).to(x.dtype)

            a = a * self.slopes.view(self.slopes.shape[0], 1, 1)

            self.alibi_cache = a[:, seq_len_k - 1, :].view(a.shape[0], 1, a.shape[2])
            self.alibi_cache = self.alibi_cache.masked_fill(
                self.mask[:, :, :T, :T] == 0, float("-inf")
            )

        y = nn.functional.scaled_dot_product_attention(q, k, v, self.alibi_cache)

        y = (
            rearrange(y, "b n t h -> b t n h").contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.fc_resid(y))
        return y, present


class GPT2Block(nn.Module):
    """
    Standard Transformer block

    Based on `https://github.com/karpathy/minGPT/` with modifications
    """

    def __init__(
        self,
        embedding_dim: int,
        num_head: int,
        block_size: int,
        resid_dropout: float,
        num_layers: int,
        parallel_residual: bool = True,
    ) -> None:
        super().__init__()

        self.parallel_residual = parallel_residual
        if self.parallel_residual:
            self.ln = nn.LayerNorm(embedding_dim)
        else:
            self.ln1 = nn.LayerNorm(embedding_dim)
            self.ln2 = nn.LayerNorm(embedding_dim)

        self.attn = ALiBi(
            embedding_dim,
            num_head,
            block_size,
            resid_dropout,
            num_layers,
        )

        self.mlp = MLPBlock(
            embedding_dim,
            4 * embedding_dim,
            resid_dropout,
            num_layers,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        layer_past: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.parallel_residual:
            x_ln = self.ln(x)
            attn_out = self.attn(x_ln, use_cache, layer_past)
            mlp_out = self.mlp(x_ln)
            return x + mlp_out + attn_out[0], attn_out[1]

        else:
            attn_out = self.attn(self.ln1(x), use_cache, layer_past)
            x = x + attn_out[0]
            x = x + self.mlp(self.ln2(x))

            return x, attn_out[1]


class GPT2(nn.Module):
    def __init__(
        self,
        num_ctx: int,
        embedding_dim: int,
        N: int,
        vocab_size: int,
        num_head: int = 12,
        mlp_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        tied_embeddings: bool = False
    ):
        super().__init__()
        self.num_ctx = num_ctx
        self.embedding_dim = embedding_dim
        self.N = N
        self.vocab_size = vocab_size
        self.mlp_dropout = mlp_dropout
        self.resid_dropout = resid_dropout
        self.embedding_dropout = embedding_dropout
        self.num_head = num_head

        """
        Basic GPT2 transformer module
        """

        self.wte = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.dropout = nn.Dropout(p=self.embedding_dropout)

        self.blocks = nn.ModuleList(
            [
                copy.deepcopy(
                    GPT2Block(
                        embedding_dim=embedding_dim,
                        num_head=self.num_head,
                        block_size=self.num_ctx,
                        resid_dropout=resid_dropout,
                        num_layers=N,
                    )
                )
                for i in range(self.N)
            ]
        )

        self.norm = nn.LayerNorm(self.embedding_dim)

        embed_shape = self.wte.weight.shape
        self.lm_head = nn.Linear(
            in_features=embed_shape[1], out_features=embed_shape[0], bias=False
        )

        # Tying embedding weights
        if tied_embeddings:
            self.lm_head.weight = self.wte.weight

        self.apply(_embedding_init)

    def generate(
        self, context: torch.Tensor, max_length: int, sample: bool = False
    ) -> torch.Tensor:
        """
        Small generation method for compatibility with LM-Eval harness. Defaults
        to greedy decoding

        Parameters:
            context ('torch.Tensor'):
                Input context to prime the model

            max_length ('int'):
                The maximum length of tokens to generate (sum of context + *generated tokens*)

            sample ('bool'):
                Bool whether to sample from logits distribution
        """

        context = torch.tensor(context, dtype=torch.long).to(self.wte.weight.device)

        x = context.view(1, -1)

        num_generation_steps = max_length - x.shape[1]

        for _ in range(num_generation_steps):

            if x.shape[1] > self.num_ctx:
                x_cond = x[:, -self.num_ctx :]
            else:
                x_cond = x

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits = self.forward(x_cond)

                logits = logits[:, -1, :]

                probs = F.softmax(logits, dim=-1)

                if not sample:
                    out = torch.topk(probs, k=1)
                    x = torch.cat((x[:, :], out.indices), axis=1)
                else:
                    out = torch.multinomial(probs, num_samples=1)
                    x = torch.cat((x[:, :], out), axis=1)

        return x

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor = None,
        use_cache: bool = False,
        past_states: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        x = self.wte(x)
        x = self.dropout(x)
        present_states = []
        if not use_cache:
            past_states = [None] * self.N

        if past_states is None:
            past_states = [None] * self.N

        for block, past_state in zip(self.blocks, past_states):
            x, layer_past = block(x, use_cache, past_state)

            present_states.append(layer_past)

        x = self.norm(x)

        logits_lm = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits_lm[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            return logits_lm, loss
        else:
            if use_cache:
                return logits_lm, present_states
            else:
                return logits_lm


def model_getter(
    model_size: str,
    config_path: str = "torch_compatability/model_config.yaml",
    model_checkpoint: str = None,
) -> nn.Module:
    """Loads model configuration from YAML files
    and returns models

    Args:
        model_size (str): model name
            This is checked against all top-level model names in the
            YAML file (defaults to 'conf/model_config.yaml')
    """

    configs = OmegaConf.load(config_path)
    assert model_size in list(configs.keys()), "Invalid model name provided"
    model = GPT2(**configs[model_size])

    if model_checkpoint is not None:
        state_dict = torch.load(
            model_checkpoint,
            map_location="cpu",
        )

        model.load_state_dict(state_dict)

    return model
