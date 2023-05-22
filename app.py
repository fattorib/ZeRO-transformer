import argparse
from functools import partial
from typing import Any, Callable

import gradio as gr
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPTNeoXTokenizerFast

from torch_compatability.GPT2 import model_getter


def parse():
    parser = argparse.ArgumentParser(description="Gradio Inference App")
    parser.add_argument("--model-size", default="medium", type=str)
    parser.add_argument("--share", default=False, action="store_true")
    parser.add_argument("--model-path", default="medium", type=str)
    args = parser.parse_args()
    return args


if torch.cuda.is_available():
    DEVICE = "cuda"


tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")


def model_creator(size: str, path: str) -> torch.nn.Module:

    model = model_getter(size, model_checkpoint=path)

    model.to(DEVICE)
    model.half()
    torch.cuda.empty_cache()
    model.eval()

    return model


@torch.no_grad()
def generate_from_prompt(
    prompt: str,
    model: Any,
    tokenizer: Any,
    sampling_func: Callable,
    logit_processor: Callable,
    sample: bool,
    steps: int,
    device: Any,
    return_on_eos: bool,
):

    tokens = torch.tensor(
        tokenizer.encode(prompt.strip()),
        dtype=torch.long,
    )

    x = tokens.view(1, -1).to(device)
    if x.shape[1] > model.num_ctx:
        x_cond = x[:, -model.num_ctx :]
    else:
        x_cond = x

    layer_past = None
    generated_tokens = []

    for _ in tqdm(range(steps), disable=True):
        with torch.cuda.amp.autocast(cache_enabled=False):
            logits, layer_past = model(x_cond, use_cache=True, past_states=layer_past)

        logits = logit_processor(logits, generated_tokens)
        logits = sampling_func(logits)
        probs = F.softmax(logits, dim=-1)

        if sample:
            x_cond = torch.multinomial(probs, num_samples=1)
            if return_on_eos:
                if x_cond.item() == tokenizer.eos_token_id:
                    return x

            x = torch.cat((x[:, :], x_cond), axis=1)

            if x_cond.item() not in generated_tokens:
                generated_tokens.append(x_cond.item())
        else:
            x_cond = torch.topk(probs, k=1).indices
            if return_on_eos:
                if x_cond.item() == tokenizer.eos_token_id:
                    return x
            x = torch.cat((x[:, :], x_cond), axis=1)

        yield x_cond


def process_logits(
    logits: torch.tensor, generated_tokens: list, rep_pen: float, temperature: float
) -> torch.tensor:
    logits = logits[:, -1, :] / temperature

    for prev_gen_token in generated_tokens:
        if logits[:, prev_gen_token] < 0:
            logits[:, prev_gen_token] *= rep_pen
        else:
            logits[:, prev_gen_token] /= rep_pen

    return logits


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


def top_p_logits(
    logits: torch.Tensor,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Filter a distribution of logits using nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
    return logits


def generate_text(
    prompt,
    steps,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    sampling_choice,
    eos_return,
):
    if sampling_choice == "Top-k":
        sampling_func = partial(top_k_logits, k=top_k)

    elif sampling_choice == "Nucleus":
        sampling_func = partial(top_p_logits, top_p=top_p)

    elif sampling_choice == "Greedy":
        sampling_func = partial(top_k_logits, k=1)

    processer_partial = partial(
        process_logits, rep_pen=repetition_penalty, temperature=temperature
    )

    text_generator = generate_from_prompt(
        prompt,
        model,
        tokenizer,
        sampling_func=sampling_func,
        logit_processor=processer_partial,
        sample=True,
        steps=steps,
        device=DEVICE,
        return_on_eos=eos_return,
    )

    text = []
    for token in text_generator:
        text.append(tokenizer.decode(token.tolist()[0]))

    generated_text = "".join(text)

    return [
        (prompt, None),
        (generated_text, "Generated Text"),
    ]


if __name__ == "__main__":
    args = parse()

    model = model_creator(args.model_size, args.model_path)

    # model = torch.compile(model)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                input_txt = gr.Textbox(lines=10, label="Enter your text here")
                token_slider = gr.Slider(
                    0, 1000, value=100, label="Number of tokens to generate"
                )

                with gr.Accordion("Generation Parameters", open=False):
                    temp_slider = gr.Slider(0, 2, value=0.80, label="Temperature")

                    topk_slider = gr.Slider(
                        0,
                        50,
                        value=40,
                        label="k (Top-k Sampling)",
                    )
                    topp_slider = gr.Slider(
                        0,
                        1,
                        value=0.96,
                        label="p (Nucleus Sampling)",
                    )
                    rep_slider = gr.Slider(
                        0.0,
                        1.3,
                        value=1.2,
                        label="Repetition Penalty",
                    )
                    radio = gr.Dropdown(
                        choices=["Top-k", "Nucleus", "Greedy"],
                        label="Sampling Method",
                        value="Nucleus",
                    )
                    eos_return = gr.Checkbox(
                        value=True, label="Terminate generation on EOS token."
                    )

            with gr.Column():
                output_txt = gr.HighlightedText(
                    label="Generated Text",
                    combine_adjacent=True,
                    color_map=["Generated Text", "blue"],
                )

                generate_btn = gr.Button("Generate Text")

        generate_btn.click(
            generate_text,
            [
                input_txt,
                token_slider,
                temp_slider,
                topk_slider,
                topp_slider,
                rep_slider,
                radio,
                eos_return,
            ],
            [output_txt],
        )

    demo.launch()
