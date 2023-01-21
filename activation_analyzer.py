""" 
Hack of https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/Interactive%20Neuroscope.ipynb
"""
import argparse

from typing import Any, List
import gradio as gr
import torch
from transformers import GPTNeoXTokenizerFast
from functools import partial
from torch_compatability.GPT2 import model_getter

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"


def parse():
    parser = argparse.ArgumentParser(description="Gradio Inspection App")
    parser.add_argument("--model-size", default="medium", type=str)
    parser.add_argument("--model-path", default="medium", type=str)
    args = parser.parse_args()
    return args

def model_creator(size: str, path: str) -> torch.nn.Module:

    tokenizer = None
    if "distill" in size:
        num_ctx = 2048
        tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
        vocab_size=50304
    
    elif "byte" in size:
        num_ctx = 4096
        tokenizer = ByteTokenizer()
        vocab_size=257
    
    elif "solu" in size:
        num_ctx = 1024
        tokenizer = ByteTokenizer()
        vocab_size=257

    else:
        num_ctx = 1024 
        tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
        vocab_size=50304

    model = model_getter(
        size,
        vocab_size=vocab_size,
        num_ctx=num_ctx,
        model_checkpoint=path,
    )

    model.to(DEVICE)
    model.eval()

    return model, tokenizer




style_string = """<style> 
    span.token {
        border: 1px solid rgb(123, 123, 123)
        } 
    </style>"""


def calculate_color(val, max_val, min_val):
    # Hacky code that takes in a value val in range [min_val, max_val], normalizes it to [0, 1] and returns a color which interpolates between slightly off-white and red (0 = white, 1 = red)
    # We return a string of the form "rgb(240, 240, 240)" which is a color CSS knows
    normalized_val = (val - min_val) / max_val
    return f"rgb(240, {240*(1-normalized_val)}, {240*(1-normalized_val)})"

class ByteTokenizer:
    def __init__(self) -> None:
        self.eos_token_id = 256

    def encode(self, text: str) -> bytes:
        # encode a string of text as a bytearray
        return torch.tensor([list(text.encode("utf-8"))])

    def decode(self, tokens: List[str]) -> str:
        # decode a list of bytes to string
        return bytearray(tokens).decode("utf-8")


@torch.no_grad()
def pull_act_from_text(
    model: torch.nn.Module, text: str, neuron_idx: int, layer_idx: int, tokenizer: Any
):

    activation = {}

    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()[:, :, neuron_idx]

        return hook

    h = model.blocks[layer_idx].mlp.solu.register_forward_hook(
        getActivation("neuron_act")
    )

    tokens = torch.tensor(tokenizer.encode(text)).cuda()

    tokens = tokens.view(1, -1)

    model(tokens).cuda()

    h.remove()

    if model.vocab_size > 257:
        return activation["neuron_act"].cpu().numpy()[0, :], [
            tokenizer.decode(tokens[0, i]) for i in range(tokens.shape[1])
        ]
    else:
        return activation["neuron_act"].cpu().numpy()[0, :], [
            chr(tokens[0, i]) for i in range(tokens.shape[1])
        ]


if __name__ == "__main__":

    args = parse()

    model, tokenizer = model_creator(args.model_size, args.model_path)


    default_text = "Default Text."
    default_layer = 0
    default_neuron_index = 0

    default_max_val = 4.0
    default_min_val = 0.0

    def basic_neuron_vis(text, layer, neuron_index, max_val=None, min_val=None):
        """
        text: The text to visualize
        layer: The layer index
        neuron_index: The neuron index
        max_val: The top end of our activation range, defaults to the maximum activation
        min_val: The top end of our activation range, defaults to the minimum activation

        Returns a string of HTML that displays the text with each token colored according to its activation

        Note: It's useful to be able to input a fixed max_val and min_val, because otherwise the colors will change as you edit the text, which is annoying.
        """
        if layer is None:
            return "Please select a Layer"
        if neuron_index is None:
            return "Please select a Neuron"

        acts, str_tokens = pull_act_from_text(model, text, neuron_index, layer, tokenizer)

        act_max = acts.max()
        act_min = acts.min()
        # Defaults to the max and min of the activations
        if max_val is None:
            max_val = act_max
        if min_val is None:
            min_val = act_min
        # We want to make a list of HTML strings to concatenate into our final HTML string
        # We first add the style to make each token element have a nice border
        htmls = [style_string]
        # We then add some text to tell us what layer and neuron we're looking at - we're just dealing with strings and can use f-strings as normal
        # h4 means "small heading"
        htmls.append(f"<h4>Layer: <b>{layer}</b>. Neuron Index: <b>{neuron_index}</b></h4>")
        # We then add a line telling us the limits of our range
        htmls.append(
            f"<h4>Max Range: <b>{max_val:.4f}</b>. Min Range: <b>{min_val:.4f}</b></h4>"
        )
        # If we added a custom range, print a line telling us the range of our activations too.
        if act_max != max_val or act_min != min_val:
            htmls.append(
                f"<h4>Custom Range Set. Max Act: <b>{act_max:.4f}</b>. Min Act: <b>{act_min:.4f}</b></h4>"
            )

        # Convert the text to a list of tokens
        for tok, act in zip(str_tokens, acts):
            # A span is an HTML element that lets us style a part of a string (and remains on the same line by default)
            # We set the background color of the span to be the color we calculated from the activation
            # We set the contents of the span to be the token
            htmls.append(
                f"<span class='token' style='background-color:{calculate_color(act, max_val, min_val)}' >{tok}</span>"
            )

        return "".join(htmls)


    default_html_string = basic_neuron_vis(
        default_text,
        default_layer,
        default_neuron_index,
        max_val=default_max_val,
        min_val=default_min_val,
    )

    with gr.Blocks() as demo:
        gr.HTML(value=f"Hacky Interactive Neuroscope for model")
        # The input elements
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text", value=default_text)
                # Precision=0 makes it an int, otherwise it's a float
                # Value sets the initial default value
                layer = gr.Number(label="Layer", value=default_layer, precision=0)
                neuron_index = gr.Number(
                    label="Neuron Index", value=default_neuron_index, precision=0
                )
                # If empty, these two map to None
                max_val = gr.Number(label="Max Value", value=default_max_val)
                min_val = gr.Number(label="Min Value", value=default_min_val)
                inputs = [text, layer, neuron_index, max_val, min_val]
            with gr.Column():
                # The output element
                out = gr.HTML(label="Neuron Acts", value=default_html_string)
        for inp in inputs:
            inp.change(basic_neuron_vis, inputs, out)

    demo.launch(share=False, height=1000)

