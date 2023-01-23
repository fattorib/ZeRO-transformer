""" 
From a corpus of tokens, run through model and keep largest activations for 
a specific neuron and layer
"""
import argparse
from functools import partial
from typing import Any, List

import gradio as gr
import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm

from activation_analyzer import ByteTokenizer, model_creator

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"


def parse():
    parser = argparse.ArgumentParser(description="Gradio Inspection App")
    parser.add_argument("--model-size", default="medium", type=str)
    parser.add_argument("--model-path", default="medium", type=str)
    parser.add_argument("--layer-idx", type=int)
    parser.add_argument("--neuron-idx", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    return args


@torch.no_grad()
def pull_act_from_text(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    neuron_idx: int,
    layer_idx: int,
    tokenizer: Any,
):

    activation = {}

    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()[:, :, neuron_idx]

        return hook

    if model.use_solu:
        h = model.blocks[layer_idx].mlp.solu.register_forward_hook(
            getActivation("neuron_act")
        )

    else:
        h = model.blocks[layer_idx].mlp.gelu.register_forward_hook(
            getActivation("neuron_act")
        )

    # tokens = torch.tensor(tokenizer.encode(text)).cuda()

    tokens = tokens.cuda()

    with torch.cuda.amp.autocast():
        model(tokens)

    h.remove()

    if model.vocab_size > 257:
        return activation["neuron_act"].cpu().numpy()[0, :], [
            tokenizer.decode(tokens[0, i]) for i in range(tokens.shape[1])
        ]
    else:
        return activation["neuron_act"].cpu().numpy()[0, :], [
            chr(tokens[0, i]) for i in range(tokens.shape[1])
        ]


style_string = """<style> 
    span.token {
        border: 0px solid rgb(123, 123, 123)
        } 
    </style>"""


def calculate_color(val, max_val, min_val):
    # Hacky code that takes in a value val in range [min_val, max_val], normalizes it to [0, 1] and returns a color which interpolates between slightly off-white and red (0 = white, 1 = red)
    # We return a string of the form "rgb(240, 240, 240)" which is a color CSS knows
    normalized_val = (val - min_val) / max_val
    return f"rgb(240, {240*(1-normalized_val)}, {240*(1-normalized_val)})"


if __name__ == "__main__":

    args = parse()

    NEURON_IDX = args.neuron_idx
    LAYER_IDX = args.layer_idx

    model, tokenizer = model_creator(args.model_size, args.model_path)

    def preprocess(batch):
        x = batch["input_id.pth"][:1024]
        return torch.from_numpy(x.astype(np.int32)).long()

    validation_dataset = wds.DataPipeline(
        wds.SimpleShardList("data/processed/openwebtext2_validation-000000.tar.gz"),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.decode(handler=wds.warn_and_continue),
        wds.map(preprocess),
    )

    tl = DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        drop_last=True,
    )

    def create_html(text, activations):
        """
        From a collection of tuples and activations, turn this into
        visible HTML for rendering
        """
        act_max = activations.max()
        act_min = activations.min()

        max_val = act_max
        min_val = act_min

        htmls = [style_string]
        # We then add some text to tell us what layer and neuron we're looking at - we're just dealing with strings and can use f-strings as normal
        # h4 means "small heading"
        htmls.append(
            f"<h4>Layer: <b>{LAYER_IDX}</b>. Neuron Index: <b>{NEURON_IDX}</b></h4>"
        )
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
        for tok, act in zip(text, activations):
            # A span is an HTML element that lets us style a part of a string (and remains on the same line by default)
            # We set the background color of the span to be the color we calculated from the activation
            # We set the contents of the span to be the token
            htmls.append(
                f"<span class='token' style='background-color:{calculate_color(act, max_val, min_val)}' >{tok}</span>"
            )

        return "".join(htmls)

    topk = args.topk
    topk_activations = []  # tuple of (max_activation, activations, text)
    max_evaluation = args.max_samples

    for i, text in tqdm(enumerate(tl), total=max_evaluation):

        acts, text = pull_act_from_text(
            model,
            text,
            neuron_idx=NEURON_IDX,
            layer_idx=LAYER_IDX,
            tokenizer=ByteTokenizer(),
        )

        max_act, min_act = acts.max(), acts.min()

        if len(topk_activations) < topk:
            topk_activations.append((max_act, acts, text))

            topk_activations.sort(
                key=lambda x: x[0], reverse=True
            )  # sorted smallest to largest

        else:
            min_topk_act = topk_activations[-1][0]
            if max_act > min_topk_act:
                topk_activations.pop(-1)
                topk_activations.append((max_act, acts, text))
                topk_activations.sort(key=lambda x: x[0], reverse=True)

        if i > max_evaluation:
            break

    html = ""

    header = f"<h1>Model: SoLU Model: {model.N} Layer(s), 2048 Neurons per Layer</h1><h1>Dataset: OpenWebText2</h1><h2>Neuron {NEURON_IDX} in Layer {LAYER_IDX} </h2>"
    for activation_data in topk_activations:
        html += create_html(activation_data[-1], activation_data[1])

    with open(f"neurons/layer_{LAYER_IDX}_{NEURON_IDX}.html", "w") as f:
        f.write(header + html)
