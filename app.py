import argparse

import gradio as gr
import torch
from transformers import GPTNeoXTokenizerFast

from torch_compatability.generation import TextGenerator
from torch_compatability.GPT2 import model_getter


def parse():
    parser = argparse.ArgumentParser(description="Gradio Inference App")
    parser.add_argument("--model-size", default="medium", type=str)
    parser.add_argument("--share", default=False, action="store_true")
    parser.add_argument("--model-path", default="medium", type=str)
    args = parser.parse_args()
    return args


DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"


tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
generator = TextGenerator(seq_len=2048, tokenizer=tokenizer)


def model_creator(size: str, path: str) -> torch.nn.Module:

    model = model_getter(
        size,
        vocab_size=50304,
        num_ctx=1024 if "distill" not in size else 2048,
        model_checkpoint=path,
    )

    model.to(DEVICE)
    model.eval()

    return model


def generate_text(
    prompt,
    steps,
    temperature,
    top_k,
    top_p,
    tau,
    repetition_penalty,
    epsilon,
    sampling_choice,
    beta,
):
    if sampling_choice == "Top-k":
        sampling_method = "topk"

    elif sampling_choice == "Nucleus":
        sampling_method = "nucleus"

    elif sampling_choice == "Typical":
        sampling_method = "typical"

    elif sampling_choice == "Greedy":
        sampling_method = "greedy"

    elif sampling_choice == "$\eta$":
        sampling_method = "eta"

    generated_text, new_gen, logprobs = generator.generate_text_from_prompt(
        model=model,
        prompt=prompt,
        steps=int(steps),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        tau=tau,
        repetition_penalty=repetition_penalty,
        epsilon=epsilon,
        sampling_method=sampling_method,
        device=DEVICE,
        beta=beta,
    )

    original_gen_length = len(generated_text) - len(new_gen)

    return [
        (generated_text[:original_gen_length], None),
        (generated_text[original_gen_length:], "Generated Text"),
    ]


if __name__ == "__main__":
    args = parse()

    assert len(args.model_path) > 0, "Must provide a valid model checkpoint"

    model = model_creator(args.model_size, args.model_path)

    description = "WIP DESCRIPTION"

    iface = gr.Interface(
        fn=generate_text,
        inputs=[
            gr.inputs.Textbox(lines=10, label="Enter your text here"),
            gr.inputs.Slider(
                0, 1000, default=100, label="Number of tokens to generate"
            ),
            gr.inputs.Slider(0, 2, default=0.70, label="Temperature"),
            gr.inputs.Slider(
                0,
                50,
                default=40,
                label="k (Top-k Sampling)",
            ),
            gr.inputs.Slider(
                0,
                1,
                default=0.96,
                label="p (Nucleus Sampling)",
            ),
            gr.inputs.Slider(
                0,
                1,
                default=0.2,
                label="Tau (Typical Sampling)",
            ),
            gr.inputs.Slider(
                0.0,
                1.3,
                default=1.2,
                label="Repetition Penalty",
            ),
            gr.inputs.Slider(
                0.0,
                0.001,
                default=0.0006,
                label="$\epsilon$",
            ),
            gr.inputs.Radio(
                choices=["Top-k", "Nucleus", "Typical", "Greedy", "$\eta$"],
                label="Sampling Method",
                default="Nucleus",
            ),
            gr.inputs.Slider(
                0.0,
                1.0,
                default=0.5,
                label=r"$\beta$",
            ),
        ],
        outputs=gr.HighlightedText(
            label="Generated Text",
            combine_adjacent=True,
            color_map=["Generated Text", "blue"],
        ),
        live=False,
        title="WIP Title",
        description=None,
        article="For more details check out the model repo [here](https://github.com/fattorib/transformer)",
        allow_flagging="never",
    )
    iface.launch(share=args.share)
