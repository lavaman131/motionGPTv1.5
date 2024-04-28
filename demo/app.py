import os

os.environ["DISPLAY"] = ":0.0"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"

import gradio as gr
import random
import torch
import time
import cv2
import numpy as np
import OpenGL.GL as gl
import imageio
import pytorch_lightning as pl
import moviepy.editor as mp
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
from mgpt.nlp.utils.parse_response import (
    load_system_prompt_from_text,
    get_motion_dict_from_response,
    tokenize_and_preserve_words,
)
from mgpt.render import fast_render
from mgpt.nlp import PROMPT_LIBRARY_DIR, PRECOMPUTED_DIR
from mgpt.nlp.utils.duration_extraction import (
    MotionDurationModel,
)
import json
from subprocess import Popen
from transformers import DistilBertTokenizer
from mgpt.render import datasets_fps


# Load model
pl.seed_everything(42)

language_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

language_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

language_tokenizer = AutoTokenizer.from_pretrained(
    language_model_id,
    token=os.environ["HF_ACCESS_TOKEN"],
)
language_model = AutoModelForCausalLM.from_pretrained(
    language_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=language_bnb_config,
    token=os.environ["HF_ACCESS_TOKEN"],
)
language_terminators = [
    language_tokenizer.eos_token_id,
    language_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

motion_duration_model_dir = PRECOMPUTED_DIR.joinpath(
    "duration_extraction_model", "epoch=19-step=1980.ckpt"
)

motion_duration_model = MotionDurationModel.load_from_checkpoint(
    motion_duration_model_dir
)

# Tokenize the phrases and create a mapping
motion_duration_tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased"
)

system_message = load_system_prompt_from_text(
    PROMPT_LIBRARY_DIR.joinpath("system_prompt.txt")
)


# HTML Style
Video_Components = """
<div class="side-video" style="position: relative;">
    <video width="340" autoplay loop>
        <source src="file/{video_path}" type="video/mp4">
    </video>
    <a class="videodl-button" href="file/{video_path}" download="{video_fname}" title="Download Video">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-video"><path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/></svg>
    </a>
</div>
"""

Video_Components_example = """
<div class="side-video" style="position: relative;">
    <video width="340" autoplay loop controls>
        <source src="file/{video_path}" type="video/mp4">
    </video>
    <a class="npydl-button" href="file/{video_path}" download="{video_fname}" title="Download Video">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-video"><path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/></svg>
    </a>
</div>
"""

Text_Components = """
<h3 class="side-content" >{msg}</h3>
"""


def render_motion(data, method="fast", dataset="babel"):
    output_dir = "./results/output"

    if method == "fast":
        motion_save_path = fast_render(data, output_path=output_dir, dataset=dataset)
    else:
        raise NotImplementedError(f"Method {method} is not implemented.")

    return motion_save_path


def add_text(history, text, motion_uploaded, data_stored):
    data_stored = data_stored + [{"user_input": text}]

    text = f"""<h3>{text}</h3>"""
    history = history + [(text, None)]

    return history, gr.update(value="", interactive=False), motion_uploaded, data_stored


def get_no_motion_response():
    return None, None, None


def bot(history, motion_uploaded, data_stored, method, dataset="babel"):
    motion_length, motion_token_string = (
        motion_uploaded["motion_lengths"],
        motion_uploaded["motion_token_string"],
    )

    input = data_stored[-1]["user_input"]
    prompt = input
    data_stored[-1]["model_input"] = prompt
    messages = [
        system_message,
        {"role": "user", "content": input},
    ]
    input_ids = language_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(language_model.device)

    outputs = language_model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=language_terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1] :]
    text = language_tokenizer.decode(response, skip_special_tokens=True)

    start_idx = text.find("<motion>")
    end_idx = text.find("</motion>")
    generate_motion = start_idx >= 0 and end_idx >= 0

    if generate_motion:
        motion_json = get_motion_dict_from_response(
            text,
            start_idx,
            end_idx,
            motion_duration_model,
            motion_duration_tokenizer,
            motion_duration_model.device,
        )

        if motion_json:
            instructions_fname = "./instructions/llm_motion.json"
            fps = datasets_fps[dataset]
            motion_json["lengths"] = [
                int(length * fps) for length in motion_json["lengths"]
            ]
            with open(instructions_fname, "w") as f:
                json.dump(motion_json, f)

            out_lengths = motion_json["lengths"]

            output_dir = "./results/output"

            Popen(
                [
                    "python",
                    "generate.py",
                    "--model_path",
                    "./results/babel/FlowMDM/model001300000.pt",
                    "--num_repetitions",
                    "1",
                    "--bpe_denoising_step",
                    "60",
                    "--guidance_param",
                    "1.5",
                    "--instructions_file",
                    instructions_fname,
                    "--output_dir",
                    output_dir,
                ],
                shell=True,
            )

            data = np.load(output_dir, allow_pickle=True)

            output_mp4_path = render_motion(data, method=method)
            video_fname = Path(output_mp4_path).name
        else:
            generate_motion = False

    if not generate_motion:
        (
            out_lengths,
            output_mp4_path,
            video_fname,
        ) = get_no_motion_response()

    data_stored[-1]["model_output"] = {
        "length": out_lengths,
        "texts": text,
        "motion_video": output_mp4_path,
        "motion_video_fname": video_fname,
    }

    if generate_motion:
        response = [
            Video_Components.format(
                video_path=output_mp4_path,
                video_fname=video_fname,
            )
        ]
    else:
        response = f"""<h3>{text}</h3>"""

    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.02)
        yield history, motion_uploaded, data_stored


def bot_example(history, responses):
    history = history + responses
    return history


def load_css():
    with open("./assets/css/custom.css", "r", encoding="utf-8") as f:
        css = f.read()
    return css


with gr.Blocks() as demo:
    chat_instruct_sum = gr.State(
        [
            (
                None,
                """
         👋 Hi, I'm MotionGPT! I can generate realistic human motion from text, or generate text from motion.

         1. You can chat with me in pure text like generating human motion following your descriptions.
         2. After generation, you can click the button in the top right of generation human motion result to download the human motion video or feature stored in .npy format.
         3. With the human motion feature file downloaded or got from dataset, you are able to ask me to translate it!
         4. Of course, you can also purely chat with me and let me give you human motion in text, here are some examples!
         """,
            )
        ]
    )

    Init_chatbot = chat_instruct_sum.value[:1]

    data_stored = gr.State([])

    gr.Markdown("""
                # MotionGPTv2

                <p align="left">
                <a href="https://github.com/lavaman131/motionGPTv2">Github Repo</a>
                </p>
                """)

    chatbot = gr.Chatbot(
        Init_chatbot,
        elem_id="mGPT",
        height=600,
        label="MotionGPTv2",
        avatar_images=(None, ("./assets/images/avatar_bot.jpg")),
        bubble_full_width=False,
    )

    with gr.Row():
        with gr.Column(scale=0.85):
            with gr.Row():
                txt = gr.Textbox(
                    label="Text",
                    show_label=False,
                    elem_id="textbox",
                    placeholder="Enter text and press ENTER or speak to input. You can also upload motion.",
                    container=False,
                )

            with gr.Row():
                # regen = gr.Button("🔄 Regenerate", elem_id="regen")
                clear = gr.ClearButton([txt, chatbot], value="🗑️ Clear")

            with gr.Row():
                gr.Markdown("""
                ### You can get more examples (pre-generated for faster response) by clicking the buttons below:
                """)

            with gr.Row():
                instruct_eg = gr.Button("Instructions", elem_id="instruct")

        with gr.Column(scale=0.15, min_width=150):
            method = gr.Dropdown(
                ["slow", "fast"],
                label="Visulization method",
                interactive=True,
                elem_id="method",
                value="fast",
            )

    txt_msg = txt.submit(
        add_text,
        [chatbot, txt, data_stored, method],
        [chatbot, txt, data_stored],
        queue=False,
    ).then(
        bot,
        [chatbot, data_stored, method],
        [chatbot, data_stored],
    )

    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    instruct_msg = instruct_eg.click(
        bot_example, [chatbot, chat_instruct_sum], [chatbot], queue=False
    )

    chatbot.change(scroll_to_output=True)

if __name__ == "__main__":
    demo.launch(debug=True)
