import gradio as gr
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

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=os.environ["HF_ACCESS_TOKEN"],
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    token=os.environ["HF_ACCESS_TOKEN"],
)
system_message = load_system_prompt_from_text(
    PROMPT_LIBRARY_DIR.joinpath("system_prompt.txt")
)


def generate_response(user_input):
    messages = [
        system_message,
        {"role": "user", "content": user_input},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1] :]
    text = tokenizer.decode(response, skip_special_tokens=True)

    video_path = None
    if "generate" in user_input.lower() and "motion" in user_input.lower():
        model_dir = PRECOMPUTED_DIR.joinpath(
            "duration_extraction_model", "epoch=19-step=1980.ckpt"
        )
        duration_model = MotionDurationModel.load_from_checkpoint(model_dir)

        distilbert_tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert/distilbert-base-uncased"
        )
        motion_json = get_motion_dict_from_response(
            text, duration_model, distilbert_tokenizer, duration_model.device
        )

        instructions_fname = "./instructions/llm_motion.json"
        with open(instructions_fname, "w") as f:
            json.dump(motion_json, f)

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
            ]
        )

        data = np.load(output_dir, allow_pickle=True)
        fast_render(data, output_path=output_dir, dataset="babel")

        video_path = output_dir + "/motion.mp4"

    return text, video_path


iface = gr.Interface(
    fn=generate_response,
    inputs=gr.inputs.Textbox(lines=5, label="Enter your request"),
    outputs=[
        gr.outputs.Textbox(label="Generated Text"),
        gr.outputs.Video(label="Generated Motion (if applicable)"),
    ],
    title="Motion Generation Chatbot",
    description="Enter a request and the chatbot will generate a corresponding response (text and/or video).",
)

iface.launch()
