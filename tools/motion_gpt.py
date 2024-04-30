import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
from mgpt.nlp.utils.parse_response import (
    load_system_prompt_from_text,
    get_motion_dict_from_response,
)
from mgpt.nlp import PROMPT_LIBRARY_DIR, PRECOMPUTED_DIR
from mgpt.nlp.utils.duration_extraction import (
    MotionDurationModel,
)
from mgpt.nlp.utils.similarity_search import init_vector_db
import time
from mgpt.utils.parser_util import generate_args
from pathlib import Path
import json
import pytorch_lightning as pl
from mgpt.render import datasets_fps, fast_render
from mgpt.model.generate import generate
from mgpt.utils.model_util import load_model
from mgpt.utils import dist_util
from mgpt.diffusion.diffusion_wrappers import (
    DiffusionWrapper_FlowMDM as DiffusionWrapper,
)


def main():
    # Load model
    args = generate_args()

    pl.seed_everything(args.seed)
    user_prompt = args.user_prompt

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
    motion_duration_tokenizer = AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased"
    )

    args.bpe_denoising_step = 60
    args.guidance_param = 1.5
    # ================= Load model and diffusion wrapper ================
    print("Creating model and diffusion...")
    model, diffusion = load_model(args, dist_util.dev())
    diffusion = DiffusionWrapper(args, diffusion, model)

    system_message = load_system_prompt_from_text(
        PROMPT_LIBRARY_DIR.joinpath("improved_system_prompt.txt")
    )

    db = init_vector_db("action_vocab.txt", PRECOMPUTED_DIR)
    # get chunks
    docs = db.similarity_search(user_prompt, k=4)
    motion_dictionary = "\n".join(
        ["".join(docs[i].page_content) for i in range(len(docs))]
    )
    system_message["content"] = system_message["content"].replace(
        r"{motion_dictionary}", motion_dictionary
    )

    messages = [
        system_message,
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    print(f"Raw model input:\n {messages}")

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
            save_fname = time.strftime(
                "%Y-%m-%d-%H_%M_%S", time.localtime(time.time())
            ) + str(np.random.randint(10000, 99999))
            instructions_fname = f"./instructions/{save_fname}.json"
            fps = datasets_fps["babel"]
            motion_json["lengths"] = [
                int(length * fps) for length in motion_json["lengths"]
            ]
            Path(instructions_fname).parent.mkdir(parents=True, exist_ok=True)
            with open(instructions_fname, "w") as f:
                json.dump(motion_json, f)

            out_lengths = motion_json["lengths"]

            output_dir = "./results/output"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            generate(
                args=args,
                save_fname=save_fname,
                out_path=output_dir,
                instructions_file=instructions_fname,
                diffusion=diffusion,
                num_repetitions=1,
            )

            npy_path = Path(output_dir).joinpath(save_fname).with_suffix(".npy")

            data = np.load(npy_path, allow_pickle=True)

            fast_render(data, save_fname, output_dir, dataset="babel")
            video_fname = save_fname
        else:
            generate_motion = False

    if not generate_motion:
        (
            out_lengths,
            output_mp4_path,
            video_fname,
        ) = None, None, None


if __name__ == "__main__":
    main()
