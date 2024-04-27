from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
from mgpt.nlp.utils.parse_response import (
    load_system_prompt_from_text,
    get_motion_dict_from_response,
)
from mgpt.nlp import PROMPT_LIBRARY_DIR

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

messages = [
    system_message,
    {
        "role": "user",
        "content": "Hello! Can you generate the hip abduction motion?",
    },
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

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

print(text)

motion_json = get_motion_dict_from_response(text)

print(motion_json)
