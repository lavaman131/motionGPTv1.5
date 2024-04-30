from typing import TypedDict, List, Optional, Dict, Union
import json
import numpy as np
import torch
from mgpt.nlp.utils.duration_extraction import (
    MotionDurationModel,
)
from transformers import DistilBertTokenizer
from mgpt.nlp import FileName


def postprocess_model_preds(output, word_ids):
    """
    Averages the tokenizer output based on word_ids.
    """
    n = len(np.unique(word_ids[1:-1]))
    predicted_durations = [[] for _ in range(n)]
    for i, word_id in enumerate(word_ids[1:-1]):  # ignore the special tokens
        predicted_durations[word_id].append(output[i])
    return [np.mean(p) for p in predicted_durations]


def load_system_prompt_from_text(
    text_file: FileName,
) -> Dict[str, str]:
    """
    Loads the system prompt from the text.
    """
    with open(text_file, "r") as f:
        return {"role": "system", "content": f.read()}


class MotionInstruction(TypedDict):
    text: List[str]
    lengths: List[float]


def get_motion_dict_from_response(
    response: str,
    start_idx: int,
    end_idx: int,
    model: MotionDurationModel,
    tokenizer: DistilBertTokenizer,
    device: Union[str, torch.device],
) -> Optional[MotionInstruction]:
    """
    Extracts the motion json from the response.
    """
    assert start_idx < end_idx and start_idx >= 0 and end_idx >= 0, "Invalid indices"
    try:
        raw_motion_json = json.loads(response[start_idx + len("<motion>") : end_idx])
        motion_json: MotionInstruction = {"text": [], "lengths": []}
        for action in raw_motion_json["actions"]:
            motion_json["text"].append(action["action"])
        model.eval()
        tokenized_inputs = tokenizer(
            motion_json["text"],
            return_tensors="pt",
            truncation=True,
            padding=False,
            add_special_tokens=True,
            is_split_into_words=True,
        ).to(device)
        with torch.no_grad():
            preds = model(**tokenized_inputs)
        durations = postprocess_model_preds(
            preds.view(-1).detach().cpu().numpy(), tokenized_inputs.word_ids()
        )
        motion_json["lengths"] = durations
        return motion_json
    except json.JSONDecodeError:
        return None
