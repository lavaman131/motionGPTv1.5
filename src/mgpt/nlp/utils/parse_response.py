from typing import TypedDict, List, Optional, Dict, Union
import json
import numpy as np
import torch
from mgpt.nlp.utils.duration_extraction import (
    MotionDurationModel,
)
from transformers import DistilBertTokenizer
from mgpt.nlp import FileName, PRECOMPUTED_DIR


def tokenize_and_preserve_words(sentence, tokenizer, add_special_tokens=True):
    tokenized_sentence = []
    lengths = []

    for word in sentence:
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)
        lengths.append(n_subwords)

    if add_special_tokens:
        tokenized_sentence = (
            [f"{tokenizer.cls_token}"] + tokenized_sentence + [f"{tokenizer.sep_token}"]
        )
        lengths = [1] + lengths + [1]

    return tokenized_sentence, lengths


def postprocess_model_preds(preds, lengths):
    """
    Averages the model output based on the lengths.
    """
    predicted_durations = []
    current_idx = 0
    for length in lengths[1:-1]:
        predicted_duration = np.nanmean(preds[current_idx : current_idx + length])
        predicted_duration = max(0.5, predicted_duration)
        predicted_durations.append(predicted_duration)
        current_idx += length
    return predicted_durations


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
    model: MotionDurationModel,
    tokenizer: DistilBertTokenizer,
    device: Union[str, torch.device],
) -> Optional[MotionInstruction]:
    """
    Extracts the motion json from the response.
    """
    start_idx = response.find("<motion>")
    end_idx = response.find("</motion>")
    if start_idx == -1 or end_idx == -1:
        return None
    try:
        raw_motion_json = json.loads(response[start_idx + len("<motion>") : end_idx])
        motion_json: MotionInstruction = {"text": [], "lengths": []}
        for action in raw_motion_json["actions"]:
            motion_json["text"].append(action["action"])
        tokenized_sentence, lengths = tokenize_and_preserve_words(
            motion_json["text"], tokenizer
        )
        model.eval()
        output = tokenizer(
            tokenized_sentence,
            is_split_into_words=True,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            preds = model(**output)
        durations = postprocess_model_preds(
            preds.view(-1).detach().cpu().numpy(), lengths
        )
        motion_json["lengths"] = durations
        return motion_json
    except json.JSONDecodeError:
        return None
