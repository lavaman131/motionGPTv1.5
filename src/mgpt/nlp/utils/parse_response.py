from typing import TypedDict, List, Optional, Dict
import json
from mgpt.nlp import FileName, PRECOMPUTED_DIR
import pickle
from mgpt.nlp.utils.stats import extract_motion_and_duration


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


def get_motion_dict_from_response(response: str) -> Optional[MotionInstruction]:
    """
    Extracts the motion json from the response.
    """
    start_idx = response.find("<motion>")
    end_idx = response.find("</motion>")
    if start_idx == -1 or end_idx == -1:
        return None
    try:
        raw_motion_json = json.loads(response[start_idx + len("<motion>") : end_idx])
        with open(PRECOMPUTED_DIR.joinpath("babel_action_stats.pkl"), "rb") as f:
            stats = pickle.load(f)
        motion_json: MotionInstruction = {"text": [], "lengths": []}
        for action in raw_motion_json["actions"]:
            text, length = extract_motion_and_duration(
                action=action["action"], stats=stats
            )
            motion_json["text"].append(text)
            motion_json["lengths"].append(round(length, 3))
        return motion_json
    except json.JSONDecodeError:
        return None
