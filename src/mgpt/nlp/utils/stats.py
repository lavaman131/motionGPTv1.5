from pathlib import Path
import json
import pickle
import numpy as np
from typing import Dict, Any, Tuple
from mgpt.nlp import PRECOMPUTED_DIR, FileName
from thefuzz import process
import numpy as np
from hmmlearn import hmm


def extract_motion_and_duration(
    action: str, stats: Dict[str, Any], seed: int = 42
) -> Tuple[str, float]:
    rng = np.random.default_rng(seed)

    closest_action, _ = process.extractOne(action, stats["actions"])
    duration = rng.normal(
        loc=stats["avg_duration"][closest_action],
        scale=stats["var_duration"][closest_action],
    )
    return closest_action, duration


def save_stats(file_path: FileName, precomputed_dir: FileName) -> None:
    with open(file_path, "r") as f:
        data = json.load(f)
        actions = set()
        action_sequences = []
        duration_sequences = []
        for key in data.keys():
            frame_ann = data[key]["frame_ann"]
            if frame_ann:
                action_sequence = []
                duration_sequence = []
                sorted_labels = sorted(frame_ann["labels"], key=lambda x: x["start_t"])
                for label in sorted_labels:
                    actions.add(label["proc_label"])
                    action_sequence.append(label["proc_label"])
                    duration_sequence.append(abs(label["end_t"] - label["start_t"]))

                action_sequences.append(action_sequence)
                duration_sequences.append(duration_sequence)

    stats = dict()
    stats["id_to_action"] = {i: k for i, k in enumerate(actions)}
    stats["action_to_id"] = {k: i for i, k in stats["id_to_action"].items()}
    stats["action_sequences"] = [
        [stats["action_to_id"][a] for a in action_sequence]
        for action_sequence in action_sequences
    ]
    stats["duration_sequences"] = duration_sequences

    precomputed_dir = Path(precomputed_dir)
    precomputed_dir.mkdir(parents=True, exist_ok=True)

    with open(precomputed_dir.joinpath("babel_action_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)


def train_hmm(
    precomputed_dir: FileName, stats: Dict[str, Any], bin_size: float = 0.5
) -> None:
    """Train a Gaussian HMM on the action sequences

    Parameters
    ----------
    precomputed_dir : FileName
        directory to save the precomputed files
    stats : Dict[str, Any]
        stats dictionary
    bin_size : float, optional
        discretize time into `bin_size` second bins, by default 0.5
    """
    # Step 1: Prepare the training data
    h = np.concatenate(stats["duration_sequences"]).reshape(-1, 1)
    bins = np.arange(bin_size, np.max(h) + bin_size, bin_size)
    bins_mapping = {i: bins[i] for i in range(bins.size)}
    n_components = bins.size
    X = np.concatenate(stats["action_sequences"]).reshape(-1, 1)
    lengths = [len(seq) for seq in stats["action_sequences"]]

    # Step 2: Create an instance of the Gaussian HMM
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")

    # # Step 3: Train the HMM
    model.fit(X, lengths)

    precomputed_dir = Path(precomputed_dir)

    model_file = precomputed_dir.joinpath("hmm_model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    stats["id_to_time"] = bins_mapping
    stats["time_to_id"] = {action: i for i, action in stats["id_to_time"].items()}

    with open(precomputed_dir.joinpath("babel_action_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)


if __name__ == "__main__":
    # babel_dir = Path("/projectnb/ivc-ml/alavaee/data/babel/babel-teach")
    # file_path = babel_dir.joinpath("train.json")
    # save_stats(file_path, PRECOMPUTED_DIR)
    with open(PRECOMPUTED_DIR.joinpath("babel_action_stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    # print(stats.keys())
    # max_idx = 0
    # max_duration = 0
    # for i, action_sequence in enumerate(stats["duration_sequences"]):
    #     for duration in action_sequence:
    #         if duration > max_duration:
    #             max_idx = i
    #             max_duration = duration
    # print(stats["duration_sequences"][max_idx])
    # print([stats["id_to_action"][s] for s in stats["action_sequences"][max_idx]])
    # print(np.max([s for dur in stats["duration_sequences"] for s in dur]))
    # train_hmm(precomputed_dir=PRECOMPUTED_DIR, stats=stats)
    with open(PRECOMPUTED_DIR.joinpath("hmm_model.pkl"), "rb") as f:
        model = pickle.load(f)
    res = model.predict([stats["action_sequences"][10][:-4]])
    lbl = stats["id_to_time"][res[0]]
    print(lbl)
