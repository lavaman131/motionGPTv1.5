from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from mgpt.constants.transforms import SMPL_DATA_PATH
from mgpt.render import fast_render, slow_render


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--joints_path", type=str, help="Path to joints data")
    parser.add_argument("--method", type=str, help="Method for rendering")
    parser.add_argument("--output_path", type=str, help="Path to output folder")
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to use",
        default="babel",
        choices=["humanml", "babel"],
    )
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="if model uses unconstrained generation (not conditioned on text).",
        default=False,
    )

    args = parser.parse_args()
    joints_path = args.joints_path
    method = args.method
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    smpl_model_path = SMPL_DATA_PATH
    dataset = args.dataset
    unconstrained = args.unconstrained

    data = np.load(joints_path, allow_pickle=True)

    if method == "slow":
        slow_render(data, output_path, smpl_model_path)

    elif method == "fast":
        fast_render(data, output_path, dataset)
