# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

from mgpt.utils.fixseed import fixseed
import os
import numpy as np
import torch
from mgpt.utils.parser_util import generate_args
from mgpt.utils.model_util import load_model
from mgpt.utils import dist_util
from mgpt.data_loaders.humanml.scripts.motion_process import recover_from_ric
import json
from mgpt.diffusion.diffusion_wrappers import (
    DiffusionWrapper_FlowMDM as DiffusionWrapper,
)
from mgpt.constants.transforms import DATASET_STATS_DIR


def feats_to_xyz(sample, dataset, batch_size=1):
    if dataset == "humanml":  # for HumanML3D
        n_joints = 22
        mean = np.load(DATASET_STATS_DIR.joinpath("HML_Mean_Gen.npy"))
        std = np.load(DATASET_STATS_DIR.joinpath("HML_Std_Gen.npy"))
        sample = sample.cpu().permute(0, 2, 3, 1)
        sample = (sample * std + mean).float()
        sample = recover_from_ric(sample, n_joints)  # --> [1, 1, seqlen, njoints, 3]
        sample = sample.view(-1, *sample.shape[2:]).permute(
            0, 2, 3, 1
        )  # --> [1, njoints, 3, seqlen]
    elif dataset == "babel":  # [bs, 135, 1, seq_len] --> 6 * 22 + 3 for trajectory
        from mgpt.data_loaders.amass.transforms import SlimSMPLTransform

        transform = SlimSMPLTransform(
            batch_size=batch_size,
            name="SlimSMPLTransform",
            ename="smplnh",
            normalization=True,
        )
        all_feature = sample  # [bs, nfeats, 1, seq_len]
        all_feature_squeeze = all_feature.squeeze(2)  # [bs, nfeats, seq_len]
        all_feature_permutes = all_feature_squeeze.permute(
            0, 2, 1
        )  # [bs, seq_len, nfeats]
        splitted = torch.split(
            all_feature_permutes, all_feature.shape[0]
        )  # [list of [seq_len,nfeats]]
        sample_list = []
        for seq in splitted[0]:
            all_features = seq
            Datastruct = transform.SlimDatastruct
            datastruct = Datastruct(features=all_features)
            sample = datastruct.joints

            sample_list.append(sample.permute(1, 2, 0).unsqueeze(0))
        sample = torch.cat(sample_list)
    else:
        raise NotImplementedError("'feats_to_xyz' not implemented for this dataset")
    return sample


def generate(
    args,
    save_fname,
    out_path,
    instructions_file,
    diffusion,
    num_repetitions=1,
    seed=10,
):
    args.seed = seed
    args.num_repetitions = num_repetitions
    fixseed(args.seed)

    animation_out_path = out_path
    os.makedirs(animation_out_path, exist_ok=True)

    # ================= Load texts + lengths and adapt batch size ================
    # this block must be called BEFORE the dataset is loaded
    assert os.path.exists(instructions_file)
    # load json
    with open(instructions_file, "r") as f:
        instructions = json.load(f)
        assert (
            "text" in instructions and "lengths" in instructions
        ), "Instructions file must contain 'text' and 'lengths' keys"
        assert (
            len(instructions["text"]) == len(instructions["lengths"])
        ), "Instructions file must contain the same number of 'text' and 'lengths' elements"
    num_instructions = len(instructions["text"])
    args.batch_size = num_instructions
    args.num_samples = 1

    # from instructions
    json_lengths = instructions["lengths"]
    json_texts = instructions["text"]
    mask = torch.ones((len(json_texts), max(json_lengths)))
    for i, length in enumerate(json_lengths):
        mask[i, length:] = 0
    model_kwargs = {
        "y": {
            "mask": mask,
            "lengths": torch.tensor(json_lengths),
            "text": list(json_texts),
            "tokens": [""],
        }
    }
    with open(os.path.join(animation_out_path, "prompted_texts.json"), "w") as f:
        json.dump(instructions, f)

    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = (
            torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        )

    print(
        list(
            zip(
                list(model_kwargs["y"]["text"]),
                list(model_kwargs["y"]["lengths"].cpu().numpy()),
            )
        )
    )

    # ================= Sample ================
    all_motions = []
    all_lengths = []
    all_text = []
    for rep_i in range(args.num_repetitions):
        print(f"### Sampling [repetition #{rep_i}]")
        sample = diffusion.p_sample_loop(
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
        )
        sample = feats_to_xyz(sample, args.dataset)

        c_text = ""
        for i in range(num_instructions):
            c_text += model_kwargs["y"]["text"][i] + " /// "

        all_text.append(c_text)
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(
            model_kwargs["y"]["lengths"].sum().unsqueeze(0)
        )  # .cpu().numpy())

        print(f"created {rep_i+1}/{args.num_repetitions} human motion compositions.")

    all_motions = np.concatenate(all_motions, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)

    # ================= Save results + visualizations ================
    npy_path = os.path.join(out_path, f"{save_fname}.npy")
    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path,
        {
            "motion": all_motions,
            "text": all_text,
            "lengths": model_kwargs["y"]["lengths"].cpu().numpy(),
            "num_samples": args.num_samples,
            "num_repetitions": args.num_repetitions,
        },
    )
    with open(npy_path.replace(".npy", ".txt"), "w") as fw:
        fw.write("\n".join(all_text))
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in all_lengths]))
