from mgpt.utils.parser_util import evaluation_parser
from mgpt.utils.fixseed import fixseed
from datetime import datetime
from mgpt.data_loaders.model_motion_loaders import get_mdm_loader

# from mgpt.data_loaders.humanml.utils.metrics import
from mgpt.data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
import os

# from mgpt.data_loaders.humanml.scripts.motion_process import
# from mgpt.data_loaders.humanml.utils.utils
from mgpt.utils.model_util import load_model
import torch
from mgpt.diffusion import logger
from mgpt.utils import dist_util
from mgpt.data_loaders.get_data import get_dataset_loader
from mgpt.diffusion.diffusion_wrappers import (
    DiffusionWrapper_FlowMDM as DiffusionWrapper,
)
import numpy as np
from mgpt.utils.metrics import (
    evaluate_jerk,
    evaluate_matching_score,
    evaluate_fid,
    evaluate_diversity,
    get_metric_statistics,
    generate_plot_PJ,
)
from mgpt.evaluation.evaluate import evaluation
from mgpt.constants.evaluation import EVAL_FILES, SEQ_LENS


def get_log_filename(args, target, only_method=False):
    assert target in ["motion", "transition"]

    folder = os.path.join(os.path.dirname(args.model_path), "evaluation")
    os.makedirs(folder, exist_ok=True)
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    log_file = os.path.join(folder, "{}_{}_{}".format(target.capitalize(), name, niter))
    if args.guidance_param != 1.0:
        log_file += f"_gscale{args.guidance_param}"
    log_file += f"_{args.eval_mode}"
    log_file += f"_transLen{args.transition_length}" if not only_method else ""

    if args.extrapolation:
        log_file += "_extrapolation"
    if (
        not only_method
        and args.scenario is not None
        and args.scenario not in ["", "test", "val"]
    ):
        # it will be a targetted experiment --> in-distribution, out-distribution, etc.
        log_file += f"_{args.scenario}"
    return log_file + f"_s{args.seed}.log"


def get_log_filename_precomputed(args, target, only_method=False):
    # only_method --> only arguments related to the method, not the evaluation
    # this is to create the folder for precomputed motions (reused for different evaluations)
    assert target in ["motion", "transition"]

    folder = os.path.join(args.model_path, "evaluation")
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, "{}".format(target))
    log_file += f"_transLen{args.transition_length}" if not only_method else ""
    if args.extrapolation:
        log_file += "_extrapolation"
    return log_file + f"_s{args.seed}.log"


def get_summary_filename(args):
    # this is SPECIFIC TO AN EVALUATION
    folder = os.path.join(os.path.dirname(args.model_path), "evaluations_summary")
    os.makedirs(folder, exist_ok=True)
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    suffix = "_transLen{}".format(args.transition_length)
    log_file = os.path.join(
        folder, "{}_{}_{}{}".format(niter, args.eval_mode, args.seed, suffix)
    )
    if args.extrapolation:
        log_file += "_extrapolation"
    if args.scenario is not None and args.scenario not in ["", "test", "val"]:
        # it will be a targetted experiment --> in-distribution, out-distribution, etc.
        log_file += f"_{args.scenario}"
    return log_file + ".json"


def run(args, shuffle=True):
    fixseed(args.seed)
    drop_last = False
    args.batch_size = 32  # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!

    print(f"Eval mode [{args.eval_mode}]")
    if args.eval_mode == "debug":
        diversity_times = 30
        replication_times = 1  # about 3 Hrs
    elif args.eval_mode == "fast":
        diversity_times = 300
        replication_times = 3
    elif args.eval_mode == "final":
        diversity_times = 300
        replication_times = 10  # about 12 Hrs
    else:
        raise ValueError()

    if args.scenario == "short":  # not enough samples in the other scenarios <300
        diversity_times = 175
    elif args.scenario == "medium":
        diversity_times = 275

    logger.configure()

    logger.log("creating data loader...")
    dist_util.setup_dist(args.device)
    print("Creating model and diffusion...")
    model, diffusion = load_model(args, dist_util.dev())
    diffusion = DiffusionWrapper(args, diffusion, model)

    split = "test" if args.dataset != "babel" else "val"  # no test set for babel
    if args.extrapolation:
        eval_file = EVAL_FILES[args.dataset]["extrapolation"]
    else:
        eval_file = EVAL_FILES[args.dataset][split]

    min_seq_len, max_seq_len = SEQ_LENS[args.dataset]
    precomputed_folder = (
        get_log_filename(args, "motion", only_method=True)
        .replace(".log", "")
        .replace("evaluation", "evaluation_precomputed")
    )
    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())

    print("Loading motion GT...")
    motion_gt_loader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=(min_seq_len, max_seq_len),
        split=split,
        load_mode="gt",
        shuffle=shuffle,
        drop_last=drop_last,
        cropping_sampler=False,
    )
    print(f"Done! Motion dataset size: {len(motion_gt_loader.dataset)}")
    print("Loading transitions GT...")
    transition_gt_loader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=(args.transition_length, args.transition_length),
        split=split,
        load_mode="gt_transitions",
        shuffle=shuffle,
        drop_last=drop_last,
        cropping_sampler=True,
    )
    print(f"Done! Transition dataset size: {len(transition_gt_loader.dataset)}")
    gt_loaders = [motion_gt_loader, transition_gt_loader]
    targets = ["motion", "transition"]
    log_files = [get_log_filename(args, t) for t in targets]

    eval_motion_loader = lambda rep: get_mdm_loader(
        args,
        model,
        diffusion,
        args.batch_size,
        eval_file,
        transition_gt_loader.dataset,
        max_motion_length=max_seq_len,
        precomputed_folder=os.path.join(
            precomputed_folder,
            f"{rep:02d}",
        ),
        scenario=args.scenario,  # to evaluate a subset of them
    )

    print("=" * 10, "Motion evaluation", "=" * 10)
    for mode, log_file in zip(targets, log_files):
        print(f"[{mode}] Will save to log file {log_file}")
    mean_dict, plots_dict = evaluation(
        eval_wrapper,
        gt_loaders,
        eval_motion_loader,
        targets,
        log_files,
        replication_times,
        diversity_times,
        extrapolation=args.extrapolation,
    )

    # store to folder "evaluations_summary"
    log_file = get_summary_filename(args)
    plot_file = log_file.replace(".json", "_plots.json")
    print(f"Saving summary to {log_file}")
    # store mean_dict
    import json

    with open(log_file, "w") as f:
        json.dump(mean_dict, f, default=str)
    # store plots_dict
    with open(plot_file, "w") as f:
        json.dump(plots_dict, f, default=str)

    generate_plot_PJ(plot_file)

    return mean_dict


if __name__ == "__main__":
    args = evaluation_parser()
    run(args)
