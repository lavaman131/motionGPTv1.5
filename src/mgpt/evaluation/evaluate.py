from datetime import datetime

# from mgpt.data_loaders.humanml.utils.metrics import
from collections import OrderedDict

# from mgpt.data_loaders.humanml.scripts.motion_process import
# from mgpt.data_loaders.humanml.utils.utils
import torch
import numpy as np
from mgpt.utils.metrics import (
    evaluate_jerk,
    evaluate_matching_score,
    evaluate_fid,
    evaluate_diversity,
    get_metric_statistics,
)


torch.multiprocessing.set_sharing_strategy("file_system")


def update_metrics_dict(all_metrics_dict, metric, current_metrics_dict):
    for key, item in current_metrics_dict.items():
        if key not in all_metrics_dict[metric]:
            all_metrics_dict[metric][key] = [item]
        else:
            all_metrics_dict[metric][key] += [item]


# print to both stdout and file
def pprint(f, *args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=f, flush=True)


def evaluation(
    eval_wrapper,
    gt_loaders,
    eval_motion_loader,
    targets,
    log_files,
    replication_times,
    diversity_times,
    extrapolation=False,
):
    assert len(targets) == len(
        log_files
    ), f"len(targets)={len(targets)} != len(log_files)={len(log_files)}"
    files = [open(log_file, "w") for log_file in log_files]

    targets_metrics = {}
    for target in targets:
        if "transition" in target.lower():
            targets_metrics[target] = OrderedDict(
                {
                    "FID": OrderedDict({}),
                    "Diversity": OrderedDict({}),
                    "PeakJerk": OrderedDict({}),
                    "AUJ": OrderedDict({}),
                }
            )
        else:
            targets_metrics[target] = OrderedDict(
                {
                    "MatchingScore": OrderedDict({}),
                    "R_precision": OrderedDict({}),
                    "FID": OrderedDict({}),
                    "Diversity": OrderedDict({}),
                    "MultiModality": OrderedDict({}),
                }
            )

    targets_plots = {
        target: OrderedDict(
            {
                "PeakJerk": OrderedDict({}),
            }
        )
        for target in targets
    }
    for replication in range(replication_times):
        # we use the same generated motions for both targets (motion and transition)
        motion_loader = eval_motion_loader(replication)[0]
        for target, gt_loader, f in zip(targets, gt_loaders, files):
            motion_loader.dataset.switch_target(target)
            motion_loaders = {f"{target}": motion_loader}
            all_metrics = targets_metrics[target]
            all_plots = targets_plots[target]

            pprint(
                f,
                f"==================== [{target}] Replication {replication} ====================",
            )

            pprint(f, f"Time: {datetime.now()}")

            if (
                "FID" in all_metrics
                or "Diversity" in all_metrics
                or "MatchingScore" in all_metrics
                or "R_precision" in all_metrics
            ):
                compute_precision = "R_precision" in all_metrics and not extrapolation
                compute_matching = "MatchingScore" in all_metrics and not extrapolation
                mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(
                    eval_wrapper,
                    motion_loaders,
                    f,
                    compute_precision=compute_precision,
                    compute_matching=compute_matching,
                )
                if compute_matching:
                    update_metrics_dict(all_metrics, "MatchingScore", mat_score_dict)
                if compute_precision:
                    update_metrics_dict(all_metrics, "R_precision", R_precision_dict)

            if "FID" in all_metrics:
                fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)
                update_metrics_dict(all_metrics, "FID", fid_score_dict)

            if "Diversity" in all_metrics:
                div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)
                update_metrics_dict(all_metrics, "Diversity", div_score_dict)

            if "Jerk" in all_metrics or "AUJ" in all_metrics:
                jerk_score_dict, auj_score_dict, auj_plot_values = evaluate_jerk(
                    motion_loaders
                )
                update_metrics_dict(all_metrics, "PeakJerk", jerk_score_dict)
                update_metrics_dict(all_metrics, "AUJ", auj_score_dict)
                update_metrics_dict(all_plots, "PeakJerk", auj_plot_values)
            pprint(f, "!!! DONE !!!")

    mean_dict = {}
    for target, f in zip(targets, files):
        all_metrics = targets_metrics[target]
        for metric_name, metric_dict in all_metrics.items():
            pprint(f, "========== %s Summary ==========" % metric_name)
            for model_name, values in metric_dict.items():
                if model_name not in mean_dict:
                    mean_dict[model_name] = {}
                mean, conf_interval = get_metric_statistics(
                    np.array(values), replication_times
                )
                if isinstance(mean, np.floating):
                    pprint(
                        f,
                        f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}",
                    )
                    mean_dict[model_name][metric_name] = [
                        float(mean),
                        float(conf_interval),
                    ]
                elif isinstance(mean, np.ndarray):
                    line = f"---> [{model_name}]"
                    for i in range(len(mean)):
                        line += "(top %d) Mean: %.4f CInt: %.4f;" % (
                            i + 1,
                            mean[i],
                            conf_interval[i],
                        )
                    pprint(f, line)
                    for i in range(len(mean)):
                        mean_dict[model_name][metric_name + "_top" + str(i + 1)] = [
                            float(mean[i]),
                            float(conf_interval[i]),
                        ]
        f.close()

    # handle plot values. Keep mean across repetitions
    plots_dict = {}
    for target, all_plots in targets_plots.items():
        plots_dict[target] = {}
        for metric_name, metric_dict in all_plots.items():
            plots_dict[target][metric_name] = {}
            for model_name, values in metric_dict.items():
                if model_name not in plots_dict[target][metric_name]:
                    plots_dict[model_name][metric_name] = {}
                plots_dict[model_name][metric_name] = np.mean(
                    np.array(values), axis=0
                ).tolist()

    return mean_dict, plots_dict
