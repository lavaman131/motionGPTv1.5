import os

os.environ["DISPLAY"] = ":0.0"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
from argparse import ArgumentParser
import numpy as np
import OpenGL.GL as gl
import imageio
import random
import torch
import moviepy.editor as mp
from scipy.spatial.transform import Rotation as RRR
from mgpt.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from mgpt.render.pyrender.smpl_render import SMPLRender
from mgpt.constants.transforms import SMPL_DATA_PATH
from mgpt.render.matplot.plot_script import (
    construct_template_variables,
    save_multiple_samples,
    plot_3d_motion_mix,
)
from mgpt.data_loaders.humanml.utils import paramUtil
from pathlib import Path

datasets_fps = {"humanml": 20, "babel": 30}


def fast_render(data, save_fname, output_path, unconstrained=False, dataset="babel"):
    all_motions = data.item()["motion"]
    all_text = data.item()["text"]
    lengths_list = data.item()["lengths"]
    num_repetitions = data.item()["num_repetitions"]
    num_samples = data.item()["num_samples"]

    skeleton = paramUtil.t2m_kinematic_chain

    (
        sample_print_template,
        row_print_template,
        sample_file_template,
        row_file_template,
    ) = construct_template_variables(save_fname, unconstrained)

    fps = datasets_fps[dataset]

    output_path = Path(output_path)

    try:
        rep_files = []
        for rep_i in range(num_repetitions):
            caption = all_text[rep_i * num_samples]
            motion = all_motions[rep_i * num_samples].transpose(2, 0, 1)
            save_file = sample_file_template.format(rep_i)
            print(sample_print_template.format(rep_i, save_file))
            animation_save_path = output_path.joinpath(save_file)
            captions_list = []
            for c, l in zip(caption.split(" /// "), lengths_list):
                captions_list += [
                    c,
                ] * l
            plot_3d_motion_mix(
                str(animation_save_path),
                skeleton,
                motion,
                dataset=dataset,
                title=captions_list,
                fps=fps,
                vis_mode="alternate",
                lengths=lengths_list,
            )
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(str(animation_save_path))
    except Exception as e:
        print(f"Error while processing sample: {e}")

    save_multiple_samples(
        num_repetitions,
        str(output_path),
        row_print_template,
        row_file_template,
        rep_files,
    )


def slow_render(data, output_path, smpl_model_path):
    all_motions = data.item()["motion"]
    if all_motions.ndim == 4:
        all_motions = all_motions[0]
    all_motions = all_motions.transpose(2, 0, 1)
    all_motions = all_motions - all_motions[0, 0]
    pose_generator = HybrIKJointsToRotmat()
    pose = pose_generator(all_motions)
    pose = np.concatenate(
        [pose, np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)], 1
    )
    shape = [768, 768]
    render = SMPLRender(smpl_model_path)

    r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
    pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
    vid = []
    aroot = all_motions[:, 0]
    aroot[:, 1:] = -aroot[:, 1:]
    params = dict(pred_shape=np.zeros([1, 10]), pred_root=aroot, pred_pose=pose)
    render.init_renderer([shape[0], shape[1], 3], params)
    for i in range(all_motions.shape[0]):
        renderImg = render.render(i)
        vid.append(renderImg)

    out = np.stack(vid, axis=0)
    output_gif_path = str(output_path.joinpath("output.gif"))
    output_mp4_path = str(output_path.joinpath("output.mp4"))
    imageio.mimwrite(output_gif_path, out, duration=50)
    out_video = mp.VideoFileClip(output_gif_path)
    out_video.write_videofile(output_mp4_path, codec="libx264")
