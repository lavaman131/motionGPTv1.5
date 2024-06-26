# MotionGPTv1.5

# 🚀 Get started

## 🛠️ Installation

```bash
# install ffmpeg
sudo apt-get install ffmpeg

# recommended to use gcc 10.2.0 for torch

# install packages
conda env create -f environment.yml

conda activate mgpt

pip install -r requirements.txt

python -m spacy download en_core_web_sm
```

## 🗂️ Data preparation (required only for evaluation/training)

<details>

**Babel dataset**:

1. Download the processed version [here](https://drive.google.com/file/d/18a4eRh8mbIFb55FMHlnmI8B8tSTkbp4t/view?usp=share_link), and place it at `./dataset/babel`.

2. Download the following [here](https://drive.google.com/file/d/1PBlbxawaeFTxtKkKDsoJwQGuDTdp52DD/view?usp=sharing), and place it at `./dataset/babel`.

</details>

# 🎬 Visualization

To generate the motions use the following scripts:

For Babel dataset run:

```bash
python tools/motion_gpt.py \
--user_prompt "Can you generate the hip abduction motion?" \
--model_path ./pretrained_models/babel/model001300000.pt
```

## 🎨 Rendering

### Render keypoints

To render keypoints per frame (replacing `--dataset` with the correct name) run:

```bash
python tools/render.py \
--joints_path ./results/output/results.npy \
--method fast \
--output_path ./results/output
```

### Render SMPL mesh

To create SMPL mesh per frame (replacing `--dataset` with the correct name) run:

```bash
python tools/render.py \
--joints_path ./results/output/results.npy \
--method slow \
--dataset babel \
--output_path ./results/output
```

**This script outputs:**
* `sample_rep##_smpl_params.npy` - SMPL parameters (thetas, root translations, vertices and faces)
* `sample_rep##_obj` - Mesh per frame in `.obj` format.

**Notes:**
* The `.obj` can be integrated into Blender/Maya/3DS-MAX and rendered using them.
* This script is running [SMPLify](https://smplify.is.tue.mpg.de/) and needs GPU as well (can be specified with the `--device` flag).
* **Important** - Do not change the original `.mp4` path before running the script.

**Notes for 3d makers:**
* You have two ways to animate the sequence:
  1. Use the [SMPL add-on](https://smpl.is.tue.mpg.de/index.html) and the theta parameters saved to `sample_rep##_smpl_params.npy` (we always use beta=0 and the gender-neutral model).
  2. A more straightforward way is using the mesh data itself. All meshes have the same topology (SMPL), so you just need to keyframe vertex locations. 
     Since the OBJs are not preserving vertices order, we also save this data to the `sample_rep##_smpl_params.npy` file for your convenience.

# 📊 FlowMDM Evaluation

To reproduce the Babel evaluation over the motion and transition run:

```bash
python tools/evaluate.py \
--model_path ./pretrained_models/babel/model001300000.pt \
--dataset babel \
--eval_mode final \
--bpe_denoising_step 60 \
--guidance_param 1.5 \
--transition_length 30
```

Add `--use_chunked_att` to accelerate inference for very long compositions (imported from LongFormer, and recommended for HumanML3D). Evaluation can take >12h for the 10 repetitions depending on the GPU power. Use `--eval_mode fast` for a quick evaluation run (3 rep.). 

> [!NOTE]
> During evaluation, generated motions are backed up to be used in successive evaluations. This is useful in case you want to change some evaluation parameters such as the `transition_length` and avoid regenerating all evaluation sequences. 

# 🏋️‍♂️ FlowMDM Training

Configure the following paths in `./src/mgpt/constants/__init__.py` to your configuration:

```python
# CHANGE HERE TO PATHS FOR YOUR CONFIGURATION
BABEL_DATA_DIR = Path("/projectnb/ivc-ml/alavaee/data/babel")
HUMANML_DATA_DIR = Path("/projectnb/ivc-ml/alavaee/data/humanml")
EXTRAPOLATION_DIR = Path("/projectnb/ivc-ml/alavaee/data/babel/extrapolation")
PRECOMPUTED_DIR = Path("/projectnb/ivc-ml/alavaee/data/precomputed")
#############################################
```

To retrain FlowMDM with Babel dataset run:

```bash
python tools/train.py \
--save_dir ./results/babel/FlowMDM_retrained \
--dataset babel \
--batch_size 64 \
--num_steps 1300000 \
--rpe_horizon 100 \
--train_platform_type WandbPlatform \
--wandb_project mgpt \
--wandb_entity lavaalex
```

# 🤝 Acknowledgements

* [MotionGPT](https://github.com/OpenMotionLab/MotionGPT): Inspired ideas for the project.
* [FlowMDM](https://github.com/BarqueroGerman/FlowMDM/tree/main): Inherited for 3D motion synthesis.
* [llama3](https://github.com/meta-llama/llama3)