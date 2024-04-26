import os
from .. import THIRD_PARTY_DIR


SMPL_MODEL_DIR = THIRD_PARTY_DIR.joinpath("body_models")

SMPL_DATA_PATH = SMPL_MODEL_DIR.joinpath("smpl")
SMPLX_DATA_PATH = SMPL_MODEL_DIR.joinpath("smplx")

SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
SMPLX_MODEL_PATH = os.path.join(SMPLX_DATA_PATH, "SMPLX_NEUTRAL.npz")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, "J_regressor_extra.npy")

ROT_CONVENTION_TO_ROT_NUMBER = {
    "legacy": 23,
    "no_hands": 21,
    "full_hands": 51,
    "mitten_hands": 33,
}

GENDERS = ["neutral", "male", "female"]
NUM_BETAS = 10

TRANSFORMS_DIR = THIRD_PARTY_DIR.joinpath("transforms")
GMM_MODEL_DIR = TRANSFORMS_DIR.joinpath("joints2smpl", "smpl_models")
SMPL_MEAN_FILE = TRANSFORMS_DIR.joinpath(
    "joints2smpl", "smpl_models", "neutral_smpl_mean_params.h5"
)

# for collision
PART_SEG_DIR = TRANSFORMS_DIR.joinpath(
    "joints2smpl", "smpl_models", "smplx_parts_segm.pkl"
)
JOINTS2JFEATS_DIR = TRANSFORMS_DIR.joinpath(
    "joints2jfeats", "joints2jfeats", "rifke", "babel-amass"
)

DATASET_STATS_DIR = THIRD_PARTY_DIR.joinpath("dataset")


GLOBAL_VELHANDY_PATH = TRANSFORMS_DIR.joinpath(
    "rots2rfeats",
    "globalvelandy",
    "rot6d",
    "babel-amass",
)
SMPLH_PATH = SMPL_MODEL_DIR.joinpath("smplh")
