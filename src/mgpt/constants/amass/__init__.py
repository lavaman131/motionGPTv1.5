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


GMM_MODEL_DIR = THIRD_PARTY_DIR.joinpath("joints2smpl", "smpl_models")
SMPL_MEAN_FILE = THIRD_PARTY_DIR.joinpath(
    "joints2smpl", "smpl_models", "neutral_smpl_mean_params.h5"
)
# for collision
PART_SEG_DIR = THIRD_PARTY_DIR.joinpath(
    "transforms", "joints2smpl", "smpl_models", "smplx_parts_segm.pkl"
)

GLOBAL_VELHANDY_PATH = THIRD_PARTY_DIR.joinpath(
    "transforms",
    "rots2rfeats",
    "globalvelandy",
    "rot6d",
    "babel-amass",
    "separate_pairs",
)
SMPLH_PATH = SMPL_MODEL_DIR.joinpath("smplh")
