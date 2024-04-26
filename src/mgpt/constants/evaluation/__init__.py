from .. import EXTRAPOLATION_DIR, BABEL_DATA_DIR, HUMANML_DATA_DIR

EVAL_FILES = {
    "humanml": {
        "test": HUMANML_DATA_DIR.joinpath("test.json"),
        "extrapolation": EXTRAPOLATION_DIR.joinpath("humanml_extrapolation.json"),
    },
    "babel": {
        "val": BABEL_DATA_DIR.joinpath("babel-teach", "val.json"),
        "extrapolation": EXTRAPOLATION_DIR.joinpath("babel_extrapolation.json"),
    },
}
SCENARIOS = {
    "humanml": ["short", "medium", "long", "all"],
    "babel": ["in-distribution", "out-of-distribution"],
}
SEQ_LENS = {
    "humanml": (70, 200),
    "babel": (30, 200),
}
BATCH_SIZE = 32
