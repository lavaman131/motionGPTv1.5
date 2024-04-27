from pathlib import Path
from typing import Union, TypeAlias
import os


FileName: TypeAlias = Union[str, bytes, os.PathLike]
PROMPT_LIBRARY_DIR = Path(__file__).parent.joinpath("prompt_library")
PRECOMPUTED_DIR = Path(__file__).parent.joinpath("precomputed")
