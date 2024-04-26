# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from typing import Optional, Union
import torch
from torch import Tensor, nn
from pathlib import Path
from mgpt.constants.transforms import GLOBAL_VELHANDY_PATH


class Rots2Rfeats(nn.Module):
    def __init__(
        self,
        path: Optional[Union[str, Path]] = GLOBAL_VELHANDY_PATH,
        normalization: bool = True,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        if normalization and path is None:
            raise TypeError("You should provide a path if normalization is on.")

        super().__init__()
        self.normalization = normalization
        self.eps = eps
        if normalization:
            mean_path = Path(path) / "rfeats_mean.pt"
            std_path = Path(path) / "rfeats_std.pt"

            self.register_buffer("mean", torch.load(mean_path))
            self.register_buffer("std", torch.load(std_path))

    def normalize(self, features: Tensor) -> Tensor:
        if self.normalization:
            features = (features - self.mean) / (self.std + self.eps)
        return features

    def unnormalize(self, features: Tensor) -> Tensor:
        if self.normalization:
            features = features * self.std + self.mean
        return features
