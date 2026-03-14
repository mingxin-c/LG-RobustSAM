# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .build_sam import (
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    build_sam_tiny_vit,
    sam_model_registry,
)
from .predictor import SamPredictor
from .automatic_mask_generator import SamAutomaticMaskGenerator
from .config import get_config
from .build_sam import build_sam_from_config

# 为了与 EdgeSAM 兼容，提供 build_sam 别名（指向默认的 tiny_vit）
build_sam = build_sam_tiny_vit
