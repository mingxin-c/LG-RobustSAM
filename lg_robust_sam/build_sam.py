# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, TinyViT

# Use local batch training variants (already copied into this repo)
from .modeling.sam_batch import SamBatch, PromptEncoderBatch, MaskDecoderBatch

# Constants used across builders
prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_sz = image_size // vit_patch_size

def build_sam_vit_h(opt=None, checkpoint=None, train=False, **kwargs):
    raise NotImplementedError("ViT-H encoder not included in this repo; use tiny_vit or add ImageEncoderViT locally.")

def build_sam_vit_l(opt=None, checkpoint=None, train=False, **kwargs):
    raise NotImplementedError("ViT-L encoder not included in this repo; use tiny_vit or add ImageEncoderViT locally.")


def build_sam_vit_b(opt=None, checkpoint=None, train=False, **kwargs):
    raise NotImplementedError("ViT-B encoder not included in this repo; use tiny_vit or add ImageEncoderViT locally.")


def build_sam_tiny_vit(opt=None, checkpoint=None, train=False,
                        img_size: int = 1024,
                        tiny_embed_dims=(96, 192, 384, 576),
                        tiny_depths=(2, 2, 6, 2),
                        tiny_num_heads=(3, 6, 12, 18),
                        tiny_window_sizes=(7, 7, 14, 7),
                        drop_path_rate: float = 0.1,
                        **kwargs):
    enc = TinyViT(
        img_size=img_size,
        embed_dims=list(tiny_embed_dims),
        depths=list(tiny_depths),
        num_heads=list(tiny_num_heads),
        window_sizes=list(tiny_window_sizes),
        mlp_ratio=4.,
        drop_path_rate=drop_path_rate,
    )
    encoder_only_flag = kwargs.pop('encoder_only', False)
    if encoder_only_flag:
        # 移除分类头，以免在 ENCODER_ONLY 阶段成为未参与梯度的参数
        import torch.nn as nn
        if hasattr(enc, 'head') and isinstance(getattr(enc, 'head'), nn.Module):
            enc.head = nn.Identity()
        if hasattr(enc, 'norm_head') and isinstance(getattr(enc, 'norm_head'), nn.Module):
            enc.norm_head = nn.Identity()
        return enc
    return _build_sam_unified(enc, checkpoint, vit_dim=1280, opt=opt, **kwargs)


sam_model_registry = {
    "default": build_sam_tiny_vit,
    "tiny_vit": build_sam_tiny_vit,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}



# --- Unified builder that accepts an encoder instance and can return batch variants ---

def _build_sam_unified(
    image_encoder,
    checkpoint=None,
    enable_batch: bool = False,
    enable_distill: bool = False,
    lora: bool = False,
    vit_dim: int = 1280,
    opt=None,
):
    SamCls = SamBatch if (enable_batch and SamBatch is not None) else Sam
    PECls  = PromptEncoderBatch if (enable_batch and PromptEncoderBatch is not None) else PromptEncoder
    # Always use the robust MaskDecoder; the Batch variant is EdgeSAM-style and
    # not compatible with our AMFG/AOTG signatures.
    DecCls = MaskDecoder

    sam = SamCls(
        image_encoder=image_encoder,
        prompt_encoder=PECls(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_sz, image_embedding_sz),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=DecCls(
            opt=opt,
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
                # lora=lora,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=vit_dim,
            # yield_kd_targets is only supported in batch variant; DecCls may ignore
            # this kwarg if not implemented
            # For compatibility, only pass when DecCls has this arg
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location='cpu')
        try:
            print(sam.load_state_dict(state_dict, strict=False))
        except Exception:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v if k.startswith('module.') else v
            print(sam.load_state_dict(new_state_dict, strict=False))

    if not enable_distill:
        sam.eval()

    return sam

# --- Build from config (EdgeSAM-style) ---

def build_sam_from_config(cfg, checkpoint=None, enable_distill=False, enable_batch=False, **kwargs):
    """EdgeSAM-style entry: given a yacs config object, build the model.
    We expect cfg.MODEL.TYPE in {tiny_vit, vit_h, vit_l, vit_b} and
    cfg.DISTILL.ENCODER_ONLY flag. Other options are passed to builders.
    """
    model_type   = getattr(cfg.MODEL, 'TYPE', 'tiny_vit')
    encoder_only = getattr(cfg.DISTILL, 'ENCODER_ONLY', False)
    lora         = getattr(cfg.DISTILL, 'LORA', False)

    common_kwargs = dict(
        opt=cfg,
        checkpoint=checkpoint,
        encoder_only=encoder_only,
        enable_distill=enable_distill,
        enable_batch=enable_batch,
        lora=lora,
    )

    builder = sam_model_registry.get(model_type, sam_model_registry['default'])
    return builder(**common_kwargs)
