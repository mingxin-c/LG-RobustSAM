# --------------------------------------------------------
# lgrobustsam trainign script
# Based on the code: TinyViT
#   (https://github.com/microsoft/Cream/tree/main/TinyViT)
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
from collections import defaultdict
import numpy as np
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from my_meter import AverageMeter
from data.build import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, \
    NativeScalerWithGradNormCount, \
    auto_resume_helper, is_main_process, \
    add_common_args, \
    get_git_info, \
    dice_loss, sigmoid_focal_loss, sigmoid_ce_loss, calculate_uncertainty, \
    robust_three_stage_loss, robust_seg_consistency_loss, \
    seg_loss_on_logits, smoothness_regularizer

from lg_robust_sam import build_sam_from_config, get_config
from lg_robust_sam.utils.common import sample_point_in_mask, get_uncertain_point_coords_with_randomness, point_sample

# Official RobustSAM teacher modules (used only for teacher inference)
from robust_segment_anything.modeling.prompt_encoder import PromptEncoder as TPromptEncoder
from robust_segment_anything.modeling.mask_decoder import MaskDecoder as TMaskDecoder
from robust_segment_anything.modeling.transformer import TwoWayTransformer as TTransformer

import loralib

try:
    import wandb
except ImportError:
    wandb = None
NORM_ITER_LEN = 100


VIS = False
if VIS:
    from lg_robust_sam.utils.common import make_fig


def parse_option():
    parser = argparse.ArgumentParser(
        'LG-RobustSAM training script', add_help=False)
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(args, config):
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_sam_from_config(config, None, True, True)
    if not args.only_cpu:
        model.cuda()

    if args.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    logger.info(str(model))

    optimizer = build_optimizer(config, model)

    teacher_model = dict()

    if not args.only_cpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,
            find_unused_parameters=config.TRAIN.FIND_UNUSED_PARAMETERS)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Build official-teacher after model_without_ddp is available (need img_size etc.)
    if not config.DISTILL.ENCODER_ONLY:
        # Small wrappers to adapt official modules to current call signature
        class _TeacherPromptWrapper(nn.Module):
            def __init__(self, prompt_encoder: nn.Module):
                super().__init__()
                self.pe = prompt_encoder
            def forward(self, points=None, boxes=None, masks=None, num_prompts=None):
                return self.pe(points=points, boxes=boxes, masks=masks)

        class _TeacherMaskWrapper(nn.Module):
            def __init__(self, mask_decoder: nn.Module):
                super().__init__()
                self.md = mask_decoder
            def forward(self,
                        image_embeddings: torch.Tensor,
                        image_pe: torch.Tensor,
                        sparse_prompt_embeddings: torch.Tensor,
                        dense_prompt_embeddings: torch.Tensor,
                        num_multimask_outputs: int,
                        num_prompts=None,
                        encoder_features: torch.Tensor = None,
                        clear: bool = True):
                # Map to official signature
                multimask_output = bool(num_multimask_outputs) and (num_multimask_outputs != 1)
                # Official expects encoder_features as a list where encoder_features[0] is [B,H,W,C]
                if not isinstance(encoder_features, (list, tuple)):
                    encoder_features = [encoder_features]
                masks, iou_pred, up_feat, robust_token = self.md(
                    image_embeddings=image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_prompt_embeddings,
                    dense_prompt_embeddings=dense_prompt_embeddings,
                    multimask_output=multimask_output,
                    encoder_features=encoder_features,
                    robust_token_only=False,
                    clear=clear,
                )
                kd_targets = {'feat': up_feat, 'query': robust_token}
                return masks, iou_pred, kd_targets

        try:
            img_size = model_without_ddp.image_encoder.img_size
        except Exception:
            img_size = 1024

        try:
            num_mm = model_without_ddp.mask_decoder.num_multimask_outputs
        except Exception:
            num_mm = 3

        # Official teacher prompt encoder
        prompt_encoder_t = TPromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        )

        # Official teacher transformer + mask decoder
        transformer_t = TTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048)

        # vit_dim should match saved_mid channel dim (commonly 1280; change if your mid is 1024)
        mask_decoder_t = TMaskDecoder(
            transformer_dim=256,
            transformer=transformer_t,
            num_multimask_outputs=num_mm,
            vit_dim=1280,
        )

        # Load official weights
        for p in prompt_encoder_t.parameters():
            p.requires_grad = False
        pe_w = torch.load('weights/robustsam_prompt_encoder.pth', map_location='cpu')
        msg = prompt_encoder_t.load_state_dict(pe_w, strict=False)
        logger.warning(msg)
        logger.info("=> loaded successfully 'prompt_encoder_teacher' (official)")

        for p in mask_decoder_t.parameters():
            p.requires_grad = False
        md_w = torch.load('weights/robustsam_mask_decoder.pth', map_location='cpu')
        msg = mask_decoder_t.load_state_dict(md_w, strict=False)
        logger.warning(msg)
        logger.info("=> loaded successfully 'mask_decoder_teacher' (official)")

        if not args.only_cpu:
            prompt_encoder_t = prompt_encoder_t.cuda()
            mask_decoder_t = mask_decoder_t.cuda()
        prompt_encoder_t.eval()
        mask_decoder_t.eval()

        # Wrap to match current training call signature
        teacher_model['prompt_encoder'] = _TeacherPromptWrapper(prompt_encoder_t)
        teacher_model['mask_decoder'] = _TeacherMaskWrapper(mask_decoder_t)

    loss_scaler = NativeScalerWithGradNormCount(grad_scaler_enabled=config.AMP_ENABLE)

    lr_scheduler = build_scheduler(config, optimizer, len(
        data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        if not config.DISTILL.ENCODER_ONLY:
            load_pretrained(config, model_without_ddp.image_encoder, logger)

            if config.DISTILL.INIT_FROM_TEACHER:
                prompt_encoder_weights = torch.load('weights/robustsam_prompt_encoder.pth', map_location='cpu')
                msg = model_without_ddp.prompt_encoder.load_state_dict(prompt_encoder_weights, strict=False)
                logger.warning(msg)
                logger.info(f"=> loaded successfully 'prompt_encoder'")
                del prompt_encoder_weights

                mask_decoder_weights = torch.load('weights/robustsam_mask_decoder.pth', map_location='cpu')
                msg = model_without_ddp.mask_decoder.load_state_dict(mask_decoder_weights, strict=False)
                logger.warning(msg)
                logger.info(f"=> loaded successfully 'mask_decoder'")
                del mask_decoder_weights
                torch.cuda.empty_cache()

            if config.DISTILL.FREEZE_IMAGE_ENCODER:
                for param in model_without_ddp.image_encoder.parameters():
                    param.requires_grad = False

            if config.DISTILL.FREEZE_PROMPT_ENCODER:
                for param in model_without_ddp.prompt_encoder.parameters():
                    param.requires_grad = False

            if config.DISTILL.FREEZE_MASK_DECODER:
                for param in model_without_ddp.mask_decoder.parameters():
                    param.requires_grad = False
            if config.DISTILL.LORA:
                loralib.mark_only_lora_as_trainable(model_without_ddp.mask_decoder)
        else:
            load_pretrained(config, model_without_ddp, logger)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    loss_writer = None
    if dist.get_rank() == 0:
        loss_writer = SummaryWriter(f'{config.OUTPUT}/{datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")}')

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        teacher_epoch = 0 if config.DISTILL.NO_RAND else epoch
        # set_epoch for dataset_train when distillation
        if hasattr(dataset_train, 'set_epoch'):
            dataset_train.set_epoch(teacher_epoch)
        data_loader_train.sampler.set_epoch(teacher_epoch)

        train_one_epoch_distill_using_saved_embeddings(
            args, config, model, data_loader_train, optimizer, epoch,
            lr_scheduler, loss_scaler, teacher_model, loss_writer)

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp,
                            max_accuracy, optimizer, lr_scheduler, loss_scaler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def is_valid_grad_norm(num):
    if num is None:
        return False
    return not bool(torch.isinf(num)) and not bool(torch.isnan(num))


def set_bn_state(config, model):
    if config.TRAIN.EVAL_BN_WHEN_TRAINING:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()


# 🔥 删除parse_stage*函数，改用统一的EdgeSAM契约解包


def train_one_epoch_distill_using_saved_embeddings(args, config, model, data_loader, optimizer, epoch,
                                                   lr_scheduler, loss_scaler, teacher_model, loss_writer):
    model.train()
    set_bn_state(config, model)
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()
    data_tic = time.time()

    # 🔥 判断训练阶段
    is_encoder_only = config.DISTILL.ENCODER_ONLY
    stage_name = "第一阶段编码器蒸馏" if is_encoder_only else "第二阶段完整蒸馏"

    if VIS and dist.get_rank() == 0:
        vis_writer = SummaryWriter('vis')

    for idx, batch_data in enumerate(data_loader):
        # 🔥 统一EdgeSAM契约：((samples, annos), saved_embeddings_tuple)
        (samples, annos), saved_embeddings_tuple = batch_data

        samples = torch.stack(samples, dim=0).cuda(non_blocking=True)

        # 🔥 统一处理双特征：saved_embeddings_tuple 包含 (final, mid) 的列表
        saved_final = torch.stack([x[0] for x in saved_embeddings_tuple], dim=0).float().cuda(non_blocking=True)
        saved_mid = torch.stack([x[1] for x in saved_embeddings_tuple], dim=0).float().cuda(non_blocking=True)

        # Enforce teacher early feature shape to BHWC without silent tolerance
        # Expect [B,H,W,C] with C == vit_dim (e.g., 1280 or 1024)
        if saved_mid.dim() == 3:
            saved_mid = saved_mid.unsqueeze(0)
        if saved_mid.dim() != 4:
            raise ValueError(f"saved_mid must be 4D BHWC, got shape={tuple(saved_mid.shape)}")
        # If provided as NCHW, convert to BHWC deterministically
        if saved_mid.shape[-1] not in (1024, 1280) and saved_mid.shape[1] in (1024, 1280):
            saved_mid = saved_mid.permute(0, 2, 3, 1).contiguous()
        if saved_mid.shape[-1] not in (1024, 1280):
            raise ValueError(f"saved_mid channel(C) must be 1024 or 1280 in BHWC, got shape={tuple(saved_mid.shape)}")

        meters['data_time'].update(time.time() - data_tic)

        img_bs = samples.shape[0]
        img_size_before_pad = annos['img_size_before_pad']

        if not args.only_cpu:
            model_without_ddp = model.module
        else:
            model_without_ddp = model
        if config.DISTILL.ENCODER_ONLY:
            img_size_pad = (model_without_ddp.img_size, model_without_ddp.img_size)
        else:
            img_size_pad = (model_without_ddp.image_encoder.img_size, model_without_ddp.image_encoder.img_size)
            mask_threshold = model_without_ddp.mask_threshold

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            if config.DISTILL.ENCODER_ONLY:
                encoder_out = model(samples)
            else:
                encoder_out = model(mode='image_encoder', x=samples)

        # 统一双输出：编码器固定返回 (final, mid)，离线教师同样提供 (final, mid)
        encoder_embeddings, student_mid = encoder_out

        # 🔥 A2修复：提前创建通用的valid掩膜，避免作用域问题
        valid_img = torch.zeros(img_bs, 1, *img_size_pad, device=samples.device)
        for i in range(img_bs):
            h, w = img_size_before_pad[i][1:]
            valid_img[i, :, :h, :w] = 1

        loss = dict()
        if config.DISTILL.PIXEL_WISE > 0:
            _tmp = F.mse_loss(encoder_embeddings, saved_final, reduction='none') * config.DISTILL.PIXEL_WISE
            # Get rid of padding with masking
            valid_downsample = F.interpolate(valid_img, _tmp.shape[-2:], mode='bilinear', align_corners=False)
            valid_downsample = (valid_downsample > 0.5).flatten(2)
            _tmp = _tmp.flatten(2) * valid_downsample
            _tmp = _tmp.mean(1).sum(-1) / valid_downsample.sum(-1)
            _tmp = _tmp.mean()
            loss['pixel'] = (loss['pixel'] + _tmp) if 'pixel' in loss else _tmp

        # ========== 中间特征：与最终特征相同的三个分支 ==========
        if getattr(config.DISTILL, 'MID_PIXEL_WISE', 0) > 0:
            _tmp = F.mse_loss(student_mid, saved_mid, reduction='none') * float(getattr(config.DISTILL, 'MID_PIXEL_WISE', 0))
            valid_downsample = F.interpolate(valid_img, _tmp.shape[-2:], mode='bilinear', align_corners=False)
            valid_downsample = (valid_downsample > 0.5).flatten(2)
            _tmp = _tmp.flatten(2) * valid_downsample
            _tmp = _tmp.mean(1).sum(-1) / valid_downsample.sum(-1)
            _tmp = _tmp.mean()
            loss['mid_pixel'] = (loss.get('mid_pixel', 0) + _tmp)

        if getattr(config.DISTILL, 'MID_CHANNEL_WISE', 0) > 0:
            temperature = 4.0
            s = (student_mid / temperature).flatten(-2, -1).softmax(dim=-1).log()
            t = (saved_mid / temperature).flatten(-2, -1).softmax(dim=-1)
            _tmp = F.kl_div(s, t) * float(getattr(config.DISTILL, 'MID_CHANNEL_WISE', 0)) * temperature ** 2
            loss['mid_chn'] = (loss.get('mid_chn', 0) + _tmp)

        if getattr(config.DISTILL, 'MID_CORRELATION', 0) > 0:
            s = F.normalize(student_mid, p=2).flatten(-2, -1)
            student_corr = s.transpose(-2, -1) @ s
            t = F.normalize(saved_mid, p=2).flatten(-2, -1)
            teacher_corr = t.transpose(-2, -1) @ t
            _tmp = F.mse_loss(student_corr, teacher_corr) * float(getattr(config.DISTILL, 'MID_CORRELATION', 0))
            loss['mid_corr'] = (loss.get('mid_corr', 0) + _tmp)
        # ======================================================

        # TODO Doesn't make sense to apply KL_DIV on features. Need to support valid_loss.
        if config.DISTILL.CHANNEL_WISE > 0:
            temperature = 4.0
            s = (encoder_embeddings / temperature).flatten(-2, -1).softmax(dim=-1).log()
            t = (saved_final / temperature).flatten(-2, -1).softmax(dim=-1)
            _tmp = F.kl_div(s, t) * config.DISTILL.CHANNEL_WISE * temperature ** 2
            loss['chn'] = (loss['chn'] + _tmp) if 'chn' in loss else _tmp

        if config.DISTILL.CORRELATION > 0:
            s = F.normalize(encoder_embeddings, p=2).flatten(-2, -1)
            student_corr = s.transpose(-2, -1) @ s
            t = F.normalize(saved_final, p=2).flatten(-2, -1)
            teacher_corr = t.transpose(-2, -1) @ t
            _tmp = F.mse_loss(student_corr, teacher_corr) * config.DISTILL.CORRELATION
            loss['corr'] = (loss['corr'] + _tmp) if 'corr' in loss else _tmp

        if not config.DISTILL.ENCODER_ONLY:
            # 🔥 A3修复：统一使用model_without_ddp，避免CPU/单卡模式崩溃
            dense_pe = model_without_ddp.prompt_encoder.get_dense_pe()


            # 🔥 提示处理 - 完全对齐 EdgeSAM COCO 的做法
            if 'prompt_point' in annos:
                points = annos['prompt_point']
                points = torch.cat(points, dim=0)
                points = points.cuda(non_blocking=True)
                labels = torch.ones(points.shape[:2], device=samples.device)
                points = (points, labels)
            else:
                points = None

            boxes = annos['prompt_box']
            masks = None

            num_prompts = []
            for box in boxes:
                num_prompts.append(box.size(0))

            boxes = torch.cat(boxes, dim=0)
            boxes = boxes.cuda(non_blocking=True)

            # 可选：将box转为中心点
            if config.DISTILL.PROMPT_BOX_TO_POINT:
                center_x = (boxes[:, 0] + boxes[:, 1]) / 2
                center_y = (boxes[:, 2] + boxes[:, 3]) / 2
                points = torch.stack([center_x, center_y], dim=1)[:, None]
                labels = torch.ones(points.shape[:2], device=samples.device)
                points = (points, labels)

            # 🔥 恢复 PROMPT_MASK_TO_POINT 逻辑 - 与 EdgeSAM COCO 完全一致
            if config.DISTILL.PROMPT_MASK_TO_POINT:
                point_list = []
                label_list = []
                gt_mask = annos['gt_mask']
                gt_mask = torch.cat(gt_mask, dim=0)
                gt_mask = gt_mask.cuda(non_blocking=True).squeeze(1)
                for g in gt_mask:
                    candidate_indices = g.nonzero()
                    if len(candidate_indices) > 0:
                        selected_index = random.randint(0, len(candidate_indices) - 1)
                        p = candidate_indices[selected_index].flip(0)
                        l = torch.tensor(1, device=samples.device)
                    else:
                        p = torch.zeros(2, device=samples.device)
                        l = torch.tensor(-2, device=samples.device)
                    point_list.append(p)
                    label_list.append(l)
                points = torch.stack(point_list, dim=0)[:, None]
                labels = torch.stack(label_list, dim=0)[:, None]
                points = (points, labels)

            if 'point' not in config.DISTILL.PROMPT_TYPE:
                points = None

            if 'box' not in config.DISTILL.PROMPT_TYPE:
                boxes = None

            cur_prompt_type = config.DISTILL.PROMPT_TYPE
            cur_decoder_iters = config.DISTILL.DECODE_ITERS
            # 统一走多掩码：无论 box/point，始终使用配置的 MULTIMASK_OUTPUT（与 EdgeSAM 对齐）
            cur_multimask_output = config.DISTILL.MULTIMASK_OUTPUT
            if 'point' in config.DISTILL.PROMPT_TYPE and 'box' in config.DISTILL.PROMPT_TYPE:
                if torch.rand(1) > 0.5:
                    points = None
                    cur_prompt_type = 'box'
                    # 不再在 box 分支降成单通道
                    if not config.DISTILL.ITER_ON_BOX:
                        cur_decoder_iters = 1
                else:
                    boxes = None
                    cur_prompt_type = 'point'

            # Get rid of padding with masking
            valid = torch.zeros(img_bs, cur_multimask_output, *img_size_pad, device=samples.device)
            valid_list = []
            for img_i in range(img_bs):
                h, w = img_size_before_pad[img_i][1:]
                valid[img_i, :, :h, :w] = 1
                valid_list.append(valid[img_i:img_i + 1].expand(num_prompts[img_i], *valid.shape[1:]))
            valid = torch.cat(valid_list, dim=0)

            prev_point = points
            for iter_i in range(cur_decoder_iters):
                if iter_i > 0:
                    with torch.no_grad():
                        valid_down = F.interpolate(valid, mask_s.shape[2:], mode="bilinear", align_corners=False)
                        mask_s = (mask_s.detach() > mask_threshold) * valid_down
                        mask_t = (mask_t.detach() > mask_threshold) * valid_down

                        if mask_t.shape[1] > 1:
                            max_iou_idx = iou_t.argmax(dim=1)
                            batch_range = torch.arange(mask_s.shape[0], device=mask_s.device)
                            mask_s = mask_s[batch_range, max_iou_idx].unsqueeze(1)
                            mask_t = mask_t[batch_range, max_iou_idx].unsqueeze(1)
                        point, label = sample_point_in_mask(mask_s, mask_t, config.DISTILL.POINTS_PER_REFINE_ITER)

                        point[:, :, 0] = point[:, :, 0] / mask_s.shape[3] * img_size_pad[1]
                        point[:, :, 1] = point[:, :, 1] / mask_s.shape[2] * img_size_pad[0]

                        del mask_s, mask_t
                        if prev_point is not None:
                            point = torch.cat([prev_point[0], point], dim=1)
                            label = torch.cat([prev_point[1], label], dim=1)
                        points = (point, label)
                        prev_point = points

                with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                    sparse_emb_s, dense_emb_s = model(
                        mode='prompt_encoder',
                        points=points, boxes=boxes,
                        masks=masks, num_prompts=num_prompts
                    )
                    mask_s, iou_s, kd_targets_s = model(
                        mode='mask_decoder',
                        image_embeddings=encoder_embeddings,
                        image_pe=dense_pe,
                        sparse_prompt_embeddings=sparse_emb_s,
                        dense_prompt_embeddings=dense_emb_s,
                        num_multimask_outputs=cur_multimask_output,
                        num_prompts=num_prompts,
                        encoder_features=student_mid,  # 🔥 学生中间特征，[B,64,64,1280]
                        clear=False  # 🔥 学生使用退化图像，设置为False
                    )

                with torch.no_grad():
                    sparse_emb_t, dense_emb_t = teacher_model['prompt_encoder'](
                        points=points, boxes=boxes,
                        masks=masks, num_prompts=num_prompts
                    )
                    mask_t, iou_t, kd_targets_t = teacher_model['mask_decoder'](
                        image_embeddings=saved_final,
                        image_pe=dense_pe,
                        sparse_prompt_embeddings=sparse_emb_t,
                        dense_prompt_embeddings=dense_emb_t,
                        num_multimask_outputs=cur_multimask_output,
                        num_prompts=num_prompts,
                        encoder_features=saved_mid,  # 🔥 教师中间特征，[B,64,64,1280]
                        clear=False  # 🔥 教师启用去退化模块，学生学习去退化能力
                    )

                # === KD 目标一致性自检（仅首个 step 打印）===
                if idx == 0:
                    def _shape_dict(d):
                        out = {}
                        if isinstance(d, dict):
                            for k, v in d.items():
                                try:
                                    out[k] = list(v.shape)
                                except Exception:
                                    try:
                                        out[k] = str(type(v))
                                    except Exception:
                                        out[k] = 'N/A'
                        return out

                    def _check_key(k, ds, dt, name):
                        if k not in ds or k not in dt:
                            logger.warning(f"[KD-Check] missing key '{k}' in {name}: S has {k in ds}, T has {k in dt}")
                            return False
                        try:
                            if ds[k].shape != dt[k].shape:
                                logger.warning(f"[KD-Check] shape mismatch for '{k}': S {tuple(ds[k].shape)} vs T {tuple(dt[k].shape)}")
                                return False
                        except Exception:
                            logger.warning(f"[KD-Check] cannot read shape for key '{k}'")
                            return False
                        return True

                    if kd_targets_s is None or kd_targets_t is None:
                        logger.warning("[KD-Check] kd_targets_s/t is None; decoder may not yield KD targets")
                    else:
                        # 打印主要键的形状
                        logger.info(f"[KD-Check] kd_targets_s keys: {list(kd_targets_s.keys())}")
                        logger.info(f"[KD-Check] kd_targets_t keys: {list(kd_targets_t.keys())}")
                        logger.info(f"[KD-Check] S shapes: {_shape_dict(kd_targets_s)}")
                        logger.info(f"[KD-Check] T shapes: {_shape_dict(kd_targets_t)}")

                        # 检查 feat / query
                        _check_key('feat', kd_targets_s, kd_targets_t, 'feat')
                        _check_key('query', kd_targets_s, kd_targets_t, 'query')

                        # 检查注意力键（包含 t2t / i2t / t2i）
                        for tag in ['t2t', 'i2t', 't2i']:
                            for k in list(kd_targets_s.keys()):
                                if tag in k:
                                    if not _check_key(k, kd_targets_s, kd_targets_t, f'attn:{tag}'):
                                        pass

                if VIS:
                    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=samples.device).view(1, 3, 1, 1)
                    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=samples.device).view(1, 3, 1, 1)
                    _samples = (samples * pixel_std + pixel_mean).detach().int()
                    for img_i in range(img_bs):
                        if img_i == 0:
                            cur = slice(0, num_prompts[img_i])
                        else:
                            cur = slice(sum(num_prompts[:img_i]), sum(num_prompts[:img_i + 1]))
                        _boxes = boxes[cur].detach()
                        _mask_t = (mask_t[cur].detach().squeeze(1) > 0).int()
                        _mask_s = (mask_s[cur].detach().squeeze(1) > 0).int()

                        fig = make_fig(_samples[img_i], _boxes, _mask_t, _mask_s, 'gt', 'pred')
                        file_name = annos['info']['file_name']
                        file_name = file_name.split('.')[0]
                        vis_writer.add_figure(f'{file_name}/{iter_i + 1}', fig)

                if config.DISTILL.DECODER_BCE > 0 or config.DISTILL.DECODER_FOCAL > 0 or config.DISTILL.DECODER_DICE > 0:
                    valid_down = F.interpolate(valid, mask_s.shape[2:], mode='bilinear', align_corners=False)
                    _mask_s = mask_s.float()
                    _mask_t = mask_t
                    if config.DISTILL.POINT_REND_SAMPLING:
                        valid_down[valid_down < 0.5] = -torch.inf
                        valid_down[valid_down >= 0.5] = 0
                        with torch.no_grad():
                            point_coords = get_uncertain_point_coords_with_randomness(
                                mask_s + valid_down,
                                lambda logits: calculate_uncertainty(logits),
                                num_points=112 * 112, oversample_ratio=3, importance_sample_ratio=0.75
                            )
                            _mask_t = point_sample(mask_t, point_coords, align_corners=False).squeeze(1)
                        _mask_s = point_sample(mask_s, point_coords, align_corners=False).squeeze(1)
                        valid_down = None

                    temperature = config.DISTILL.TEMPERATURE
                    _mask_s /= temperature
                    _mask_t /= temperature

                    target_logit = True
                    if not config.DISTILL.USE_TEACHER_LOGITS:
                        _mask_t = (_mask_t > mask_threshold).float()
                        target_logit = False

                    if config.DISTILL.DECODER_BCE > 0:
                        _tmp = sigmoid_ce_loss(_mask_s, _mask_t, valid_down,
                                               target_logit) * config.DISTILL.DECODER_BCE / cur_decoder_iters
                        key = f'dec_bce_{iter_i}'
                        loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                    if config.DISTILL.DECODER_FOCAL > 0:
                        _tmp = sigmoid_focal_loss(_mask_s, _mask_t, valid_down,
                                                  target_logit) * config.DISTILL.DECODER_FOCAL / cur_decoder_iters
                        key = f'dec_focal_{iter_i}'
                        loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                    if config.DISTILL.DECODER_DICE > 0:
                        _tmp = dice_loss(_mask_s, _mask_t, valid_down,
                                         target_logit) * config.DISTILL.DECODER_DICE / cur_decoder_iters
                        key = f'dec_dice_{iter_i}'
                        loss[key] = (loss[key] + _tmp) if key in loss else _tmp



                # --- Added: logits-level segmentation loss + smoothness on decoder outputs ---
                if getattr(config.DISTILL, 'LOGITS_SEG_LOSS', 0.0) > 0 or \
                   getattr(config.DISTILL, 'LOGITS_TV', 0.0) > 0 or \
                   getattr(config.DISTILL, 'LOGITS_LAP', 0.0) > 0:
                    # Use logits directly (no thresholding). Align valid mask to logits size
                    valid_down_for_logits = F.interpolate(valid, mask_s.shape[2:], mode='bilinear', align_corners=False)
                    # Soft target logits if configured, scaled by temperature if any
                    student_logits = mask_s
                    teacher_logits = mask_t
                    if hasattr(config.DISTILL, 'TEMPERATURE') and config.DISTILL.TEMPERATURE and config.DISTILL.TEMPERATURE != 1.0:
                        T = float(config.DISTILL.TEMPERATURE)
                        student_logits = student_logits / T
                        teacher_logits = teacher_logits / T

                    if getattr(config.DISTILL, 'LOGITS_SEG_LOSS', 0.0) > 0:
                        seg_total, seg_detail = seg_loss_on_logits(
                            student_logits, teacher_logits,
                            valid=valid_down_for_logits,
                            use_soft_target=getattr(config.DISTILL, 'USE_TEACHER_LOGITS', True),
                            bce_weight=float(getattr(config.DISTILL, 'LOGITS_SEG_BCE', 0.5)),
                            dice_weight=float(getattr(config.DISTILL, 'LOGITS_SEG_DICE', 0.5)),
                        )
                        seg_total = seg_total * float(getattr(config.DISTILL, 'LOGITS_SEG_LOSS', 1.0)) / cur_decoder_iters
                        loss[f'logits_seg_{iter_i}'] = (loss.get(f'logits_seg_{iter_i}', 0) + seg_total)
                        meters['seg_bce'].update(seg_detail['seg_bce'].item(), n=1)
                        meters['seg_dice'].update(seg_detail['seg_dice'].item(), n=1)

                    tv_w = float(getattr(config.DISTILL, 'LOGITS_TV', 0.0))
                    lap_w = float(getattr(config.DISTILL, 'LOGITS_LAP', 0.0))

                    # Apply a simple epoch-based warm-up to TV weight for stability
                    if tv_w > 0 and getattr(config.TRAIN, 'WARMUP_EPOCHS', 0) and config.TRAIN.WARMUP_EPOCHS > 0:
                        _tv_factor = min(1.0, float(epoch + 1) / float(config.TRAIN.WARMUP_EPOCHS))
                    else:
                        _tv_factor = 1.0
                    tv_w_eff = tv_w * _tv_factor

                    if tv_w_eff > 0 or lap_w > 0:
                        smooth_total, smooth_detail = smoothness_regularizer(
                            student_logits, valid=valid_down_for_logits,
                            tv_weight=tv_w_eff, lap_weight=lap_w
                        )
                        smooth_total = smooth_total / cur_decoder_iters
                        loss[f'logits_smooth_{iter_i}'] = (loss.get(f'logits_smooth_{iter_i}', 0) + smooth_total)
                        if 'tv' in smooth_detail: meters['tv'].update(smooth_detail['tv'].item(), n=1)
                        if 'lap' in smooth_detail: meters['lap'].update(smooth_detail['lap'].item(), n=1)
                # --- end Added ---



                    if config.DISTILL.DECODER_IOU > 0:
                        _tmp = F.mse_loss(iou_s, iou_t) * config.DISTILL.DECODER_IOU / cur_decoder_iters
                        key = f'dec_iou_{iter_i}'
                        loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                if config.DISTILL.DECODER_ATTN > 0:
                    _tmp, count = 0, 0
                    looking_for = ['t2t', 'i2t', 't2i']
                    for key in kd_targets_s:
                        for tgt in looking_for:
                            if tgt in key:
                                count += 1
                                _tmp += F.mse_loss(kd_targets_s[key], kd_targets_t[key]) / cur_decoder_iters
                    _tmp = _tmp / count * config.DISTILL.DECODER_ATTN
                    key = f'dec_attn_{iter_i}'
                    loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                # The magnitude of the feature and query are very different, so normalization is needed.
                if config.DISTILL.DECODER_FEAT > 0:
                    feat_s = kd_targets_s['feat']
                    feat_t = kd_targets_t['feat']
                    # Cosine similarity (non-linear)
                    # feat_s = F.normalize(feat_s, dim=1)
                    # feat_t = F.normalize(feat_t, dim=1)
                    # _tmp = (1 - torch.einsum('bchw,bchw->bhw', feat_s, feat_t)).mean() * config.DISTILL.DECODER_FEAT

                    # L1 norm (linear)
                    # feat_s = F.normalize(feat_s, dim=1, p=1)
                    # feat_t = F.normalize(feat_s, dim=1, p=1)
                    _tmp = F.mse_loss(feat_s, feat_t) * config.DISTILL.DECODER_FEAT / cur_decoder_iters

                    key = f'dec_feat_{iter_i}'
                    loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                if config.DISTILL.DECODER_QUERY > 0:
                    query_s = kd_targets_s['query']
                    query_t = kd_targets_t['query']
                    # Cosine similarity (non-linear)
                    # query_s = F.normalize(query_s, dim=-1)
                    # query_t = F.normalize(query_t, dim=-1)
                    # _tmp = (1 - torch.einsum('bnc,bnc->bn', query_s, query_t)).mean() * config.DISTILL.DECODER_QUERY

                    # L1 norm (linear)
                    query_s = F.normalize(query_s, dim=1, p=1)
                    query_t = F.normalize(query_t, dim=1, p=1)
                    _tmp = F.mse_loss(query_s, query_t) * config.DISTILL.DECODER_QUERY / cur_decoder_iters

                    key = f'dec_query_{iter_i}'
                    loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                # 🔥 RobustSAM风格的三段式一致性损失
                if any(getattr(config.DISTILL, f'ROBUST_{x}', 0) > 0 for x in [
                    'MFC_MID', 'MFC_DECODER', 'MFC_CORRELATION',
                    'TC_QUERY', 'TC_ATTN',
                    'SEG_DICE', 'SEG_FOCAL', 'SEG_BCE'
                ]):
                    # 构造学生输出字典（严格对齐 RobustSAM：不包含中间特征）
                    student_outputs = {
                        'mask': mask_s,
                        'decoder_feat': kd_targets_s.get('feat') if kd_targets_s else None,
                        'query': kd_targets_s.get('query') if kd_targets_s else None,
                        'attn': {k: v for k, v in kd_targets_s.items()
                                if kd_targets_s and any(t in k for t in ['t2t', 'i2t', 't2i'])}
                    }

                    # 构造教师输出字典（严格对齐 RobustSAM：不包含中间特征）
                    teacher_outputs = {
                        'mask': mask_t,
                        'decoder_feat': kd_targets_t.get('feat') if kd_targets_t else None,
                        'query': kd_targets_t.get('query') if kd_targets_t else None,
                        'attn': {k: v for k, v in kd_targets_t.items()
                                if kd_targets_t and any(t in k for t in ['t2t', 'i2t', 't2i'])}
                    }

                    # 计算特征和token一致性损失
                    robust_losses = robust_three_stage_loss(
                        student_outputs, teacher_outputs, config, valid_down, cur_decoder_iters
                    )

                    # 计算分割损失：改为学生 vs 真实 GT（避免与迭代处学生 vs 教师重复）
                    # GT 尺度对齐到当前 logits，并在多掩码输出时按通道广播
                    gt_mask = annos['gt_mask']
                    gt_mask = torch.cat(gt_mask, dim=0).cuda(non_blocking=True)  # [N,1,1024,1024]
                    gt_down = F.interpolate(gt_mask, size=mask_s.shape[2:], mode='nearest')
                    target = gt_down
                    if mask_s.shape[1] > 1:
                        target = gt_down.expand(-1, mask_s.shape[1], -1, -1)

                    # 添加到总损失字典
                    for key, value in robust_losses.items():
                        loss[f'{key}_{iter_i}'] = value

                    if getattr(config.DISTILL, 'ROBUST_SEG_DICE', 0) > 0:
                        key = f'robust_seg_dice_{iter_i}'
                        _tmp = dice_loss(mask_s, target, valid_down, target_logit=False) \
                               * float(config.DISTILL.ROBUST_SEG_DICE) / cur_decoder_iters
                        loss[key] = (loss.get(key, 0) + _tmp)

                    if getattr(config.DISTILL, 'ROBUST_SEG_FOCAL', 0) > 0:
                        key = f'robust_seg_focal_{iter_i}'
                        _tmp = sigmoid_focal_loss(mask_s, target, valid_down, target_logit=False) \
                               * float(config.DISTILL.ROBUST_SEG_FOCAL) / cur_decoder_iters
                        loss[key] = (loss.get(key, 0) + _tmp)

                    if getattr(config.DISTILL, 'ROBUST_SEG_BCE', 0) > 0:
                        key = f'robust_seg_bce_{iter_i}'
                        _tmp = sigmoid_ce_loss(mask_s, target, valid_down, target_logit=False) \
                               * float(config.DISTILL.ROBUST_SEG_BCE) / cur_decoder_iters
                        loss[key] = (loss.get(key, 0) + _tmp)

        for key in loss:
            loss[key] = loss[key] / config.TRAIN.ACCUMULATION_STEPS
            meters[key].update(loss[key].item(), len(samples))

        total_loss = sum(loss.values())

        if loss_writer is not None:
            display_dict = {'total': total_loss}
            for key in loss:
                display_dict[key] = loss[key].item()

            loss_writer.add_scalars('loss', display_dict, epoch * num_steps + idx)


            # Also log auxiliary breakdown metrics if available (not counted into total)
            aux_dict = {}
            for k in ['seg_bce', 'seg_dice', 'tv', 'lap']:
                if k in meters:
                    try:
                        aux_dict[k] = float(meters[k].val)
                    except Exception:
                        pass
            if aux_dict:
                loss_writer.add_scalars('aux', aux_dict, epoch * num_steps + idx)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(total_loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        torch.cuda.synchronize()

        loss_meter.update(total_loss.item(), len(samples))
        if is_valid_grad_norm(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()
        data_tic = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            extra_meters_str = ''
            for k, v in meters.items():
                extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'{extra_meters_str}'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    extra_meters_str = f'Train-Summary: [{epoch}/{config.TRAIN.EPOCHS}]\t'
    for k, v in meters.items():
        v.sync()
        extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
    logger.info(extra_meters_str)
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    args, config = parse_option()
    config.defrost()
    config.freeze()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    if args.only_cpu:
        ddp_backend = 'gloo'
    else:
        torch.cuda.set_device(config.LOCAL_RANK)
        ddp_backend = 'nccl'

    torch.distributed.init_process_group(
        backend=ddp_backend, init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # Disable linear LR scaling: use config values directly without scaling
    config.defrost()
    # Keep BASE_LR / WARMUP_LR / MIN_LR exactly as provided via YAML/CLI
    config.TRAIN.BASE_LR = config.TRAIN.BASE_LR
    if hasattr(config.TRAIN, 'WARMUP_LR'):
        config.TRAIN.WARMUP_LR = config.TRAIN.WARMUP_LR
    if hasattr(config.TRAIN, 'MIN_LR'):
        config.TRAIN.MIN_LR = config.TRAIN.MIN_LR
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if is_main_process():
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        config_dict = dict(config)
        config_dict['git'] = get_git_info()
        if args.use_wandb:
            wandb_output_path = config.OUTPUT
            wandb.init(project="EdgeSAM", config=config_dict,
                       dir=wandb_output_path)

    # print git info
    logger.info('===== git =====')
    logger.info(str(get_git_info()))

    # print config
    logger.info(config.dump())

    main(args, config)
