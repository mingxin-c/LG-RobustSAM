# --------------------------------------------------------
# TinyViT Utils (save/load checkpoints, etc.)
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Adapted for TinyViT
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import subprocess
import copy


def add_common_args(parser):
    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--only-cpu', action='store_true',
                        help='Perform evaluation on CPU')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')
    parser.add_argument('--use-sync-bn', action='store_true',
                        default=False, help='sync bn')
    parser.add_argument('--use-wandb', action='store_true',
                        default=False, help='use wandb to record log')

    # distributed training
    parser.add_argument("--local-rank", type=int,
                        help='local rank for DistributedDataParallel')


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(
        f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    if config.MODEL.TYPE == 'vit_h':
        new_checkpoint = dict()
        for key in checkpoint.keys():
            if 'image_encoder' in key:
                new_key = key[len('image_encoder.'):]
                new_checkpoint[new_key] = checkpoint[key]
        msg = model.load_state_dict(new_checkpoint, strict=False)
        logger.info(msg)
    else:
        params = checkpoint['model']
        now_model_state = model.state_dict()
        mnames = ['head.weight', 'head.bias']  # (cls, 1024), (cls, )
        # 只在模型确实有线性分类头且checkpoint提供head参数时才对齐
        if mnames[-1] in params and mnames[-1] in now_model_state:
            ckpt_head_bias = params[mnames[-1]]
            now_model_bias = now_model_state[mnames[-1]]
            if ckpt_head_bias.shape != now_model_bias.shape:
                num_classes = 1000

                if len(ckpt_head_bias) == 21841 and len(now_model_bias) == num_classes:
                    logger.info("Convert checkpoint from 21841 to 1k")
                    # convert 22kto1k
                    fname = './imagenet_1kto22k.txt'
                    with open(fname) as fin:
                        mapping = torch.Tensor(
                            list(map(int, fin.readlines()))).to(torch.long)
                    for name in mnames:
                        v = params[name]
                        shape = list(v.shape)
                        shape[0] = num_classes
                        mean_v = v[mapping[mapping != -1]].mean(0, keepdim=True)
                        v = torch.cat([v, mean_v], 0)
                        v = v[mapping]
                        params[name] = v
        msg = model.load_state_dict(params, strict=False)
        logger.info(msg)

    max_accuracy = 0.0
    if not config.EVAL_MODE:
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            logger.info(
                f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            if 'max_accuracy' in checkpoint:
                max_accuracy = checkpoint['max_accuracy']

        if 'epoch' in checkpoint:
            config.defrost()
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
            config.freeze()

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    logger.info(
        f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        raise NotImplementedError()

    if 'efficient_vit' in config.MODEL.TYPE:
        new_state_dict = dict()
        for key in state_dict:
            new_state_dict[key.replace('backbone.', '')] = state_dict[key]

        msg = model.load_state_dict(new_state_dict, strict=False)
        logger.warning(msg)

        logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

        del checkpoint
        torch.cuda.empty_cache()
        return

    if 'rep_vit' in config.MODEL.TYPE:
        msg = model.load_state_dict(state_dict, strict=False)
        logger.warning(msg)

        logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

        del checkpoint
        torch.cuda.empty_cache()
        return

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    model_state_dict = model.state_dict()

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model_state_dict[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                    nH2, L2).permute(1, 0)

    # bicubic interpolate attention_biases if not match
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "attention_biases" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model_state_dict[k]
        nH1, L1 = relative_position_bias_table_pretrained.size()
        nH2, L2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                    nH2, L2)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [
        k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                    -1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                    0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(
                    0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(
                    1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier head only when model has a real classifier (not Identity) and ckpt provides it
    if hasattr(model, 'head') and hasattr(getattr(model, 'head'), 'bias') and ('head.bias' in state_dict):
        head_bias_pretrained = state_dict['head.bias']
        Nc1 = head_bias_pretrained.shape[0]
        Nc2 = model.head.bias.shape[0]
        if (Nc1 != Nc2):
            if Nc1 == 21841 and Nc2 == 1000:
                logger.info("loading ImageNet-21841 weight to ImageNet-1K ......")
                map22kto1k_path = f'./imagenet_1kto22k.txt'
                with open(map22kto1k_path) as fin:
                    mapping = torch.Tensor(
                        list(map(int, fin.readlines()))).to(torch.long)
                for name in ['head.weight', 'head.bias']:
                    v = state_dict[name]
                    mean_v = v[mapping[mapping != -1]].mean(0, keepdim=True)
                    v = torch.cat([v, mean_v], 0)
                    v = v[mapping]
                    state_dict[name] = v
            else:
                # model has a linear head but shapes mismatch; re-init model head and drop ckpt head
                try:
                    torch.nn.init.constant_(model.head.bias, 0.)
                    torch.nn.init.constant_(model.head.weight, 0.)
                except Exception:
                    pass
                if 'head.weight' in state_dict:
                    del state_dict['head.weight']
                if 'head.bias' in state_dict:
                    del state_dict['head.bias']
                logger.warning(
                    f"Error in loading classifier head, re-init classifier head to 0")
    # If model.head has no bias/weight (e.g., Identity), skip classifier alignment safely

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d)
                                 for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, grad_scaler_enabled=True):
        self._scaler = torch.cuda.amp.GradScaler(enabled=grad_scaler_enabled)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None and clip_grad > 0.0:
                assert parameters is not None
                # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def is_main_process():
    return dist.get_rank() == 0


def run_cmd(cmd, default=None):
    try:
        return subprocess.check_output(cmd.split(), universal_newlines=True).strip()
    except:
        if default is None:
            raise
        return default


def get_git_info():
    return dict(
        branch=run_cmd('git rev-parse --abbrev-ref HEAD', 'custom'),
        git_hash=run_cmd('git rev-parse --short HEAD', 'custom'),
    )


def _reshape_mask(mask):
    return mask.reshape(mask.shape[0] * mask.shape[1], mask.shape[2] * mask.shape[3])


def sigmoid_focal_loss(inputs, targets, valid=None, target_logit=False, alpha=0.25, gamma=2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    inputs = inputs.sigmoid()
    if target_logit:
        targets = targets.sigmoid()
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if valid is not None:
        loss = loss.flatten(2)
        valid = valid.flatten(2)
        loss = loss * valid
        loss = loss.sum(-1) / valid.sum(-1)

    return loss.mean()


def sigmoid_ce_loss(inputs, targets, valid=None, target_logit=False):
    """
    Stable BCE-on-logits implementation (no容错):
    - inputs: raw logits
    - targets: probabilities in [0,1]; if target_logit=True, apply sigmoid first
    """
    if target_logit:
        targets = targets.sigmoid()
    # 强校验：禁止 NaN/Inf 与越界目标
    assert torch.isfinite(inputs).all(), "BCE logits contains NaN/Inf"
    assert torch.isfinite(targets).all(), "BCE targets contains NaN/Inf"
    if targets.min() < 0 or targets.max() > 1:
        raise ValueError(f"BCE targets out of [0,1]: min={targets.min().item():.4f}, max={targets.max().item():.4f}")

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    if valid is not None:
        loss = loss.flatten(1)
        valid = valid.flatten(1)
        loss = loss * valid
        denom = valid.sum(-1)
        # 强校验：有效像素必须>0
        if torch.any(denom <= 0):
            raise ValueError("valid mask has zero valid pixels for some samples")
        loss = loss.sum(-1) / denom

    return loss.mean()


def dice_loss(inputs, targets, valid=None, target_logit=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = _reshape_mask(inputs)

    if target_logit:
        targets = targets.sigmoid()
    targets = _reshape_mask(targets)

    if valid is not None:
        valid = _reshape_mask(valid)
        inputs = inputs * valid
        targets = targets * valid

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def robust_three_stage_loss(student_outputs, teacher_outputs, config, valid_mask=None, cur_decoder_iters=1):
    """
    RobustSAM风格的三段式一致性损失（师生版本）

    Args:
        student_outputs: dict with keys ['mask', 'decoder_feat', 'query', 'attn']
        teacher_outputs: dict with same keys as student_outputs
        config: 配置对象
        valid_mask: 有效区域掩码 [N, 1, H, W]
        cur_decoder_iters: 当前解码器迭代次数，用于归一化损失

    Returns:
        dict: 各项损失的字典
    """
    loss_dict = {}

    # ========== L_MFC: Mask Feature Consistency Loss ==========

    # 仅解码器特征一致性（严格对齐 RobustSAM）+ 有效域遮罩
    if config.DISTILL.ROBUST_MFC_DECODER > 0:
        student_feat = student_outputs.get('decoder_feat')
        teacher_feat = teacher_outputs.get('decoder_feat')
        if student_feat is not None and teacher_feat is not None:
            if valid_mask is None:
                mfc = F.mse_loss(student_feat, teacher_feat)
            else:
                # 对齐有效域到特征分辨率，并在通道维广播
                vm = valid_mask
                try:
                    if vm.shape[-2:] != student_feat.shape[-2:]:
                        vm = F.interpolate(vm, size=student_feat.shape[-2:], mode='nearest')
                except Exception:
                    pass
                # vm 形状可能为 [N, M, H, W]（M=多掩码数）。先在掩码维做聚合→[N,1,H,W]
                if vm.dim() == 4 and vm.shape[1] != 1:
                    vm = vm.max(dim=1, keepdim=True)[0]
                # 广播到通道维 C
                if vm.dim() == 4 and vm.shape[1] == 1:
                    vm = vm.expand(-1, student_feat.shape[1], -1, -1)
                # dtype 对齐，防止 AMP 下半精-单精混算
                vm = vm.to(student_feat.dtype)
                diff = (student_feat - teacher_feat) ** 2
                masked = diff * vm
                denom = vm.sum()
                mfc = masked.sum() / denom.clamp_min(1.0)

            loss_dict['robust_mfc_decoder'] = mfc * config.DISTILL.ROBUST_MFC_DECODER / cur_decoder_iters

    # ========== L_TC: Token Consistency Loss ==========

    # 1) 查询token一致性
    if config.DISTILL.ROBUST_TC_QUERY > 0:
        student_query = student_outputs.get('query')
        teacher_query = teacher_outputs.get('query')
        if student_query is not None and teacher_query is not None:
            # 直接对原始 token 做 MSE（对齐 RobustSAM 的实现风格）
            loss_dict['robust_tc_query'] = F.mse_loss(
                student_query, teacher_query
            ) * config.DISTILL.ROBUST_TC_QUERY / cur_decoder_iters

    # 2) 注意力一致性
    if config.DISTILL.ROBUST_TC_ATTN > 0:
        student_attn = student_outputs.get('attn', {})
        teacher_attn = teacher_outputs.get('attn', {})
        if student_attn and teacher_attn:
            attn_loss = 0
            count = 0
            looking_for = ['t2t', 'i2t', 't2i']
            for key in student_attn:
                for tgt in looking_for:
                    if tgt in key and key in teacher_attn:
                        count += 1
                        attn_loss += F.mse_loss(student_attn[key], teacher_attn[key])
            if count > 0:
                loss_dict['robust_tc_attn'] = (attn_loss / count) * config.DISTILL.ROBUST_TC_ATTN / cur_decoder_iters

    return loss_dict


def robust_seg_consistency_loss(student_mask, teacher_mask, config, valid_mask=None, cur_decoder_iters=1):
    """
    RobustSAM风格的分割一致性损失（师生版本）

    Args:
        student_mask: 学生掩码预测 [N, 1, H, W]
        teacher_mask: 教师掩码预测 [N, 1, H, W]
        config: 配置对象
        valid_mask: 有效区域掩码 [N, 1, H, W]
        cur_decoder_iters: 当前解码器迭代次数

    Returns:
        dict: 分割一致性损失字典
    """
    loss_dict = {}

    # 使用教师logits作为目标（软目标）
    target_logit = True
    if not config.DISTILL.USE_TEACHER_LOGITS:
        teacher_mask = (teacher_mask > config.MODEL.get('MASK_THRESHOLD', 0.0)).float()
        target_logit = False

    # L_Seg: Dice一致性
    if config.DISTILL.ROBUST_SEG_DICE > 0:
        loss_dict['robust_seg_dice'] = dice_loss(
            student_mask, teacher_mask, valid_mask, target_logit
        ) * config.DISTILL.ROBUST_SEG_DICE / cur_decoder_iters

    # L_Seg: Focal一致性
    if config.DISTILL.ROBUST_SEG_FOCAL > 0:
        loss_dict['robust_seg_focal'] = sigmoid_focal_loss(
            student_mask, teacher_mask, valid_mask, target_logit
        ) * config.DISTILL.ROBUST_SEG_FOCAL / cur_decoder_iters

    # L_Seg: BCE一致性
    if config.DISTILL.ROBUST_SEG_BCE > 0:
        loss_dict['robust_seg_bce'] = sigmoid_ce_loss(
            student_mask, teacher_mask, valid_mask, target_logit
        ) * config.DISTILL.ROBUST_SEG_BCE / cur_decoder_iters

    return loss_dict


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class LRSchedulerWrapper:
    """
    LR Scheduler Wrapper

    This class attaches the pre-hook on the `step` functions (including `step`, `step_update`, `step_frac`) of a lr scheduler.
    When `step` functions are called, the learning rates of all layers are updated.

    Usage:
    ```
        lr_scheduler = LRSchedulerWrapper(lr_scheduler, optimizer)
    ```
    """

    def __init__(self, lr_scheduler, optimizer):
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer

    def step(self, epoch):
        self.lr_scheduler.step(epoch)
        self.update_lr()

    def step_update(self, it):
        self.lr_scheduler.step_update(it)
        self.update_lr()

    def step_frac(self, frac):
        if hasattr(self.lr_scheduler, 'step_frac'):
            self.lr_scheduler.step_frac(frac)
            self.update_lr()

    def update_lr(self):
        param_groups = self.optimizer.param_groups
        for group in param_groups:
            if 'lr_scale' not in group:
                continue
            params = group['params']
            # update lr scale
            lr_scale = None
            for p in params:
                if hasattr(p, 'lr_scale'):
                    if lr_scale is None:
                        lr_scale = p.lr_scale
                    else:
                        assert lr_scale == p.lr_scale, (lr_scale, p.lr_scale)
            if lr_scale != group['lr_scale']:
                if is_main_process():
                    print('=' * 30)
                    print("params:", [e.param_name for e in params])
                    print(
                        f"change lr scale: {group['lr_scale']} to {lr_scale}")
            group['lr_scale'] = lr_scale
            if lr_scale is not None:
                group['lr'] *= lr_scale

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, *args, **kwargs):
        self.lr_scheduler.load_state_dict(*args, **kwargs)


def divide_param_groups_by_lr_scale(param_groups):
    """
    Divide parameters with different lr scale into different groups.

    Inputs
    ------
    param_groups: a list of dict of torch.nn.Parameter
    ```
    # example:
    param1.lr_scale = param2.lr_scale = param3.lr_scale = 0.6
    param4.lr_scale = param5.lr_scale = param6.lr_scale = 0.3
    param_groups = [{'params': [param1, param2, param4]},
                    {'params': [param3, param5, param6], 'weight_decay': 0.}]

    param_groups = divide_param_groups_by_lr_scale(param_groups)
    ```

    Outputs
    -------
    new_param_groups: a list of dict containing the key `lr_scale`
    ```
    param_groups = [
        {'params': [param1, param2], 'lr_scale': 0.6},
        {'params': [param3], 'weight_decay': 0., 'lr_scale': 0.6}
        {'params': [param4], 'lr_scale': 0.3},
        {'params': [param5, param6], 'weight_decay': 0., 'lr_scale': 0.3}
    ]
    ```
    """
    new_groups = []
    for group in param_groups:
        params = group.pop('params')

        '''
        divide parameters to different groups by lr_scale
        '''
        lr_scale_groups = dict()
        for p in params:
            lr_scale = getattr(p, 'lr_scale', 1.0)

            # create a list if not existed
            if lr_scale not in lr_scale_groups:
                lr_scale_groups[lr_scale] = list()

            # add the parameter with `lr_scale` into the specific group.
            lr_scale_groups[lr_scale].append(p)

        for lr_scale, params in lr_scale_groups.items():
            # copy other parameter information like `weight_decay`
            new_group = copy.copy(group)
            new_group['params'] = params
            new_group['lr_scale'] = lr_scale
            new_groups.append(new_group)
    return new_groups


def set_weight_decay(model):
    skip_list = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip_list = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


# ========================= Added: Logits-level segmentation + smoothness losses =========================

def seg_loss_on_logits(student_logits: torch.Tensor,
                        target: torch.Tensor,
                        valid: torch.Tensor = None,
                        use_soft_target: bool = True,
                        bce_weight: float = 0.5,
                        dice_weight: float = 0.5):
    """
    Compute segmentation loss directly on low-res logits (before threshold/postprocess).
    - student_logits: [N, 1, H, W] raw logits from decoder (e.g., low_res_logits)
    - target:         [N, 1, H, W] teacher logits or downsampled GT mask (float)
    - valid:          [N, 1, H, W] optional valid mask (1.0 valid, 0.0 ignore)
    - use_soft_target: if True, treat target as logits and apply sigmoid inside loss
    Returns: total_loss, detail_dict
    """
    # BCE on probabilities
    bce = sigmoid_ce_loss(student_logits, target, valid=valid, target_logit=use_soft_target)
    # Dice on probabilities
    dice = dice_loss(student_logits, target, valid=valid, target_logit=use_soft_target)
    total = bce_weight * bce + dice_weight * dice
    return total, {"seg_bce": bce, "seg_dice": dice}


def total_variation_l1(x: torch.Tensor, valid: torch.Tensor = None, reduction: str = 'mean') -> torch.Tensor:
    """
    L1 Total Variation on logits to suppress checkerboard/high-frequency noise.
    x: [N, C, H, W] logits (no sigmoid)
    valid: optional [N, 1, H, W] or [N, C, H, W] mask of valid locations
    """
    # Horizontal and vertical differences
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]

    if valid is not None:
        # Align valid masks to dx/dy shapes
        if valid.dim() == 4 and valid.shape[1] == 1 and x.shape[1] > 1:
            valid = valid.expand(-1, x.shape[1], -1, -1)
        vx = valid[..., :, 1:] * valid[..., :, :-1] if valid is not None else None
        vy = valid[..., 1:, :] * valid[..., :-1, :] if valid is not None else None
        dx = dx * vx
        dy = dy * vy

    tv = dx.abs().mean() + dy.abs().mean() if reduction == 'mean' else dx.abs().sum() + dy.abs().sum()
    return tv


def laplacian_smoothness_l2(x: torch.Tensor, valid: torch.Tensor = None, reduction: str = 'mean') -> torch.Tensor:
    """
    L2 Laplacian smoothness on logits: ||Δx||^2 to penalize oscillations.
    x: [N, C, H, W] logits
    """
    n, c, h, w = x.shape
    device = x.device
    dtype = x.dtype
    kernel = torch.tensor([[0., 1., 0.],
                           [1., -4., 1.],
                           [0., 1., 0.]], dtype=dtype, device=device).view(1, 1, 3, 3)
    weight = kernel.repeat(c, 1, 1, 1)  # depthwise
    lap = F.conv2d(x, weight=weight, bias=None, stride=1, padding=1, groups=c)

    if valid is not None:
        if valid.dim() == 4 and valid.shape[1] == 1 and c > 1:
            valid = valid.expand(-1, c, -1, -1)
        lap = lap * valid

    if reduction == 'mean':
        return (lap.pow(2)).mean()
    else:
        return (lap.pow(2)).sum()


def smoothness_regularizer(logits: torch.Tensor,
                            valid: torch.Tensor = None,
                            tv_weight: float = 0.0,
                            lap_weight: float = 0.0,
                            reduction: str = 'mean'):
    """
    Combine TV and Laplacian regularizers on logits. Returns total, detail.
    """
    total = logits.new_tensor(0.0)
    details = {}
    if tv_weight and tv_weight > 0:
        tv = total_variation_l1(logits, valid=valid, reduction=reduction)
        total = total + tv_weight * tv
        details['tv'] = tv
    if lap_weight and lap_weight > 0:
        lap = laplacian_smoothness_l2(logits, valid=valid, reduction=reduction)
        total = total + lap_weight * lap
        details['lap'] = lap
    return total, details
