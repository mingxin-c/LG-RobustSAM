"""
RobustSAM教师模型特征预保存脚本
完全基于EdgeSAM的设计和逻辑，支持清晰图像和15种退化类型
"""

import os
import sys
import time
import argparse
import datetime
import numpy as np

import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_segment_anything import build_sam_vit_h
from training.data import RobustSegDataset, DEGRADATION_TYPES
from training.logger import setup_logger

try:
    from mmengine import Config
except ImportError:
    import yaml
    class Config:
        def __init__(self, cfg_dict):
            for k, v in cfg_dict.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)
        
        @classmethod
        def fromfile(cls, filename):
            with open(filename, 'r') as f:
                cfg_dict = yaml.safe_load(f)
            return cls(cfg_dict)


def parse_args():
    parser = argparse.ArgumentParser('RobustSAM教师特征预保存 - 简化版')
    parser.add_argument('--cfg', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, required=True, help='教师模型权重路径')
    parser.add_argument('--output-dir', type=str, default='teacher_embeddings_simple', help='特征保存目录')
    return parser.parse_args()


@torch.no_grad()
def save_embeddings_dataset_style(config, model, data_loader, output_dir, logger):
    """保存教师特征 - 使用数据集风格的目录结构"""
    model.eval()

    num_steps = len(data_loader)
    start_time = time.time()

    logger.info(f"开始保存RobustSAM教师特征，总共 {num_steps} 个批次")
    logger.info(f"将保存清晰图像 + {len(DEGRADATION_TYPES)} 种退化类型的特征")
    logger.info(f"🔥 新增：同时保存中间层特征（encoder_features[0]）用于robust_features生成")

    # 创建数据集风格的目录结构
    os.makedirs(output_dir, exist_ok=True)

    # 创建clear目录
    clear_dir = os.path.join(output_dir, 'clear')
    os.makedirs(clear_dir, exist_ok=True)

    # 创建15种退化类型目录
    degraded_dirs = {}
    for deg_type in DEGRADATION_TYPES:
        deg_dir = os.path.join(output_dir, deg_type)
        os.makedirs(deg_dir, exist_ok=True)
        degraded_dirs[deg_type] = deg_dir

    saved_count = {'clear': 0, 'degraded': {deg: 0 for deg in DEGRADATION_TYPES}}

    for idx, batch in enumerate(data_loader):
        # 获取清晰图像
        clear_images = batch['clear_img']
        image_names = batch['image_name']

        if isinstance(clear_images, list):
            clear_images = torch.stack(clear_images)
        if isinstance(image_names, str):
            image_names = [image_names]

        clear_images = clear_images.cuda(non_blocking=True)

        # 1. 保存清晰图像特征（包含中间层特征）
        with torch.cuda.amp.autocast(enabled=config.get('AMP_ENABLE', False)):
            clear_encoder_result = model.image_encoder(clear_images)

        # 🔥 修复：处理image_encoder返回tuple的情况，同时保存中间层特征
        if isinstance(clear_encoder_result, tuple):
            clear_outputs = clear_encoder_result[0]  # 主要特征 [B, 256, 64, 64]
            clear_encoder_features = clear_encoder_result[1]  # 中间层特征列表
        else:
            clear_outputs = clear_encoder_result
            clear_encoder_features = None

        # 转换为CPU并保存主要特征
        clear_outputs = clear_outputs.detach().to(device='cpu', dtype=torch.float16).numpy()

        # 转换中间层特征（只保存第一个，因为mask_decoder只用encoder_features[0]）
        if clear_encoder_features is not None and len(clear_encoder_features) > 0:
            clear_first_layer = clear_encoder_features[0].detach().to(device='cpu', dtype=torch.float16).numpy()
        else:
            clear_first_layer = None

        # 创建中间层特征目录
        clear_encoder_dir = os.path.join(output_dir, 'clear_encoder_features')
        os.makedirs(clear_encoder_dir, exist_ok=True)

        for i, (image_name, clear_output) in enumerate(zip(image_names, clear_outputs)):
            # 🔥 保持和数据集完全一样的文件名，只改扩展名
            # 如：原始文件 whatever_name.jpg -> 特征文件 whatever_name.npy
            feature_filename = os.path.splitext(image_name)[0] + '.npy'
            clear_path = os.path.join(clear_dir, feature_filename)

            # 保存主要特征
            if not os.path.exists(clear_path):
                np.save(clear_path, clear_output)
                saved_count['clear'] += 1

            # 🔥 新增：保存中间层特征（encoder_features[0]）
            if clear_first_layer is not None:
                clear_encoder_path = os.path.join(clear_encoder_dir, feature_filename)
                if not os.path.exists(clear_encoder_path):
                    np.save(clear_encoder_path, clear_first_layer[i])  # 保存第i个样本的中间层特征

        # 2. 保存所有15种退化图像特征
        for deg_type in DEGRADATION_TYPES:
            # 🔥 使用随机管理器确保与训练时的随机性一致
            from training.random_manager import ImageRandomContext

            # 为每种退化类型加载对应的退化图像
            degraded_images = []
            for image_name in image_names:
                # 在相同的随机上下文中加载退化图像
                with ImageRandomContext(image_name):
                    degraded_img = load_degraded_image_by_name(image_name, deg_type, config.DATA.DATA_PATH)
                    degraded_images.append(degraded_img)

            degraded_images = torch.stack(degraded_images).cuda(non_blocking=True)

            # 提取退化图像特征（包含中间层特征）
            with torch.cuda.amp.autocast(enabled=config.get('AMP_ENABLE', False)):
                degraded_encoder_result = model.image_encoder(degraded_images)

            # 🔥 修复：处理image_encoder返回tuple的情况，同时保存中间层特征
            if isinstance(degraded_encoder_result, tuple):
                degraded_outputs = degraded_encoder_result[0]  # 主要特征 [B, 256, 64, 64]
                degraded_encoder_features = degraded_encoder_result[1]  # 中间层特征列表
            else:
                degraded_outputs = degraded_encoder_result
                degraded_encoder_features = None

            # 转换为CPU并保存主要特征
            degraded_outputs = degraded_outputs.detach().to(device='cpu', dtype=torch.float16).numpy()

            # 转换中间层特征（只保存第一个）
            if degraded_encoder_features is not None and len(degraded_encoder_features) > 0:
                degraded_first_layer = degraded_encoder_features[0].detach().to(device='cpu', dtype=torch.float16).numpy()
            else:
                degraded_first_layer = None

            # 创建退化类型的中间层特征目录
            deg_encoder_dir = os.path.join(output_dir, f'{deg_type}_encoder_features')
            os.makedirs(deg_encoder_dir, exist_ok=True)

            # 保存退化图像特征
            for i, (image_name, degraded_output) in enumerate(zip(image_names, degraded_outputs)):
                # 🔥 保持和数据集完全一样的文件名，只改扩展名
                feature_filename = os.path.splitext(image_name)[0] + '.npy'
                deg_path = os.path.join(degraded_dirs[deg_type], feature_filename)

                # 保存主要特征
                if not os.path.exists(deg_path):
                    np.save(deg_path, degraded_output)
                    saved_count['degraded'][deg_type] += 1

                # 🔥 新增：保存中间层特征（encoder_features[0]）
                if degraded_first_layer is not None:
                    deg_encoder_path = os.path.join(deg_encoder_dir, feature_filename)
                    if not os.path.exists(deg_encoder_path):
                        np.save(deg_encoder_path, degraded_first_layer[i])  # 保存第i个样本的中间层特征

        # 进度报告
        if idx % 10 == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            elapsed = time.time() - start_time
            eta = elapsed / (idx + 1) * (num_steps - idx - 1)
            total_saved = saved_count['clear'] + sum(saved_count['degraded'].values())
            logger.info(
                f'Save: [{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(eta))}\t'
                f'saved {total_saved}\t'
                f'mem {memory_used:.0f}MB'
            )

    total_time = time.time() - start_time
    logger.info(f"特征保存完成！总用时: {datetime.timedelta(seconds=int(total_time))}")
    logger.info(f"清晰图像特征: {saved_count['clear']} 个")
    logger.info(f"🔥 清晰图像中间层特征: {saved_count['clear']} 个 (保存在 clear_encoder_features/)")
    for deg_type, count in saved_count['degraded'].items():
        if count > 0:
            logger.info(f"退化特征 {deg_type}: {count} 个")
            logger.info(f"🔥 退化中间层特征 {deg_type}: {count} 个 (保存在 {deg_type}_encoder_features/)")

    return saved_count


def load_degraded_image_by_name(image_name, degradation_type, data_root):
    """根据图像名称和退化类型加载退化图像"""
    import os
    from PIL import Image
    import torchvision.transforms as transforms

    # 🔥 修复：根据实际数据路径结构构建路径
    # data_root 应该是 "datasets/train"，所以直接使用
    degraded_path = os.path.join(data_root, degradation_type, image_name)

    if os.path.exists(degraded_path):
        degraded_img = Image.open(degraded_path).convert('RGB')
    else:
        # 如果退化图像不存在，使用清晰图像
        clear_path = os.path.join(data_root, 'clear', image_name)
        if os.path.exists(clear_path):
            degraded_img = Image.open(clear_path).convert('RGB')
            print(f"警告：退化图像不存在 {degraded_path}，使用清晰图像代替")
        else:
            raise FileNotFoundError(f"图像文件不存在: {clear_path}")

    # 应用变换
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(degraded_img)


def main():
    args = parse_args()

    # 加载配置
    config = Config.fromfile(args.cfg)

    # 创建logger
    logger = setup_logger(output_dir=args.output_dir, distributed_rank=0, name='save_embeddings')

    logger.info(f"加载配置文件: {args.cfg}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"教师权重: {args.resume}")

    # 构建数据集（模仿EdgeSAM的build_loader）
    # 🔥 修复：添加transform，将PIL图像转换为tensor
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = RobustSegDataset(
        data_root=config.DATA.DATA_PATH,
        split='',  # 🔥 修复：因为DATA_PATH已经是datasets/train，所以split为空
        transform=transform,  # 🔥 修复：添加transform
        random_degradation=True,   # 使用随机退化，获取退化图像
        paired_data=False,  # 不加载掩码，只加载图像
        stage='stage2'  # 🔥 修复：必须指定stage，并且用stage2确保获取退化图像
    )

    logger.info(f"数据集大小: {len(dataset)}")

    # 创建优化的数据加载器 - 充分利用显存和CPU
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,  # 🚀 增加batch_size，充分利用显存
        shuffle=False,
        num_workers=0,  # 🔥 禁用多进程，避免CUDA冲突  # � 启用多进程，加速数据加载
        pin_memory=True,
        drop_last=False
    )

    # 构建教师模型
    logger.info("构建教师模型...")
    teacher_model = build_sam_vit_h(opt=None, checkpoint=args.resume, train=False)
    teacher_model.cuda()

    logger.info("开始保存特征...")

    # 保存特征（使用数据集风格的目录结构）
    saved_count = save_embeddings_dataset_style(
        config, teacher_model, data_loader, args.output_dir, logger
    )

    logger.info(f'✅ 保存特征完成！')
    logger.info(f'清晰图像特征: {saved_count["clear"]} 个')
    for deg_type, count in saved_count['degraded'].items():
        if count > 0:
            logger.info(f'退化特征 {deg_type}: {count} 个')


if __name__ == '__main__':
    main()
