"""
用于加载Robust-Seg数据集的数据加载模块
支持加载清晰图像和15种不同类型的退化图像
"""

import os
import random
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from skimage.transform import resize

# 设置PIL允许加载截断的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEGRADATION_TYPES = [
    'snow', 'fog', 'rain', 'gauss_noise', 'ISO_noise', 'impulse_noise',
    'resampling_blur', 'motion_blur', 'zoom_blur', 'color_jitter',
    'compression', 'elastic_transform', 'frosted_glass_blur',
    'brightness', 'contrast'
]

class RobustSegDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None, 
                 random_degradation=True, specific_degradation=None,
                 paired_data=True, stage='stage1'):
        """
        Args:
            data_root: 数据集根目录
            split: 'train', 'val', 或 'test'
            transform: 图像变换
            random_degradation: 是否随机选择一种退化类型
            specific_degradation: 指定退化类型，当random_degradation=False时使用
            paired_data: 是否返回成对的清晰和退化图像
            stage: 'stage1' (编码器蒸馏) 或 'stage2' (完整蒸馏)
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.random_degradation = random_degradation
        self.specific_degradation = specific_degradation
        self.paired_data = paired_data
        self.stage = stage  # 🔥 新增：训练阶段标识
        
        # 验证specific_degradation是否有效
        if specific_degradation is not None and specific_degradation not in DEGRADATION_TYPES:
            raise ValueError(f"退化类型 {specific_degradation} 不存在。有效类型: {DEGRADATION_TYPES}")
        
        # 获取图像列表
        if split:
            self.clear_dir = os.path.join(data_root, split, 'clear')
            self.mask_dir = os.path.join(data_root, split, 'masks')
        else:
            # 🔥 修复：当split为空时，直接使用data_root下的clear目录
            self.clear_dir = os.path.join(data_root, 'clear')
            self.mask_dir = os.path.join(data_root, 'masks')
        
        # 获取所有清晰图像文件名
        self.image_names = [f for f in os.listdir(self.clear_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        
        # 加载清晰图像
        clear_path = os.path.join(self.clear_dir, image_name)
        clear_img = Image.open(clear_path).convert('RGB')
        
        # 加载掩码 - 处理形状为(1, height, width)的.npy文件
        mask_name = image_name.replace('.jpg', '.npy').replace('.jpeg', '.npy')
        mask_path = os.path.join(self.mask_dir, mask_name)
        # 日志：可选检查掩码路径

        if os.path.exists(mask_path):
            try:
                # 加载.npy格式的掩码
                mask_array = np.load(mask_path)
                # 调试：原始掩码形状

                # 处理不同形状的掩码
                if len(mask_array.shape) == 3:
                    if mask_array.shape[0] == 1:
                        # 单通道掩码 (1, height, width)
                        mask_array = mask_array[0]  # 现在形状是(height, width)
                    else:
                        # 多通道掩码 (N, height, width)
                        # 合并所有通道为一个掩码（任何前景都是前景）
                        # 多通道掩码
                        # 使用逻辑或合并所有通道
                        mask_array = np.any(mask_array, axis=0).astype(np.float32)
                
                # 确保掩码是二值的（0或1）
                if mask_array.max() > 1:
                    mask_array = (mask_array > 0).astype(np.float32)
                
                # 转换为PyTorch张量
                mask = torch.from_numpy(mask_array).float()
                
                # 调整为目标尺寸(1024, 1024)
                mask = mask.unsqueeze(0)  # 添加通道维度
                mask = transforms.functional.resize(
                    mask, 
                    (1024, 1024), 
                    interpolation=transforms.InterpolationMode.NEAREST
                )
                
                # 调试：调整后掩码形状
            except Exception as e:
                # 加载掩码出错，创建空掩码
                mask = torch.zeros((1, 1024, 1024), dtype=torch.float32)
        else:
            # 掩码文件不存在，创建空掩码
            # 创建空掩码张量
            mask = torch.zeros((1, 1024, 1024), dtype=torch.float32)
        
        # 选择退化类型
        if self.random_degradation:
            # 与教师训练一致：每次随机选择一种退化类型
            degradation_type = random.choice(DEGRADATION_TYPES)
        else:
            degradation_type = self.specific_degradation or DEGRADATION_TYPES[0]

        # 加载退化图像
        degraded_dir = os.path.join(self.data_root, self.split, degradation_type)
        degraded_path = os.path.join(degraded_dir, image_name)
        
        if os.path.exists(degraded_path):
            degraded_img = Image.open(degraded_path).convert('RGB')
        else:
            # 如果没有对应的退化图像，使用清晰图像代替
            degraded_img = clear_img.copy()
        
        # 应用变换
        if self.transform:
            clear_img = self.transform(clear_img)
            degraded_img = self.transform(degraded_img)
            
            # 掩码已经是张量，无需额外转换
            # 确保维度正确
            if mask.shape[0] != 1:
                mask = mask.unsqueeze(0)
        
        # ========== 按 EdgeSAM 格式生成提示 & anno ==========
        H = W = 1024
        img_size_before_pad = torch.tensor([3, H, W], dtype=torch.int64)

        mask_bin = (mask > 0.5).float()        # mask 已经 Resize 到 1024
        if mask_bin.ndim == 3:                 # (1,H,W) -> (1,1,H,W)
            mask_bin = mask_bin.unsqueeze(0)

        boxes, masks_out = [], []
        for i in range(mask_bin.shape[0]):
            m = mask_bin[i, 0]                 # (H,W)
            ys, xs = torch.where(m > 0)
            if ys.numel() == 0:                # 没前景，跳过
                continue

            # 生成 bbox（与 EdgeSAM COCO 一致）
            y1, y2 = ys.min().item(), ys.max().item()
            x1, x2 = xs.min().item(), xs.max().item()
            boxes.append([x1, y1, x2, y2])

            masks_out.append(m.unsqueeze(0))   # (1,H,W)

        if len(boxes) == 0:
            prompt_box   = torch.zeros((0,4),    dtype=torch.float32)
            gt_mask_i    = torch.zeros((0,1,H,W),dtype=torch.float32)
        else:
            prompt_box   = torch.tensor(boxes,   dtype=torch.float32)
            gt_mask_i    = torch.stack(masks_out,dim=0).float()

        # 🔥 完全对齐 EdgeSAM COCO：只提供 prompt_box，不生成 prompt_point
        # 点提示由训练代码根据配置生成（PROMPT_BOX_TO_POINT 或 PROMPT_MASK_TO_POINT）
        anno = {
            'prompt_box': prompt_box,              # (Ni,4) XYXY
            'gt_mask': gt_mask_i,                  # (Ni,1,H,W)
            'img_size_before_pad': img_size_before_pad
        }

        # 🔥 根据训练阶段返回不同格式的数据
        if self.stage == 'stage1':
            # 第一阶段：编码器蒸馏，随机返回清晰或退化图像
            if torch.rand(1) > 0.5:
                return clear_img, anno, {'image_name': image_name, 'type': 'clear', 'degradation_type': None}
            else:
                return degraded_img, anno, {'image_name': image_name, 'type': 'degraded', 'degradation_type': degradation_type}
        else:
            # 第二阶段：完整蒸馏，返回清晰-退化配对
            return {
                'clear_img': clear_img,
                'degraded_img': degraded_img, 
                'mask': mask,
                'anno': anno,
                'image_name': image_name,
                'degradation_type': degradation_type
            }

def pseudo_collate(batch):
    """支持两阶段的pseudo-collate函数"""
    # 检查是否为第二阶段的dict格式
    if batch and isinstance(batch[0], dict) and 'clear_img' in batch[0]:
        # 第二阶段：处理清晰-退化配对数据
        return collate_stage2_batch(batch)
    else:
        # 第一阶段：EdgeSAM风格的collate
        return collate_stage1_batch(batch)


def collate_stage1_batch(batch):
    """第一阶段编码器蒸馏的collate函数
    兼容：
    - (img, anno, meta)
    - (img, anno)
    - ((img, anno), (teacher_final, teacher_mid))  # 包装器(read)返回
    """
    # 包装器(read)返回的嵌套双元组：((img, anno), (t_final, t_mid))
    if batch and isinstance(batch[0], tuple) and len(batch[0]) == 2 \
       and isinstance(batch[0][0], tuple) and isinstance(batch[0][1], tuple):
        imgs, annos, saved = [], [], []
        for ((img, anno), teacher_pair) in batch:
            imgs.append(img)
            annos.append(anno)
            saved.append(teacher_pair)  # (final_256[BCHW], mid_1280[BHWC])
        # 将 anno 列表转为 dict of lists
        annos = {k: [a[k] for a in annos] for k in annos[0].keys()} if annos else {}
        return (imgs, annos), saved

    # 原始第一阶段格式
    imgs, annos, metadata = [], [], []
    for item in batch:
        if isinstance(item, tuple) and len(item) == 3:
            img, anno, meta = item
            metadata.append(meta)
        else:
            img, anno = item
            metadata.append({})
        imgs.append(img)
        annos.append(anno)

    # 将 anno 列表转为 dict of lists
    annos = {k: [a[k] for a in annos] for k in annos[0].keys()} if annos else {}
    return (imgs, annos), metadata


def collate_stage2_batch(batch):
    """第二阶段完整蒸馏的collate函数
    兼容：
    - dict 格式 {'degraded_img','anno',...}
    - ((degraded_img, anno), (teacher_final, teacher_mid))  # 包装器(read)返回
    """
    # 包装器(read)返回的嵌套双元组：((degraded_img, anno), (t_final, t_mid))
    if batch and isinstance(batch[0], tuple) and len(batch[0]) == 2 \
       and isinstance(batch[0][0], tuple) and isinstance(batch[0][1], tuple):
        degraded_imgs, annos, saved = [], [], []
        for ((deg_img, anno), teacher_pair) in batch:
            degraded_imgs.append(deg_img)
            annos.append(anno)
            saved.append(teacher_pair)
        # 将 anno 列表转为 dict of lists
        annos = {k: [a[k] for a in annos] for k in annos[0].keys()} if annos else {}
        return (degraded_imgs, annos), saved

    # 原始第二阶段 dict 格式
    degraded_imgs, annos, metadata = [], [], []
    for item in batch:
        degraded_imgs.append(item['degraded_img'])  # 只取退化图像作为学生输入
        annos.append(item['anno'])
        metadata.append({
            'image_name': item['image_name'],
            'degradation_type': item['degradation_type']
        })

    # 将 anno 列表转为 dict of lists
    annos = {k: [a[k] for a in annos] for k in annos[0].keys()} if annos else {}
    return (degraded_imgs, annos), metadata


def build_dataloader(data_path, batch_size, is_train=True,
                    paired_data=True, random_degradation=True,
                    specific_degradation=None, num_workers=4, distributed=False):
    """构建Robust-Seg数据集的DataLoader (EdgeSAM风格batch输出)"""

    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        # 对齐保存教师特征时的归一化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    split = 'train' if is_train else 'val'
    dataset = RobustSegDataset(
        data_root=data_path,
        split=split,
        transform=transform,
        random_degradation=random_degradation,
        specific_degradation=specific_degradation,
        paired_data=paired_data
    )

    # 创建sampler
    sampler = DistributedSampler(dataset) if distributed else None

    # 创建DataLoader（绑定pseudo_collate）
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and is_train),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=pseudo_collate
    )

    return dataloader

class RobustSAMDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='encoder'):
        """数据集类，支持加载清晰和退化图像对
        
        Args:
            root_dir: 数据集根目录
            transform: 图像变换
            mode: 'encoder'或'decoder'，决定返回格式
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # 目录结构
        self.clear_dir = os.path.join(root_dir, 'clear')
        self.mask_dir = os.path.join(root_dir, 'masks')
        
        # 退化类型目录
        self.degradation_types = [
            'snow', 'fog', 'rain', 'gauss_noise', 'ISO_noise', 'impulse_noise', 
            'resampling_blur', 'motion_blur', 'zoom_blur', 'color_jitter', 
            'compression', 'elastic_transform', 'frosted_glass_blur', 
            'brightness', 'contrast'
        ]
        self.degraded_dirs = {}
        for deg_type in self.degradation_types:
            deg_path = os.path.join(root_dir, deg_type)
            if os.path.exists(deg_path):
                self.degraded_dirs[deg_type] = deg_path
        
        # 图像文件列表
        self.image_files = [f for f in os.listdir(self.clear_dir) 
                          if f.endswith('.jpg') or f.endswith('.png')]
