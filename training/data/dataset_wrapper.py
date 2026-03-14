"""
数据集包装器，支持预保存教师特征的加载
基于EdgeSAM的DatasetWrapper实现，适配RobustSAM的数据格式
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from .data import DEGRADATION_TYPES


class TeacherEmbeddingDatasetWrapper(Dataset):
    """
    教师特征数据集包装器
    支持两种模式：
    1. write模式：保存教师特征（预保存阶段使用）
    2. read模式：加载预保存的教师特征（训练阶段使用）
    """

    def __init__(self, dataset, embedding_path, mode='read'):
        """
        Args:
            dataset: 原始数据集
            embedding_path: 教师特征保存路径
            mode: 'read' 或 'write'
        """
        super().__init__()
        self.dataset = dataset
        self.embedding_path = Path(embedding_path)
        self.mode = mode

        # 创建目录结构（适配新的目录结构）
        if mode == 'write':
            self.embedding_path.mkdir(parents=True, exist_ok=True)
            (self.embedding_path / 'clear').mkdir(exist_ok=True)
            # 🔥 15种退化类型直接在根目录下
            for deg_type in DEGRADATION_TYPES:
                (self.embedding_path / deg_type).mkdir(exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.mode == 'write':
            return self._getitem_for_write(index)
        else:
            return self._getitem_for_read(index)

    def _getitem_for_write(self, index):
        """写模式：返回原始数据用于特征提取"""
        return self.dataset[index]

    def _getitem_for_read(self, index):
        """读模式：返回原始数据+预保存的教师双特征"""
        # 获取原始数据  
        original_data = self.dataset[index]

        # 🔥 严格要求：数据集必须提供image_names属性
        assert hasattr(self.dataset, 'image_names'), \
            "Dataset must have 'image_names' attribute for teacher feature loading"
        image_name = self.dataset.image_names[index]
        
        # 🔥 修复：根据数据确定退化类型，避免随机不匹配
        if isinstance(original_data, dict):
            # 第二阶段：从数据中获取退化类型
            degradation_type = original_data['degradation_type']
        elif isinstance(original_data, tuple) and len(original_data) >= 3:
            # 第一阶段：从metadata中获取退化类型
            metadata = original_data[2]
            degradation_type = metadata.get('degradation_type', None)
            if degradation_type is None:
                # 如果是清晰图像，随机选择一个退化类型用于加载退化特征（但不会使用）
                import random
                degradation_type = random.choice(DEGRADATION_TYPES)
        else:
            # 兜底：随机选择
            import random
            degradation_type = random.choice(DEGRADATION_TYPES)

        # 🔥 严格加载：不允许失败，让错误直接暴露
        # 加载双特征 (final_256, mid_1280)
        clear_final, clear_mid = self._load_dual_embedding(image_name, 'clear')
        degraded_final, degraded_mid = self._load_dual_embedding(image_name, degradation_type)

        # 🔥 根据数据集返回的内容决定使用的教师特征
        if isinstance(original_data, dict):
            # 第二阶段：完整蒸馏，清晰-退化配对
            assert 'clear_img' in original_data and 'degraded_img' in original_data, \
                "Stage2 data must contain both 'clear_img' and 'degraded_img'"
            assert 'anno' in original_data, "Stage2 data must contain 'anno'"
            
            # 🔥 第二阶段逻辑说明：
            # - 学生模型: 退化图像 → 学生掩码B  
            # - 教师模型: 清晰图像的预保存特征 → 教师掩码A
            # - 对比: 掩码A vs 掩码B
            img = original_data['degraded_img']  # 学生输入：退化图像
            teacher_final, teacher_mid = clear_final, clear_mid  # 教师目标：清晰图像的特征
            anno = original_data['anno']
            
        elif isinstance(original_data, tuple) and len(original_data) >= 3:
            # 第一阶段：编码器蒸馏，根据数据集返回的图像类型选择对应教师特征
            img, anno, metadata = original_data[0], original_data[1], original_data[2]
            
            # 🔥 第一阶段逻辑：图像类型和教师特征必须匹配
            img_type = metadata.get('type', 'unknown')
            degradation_type_from_data = metadata.get('degradation_type', None)
            
            if img_type == 'clear':
                # 清晰图像使用清晰教师特征
                teacher_final, teacher_mid = clear_final, clear_mid
            elif img_type == 'degraded':
                # 退化图像使用对应退化类型的教师特征
                # 🔥 修复：退化类型应该匹配，因为现在从数据中获取
                teacher_final, teacher_mid = degraded_final, degraded_mid
            else:
                raise ValueError(f"Unknown image type: {img_type}")
                
        else:
            raise ValueError(f"Unexpected data format: {type(original_data)}")

        return (img, anno), (teacher_final, teacher_mid)

    def _load_dual_embedding(self, image_name, embedding_type):
        """🔥 新增：加载双特征 (final_256, mid_1280)"""
        feature_filename = os.path.splitext(image_name)[0] + '.npy'
        
        if embedding_type == 'clear':
            # 加载清晰图像的双特征
            final_path = self.embedding_path / 'clear' / feature_filename
            mid_path = self.embedding_path / 'clear_encoder_features' / feature_filename
        else:
            # 加载退化图像的双特征
            final_path = self.embedding_path / embedding_type / feature_filename  
            mid_path = self.embedding_path / f'{embedding_type}_encoder_features' / feature_filename

        # 加载最终特征 [B, 256, 64, 64]
        final_embedding = np.load(final_path)
        final_tensor = torch.from_numpy(final_embedding).float()
        
        # 加载中间特征，保持 [64, 64, 1280] (HWC) 格式，训练时会添加batch维度
        mid_embedding = np.load(mid_path) 
        # 🔥 修复：保持 [64, 64, 1280] 格式，mask_decoder期望 [B,H,W,C] 格式
        assert mid_embedding.shape == (64, 64, 1280), \
            f"Expected mid feature shape (64, 64, 1280), got {mid_embedding.shape}"
        mid_tensor = torch.from_numpy(mid_embedding).float()  # [64, 64, 1280]
        
        return final_tensor, mid_tensor

    def _load_clear_embedding(self, image_name):
        """加载清晰图像的教师特征（保留向后兼容）"""
        final_tensor, _ = self._load_dual_embedding(image_name, 'clear')
        return final_tensor

    def _load_degraded_embedding(self, image_name, degradation_type):
        """加载退化图像的教师特征"""
        # 🔥 适配新的.npy格式和目录结构
        feature_filename = os.path.splitext(image_name)[0] + '.npy'
        deg_path = self.embedding_path / degradation_type / feature_filename  # 直接在根目录下

        # 加载.npy文件
        embedding = np.load(deg_path)
        return torch.from_numpy(embedding).float()

    def _load_clear_encoder_features(self, image_name):
        """🔥 新增：加载清晰图像的中间层特征（encoder_features[0]）"""
        feature_filename = os.path.splitext(image_name)[0] + '.npy'
        encoder_path = self.embedding_path / 'clear_encoder_features' / feature_filename

        try:
            # 加载.npy文件
            encoder_features = np.load(encoder_path)
            return torch.from_numpy(encoder_features).float()
        except FileNotFoundError:
            print(f"⚠️  警告：清晰图像 {image_name} 的中间层特征不存在")
            return None

    def _load_degraded_encoder_features(self, image_name, degradation_type):
        """🔥 新增：加载退化图像的中间层特征（encoder_features[0]）"""
        feature_filename = os.path.splitext(image_name)[0] + '.npy'
        encoder_path = self.embedding_path / f'{degradation_type}_encoder_features' / feature_filename

        try:
            # 加载.npy文件
            encoder_features = np.load(encoder_path)
            return torch.from_numpy(encoder_features).float()
        except FileNotFoundError:
            print(f"⚠️  警告：退化图像 {image_name} ({degradation_type}) 的中间层特征不存在")
            return None

    def save_embedding(self, image_name, clear_embedding, degraded_embeddings):
        """保存教师特征（写模式使用）- 使用.npy格式与save_embedding.py保持一致"""
        if self.mode != 'write':
            raise ValueError("只有写模式才能保存特征")

        # 🔥 修复：使用.npy格式，与save_embedding.py保持一致
        feature_filename = os.path.splitext(image_name)[0] + '.npy'

        # 保存清晰图像特征
        clear_path = self.embedding_path / 'clear' / feature_filename
        np.save(clear_path, clear_embedding.cpu().numpy().astype(np.float16))

        # 保存退化图像特征 - 直接在根目录下，与save_embedding.py保持一致
        for deg_type, deg_embedding in degraded_embeddings.items():
            deg_path = self.embedding_path / deg_type / feature_filename  # 🔥 修复：直接在根目录下
            np.save(deg_path, deg_embedding.cpu().numpy().astype(np.float16))

    def check_embedding_exists(self, image_name, degradation_type=None):
        """检查教师特征是否已存在"""
        # 🔥 修复：使用.npy格式
        feature_filename = os.path.splitext(image_name)[0] + '.npy'

        if degradation_type is None:
            # 检查清晰图像特征
            clear_path = self.embedding_path / 'clear' / feature_filename
            return clear_path.exists()
        else:
            # 检查退化图像特征 - 直接在根目录下
            deg_path = self.embedding_path / degradation_type / feature_filename
            return deg_path.exists()

    def get_embedding_stats(self):
        """获取已保存特征的统计信息"""
        stats = {
            'clear_count': 0,
            'degraded_count': {},
            'total_size_mb': 0
        }

        # 统计清晰图像特征 - 🔥 修复：使用.npy格式
        clear_dir = self.embedding_path / 'clear'
        if clear_dir.exists():
            clear_files = list(clear_dir.glob('*.npy'))
            stats['clear_count'] = len(clear_files)
            stats['total_size_mb'] += sum(f.stat().st_size for f in clear_files) / 1024 / 1024

        # 统计退化图像特征 - 🔥 修复：直接在根目录下，使用.npy格式
        for deg_type in DEGRADATION_TYPES:
            deg_type_dir = self.embedding_path / deg_type  # 直接在根目录下
            if deg_type_dir.exists():
                deg_files = list(deg_type_dir.glob('*.npy'))
                stats['degraded_count'][deg_type] = len(deg_files)
                stats['total_size_mb'] += sum(f.stat().st_size for f in deg_files) / 1024 / 1024

        return stats


def collate_fn_with_embeddings(batch):
    """
    支持教师特征的collate函数
    过滤掉没有教师特征的样本
    """
    # 分离有效数据和无效数据
    valid_batch = []
    invalid_indices = []

    for i, data in enumerate(batch):
        if data.get('has_teacher_features', False):
            valid_batch.append(data)
        else:
            invalid_indices.append(i)

    if not valid_batch:
        raise ValueError("批次中没有有效的教师特征数据")

    if invalid_indices:
        print(f"警告：批次中有 {len(invalid_indices)} 个样本缺少教师特征")

    # 使用有效数据构建批次
    batch_data = {}

    # 处理图像数据
    if 'clear_img' in valid_batch[0]:
        batch_data['clear_img'] = torch.stack([item['clear_img'] for item in valid_batch])

    if 'degraded_img' in valid_batch[0]:
        batch_data['degraded_img'] = torch.stack([item['degraded_img'] for item in valid_batch])

    # 处理掩码
    if 'mask' in valid_batch[0]:
        batch_data['mask'] = torch.stack([item['mask'] for item in valid_batch])

    # 处理教师特征
    batch_data['teacher_clear_embeddings'] = torch.stack([
        item['teacher_clear_embedding'] for item in valid_batch
    ])

    batch_data['teacher_degraded_embeddings'] = torch.stack([
        item['teacher_degraded_embedding'] for item in valid_batch
    ])

    # 🔥 新增：处理中间层特征两路（若存在）
    if valid_batch[0].get('teacher_clear_encoder_features', None) is not None:
        batch_data['teacher_clear_encoder_features'] = torch.stack([
            item['teacher_clear_encoder_features'] for item in valid_batch
        ])
    if valid_batch[0].get('teacher_degraded_encoder_features', None) is not None:
        batch_data['teacher_degraded_encoder_features'] = torch.stack([
            item['teacher_degraded_encoder_features'] for item in valid_batch
        ])

    # 处理其他数据
    for key in ['degradation_type', 'image_name']:
        if key in valid_batch[0]:
            batch_data[key] = [item[key] for item in valid_batch]

    return batch_data


def build_dataset_with_embeddings(config, embedding_path, is_train=True):
    """构建带有预保存教师特征的数据集"""
    from .data import RobustSegDataset

    # 创建原始数据集
    split = 'train' if is_train else 'val'
    dataset = RobustSegDataset(
        data_root=config.DATA.DATA_PATH,
        split=split,
        transform=None,  # 在数据集内部处理
        random_degradation=True,
        paired_data=True
    )

    # 包装为支持教师特征的数据集
    wrapped_dataset = TeacherEmbeddingDatasetWrapper(
        dataset=dataset,
        embedding_path=embedding_path,
        mode='read'
    )

    return wrapped_dataset



def build_dataloader_with_embeddings(dataset, batch_size=4, num_workers=4, shuffle=True):
    """
    便捷构建 DataLoader：
    - 约定 dataset 已为 TeacherEmbeddingDatasetWrapper(read 模式)
    - 绑定 collate_fn_with_embeddings
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_with_embeddings,
        drop_last=False,
    )
