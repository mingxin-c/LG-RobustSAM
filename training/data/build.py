# --------------------------------------------------------
# LGRobustSAM Data Builder  
# 支持两阶段蒸馏的数据加载器构建
# --------------------------------------------------------

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import RobustSegDataset, pseudo_collate
from .dataset_wrapper import TeacherEmbeddingDatasetWrapper


def build_loader(config):
    """
    构建数据加载器，支持两阶段蒸馏训练
    
    第一阶段：编码器蒸馏 (ENCODER_ONLY=True)
    - 清晰和退化图像混合训练
    - 加载预保存的教师双特征进行对比
    
    第二阶段：完整蒸馏 (ENCODER_ONLY=False)  
    - 清晰-退化图像严格配对
    - 执行EdgeSAM式的提示循环蒸馏
    """
    config.defrost()
    
    # 🔥 根据训练阶段选择不同的数据集配置
    stage = 'stage1' if config.DISTILL.ENCODER_ONLY else 'stage2'
    print(f"🔥 数据加载器配置: {stage} ({'编码器蒸馏' if stage == 'stage1' else '完整蒸馏'})")
    
    # 构建训练数据集
    dataset_train = build_dataset(is_train=True, config=config, stage=stage)
    config.MODEL.NUM_CLASSES = 0  # RobustSAM不需要分类
    config.freeze()
    
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    
    # 构建验证数据集
    dataset_val = build_dataset(is_train=False, config=config, stage=stage)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    # 创建采样器
    sampler_train = DistributedSampler(
        dataset_train, shuffle=True,
        drop_last=False, 
    )

    sampler_val = DistributedSampler(
        dataset_val, shuffle=False,
        drop_last=False,
    )

    # 🔥 包装数据集以支持预保存的教师特征
    if hasattr(config.DISTILL, 'TEACHER_EMBED_PATH') and config.DISTILL.TEACHER_EMBED_PATH:
        print("🔥 启用教师特征数据集包装器")
        dataset_train = TeacherEmbeddingDatasetWrapper(
            dataset=dataset_train,
            embedding_path=config.DISTILL.TEACHER_EMBED_PATH,
            mode='read'
        )
        dataset_val = TeacherEmbeddingDatasetWrapper(
            dataset=dataset_val, 
            embedding_path=config.DISTILL.TEACHER_EMBED_PATH,
            mode='read'
        )

    # 创建数据加载器
    data_loader_train = DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,  # 训练时drop_last=True保证batch大小一致
        collate_fn=pseudo_collate
    )

    data_loader_val = DataLoader(
        dataset_val, 
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        collate_fn=pseudo_collate
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val


def build_dataset(is_train, config, stage='stage1'):
    """
    构建数据集
    
    Args:
        is_train: 是否为训练集
        config: 配置对象
        stage: 'stage1' (编码器蒸馏) 或 'stage2' (完整蒸馏)
    """
    import torchvision.transforms as transforms

    # 图像变换（与教师特征保存时保持一致）
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集参数
    split = 'train' if is_train else 'val'
    if hasattr(config.DATA, 'DATA_PATH'):
        data_root = config.DATA.DATA_PATH
        # 如果DATA_PATH已经包含train，则split为空
        if data_root.endswith('train') or data_root.endswith('val'):
            split = ''
    else:
        raise ValueError("配置中缺少DATA.DATA_PATH")

    # 构建数据集
    dataset = RobustSegDataset(
        data_root=data_root,
        split=split,
        transform=transform,
        random_degradation=True,  # 随机选择退化类型
        specific_degradation=None,
        paired_data=True,
        stage=stage  # 🔥 传递训练阶段信息
    )

    print(f"🔥 构建{stage}数据集: {len(dataset)} 个样本")
    
    if stage == 'stage1':
        print("   - 编码器蒸馏：清晰/退化图像混合训练")
        print("   - 特征对比：final_256 + mid_1280")
    else:
        print("   - 完整蒸馏：清晰-退化图像配对训练") 
        print("   - 掩码蒸馏：教师掩码A vs 学生掩码B")

    return dataset


def build_teacher_embedding_dataset(config, embedding_path, is_train=True, stage='stage1'):
    """
    便捷函数：直接构建带有教师特征的数据集
    """
    # 构建基础数据集
    dataset = build_dataset(is_train, config, stage)
    
    # 包装为支持教师特征的数据集
    wrapped_dataset = TeacherEmbeddingDatasetWrapper(
        dataset=dataset,
        embedding_path=embedding_path,
        mode='read'
    )
    
    return wrapped_dataset


def create_stage1_loader(config, embedding_path, batch_size=4, num_workers=4):
    """
    便捷函数：创建第一阶段编码器蒸馏的数据加载器
    """
    dataset = build_teacher_embedding_dataset(
        config, embedding_path, is_train=True, stage='stage1'
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=pseudo_collate
    )


def create_stage2_loader(config, embedding_path, batch_size=4, num_workers=4):
    """
    便捷函数：创建第二阶段完整蒸馏的数据加载器
    """
    dataset = build_teacher_embedding_dataset(
        config, embedding_path, is_train=True, stage='stage2'
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=pseudo_collate
    )


if __name__ == '__main__':
    # 测试数据加载器
    print("测试数据加载器构建...")
    
    # 模拟配置
    class Config:
        def __init__(self):
            self.DATA = type('', (), {})()
            self.DATA.DATA_PATH = 'datasets/train'
            self.DATA.BATCH_SIZE = 4
            self.DATA.NUM_WORKERS = 2  
            self.DATA.PIN_MEMORY = True
            
            self.DISTILL = type('', (), {})()
            self.DISTILL.ENCODER_ONLY = True  # 第一阶段
            self.DISTILL.TEACHER_EMBED_PATH = 'teacher_embeddings'
            
            self.LOCAL_RANK = 0
            
        def defrost(self):
            pass
            
        def freeze(self):
            pass
    
    config = Config()
    
    try:
        # 测试构建
        dataset_train, dataset_val, loader_train, loader_val = build_loader(config)
        print(f"✅ 数据加载器构建成功!")
        print(f"   训练集: {len(dataset_train)} 样本")
        print(f"   验证集: {len(dataset_val)} 样本")
        
        # 测试数据加载
        for i, batch in enumerate(loader_train):
            print(f"   批次 {i}: {len(batch)} 个元素")
            if i >= 2:  # 只测试前3个批次
                break
                
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()