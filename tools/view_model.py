import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config_loader import load_config, build_model


def view_model_structure():
    """查看模型结构和参数信息"""
    # 模型路径 - 默认查看训练产出的 final_model.pt
    model_path = "checkpoints/final_model.pt"

    print("正在加载模型...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

    # 兼容两种保存格式：完整模型对象 或 含 model_state_dict 的字典
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        config = load_config()
        model = build_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint

    print("\n" + "=" * 60)
    print("模型结构:")
    print("=" * 60)
    print(model)

    print("\n" + "=" * 60)
    print("模型参数信息:")
    print("=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    print("\n" + "=" * 60)
    print("各层参数:")
    print("=" * 60)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape} ({param.numel():,} 参数)")


if __name__ == "__main__":
    view_model_structure()
