import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.checkpoint import load_model


def view_model_structure():
    """查看模型结构和参数信息"""
    model_path = "checkpoints/final_model.pt"
    vocab_path = "checkpoints/vocab.json"

    print("正在加载模型...")
    # 复用 load_model：从 *_config.yaml 透传增强开关，避免 state_dict 不匹配
    model, _ = load_model(model_path, vocab_path, device='cpu')

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
