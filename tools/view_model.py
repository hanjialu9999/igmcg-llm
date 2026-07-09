import torch

def view_model_structure():
    """查看模型结构和参数信息"""
    
    # 模型路径 - 根据需要修改
    model_path = "checkpoints/final_model.pt"
    
    print("正在加载模型...")
    model = torch.load(model_path)
    
    print("\n" + "="*60)
    print("模型结构:")
    print("="*60)
    print(model)
    
    print("\n" + "="*60)
    print("模型参数信息:")
    print("="*60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("\n" + "="*60)
    print("各层参数:")
    print("="*60)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape} ({param.numel():,} 参数)")

if __name__ == "__main__":
    view_model_structure()
