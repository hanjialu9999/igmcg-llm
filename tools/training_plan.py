#!/usr/bin/env python3
"""
第四轮训练计划 - 完整总结
"""

print("\n" + "="*75)
print("📋 第四轮训练计划总结")
print("="*75)

print("\n" + "─"*75)
print("📊 数据进度")
print("─"*75)

data_history = [
    ("初始", 474, "单个文件"),
    ("第2轮", 2570, "格式化+扩展"),
    ("第3轮", 3327, "新增对话文件 (562行)"),
    ("第4轮", 3767, "扩展对话文件 (1002行) + 新数据"),
]

print("\n")
for name, count, desc in data_history:
    print(f"  {name:6s} : {count:4d} 句  ({desc})")

print(f"\n  实际增长: 474 → 3767 (+{((3767-474)/474)*100:.0f}%)")

print("\n" + "─"*75)
print("🧠 模型配置")
print("─"*75)

print("\n  • 嵌入维度: 256")
print("  • 注意力头数: 8")
print("  • Transformer层数: 4")
print("  • 隐层维度: 512")
print("  • 最大序列长度: 32")
print("  • Dropout: 0.1")

print("\n" + "─"*75)
print("⏱️  训练配置")
print("─"*75)

print("\n  • 批次大小: 8")
print("  • 学习率: 0.001 (cosine annealing)")
print("  • Epochs: 100 (之前50 → 扩大2倍)")
print("  • 权重衰减: 0.0001")
print("  • 梯度裁剪: 1.0")

print("\n  预计耗时: ~2-3小时 (依赖GPU)")
print("  总批次: 100 × 471 = 47,100 批")

print("\n" + "─"*75)
print("🚀 当前状态")
print("─"*75)

import os
checkpoint_dir = 'checkpoints'
files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]
if files:
    epochs = []
    for f in files:
        try:
            epoch = int(f.split('_')[2].split('.')[0])
            epochs.append(epoch)
        except Exception:
            pass
    
    if epochs:
        latest = max(epochs)
        progress = (latest / 100) * 100
        print(f"\n  ✅ 已完成: Epoch {latest}/100 ({progress:.0f}%)")
        
        remaining = 100 - latest
        print(f"  ⏳ 剩余: {remaining} epochs")
        
        if latest >= 100:
            print("\n  🎉 训练已完成！")
        else:
            print(f"\n  💡 监控方法:")
            print(f"     python monitor_live.py    (实时监控)")
            print(f"     python check_training.py  (快速查看)")

print("\n" + "─"*75)
print("📈 预期效果对比")
print("─"*75)

comparisons = [
    ("数据量", "474 → 3767", "+695%"),
    ("模型大小", "Base → Large", "参数数量↑"),
    ("训练轮次", "20 → 100", "+400%"),
    ("预期质量", "中等 → 较好", "显著改善"),
]

print("\n")
for item, change, improvement in comparisons:
    print(f"  {item:10s} : {change:20s} → {improvement}")

print("\n" + "="*75)
print("💡 使用建议")
print("="*75)

print("\n  1️⃣ 监控训练进度:")
print("     • python monitor_live.py (推荐，实时显示)")
print("     • python check_training.py (简快查看)")

print("\n  2️⃣ 训练完成后:")
print("     • python dialogue_interactive.py (交互对话)")
print("     • python quick_demo.py (快速演示)")
print("     • python dialogue_demo.py (自动演示)")

print("\n  3️⃣ 调整参数:")
print("     • 在对话中输入 'config' 调整生成参数")
print("     • temperature, top_k, repetition_penalty等")

print("\n" + "="*75 + "\n")
