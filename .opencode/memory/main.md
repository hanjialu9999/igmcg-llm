# igmcg-llm 项目记忆

## 项目概况
- 中文 LM，自定义架构：注意力 × SSM(MambaSSM) × IGMCG 直觉引导解码 + n-gram 双轨解码
- 主干：Pre-LN + RMSNorm + RoPE + SwiGLU；支持 KV-cache 增量解码
- 设备：AMD Radeon 780M iGPU (DML) + CPU fallback
- venv：`.amd_venv`（Python 3.11.9, torch 2.4.1 + torch_directml 0.2.5）

## 测试状态
- **44 passed**（tests/test_transformer.py 11 + test_generation_pipeline.py 12 + test_config_loader.py 4 + test_regression.py 17）

## 八轮审查汇总（2026-07-17）
- 共修复 45 个问题，9 笔提交已推送到 origin/main
- 关键 bug：训练路径滑窗因果泄露、生成期 memory 每步 reset、CharMerge 非因果卷积、BOS 重复
- 新增 17 个回归测试覆盖所有已修 bug 路径
- CharMergeLayer 修复了 F.conv1d 不支持 tuple padding 的 bug（此前未触发因 layer_plan=null）

## 铁律
- 先读代码再改，证据锚定
- 测试必须覆盖修改路径
- 任何新架构特性必须同时提交测试
- 训练/推理路径的掩码逻辑必须一致
