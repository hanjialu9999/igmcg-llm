import torch


def get_device(preferred=None):
    """自动选择计算设备，优先级：显式指定 > CUDA(NVIDIA) > DirectML(AMD/Intel) > CPU。

    这样同一套代码可以在不同配置的电脑上自动适配：
      - 有 NVIDIA 显卡 → 用 CUDA
      - 有 AMD / Intel 核显或独显（Windows）→ 通过 torch-directml 用 DirectML
      - 都没有 → 自动退回 CPU

    Args:
        preferred: 来自配置的设备字段。'auto' / None 表示自动探测；
                   其它值（如 'cuda' / 'cpu' / 'dml'）则直接使用。
    """
    if preferred is not None and str(preferred).lower() not in ('auto', 'none', ''):
        return torch.device(preferred)

    # 1) NVIDIA (CUDA)
    if torch.cuda.is_available():
        return torch.device('cuda')

    # 2) AMD / Intel 等 Windows 核显/独显：通过 DirectML 后端
    #    （使用前需在对应环境安装 torch-directml，例如 Python 3.10/3.11 + torch 2.0/2.1）
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except Exception:
        pass

    # 3) 兜底 CPU
    return torch.device('cpu')


def supports_amp(device):
    """是否支持 CUDA 自动混合精度（DirectML / CPU 不支持 torch.cuda.amp）。"""
    return isinstance(device, torch.device) and device.type == 'cuda'
