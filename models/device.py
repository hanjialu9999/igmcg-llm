from __future__ import annotations

from typing import Optional, Union

import torch


def get_device(preferred: Optional[Union[str, torch.device]] = None) -> torch.device:
    """自动选择计算设备，优先级：显式指定 > CUDA(NVIDIA) > DirectML(AMD/Intel) > CPU。

    这样同一套代码可以在不同配置的电脑上自动适配：
      - 有 NVIDIA 显卡 → 用 CUDA
      - 有 AMD / Intel 核显或独显（Windows）→ 通过 torch-directml 用 DirectML
      - 都没有 → 自动退回 CPU

    Args:
        preferred: 来自配置的设备字段。'auto' / None 表示自动探测；
                   其它值（如 'cuda' / 'cpu' / 'dml'）则直接使用。
    """
    pref = str(preferred).lower() if preferred is not None else 'auto'

    # 显式指定 DirectML（AMD/Intel 核显/独显），需已安装 torch-directml
    if pref == 'dml':
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device()
        except Exception:
            pass
        return torch.device('cpu')

    if pref not in ('auto', 'none', ''):
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


def apply_cpu_threads(threads: Optional[Union[int, str]] = None) -> None:
    """限制 PyTorch 占用的 CPU 线程数，避免训练/推理吃满所有核心。

    传入 None / 0 / 负数则不改变默认设置。
    """
    if not threads or int(threads) <= 0:
        return
    n = int(threads)
    try:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, n // 2))
    except Exception:
        pass