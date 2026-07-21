from __future__ import annotations

from typing import Optional, Union

import torch


def _try_dml_device() -> Optional[torch.device]:
    """尝试返回 DirectML 设备（AMD/Intel 核显/独显）；未安装或不可用时返回 None。

    各分支均打印一行日志，避免静默回退到 CPU 让用户误以为在用 DML。
    """
    try:
        import torch_directml
    except Exception as e:
        print(f"[Device] 未启用 DirectML：torch_directml 不可用（{type(e).__name__}: {e}），将回退 CPU")
        return None
    if getattr(torch_directml, "is_available", lambda: False)():
        print("[Device] 已检测到 DirectML 设备（AMD/Intel 核显/独显）")
        return torch_directml.device()
    print("[Device] torch_directml 已安装但当前不可用，回退 CPU")
    return None


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
        return _try_dml_device() or torch.device('cpu')

    if pref not in ('auto', 'none', ''):
        return torch.device(preferred)

    # 1) NVIDIA (CUDA)
    if torch.cuda.is_available():
        return torch.device('cuda')

    # 2) AMD / Intel 等 Windows 核显/独显：通过 DirectML 后端
    #    （使用前需在对应环境安装 torch-directml，例如 Python 3.10/3.11 + torch 2.0/2.1）
    dml = _try_dml_device()
    if dml is not None:
        return dml

    # 3) 兜底 CPU
    return torch.device('cpu')


def apply_cpu_threads(threads: Optional[Union[int, str]] = None) -> None:
    """限制 PyTorch 占用的 CPU 线程数，避免训练/推理吃满所有核心。

    传入 None / 0 / 负数 / 非数字字符串则不改变默认设置。
    """
    if not threads:
        return
    try:
        n = int(threads)
    except (ValueError, TypeError):
        return
    if n <= 0:
        return
    try:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, n // 2))
    except RuntimeError:
        pass