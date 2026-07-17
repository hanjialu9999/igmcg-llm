#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并训练数据脚本（兼容性薄包装）。

实际逻辑已统一到 `scripts/data_manager.py merge` 子命令。本文件保留原命令行
接口，转发到 data_manager。新用法推荐：`python scripts/data_manager.py merge`。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data_manager import build_parser, cmd_merge


if __name__ == '__main__':
    # 复用 data_manager 的 merge 子命令解析，保持原有参数兼容
    parser = build_parser()
    parser.prog = 'merge_data.py'
    args = parser.parse_args(['merge', *sys.argv[1:]])
    args.func(args)
