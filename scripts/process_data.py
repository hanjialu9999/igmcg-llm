"""
将 data/datasets/ 下的 QA 文本转为 jsonl（兼容性薄包装）。

实际逻辑已统一到 `scripts/data_manager.py to-jsonl` 子命令。新用法推荐：
`python scripts/data_manager.py to-jsonl`。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data_manager import build_parser


if __name__ == '__main__':
    input_folder = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'data/datasets'
    output_folder = 'data/processed'
    if '-o' in sys.argv:
        output_folder = sys.argv[sys.argv.index('-o') + 1]
    parser = build_parser()
    args = parser.parse_args(['to-jsonl', '--input_folder', input_folder, '--output_folder', output_folder])
    args.func(args)
