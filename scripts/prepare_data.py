"""
数据集快速准备脚本
一键合并和准备训练数据
"""

import os
import sys
from pathlib import Path

def main():
    print("="*60)
    print("数据集准备工具")
    print("="*60)
    
    # 检查数据文件
    datasets_dir = Path('data/datasets')
    txt_files = list(datasets_dir.glob('*.txt'))
    
    print(f"\n✓ 找到 {len(txt_files)} 个数据文件")
    total_lines = 0
    for f in sorted(txt_files):
        if f.name != 'README.md':
            with open(f, 'r', encoding='utf-8') as file:
                lines = len([l for l in file if l.strip()])
            total_lines += lines
            print(f"  - {f.stem:<25} {lines:>4} 句")
    
    print(f"\n总计: {total_lines} 句话")
    
    # 合并数据
    print("\n合并数据集...")
    output_file = 'data/train_data_combined.txt'
    output_path = Path(output_file)
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for txt_file in sorted(datasets_dir.glob('*.txt')):
            if txt_file.name == 'README.md':
                continue
            with open(txt_file, 'r', encoding='utf-8') as infile:
                outfile.writelines(infile.readlines())
    
    print(f"✓ 合并完成: {output_file}")
    
    # 更新配置文件
    print("\n更新配置文件...")
    config_file = 'config/config.yaml'
    
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换路径
    new_content = content.replace(
        'train_file: "data/train_data.txt"',
        'train_file: "data/train_data_combined.txt"'
    )
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✓ 配置已更新")
    
    print("\n" + "="*60)
    print("准备完成! 可以开始训练了")
    print("="*60)
    print("\n执行以下命令开始训练:")
    print("  python scripts/train.py")
    print("\n或者使用菜单:")
    print("  python run.py")
    print("="*60)


if __name__ == '__main__':
    main()
