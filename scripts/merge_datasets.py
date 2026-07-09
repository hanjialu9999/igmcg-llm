"""
合并所有数据集文件的脚本
"""

import os
from pathlib import Path

def merge_datasets(output_file='data/train_data_combined.txt'):
    """
    合并 data/datasets/ 文件夹中的所有数据文件
    """
    datasets_dir = Path('data/datasets')
    
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 找到所有 .txt 文件
    txt_files = sorted(datasets_dir.glob('*.txt'))
    
    print(f"找到 {len(txt_files)} 个数据文件")
    for f in txt_files:
        print(f"  - {f.name}")
    
    # 合并所有文件
    total_lines = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for txt_file in txt_files:
            if txt_file.name == 'README.md':  # 跳过说明文件
                continue
                
            print(f"\n合并 {txt_file.name}...")
            with open(txt_file, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()
                outfile.writelines(lines)
                total_lines += len(lines)
                print(f"  已添加 {len(lines)} 句话")
    
    print(f"\n✓ 合并完成!")
    print(f"输出文件: {output_path}")
    print(f"总句数: {total_lines}")
    print(f"\n下次训练时使用:")
    print(f'  train_file: "{output_file}"')


def count_lines(file_path):
    """统计文件行数"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return len([l for l in lines if l.strip()])


def stats():
    """显示数据统计"""
    print("="*50)
    print("数据集统计")
    print("="*50)
    
    datasets_dir = Path('data/datasets')
    txt_files = sorted(datasets_dir.glob('*.txt'))
    
    total = 0
    for txt_file in txt_files:
        if txt_file.name == 'README.md':
            continue
        count = count_lines(txt_file)
        total += count
        print(f"{txt_file.name:30} {count:5} 句")
    
    print("-"*50)
    print(f"{'总计':30} {total:5} 句")
    print("="*50)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'stats':
        stats()
    else:
        merge_datasets()
        print("\n提示: 修改 config/config.yaml 中的 train_file 为:")
        print('      train_file: "data/train_data_combined.txt"')
