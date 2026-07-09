#!/usr/bin/env python
"""
快速启动脚本 - 选择训练或生成文本
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def show_menu():
    """显示菜单"""
    print("\n" + "="*50)
    print("AI文本生成模型 - 快速启动")
    print("="*50)
    print("1. 查看配置参数")
    print("2. 开始训练模型")
    print("3. 生成文本（交互模式）")
    print("4. 单条生成文本")
    print("5. 退出")
    print("="*50)
    choice = input("选择 (1-5): ").strip()
    return choice


def show_config():
    """显示当前配置"""
    import yaml
    config_path = 'config/config.yaml'
    if not Path(config_path).exists():
        print(f"配置文件不存在：{config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*50)
    print("当前配置参数")
    print("="*50)
    print("\n【模型配置】")
    for key, value in config['model'].items():
        print(f"  {key}: {value}")
    
    print("\n【训练配置】")
    for key, value in config['training'].items():
        print(f"  {key}: {value}")
    
    print("\n【数据配置】")
    for key, value in config['data'].items():
        print(f"  {key}: {value}")
    print("="*50)


def start_training():
    """开始训练"""
    print("\n开始训练模型...")
    os.system('python scripts/train.py --config config/config.yaml')


def interactive_generation():
    """交互生成"""
    device = input("使用GPU还是CPU? (cuda/cpu, 默认cpu): ").strip() or 'cpu'
    print(f"\n使用设备: {device}")
    os.system(f'python scripts/generate.py --interactive --device {device}')


def single_generation():
    """单条生成"""
    prompt = input("输入提示词: ").strip()
    if not prompt:
        print("提示词不能为空!")
        return
    
    max_length = input("最大长度 (默认30): ").strip()
    max_length = int(max_length) if max_length.isdigit() else 30
    
    temperature = input("温度参数 (默认0.8, 0.5-1.5): ").strip()
    temperature = float(temperature) if temperature else 0.8
    
    device = input("使用GPU还是CPU? (cuda/cpu, 默认cpu): ").strip() or 'cpu'
    
    print(f"\n生成文本 (提示词: {prompt})...")
    os.system(f'python scripts/generate.py --prompt "{prompt}" --max-length {max_length} --temperature {temperature} --device {device}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'generate', 'config'],
                        help='运行模式')
    args = parser.parse_args()
    
    if args.mode:
        if args.mode == 'config':
            show_config()
        elif args.mode == 'train':
            start_training()
        elif args.mode == 'generate':
            interactive_generation()
    else:
        # 菜单模式
        while True:
            choice = show_menu()
            
            if choice == '1':
                show_config()
            elif choice == '2':
                start_training()
            elif choice == '3':
                interactive_generation()
            elif choice == '4':
                single_generation()
            elif choice == '5':
                print("再见!")
                break
            else:
                print("无效选择，请重试!")


if __name__ == '__main__':
    main()
