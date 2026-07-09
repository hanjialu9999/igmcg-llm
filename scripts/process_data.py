import json
import os
import glob

def process_qa_data(input_file, output_file):
    """
    将奇数行问题、偶数行答案的txt文件转换为jsonl格式
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    qa_pairs = []
    for i in range(0, len(lines)-1, 2):
        question = lines[i]
        answer = lines[i+1]
        if question and answer:  # 过滤空行
            qa_pairs.append({
                "question": question,
                "answer": answer
            })
    
    # 保存成jsonl格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"✅ 处理完成！共 {len(qa_pairs)} 对问答")
    print(f"   输入: {input_file}")
    print(f"   输出: {output_file}")

def process_folder(input_folder, output_folder):
    """
    批量处理整个文件夹的txt文件
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 找到所有txt文件
    file_pattern = os.path.join(input_folder, "*.txt")
    files = glob.glob(file_pattern)
    
    print(f"找到 {len(files)} 个txt文件\n")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, filename.replace('.txt', '.jsonl'))
        
        print(f"正在处理: {filename}")
        process_qa_data(file_path, output_path)
        print()
    
    print(f"🎉 全部完成！共处理 {len(files)} 个文件")

if __name__ == "__main__":
    # 使用示例
    
    # 方式1: 处理单个文件
    # process_qa_data('your_data.txt', 'train_data.jsonl')
    
    # 方式2: 批量处理整个文件夹
    input_folder = "data/datasets"
    output_folder = "data/processed"
    process_folder(input_folder, output_folder)
