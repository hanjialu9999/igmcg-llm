import json
import torch

class SimpleTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.pad_token_id = self.word2idx['<pad>']
        self.unk_token_id = self.word2idx['<unk>']
        self.bos_token_id = self.word2idx['<bos>']
        self.eos_token_id = self.word2idx['<eos>']
    
    def encode(self, text, add_special_tokens=True):
        words = text.lower().split()
        ids = [self.word2idx.get(w, self.unk_token_id) for w in words]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids
    
    def decode(self, ids):
        return ' '.join([self.idx2word.get(i, '<unk>') for i in ids 
                        if i not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]])

def generate_answer(model, tokenizer, question, max_length=100):
    """生成答案"""
    model.eval()
    device = next(model.parameters()).device
    
    # 编码问题
    input_ids = tokenizer.encode(question)
    input_tensor = torch.tensor([input_ids]).to(device)
    
    with torch.no_grad():
        # 生成答案（这里的生成方式需要根据你的模型调整）
        output = model(input_tensor)
        # 简单取最可能的token
        predicted_ids = output.argmax(dim=-1).squeeze().tolist()
    
    # 解码
    answer = tokenizer.decode(predicted_ids)
    return answer

def main():
    # 路径 - 根据实际情况修改
    vocab_path = "checkpoints/vocab.json"
    model_path = "best_finetuned_model.pt"  # 微调后的模型
    
    print("加载tokenizer和模型...")
    tokenizer = SimpleTokenizer(vocab_path)
    model = torch.load(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"模型已加载到 {device}\n")
    print("=" * 50)
    print("可以开始提问了！（输入 'quit' 退出）")
    print("=" * 50)
    
    while True:
        question = input("\n你的问题: ").strip()
        
        if question.lower() == 'quit':
            print("再见！")
            break
        
        if not question:
            continue
        
        answer = generate_answer(model, tokenizer, question)
        print(f"模型回答: {answer}")

if __name__ == "__main__":
    main()
