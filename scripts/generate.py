import torch
import json
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.transformer import TransformerModel
from models.data_utils import Vocabulary
from models.device import get_device

def load_model(model_path, vocab_path, device='cpu'):
    """Load trained model and vocabulary"""
    
    # Load vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    vocab = Vocabulary()
    vocab.word2idx = vocab_data['word2idx']
    vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    model_config = checkpoint.get('config', {
        'vocab_size': checkpoint['vocab_size'],
        'embedding_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'hidden_dim': 256,
        'max_seq_length': 32,
        'dropout': 0.1
    })
    
    model = TransformerModel(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=model_config.get('embedding_dim', 128),
        num_heads=model_config.get('num_heads', 4),
        num_layers=model_config.get('num_layers', 2),
        hidden_dim=model_config.get('hidden_dim', 256),
        max_seq_length=model_config.get('max_seq_length', 32),
        dropout=model_config.get('dropout', 0.1)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab


def generate_text(model, vocab, prompt, max_length=30, temperature=0.8, 
                  top_k=50, device='cpu'):
    """Generate text from prompt"""
    
    # Encode prompt
    tokens = vocab.encode(prompt)
    
    # Remove EOS token for generation (we only want BOS + prompt words)
    if tokens[-1] == vocab.eos_idx:
        tokens = tokens[:-1]
    
    # Generate - model will continue from the prompt
    # Use repetition penalty to avoid repetitive output
    generated = model.generate(tokens, max_length=max_length, 
                              temperature=temperature, top_k=top_k, 
                              device=device, repetition_penalty=1.4)
    
    # Decode the entire sequence
    text = vocab.decode(generated)
    
    return text.strip()


def interactive_mode(model, vocab, device='cpu'):
    """Interactive text generation"""
    print("\n" + "="*50)
    print("Text Generation with AI Model")
    print("="*50)
    print("Type prompts to generate text (type 'quit' to exit)")
    print("-"*50 + "\n")
    
    while True:
        prompt = input("Enter prompt: ").strip()
        
        if prompt.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not prompt:
            print("Prompt cannot be empty!")
            continue
        
        # Generate with different parameters
        print("\nGenerating...")
        temp_values = [0.7, 0.9]
        
        for temp in temp_values:
            generated = generate_text(model, vocab, prompt, max_length=20, 
                                     temperature=temp, top_k=50, device=device)
            print(f"[Temperature {temp}]: {generated}\n")
        
        print("-"*50)


def batch_generate(model, vocab, prompts, max_length=30, temperature=0.8, 
                   device='cpu'):
    """Generate text for multiple prompts"""
    results = []
    for prompt in prompts:
        generated = generate_text(model, vocab, prompt, max_length=max_length, 
                                 temperature=temperature, device=device)
        results.append({
            'prompt': prompt,
            'generated': generated
        })
    return results


def main():
    parser = argparse.ArgumentParser(description='Text generation with trained model')
    parser.add_argument('--model', type=str, default='./checkpoints/final_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default='./checkpoints/vocab.json',
                        help='Path to vocabulary file')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt for generation')
    parser.add_argument('--max-length', type=int, default=30,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (0.5-1.5)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto (detect) / cuda / cpu / dml')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Model not found at {args.model}")
        print("Please train the model first using: python scripts/train.py")
        return
    
    # Load model (自动适配 CUDA / DirectML(AMD) / CPU；--device 默认 auto 自动探测)
    device = get_device(args.device)
    print(f"Loading model from {args.model}...")
    model, vocab = load_model(args.model, args.vocab, device=device)
    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Generation mode
    if args.interactive:
        interactive_mode(model, vocab, device)
    elif args.prompt:
        generated = generate_text(model, vocab, args.prompt, 
                                 max_length=args.max_length,
                                 temperature=args.temperature,
                                 top_k=args.top_k,
                                 device=device)
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {generated}\n")
    else:
        # Default example
        examples = [
            "Hello, how are you",
            "The weather is",
            "I love",
            "Machine learning"
        ]
        print("\nGenerating text for example prompts:\n")
        for prompt in examples:
            generated = generate_text(model, vocab, prompt, 
                                     max_length=20,
                                     temperature=0.8,
                                     top_k=50,
                                     device=device)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}\n")


if __name__ == '__main__':
    main()
