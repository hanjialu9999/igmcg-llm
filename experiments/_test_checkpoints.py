import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch
from models.config_loader import load_config, build_model, load_vocab

config = load_config('configs/pretrain.yaml')
model = build_model(config, device='cpu')

# Try checkpoints_dml_test models
for model_path in ['checkpoints_dml_test/final_model.pt', 'checkpoints_dml_test/model_epoch_2.pt', 'checkpoints_dml_test/model_epoch_1.pt']:
    print(f'\nTrying {model_path}...')
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f'  Loaded successfully')

        model.eval()
        vocab = load_vocab('checkpoints/vocab.json')

        prompt = 'hello world'
        tokens = vocab.encode(prompt, add_special_tokens=False)
        tokens = [vocab.bos_idx] + tokens
        with torch.no_grad():
            generated = model.generate(tokens, max_length=50, device='cpu', temperature=0.8, top_k=50, repetition_penalty=1.2)
        text = vocab.decode(generated, skip_special=True)
        print(f'  Generated: {text[:100]}')
    except Exception as e:
        print(f'  Error: {e}')
