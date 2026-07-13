import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, os, torch, glob
paths = [
    r'archive_unused\checkpoints_old_arch\final_model.pt',
    r'archive_unused\checkpoints_old_arch\best_finetuned_model.pt',
    r'archive_unused\checkpoints_backup\best_model_epoch_29.pt',
    r'archive_unused\checkpoints_backup\_stab_ckpt\final_model.pt',
    r'archive_unused\checkpoints_backup\checkpoints\final_model.pt',
    r'archive_unused\checkpoints_backup\checkpoints_4\final_model.pt',
]
for p in paths:
    print('=' * 70)
    print(p)
    try:
        ck = torch.load(p, map_location='cpu', weights_only=True)
    except Exception as e:
        print('  LOAD ERR', e); continue
    print('  top keys:', list(ck.keys())[:10])
    cfg = ck.get('config')
    if isinstance(cfg, dict):
        print('  config keys:', list(cfg.keys()))
        for k in ('model', 'training', 'paths', 'vocab_size'):
            if k in cfg:
                print('   ', k, '=', cfg[k] if not isinstance(cfg[k], dict) else list(cfg[k].keys()))
        if 'model' in cfg and isinstance(cfg['model'], dict):
            print('   model cfg:', cfg['model'])
    sd = ck.get('model_state_dict')
    if sd is None and 'state_dict' in ck:
        sd = ck['state_dict']
    if sd is None and isinstance(ck, dict) and any(k.endswith('.weight') or k.endswith('.bias') for k in ck.keys()):
        sd = ck
    if sd is not None:
        n = sum(v.numel() for v in sd.values())
        print(f'  params: {n/1e6:.2f}M, num tensors: {len(sd)}')
        prefixes = sorted(set(k.split('.')[0] if not k.startswith('blocks') else 'blocks.'+k.split('.')[1] for k in sd.keys()))
        print('  module prefixes:', prefixes[:20])
        print('  sample keys:', list(sd.keys())[:6])
    else:
        print('  NO state_dict found')
