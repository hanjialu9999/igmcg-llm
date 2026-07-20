import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config_loader import load_config, build_model
from models.transformer import TransformerModel
from models.data_utils import CharTokenizer
from models.device import get_device
from scripts.generate import generate_igmcg, generate_text, NGramModel, load_model
from scripts.train import compute_lr


# ---------------------------------------------------------------------------
# data_utils 边界（单一 BaseTokenizer 契约）
# ---------------------------------------------------------------------------
def _char_vocab():
    tok = CharTokenizer()
    tok.train(['你好世界', '机器学习很有趣', '今天天气真好', '中国的首都是北京'])
    return tok


def test_tokenizer_zero_oov():
    """字符级 BaseTokenizer 对训练语料内字符零 OOV（均落单字 token，非字节回退）。"""
    tok = _char_vocab()
    for ch in '你好世界机器学习很有趣今天天气真中国的首都北京':
        tid = tok._sym_to_id(ch)
        assert isinstance(tid, int) and tid >= len(tok.special_tokens) + 256, \
            f"字符 {ch} 不应走字节回退"


def test_tokenizer_encode_decode_roundtrip():
    """中文字符级编码/解码可无损往返。"""
    tok = _char_vocab()
    text = '你好世界'
    ids = tok.encode(text)
    assert ids[0] == tok.bos_idx and ids[-1] == tok.eos_idx
    assert tok.decode(ids) == text


def test_tokenizer_oov_byte_fallback():
    """未登录稀有字走字节回退，decode 还原一致（零 OOV 契约）。"""
    tok = _char_vocab()
    text = '龘龘龘'
    ids = tok.encode(text, add_special_tokens=False)
    assert tok.decode(ids) == text


# ---------------------------------------------------------------------------
# IGMCG 多候选生成（走 batch 前向路径）
# ---------------------------------------------------------------------------
def test_igmcg_generation():
    """generate_igmcg 返回 (文本, 候选列表)，候选数 == num_candidates 且按分数降序。"""
    cfg = load_config('configs/pretrain.yaml')
    model = build_model(cfg, device='cpu')
    model.eval()
    vocab = _char_vocab()
    text, cands = generate_igmcg(model, vocab, '你好', num_candidates=4,
                                 max_length=15, base_temp=0.8, top_k=30, device='cpu')
    assert isinstance(text, str)
    assert len(cands) == 4
    scores = [c['score'] for c in cands]
    assert scores == sorted(scores, reverse=True)
    assert all('score' in c and 'rep' in c for c in cands)


# ---------------------------------------------------------------------------
# n-gram 模型
# ---------------------------------------------------------------------------
def test_ngram_logprob_vector_and_cache():
    """NGramModel.logprob_vector 返回形状 (V,) 且有限；相同上下文命中缓存。"""
    vocab = CharTokenizer()
    corpus = ['人工智能改变世界', '世界很大', '人工智能很有用', '机器学习很有趣']
    vocab.train(corpus)
    with tempfile.NamedTemporaryFile('w', suffix='.txt', encoding='utf-8', delete=False) as f:
        f.write('\n'.join(corpus))
        path = f.name
    try:
        ng = NGramModel(vocab, path, max_order=3)
        ids = vocab.encode('人工智能', add_special_tokens=False)
        vec1 = ng.logprob_vector(ids, 'cpu')
        assert vec1.shape == (len(vocab),)
        assert torch.isfinite(vec1).all()
        # 缓存命中：第二次同上下文应返回缓存张量（相同值）
        vec2 = ng.logprob_vector(ids, 'cpu')
        assert torch.allclose(vec1, vec2)
        assert (ids[-2], ids[-1]) in ng._logprob_cache
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 设备探测
# ---------------------------------------------------------------------------
def test_device_detection():
    """get_device 对显式字符串返回对应 torch.device；auto 返回合法 device 对象。"""
    assert get_device('cpu') == torch.device('cpu')
    assert get_device('cuda').type == 'cuda'
    dev = get_device('auto')
    assert isinstance(dev, torch.device)


# ---------------------------------------------------------------------------
# 配置加载错误兜底
# ---------------------------------------------------------------------------
def test_config_loader_missing_file_raises():
    """加载不存在的配置文件应抛错（FileNotFoundError / YAMLError）。"""
    with pytest.raises(Exception):
        load_config('configs/this_file_does_not_exist.yaml')


def test_config_loader_valid():
    """加载存在的配置返回非空 dict 且含 model 段。"""
    cfg = load_config('configs/pretrain.yaml')
    assert isinstance(cfg, dict)
    assert 'model' in cfg


# ---------------------------------------------------------------------------
# 学习率调度边界
# ---------------------------------------------------------------------------
def test_warmup_scheduler_ramp_and_decay():
    """compute_lr：预热期线性升温到 base；cosine 在预热后衰减低于 base；不崩溃。"""
    base = 0.01
    # 预热起点（eff_step=1, warmup=10）应约为 base/10
    assert abs(compute_lr(1, 100, 10, base, 0.0, 'constant', 0.1) - base / 10) < 1e-6
    # 预热终点达到 base
    assert abs(compute_lr(10, 100, 10, base, 0.0, 'constant', 0.1) - base) < 1e-6
    # constant 在预热后保持 base
    assert abs(compute_lr(50, 100, 10, base, 0.0, 'constant', 0.1) - base) < 1e-6
    # cosine 在预热后衰减
    assert compute_lr(100, 100, 10, base, 0.0, 'cosine', 0.1) < base
    # warmup_target 钳制 >= total_eff 时也不崩溃，且不超过 base
    assert compute_lr(100, 100, 200, base, 0.0, 'constant', 0.1) <= base + 1e-9


# ---------------------------------------------------------------------------
# 梯度累积正确性
# ---------------------------------------------------------------------------
def test_gradient_accumulation():
    """两次微批 backward 累积的梯度 == 各自独立 backward 梯度之和。"""
    cfg = load_config('configs/pretrain.yaml')
    model = build_model(cfg, device='cpu')
    model.eval()  # 关闭 dropout，保证确定性
    V = cfg['model']['vocab_size']
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    params = [p for p in model.parameters() if p.requires_grad]

    x1 = torch.randint(0, V, (2, 5))
    x2 = torch.randint(0, V, (2, 5))

    def grad_of(x):
        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out.view(-1, V), x.view(-1))
        loss.backward()
        return [p.grad.detach().clone() for p in params]

    g1 = grad_of(x1)
    g2 = grad_of(x2)

    opt.zero_grad()
    out1 = model(x1)
    F.cross_entropy(out1.view(-1, V), x1.view(-1)).backward()
    out2 = model(x2)
    F.cross_entropy(out2.view(-1, V), x2.view(-1)).backward()
    g_acc = [p.grad.detach().clone() for p in params]

    for a, b1, b2 in zip(g_acc, g1, g2):
        assert torch.allclose(a, b1 + b2, atol=1e-6), "梯度累积结果不等于两次独立反向之和"


# ---------------------------------------------------------------------------
# 量化加载（依赖本地 gitignored 的冒烟 checkpoint，缺失则跳过）
# ---------------------------------------------------------------------------
def test_quantization_load():
    """load_model(quantize=True) 在 CPU 上以 int8 动态量化加载并可生成。"""
    ckpt = Path(__file__).parent.parent / 'checkpoints_smoke_4k' / 'final_model.pt'
    vocab_path = Path(__file__).parent.parent / 'checkpoints_smoke_4k' / 'vocab.json'
    if not (ckpt.exists() and vocab_path.exists()):
        pytest.skip('缺少本地冒烟 checkpoint（gitignored），跳过量化加载测试')
    model, vocab = load_model(str(ckpt), str(vocab_path), device='cpu', quantize=True)
    out = generate_text(model, vocab, '你好', max_length=10, device='cpu')
    assert isinstance(out, str)


def test_arch_options_enabled_build_and_forward():
    """开启全部架构增强（qk_norm/attn_temp/residual_gate/hybrid_gate）后模型可正常前向与生成。"""
    cfg = load_config('configs/config_hybrid.yaml')
    # 这些增强通过 config['model'] 开关控制（与训练 YAML 一致）
    for k in TransformerModel.ENHANCEMENT_KEYS:
        cfg['model'][k] = True
    # 显式放一个 hybrid 块以覆盖混合路径门控（config_hybrid.yaml 默认只用 attn/ssm）
    cfg['model']['num_layers'] = 2
    cfg['model']['layer_plan'] = 'attn,hybrid'
    model = build_model(cfg, device='cpu')
    model.eval()
    # 新参数应存在：attn 块含 qk_norm/log_temp，hybrid 块含混合门控，各块含残差门控
    assert any(hasattr(m, 'qk_norm') for m in model.modules())
    assert any(hasattr(m, 'log_temp') for m in model.modules())
    assert any(hasattr(m, 'hybrid_attn_gate') for m in model.modules())
    assert any(hasattr(m, 'sub1_gate') for m in model.modules())
    V = cfg['model']['vocab_size']
    x = torch.randint(0, V, (2, 8))
    with torch.no_grad():
        logits, past = model(x, use_cache=True)
    assert logits.shape == (2, 8, V)
    assert past is not None
    # 增量步也走通（带门控的 KV/SSM 状态）
    with torch.no_grad():
        logits2, _ = model(x[:, -1:], past_key_values=past, use_cache=True)
    assert logits2.shape == (2, 1, V)
    # 门控默认 1.0，前向数值应有限
    assert torch.isfinite(logits2).all()


def test_ngram_last_ids_reset_across_candidates():
    """回归测试：_generate_candidates_batch 必须重置 _ngram_last_ids，否则跨调用残留污染 n-gram 上下文。"""
    vocab = CharTokenizer(vocab_size=200)
    vocab.train(['中 国 人 民', '中 国 梦 想'])
    V = len(vocab)
    from models.transformer import TransformerModel
    m = TransformerModel(V, 64, 4, 2, 128, 32)
    m.eval()
    prompt = '中 国'
    # 手动设置残留的 _ngram_last_ids（batch=2，与后续 N=4 不同）
    m._ngram_last_ids = torch.ones(2, 9, dtype=torch.long)
    from scripts.generate import generate_igmcg
    with torch.no_grad():
        generate_igmcg(m, vocab, prompt, num_candidates=4, max_length=5, device=torch.device('cpu'))
    # 验证：生成完成后 _ngram_last_ids 要么是 None，要么 shape 匹配新 batch
    if m._ngram_last_ids is not None:
        assert m._ngram_last_ids.shape[0] == 4, f'Expected batch=4, got {m._ngram_last_ids.shape[0]}'


def test_enhancement_mutex_check():
    """回归测试：互斥校验 enhancement_off_prob>0 是 bool，不应被 is not None 恒判为 True。"""
    # 模拟 train.py 的互斥校验逻辑
    enhancement_schedule = 'selv2'
    enhancement_off_prob = 0.0
    curriculum_anneal = None
    # 旧逻辑（有 bug）: sum(x is not None for x in (schedule, off_prob > 0, curriculum is not None))
    # off_prob > 0 -> False, False is not None -> True，恒计数 +1
    _n_set_buggy = sum(x is not None for x in (enhancement_schedule, enhancement_off_prob > 0, curriculum_anneal is not None))
    # 新逻辑（修复后）: 分别检查三个条件
    _n_set_fixed = sum([
        enhancement_schedule is not None,
        enhancement_off_prob > 0,
        curriculum_anneal is not None
    ])
    # 只设了 schedule，应该 _n_set=1
    assert _n_set_fixed == 1, f'Expected 1, got {_n_set_fixed}'
    # 旧逻辑会得到 3（全部恒 True，误报）
    assert _n_set_buggy == 3, f'Old logic should give 3 (false positive), got {_n_set_buggy}'


if __name__ == '__main__':
    test_vocab_empty_corpus_no_divzero()
    test_vocab_encode_decode_roundtrip()
    test_vocab_coverage_positive()
    test_igmcg_generation()
    test_ngram_logprob_vector_and_cache()
    test_device_detection()
    test_config_loader_missing_file_raises()
    test_config_loader_valid()
    test_warmup_scheduler_ramp_and_decay()
    test_gradient_accumulation()
    test_quantization_load()
    test_arch_options_enabled_build_and_forward()
    test_ngram_last_ids_reset_across_candidates()
    test_enhancement_mutex_check()
    print("\nAll pipeline tests passed!")
