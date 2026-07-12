"""临时基准测试：对比本次优化的效果（RoPE 缓存命中、int8 动态量化速度与质量）。

不改动项目默认代码，仅 import 现有 API。运行：
  F:\Projects\.amd_venv\Scripts\python.exe experiments/_bench_speed.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import models.transformer as T
from generate import load_model, generate_text
from models.data_utils import Vocabulary

MODEL = "archive_unused/checkpoints_backup/_stab_ckpt/final_model.pt"
VOCAB = "checkpoints_dml_test/vocab.json"
PROMPT = "人工智能正在改变世界，"
MAX_LEN = 200
DEVICE = "cpu"
REF_TEXT = "人工智能正在改变世界，模型需要在本地设备上高效运行，中文语言模型可以生成流畅的文本。"


def timed_generate(model, vocab, quantize_label, seed=42):
    torch.manual_seed(seed)
    ids = vocab.encode(PROMPT, add_special_tokens=False)
    t0 = time.perf_counter()
    out = model.generate(ids, max_length=MAX_LEN, temperature=1.0, top_k=30,
                         repetition_penalty=1.4, device=DEVICE)
    dt = time.perf_counter() - t0
    new_ids = out[len(ids):]
    text = vocab.decode(new_ids, skip_special=True)
    toks = len(new_ids)
    print(f"  [{quantize_label}] 生成 {toks} token 用时 {dt:.3f}s -> {toks/dt:.1f} tok/s")
    return text, toks / dt


def count_rope_hits():
    hits = {"h": 0, "m": 0}
    orig = T._rope_cos_sin

    def wrapper(inv_freq, start_pos, seq_len, device, dtype, max_len=2048):
        key = (str(device), inv_freq.shape[0])
        need = start_pos + seq_len
        c = T._ROPE_CACHE.get(key)
        if c is not None and c[0].size(2) >= need:
            hits["h"] += 1
        else:
            hits["m"] += 1
        return orig(inv_freq, start_pos, seq_len, device, dtype, max_len=max_len)

    T._rope_cos_sin = wrapper
    return hits


def no_cache_rope():
    def wrapper(inv_freq, start_pos, seq_len, device, dtype, max_len=2048):
        # 模拟旧实现：每次都重算且不缓存（按 start_pos 单段），等价旧行为
        t = torch.arange(start_pos, start_pos + seq_len, device=device).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :].to(dtype)
        sin = emb.sin()[None, None, :, :].to(dtype)
        return cos, sin
    T._rope_cos_sin = wrapper


def perplexity(model, vocab, text):
    ids = vocab.encode(text, add_special_tokens=False)
    if len(ids) < 2:
        return float("nan")
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=DEVICE)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(x)
    logits = logits.view(-1, logits.size(-1))
    y = y.view(-1)
    loss = torch.nn.functional.cross_entropy(logits, y)
    return float(torch.exp(loss))


print("=" * 60)
print(f"PyTorch {torch.__version__} | device={DEVICE}")
print("=" * 60)

NATIVE_ROPE = T._rope_cos_sin  # 原生实现引用，便于后续恢复

# 1) RoPE 缓存命中率（新实现）
print("\n[1] RoPE 缓存命中率（新实现，逐 token KV 缓存生成）")
T._ROPE_CACHE.clear()
hits = count_rope_hits()
model_b, vocab = load_model(MODEL, VOCAB, device=DEVICE, quantize=False)
base_text, base_spd = timed_generate(model_b, vocab, "baseline-fp32")
print(f"    RoPE 缓存命中 {hits['h']} 次 / 未命中 {hits['m']} 次 "
      f"(命中率 {hits['h']/(hits['h']+hits['m'])*100:.1f}%)")
p_base = perplexity(model_b, vocab, REF_TEXT)

# 2) 旧实现（无缓存，每步重算）对比速度
print("\n[2] 旧实现（无 RoPE 缓存）同条件生成速度")
T._ROPE_CACHE.clear()
no_cache_rope()
old_text, old_spd = timed_generate(model_b, vocab, "old-nocache")
print(f"    新实现更快约 { (old_spd/base_spd - 1)*100:.1f}% （或无差异，因单步表很小）")

# 3) int8 动态量化：速度 + 质量
print("\n[3] int8 动态量化（--quantize）")
T._ROPE_CACHE.clear()
T._rope_cos_sin = NATIVE_ROPE  # 恢复原生 RoPE 实现
model_q, _ = load_model(MODEL, VOCAB, device=DEVICE, quantize=True)
q_text, q_spd = timed_generate(model_q, vocab, "quantized-int8")
print(f"    量化相对 fp32 提速约 { (q_spd/base_spd - 1)*100:.1f}%")

# 3b) torch.compile：稳定态速度（先 warmup 触发编译，再计时）
print("\n[3b] torch.compile（--compile）稳定态速度")
T._ROPE_CACHE.clear()
T._rope_cos_sin = NATIVE_ROPE
model_c, _ = load_model(MODEL, VOCAB, device=DEVICE, compile_model=True)
torch.manual_seed(0)
_ = model_c.generate(vocab.encode(PROMPT, add_special_tokens=False),
                     max_length=10, temperature=1.0, top_k=30,
                     repetition_penalty=1.4, device=DEVICE)  # warmup：触发编译
c_text, c_spd = timed_generate(model_c, vocab, "compiled-fp32", seed=7)
print(f"    编译相对 fp32 eager 提速约 { (c_spd/base_spd - 1)*100:.1f}%")

# 5) bf16 精度（默认 auto 启用）：速度 + 质量
print("\n[5] bf16 精度（默认 auto 会启用）稳定态速度 + 质量")
T._ROPE_CACHE.clear()
T._rope_cos_sin = NATIVE_ROPE
try:
    torch.set_autocast_enabled('cpu', True)
    torch.set_autocast_dtype('cpu', torch.bfloat16)
    model_b16, _ = load_model(MODEL, VOCAB, device=DEVICE)
    b_text, b_spd = timed_generate(model_b16, vocab, "bf16-fp32weights", seed=3)
    p_b = perplexity(model_b16, vocab, REF_TEXT)
    print(f"    bf16 相对 fp32 提速约 { (b_spd/base_spd - 1)*100:.1f}% | 参考文本困惑度 bf16={p_b:.3f} (fp32={p_base:.3f})")
finally:
    torch.set_autocast_enabled('cpu', False)

# 4) 质量对比（困惑度越低越好；用固定参考文本，确定性地隔离量化对模型的影响）
print("\n[4] 生成质量对比")
print(f"    baseline 文本: {base_text[:80]!r}")
print(f"    quantized 文本: {q_text[:80]!r}")
p_q = perplexity(model_q, vocab, REF_TEXT)
print(f"    固定参考文本困惑度  baseline={p_base:.3f} | quantized={p_q:.3f} "
      f"(差异 {p_q - p_base:+.3f})")
same = base_text == q_text
print(f"    与 baseline 生成文本完全一致: {same}")

print("\nDONE")
