"""标准 gated delta rule：逐步循环 vs 分块并行（WY/UT 形式）原型。只读实验，不改模型代码。

递推（S 为 D_v × D_k）：
    S_t = a_t · S_{t-1} (I - b_t k_t k_tᵀ) + b_t v_t k_tᵀ
    o_t = S_t q_t
仓库现有 GatedDeltaNet 默认循环（mixers.py:1318-1335）不是这个递推：写入 (v - Sk)⊗k 把 v 放在行上，
读出 einsum('bhd,bhde->bhe', k, S) 却按行收缩，等于取回 (k·v)k，见 gdn_orientation_check.py。
本原型只回答两件事：标准形式能不能分块并行、在 DML 上快多少、误差多大。

分块算法只用 matmul / exp / log / clamp / cat / 逐元素乘，不用 scatter、in-place、cumsum、cumprod、
torch.full、三角求解，按项目已知的 DML 坑写。块内 (I + A)^{-1} 用 (I - A)(I + A²)(I + A⁴)… 乘积展开
（A 严格下三角、幂零），块长 C=16 时数值稳定；C=32/64 在长记忆（a→1）下会炸，别用。

用法：
    python experiments/claude_review_20261003/gdn_chunk_proto.py            # CPU：误差 + 算子数 + 耗时
    python experiments/claude_review_20261003/gdn_chunk_proto.py --device dml   # DML：误差（对 CPU fp64）+ 耗时
"""
import argparse
import time
import torch


def loop_standard(q, k, v, a, b):
    """参考实现：逐步循环。q,k,v: (B,H,T,D)；a,b: (B,H,T,1)。"""
    B, H, T, D = q.shape
    S = q.new_zeros(B, H, D, D)
    outs = []
    for t in range(T):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        at, bt = a[:, :, t].unsqueeze(-1), b[:, :, t].unsqueeze(-1)
        Sk = torch.einsum('bhed,bhd->bhe', S, kt)
        S = at * S + bt * ((vt - at.squeeze(-1) * Sk).unsqueeze(-1) * kt.unsqueeze(-2))
        outs.append(torch.einsum('bhed,bhd->bhe', S, qt))
    return torch.stack(outs, dim=2), S


def _tril(C, device, dtype, strict=False):
    i = torch.arange(C, device=device)
    m = (i.unsqueeze(1) > i.unsqueeze(0)) if strict else (i.unsqueeze(1) >= i.unsqueeze(0))
    return m.to(dtype)


def _unit_lower_inverse_apply(A, X, C):
    """(I + A)^{-1} X，A 严格下三角 (…,C,C)。各因子都是 A 的多项式，可交换，依次作用到 X 上。"""
    X = X - A @ X
    P = A @ A
    p = 2
    while p < C:
        X = X + P @ X
        p *= 2
        if p < C:
            P = P @ P
    return X


def chunked_standard(q, k, v, a, b, C=16):
    """分块并行版本，结果与 loop_standard 数学等价。块间只剩 T/C 次顺序步。"""
    B, H, T, D = q.shape
    assert T % C == 0, 'T 须为 C 的整数倍（真实实现里对尾块补 a=1,b=0 的单位元）'
    n = T // C
    dev, dt = q.device, q.dtype
    qc, kc, vc = (x.reshape(B, H, n, C, D) for x in (q, k, v))
    la = torch.log(a).reshape(B, H, n, C, 1)          # 真实实现可直接用 logsigmoid(raw)
    bc = b.reshape(B, H, n, C, 1)
    L_incl = _tril(C, dev, dt)                         # 下三角含对角
    L_strict = _tril(C, dev, dt, strict=True)
    g = L_incl @ la                                    # 块内累积 log 衰减（用矩阵乘代替 cumsum）
    diff = g - g.transpose(-1, -2)                     # g_t - g_i
    G = torch.exp(diff.clamp(max=0.0)) * L_incl        # i<=t 时 γ_t/γ_i，其余 0（不用 where/full）
    KK = kc @ kc.transpose(-1, -2)
    A = bc * G * KK * L_strict                         # 严格下三角
    U_v = _unit_lower_inverse_apply(A, bc * vc, C)
    eg = torch.exp(g)
    W = _unit_lower_inverse_apply(A, bc * eg * kc, C)
    P = G * (qc @ kc.transpose(-1, -2))                # (M ⊙ Q Kᵀ)，含对角
    dec_end = torch.exp(g[:, :, :, -1:, :] - g)        # γ_C/γ_i
    g_end = torch.exp(g[:, :, :, -1, :]).unsqueeze(-1)  # (B,H,n,1,1)
    S = q.new_zeros(B, H, D, D)
    outs = []
    for c in range(n):
        U = U_v[:, :, c] - W[:, :, c] @ S.transpose(-1, -2)
        outs.append(eg[:, :, c] * (qc[:, :, c] @ S.transpose(-1, -2)) + P[:, :, c] @ U)
        S = g_end[:, :, c] * S + (U * dec_end[:, :, c]).transpose(-1, -2) @ kc[:, :, c]
    return torch.cat(outs, dim=2).reshape(B, H, T, D), S


def make_inputs(B=24, H=8, T=64, D=32, alpha_mu=-2.0, beta_mu=2.0, spread=1.0, seed=0):
    """r42 Controller 规模：B=24, H=8, T=64, head_dim=32；k 为 relu 后 L2 归一化（同仓库 _feat + 归一化）。"""
    gen = torch.Generator().manual_seed(seed)
    q = torch.relu(torch.randn(B, H, T, D, generator=gen)) + 1e-6
    k = torch.relu(torch.randn(B, H, T, D, generator=gen)) + 1e-6
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    v = torch.randn(B, H, T, D, generator=gen)
    a = torch.sigmoid(alpha_mu + spread * torch.randn(B, H, T, 1, generator=gen))
    b = torch.sigmoid(beta_mu + spread * torch.randn(B, H, T, 1, generator=gen))
    return [q, k, v, a, b]


REGIMES = [('r42 初始化附近 a≈0.12 b≈0.88', dict()),
           ('长记忆 a≈0.98 b≈0.88', dict(alpha_mu=4.0)),
           ('极端 a≈0.999 b≈0.999', dict(alpha_mu=7.0, beta_mu=7.0, spread=0.1))]


def count_ops(fn):
    from torch.utils._python_dispatch import TorchDispatchMode
    n = [0]

    class Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            n[0] += 1
            return func(*args, **(kwargs or {}))
    with Count():
        fn()
    return n[0]


def timed(fn, sync, reps=3):
    ts = []
    for _ in range(reps + 1):  # 第一遍预热不计
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    ts = sorted(ts[1:])
    return ts[len(ts) // 2], ts[0], ts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu', choices=['cpu', 'dml'])
    ap.add_argument('--chunk', type=int, default=16)
    args = ap.parse_args()
    if args.device == 'dml':
        import torch_directml
        dev = torch_directml.device()
    else:
        dev = torch.device('cpu')
        torch.set_num_threads(4)
    C = args.chunk
    print(f'device={dev}  chunk={C}')

    # 1) 误差：对 CPU fp64 逐步循环
    for name, kw in REGIMES:
        x = make_inputs(**kw)
        ref64, _ = loop_standard(*[t.double() for t in x])
        xd = [t.to(dev) for t in x]
        loop32, _ = loop_standard(*xd)
        chunk32, _ = chunked_standard(*xd, C=C)
        e_loop = (loop32.cpu().double() - ref64).abs().max().item()
        e_chunk = (chunk32.cpu().double() - ref64).abs().max().item()
        rel = ((chunk32.cpu().double() - ref64).norm() / ref64.norm()).item()
        print(f'{name:28s} max|loop32-ref64|={e_loop:.2e}  max|chunk32-ref64|={e_chunk:.2e}  rel={rel:.1e}')

    # 2) 梯度对拍（CPU fp32 循环 vs 本设备分块）
    x = make_inputs()
    x1 = [t.clone().requires_grad_(True) for t in x]
    x2 = [t.to(dev).requires_grad_(True) for t in x]
    loop_standard(*x1)[0].pow(2).sum().backward()
    chunked_standard(*x2, C=C)[0].pow(2).sum().backward()
    for nm, t1, t2 in zip('qkvab', x1, x2):
        d = (t1.grad - t2.grad.cpu()).abs().max().item()
        print(f'grad {nm}: max|Δ|={d:.2e}  （梯度量级 {t1.grad.abs().max().item():.2e}）')

    # 3) 速度：前向+反向，3 次中位数（DML 有 ±30% 波动，报范围）
    xs = [t.to(dev).requires_grad_(True) for t in make_inputs()]
    sync = (lambda: xs[0].grad.sum().item()) if args.device == 'dml' else (lambda: None)

    def run(fn):
        for t in xs:
            t.grad = None
        fn(*xs)[0].pow(2).sum().backward()
    for label, fn in [('逐步循环', loop_standard), (f'分块 C={C}', lambda *a: chunked_standard(*a, C=C))]:
        med, lo, hi = timed(lambda: run(fn), sync)
        extra = ''
        if args.device == 'cpu':
            extra = f'  前向+反向算子数={count_ops(lambda: run(fn))}'
        print(f'{label:10s} 前向+反向 {med * 1e3:7.1f} ms（{lo * 1e3:.1f}–{hi * 1e3:.1f}）{extra}')


if __name__ == '__main__':
    main()
