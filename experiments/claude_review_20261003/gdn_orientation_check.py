"""现有 GatedDeltaNet 默认循环到底算的是什么（只读，CPU 即可）。

loop_code 逐字照抄 mixers.py:1318-1335 的单步更新和读出：
    Sk  = einsum('bhd,bhde->bhe', k, S)            # 按 S 的行收缩
    S   = a*S + b*((v - Sk).unsqueeze(-1) * k.unsqueeze(-2))   # 写入时 v 在行上
    num = einsum('bhd,bhde->bhe', q, S)            # 读出也按行收缩
loop_standard 是标准 gated delta rule（S 为 D_v×D_k，读 S@q）。
1) 写一次 (k1,v1)，用 q=k1 读：标准形式应取回 v1。
2) 同一个 k1 先写 v1 再写 v2：标准形式应取回 v2（覆盖旧值）。
"""
import torch

torch.manual_seed(0)
D = 4


def loop_code(ks, vs, alpha=1.0, beta=1.0):
    # exactly the repo's per-step update and readout (mixers.py:1318-1335)
    S = torch.zeros(D, D)
    for k, v in zip(ks, vs):
        Sk = torch.einsum('d,de->e', k, S)
        S = alpha * S + beta * ((v - Sk).unsqueeze(-1) * k.unsqueeze(-2))
    return S


def read_code(S, q):
    return torch.einsum('d,de->e', q, S)  # num = qf . S  (contracts the row index)


def loop_standard(ks, vs, alpha=1.0, beta=1.0):
    # standard (gated) delta rule, S is D_v x D_k: S = a*S(I - b k k^T) + b v k^T
    S = torch.zeros(D, D)
    for k, v in zip(ks, vs):
        S = alpha * S - beta * (S @ k).unsqueeze(-1) * k.unsqueeze(-2) + beta * v.unsqueeze(-1) * k.unsqueeze(-2)
    return S


def read_standard(S, q):
    return S @ q


k1 = torch.nn.functional.normalize(torch.rand(D), dim=0)
k2 = torch.nn.functional.normalize(torch.rand(D), dim=0)
v1 = torch.randn(D)
v2 = torch.randn(D)

print('k1      ', k1.numpy().round(3))
print('v1      ', v1.numpy().round(3))
print('--- one write (k1,v1), read with q=k1')
print('standard', read_standard(loop_standard([k1], [v1]), k1).numpy().round(3), '(should be v1)')
print('repo    ', read_code(loop_code([k1], [v1]), k1).numpy().round(3), '= (k1.v1)*k1 =', ((k1 @ v1) * k1).numpy().round(3))
print('--- overwrite: (k1,v1) then (k1,v2), read q=k1')
print('v2      ', v2.numpy().round(3))
print('standard', read_standard(loop_standard([k1, k1], [v1, v2]), k1).numpy().round(3), '(should be v2)')
print('repo    ', read_code(loop_code([k1, k1], [v1, v2]), k1).numpy().round(3))

# transpose relation: repo state after writes vs standard state
S_code = loop_code([k1, k2], [v1, v2], alpha=0.9, beta=0.7)
S_std = loop_standard([k1, k2], [v1, v2], alpha=0.9, beta=0.7)
print('--- two writes, alpha=.9 beta=.7: |S_code - S_std| =', (S_code - S_std).abs().max().item(),
      ' |S_code - S_std^T| =', (S_code - S_std.T).abs().max().item())
