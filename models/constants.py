"""共享常量（单一事实来源）。

跨模块重复的魔数/特殊 token 定义集中此处，避免「改一处漏一处」的静默发散：
- 特殊 token 列表与索引（data_utils.Vocabulary / BaseTokenizer 曾各硬编码一份）
- 注意力掩码填充值、RoPE 基频（transformer / config_loader 多处约定同一魔数）
- 生成默认超参（generate / chat / gen_50mb / load_generation_config 散落默认）
"""

# —— 特殊 token（顺序即索引，新增须追加在末尾、勿改既有位置）——
SPECIAL_TOKENS: tuple = ('<pad>', '<???>', '<bos>', '<eos>', '[SEP]')
PAD_IDX: int = 0
UNK_IDX: int = 1
BOS_IDX: int = 2
EOS_IDX: int = 3
SEP_IDX: int = 4

# —— 注意力 / 位置编码魔数 ——
MASK_FILL_VALUE: float = -1e9   # 因果/窗口掩码填充（被 SDPA 视为 -inf）
ROPE_BASE: float = 10000.0      # RoPE 默认基频

# —— 生成默认超参（CLI 缺省与回退默认对齐）——
DEFAULT_GENERATION: dict = {
    'temperature': 1.0,
    'top_k': 50,
    'top_p': 0.9,
    'repetition_penalty': 1.4,
}
