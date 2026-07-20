from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.constants import (SPECIAL_TOKENS, PAD_IDX, UNK_IDX, BOS_IDX,
                              EOS_IDX, SEP_IDX)


class TextDataset(Dataset):
    """Improved dataset with lazy loading and better preprocessing"""
    
    def __init__(self, texts: List[str], vocab: 'BaseTokenizer', max_seq_length: int = 32, preprocess: bool = True):
        """
        Args:
            texts: List of text strings
            vocab: BaseTokenizer object (统一分词，训练/推理共用)
            max_seq_length: Maximum sequence length
            preprocess: Whether to preprocess texts (clean, deduplicate)
        """
        self.vocab = vocab
        self.max_seq_length = max_seq_length

        # Preprocess texts
        if preprocess:
            texts = self._preprocess_texts(texts)

        # 预分词：一次性把全部文本编码为紧凑 int16 张量并释放原始字符串，避免 3.6M 条字符串
        # 与逐样本缓存（encoded_cache）撑爆内存——在 AMD 核显（共享显存）上会触发 DML
        # “GPU 不响应更多命令 / device reset”导致 backward 崩溃。
        L = max_seq_length + 1
        pad = self.vocab.pad_idx
        arr = np.zeros((len(texts), L), dtype=np.int16)
        for i, text in enumerate(texts):
            tokens = self.vocab.encode(text)
            n = len(tokens)
            if n > L:
                tokens = tokens[:L]
            elif n < L:
                tokens = tokens + [pad] * (L - n)
            arr[i] = tokens
        self.tokens = torch.from_numpy(arr)  # (N, L) int16，取用时转 long
        self.texts = None  # 释放原始字符串，回收内存
        
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Clean and validate texts"""
        cleaned: List[str] = []
        seen: set = set()
        for text in texts:
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Skip empty texts
            if len(text.strip()) == 0:
                continue
            
            # Skip duplicate texts (O(1) lookup with set)
            if text not in seen:
                seen.add(text)
                cleaned.append(text)
        
        return cleaned
    
    def __len__(self) -> int:
        return int(self.tokens.shape[0])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回预填充并转为 long tensor 的样本（已预先编码到 self.tokens，热路径零重算）。"""
        row = self.tokens[idx]
        Lm1 = self.max_seq_length
        item = {
            'input_ids': row[:Lm1].long(),
            'target_ids': row[1:Lm1 + 1].long(),
            'length': torch.tensor(Lm1, dtype=torch.long),
        }
        return item


def load_data(data_file: str, vocab_size: int = 5000, max_seq_length: int = 32,
              min_freq: int = 1, vocab: Optional['BaseTokenizer'] = None) -> Tuple[TextDataset, 'BaseTokenizer']:
    """Load data from file with validation.

    统一走 `BaseTokenizer`（字符级 `CharTokenizer` 适配字符级 LM）；旧的 `Vocabulary`
    双轨已删除。若未传入 vocab，则按字符级构建（CharTokenizer().train(texts)）。
    """
    print(f"Loading data from {data_file}...")
    
    with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(texts)} lines")
    
    # 构建字符级词表（零 OOV，与推理 load_vocab 同一套分词语义）
    if vocab is None:
        vocab = CharTokenizer(vocab_size=vocab_size)
        vocab.train(texts, min_freq=min_freq)
    
    # Create dataset
    dataset = TextDataset(texts, vocab, max_seq_length, preprocess=True)
    del texts  # 释放原始字符串列表，避免 3.6M 条字符串常驻内存
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Max sequence length: {max_seq_length}")
    
    print(f"\nVocabulary Statistics:")
    print(f"  vocab_size: {len(vocab)}")
    print(f"  special_tokens: {len(vocab.special_tokens)}")
    print(f"  regular symbols: {len(vocab.word2idx) - len(vocab.special_tokens)}")
    
    return dataset, vocab


def create_dataloader(dataset: TextDataset, batch_size: int = 16, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """Create dataloader with better defaults and parallel data loading"""
    import os
    
    # Windows 上的多进程会导致 RuntimeError，使用 num_workers=0
    if os.name == 'nt':
        num_workers = 0
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # Windows 改为 0，Linux/Mac 可用 > 0
        pin_memory=torch.cuda.is_available(),  # Use GPU memory if available
        prefetch_factor=2 if num_workers > 0 else None,  # 仅在多进程时预取
        persistent_workers=(num_workers > 0),  # Keep workers alive
        drop_last=shuffle  # Drop last incomplete batch during training
    )


def split_dataset(dataset: TextDataset, train_ratio: float = 0.9, seed: int = 42) -> Tuple[Dataset, Dataset]:
    """Split dataset into train and validation sets"""
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, val_dataset


class BaseTokenizer:
    """分词器基类（BPE 与字符级共用的共享实现，D 项去误继承）。

    统一提供：特殊 token + 256 字节级 fallback 的词表初始化、字符有效性过滤、
    BPE 预分词、符号→id 查表（含 UTF-8 字节回退，OOV=0）、通用 encode/decode。
    具体分词策略（BPE 合并 / 纯字符）由子类 BPETokenizer / CharTokenizer 各自实现。
    """

    BYTE_PREFIX = 'bytes:'  # 字节级 token 前缀，如 bytes:0 ~ bytes:255

    def __init__(self, vocab_size: int = 8000, special_tokens: Optional[List[str]] = None):
        if special_tokens is None:
            special_tokens = list(SPECIAL_TOKENS)
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pad_idx = PAD_IDX
        self.unk_idx = UNK_IDX
        self.bos_idx = BOS_IDX
        self.eos_idx = EOS_IDX
        self.sep_idx = SEP_IDX
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        # 1) 特殊 token
        for idx, tok in enumerate(self.special_tokens):
            self.word2idx[tok] = idx
            self.idx2word[idx] = tok
        # 2) 256 个字节级 token（最底层 fallback）：任何字符都能用 UTF-8 字节表示，
        #    彻底避免未登录字变 <???>（OOV 永久为 0）。
        self.byte_tokens: List[str] = []
        for b in range(256):
            tok = f'{self.BYTE_PREFIX}{b}'
            self.byte_tokens.append(tok)
            self.word2idx[tok] = len(self.word2idx)
            self.idx2word[len(self.word2idx) - 1] = tok
        # 单字/子词可用的额度 = 总词表 - 已占用（special + 256 byte）
        self._symbol_cap = self.vocab_size - len(self.word2idx)

    # ---------- 训练 ----------
    def train(self, texts: List[str], min_freq: int = 2, cjk_boost: float = 5.0) -> None:
        """对训练语料跑 BPE，生成子词词表。

        min_freq: 单字/子词进入词表的最低出现次数，过滤一次性噪声。
        cjk_boost: 中文相邻字对的合并优先级加成（默认 5），让中文词组优先
            合并为子词，避免有限的子词额度被英文/数字碎片抢光导致中文几乎
            不合并（实测原版中文多字词为 0）。
        """
        self._cjk_boost = cjk_boost
        # 1) 预分词（BPE 专用：CJK 连续不拆散，使中文相邻字对可合并）
        counter: Counter = Counter()
        for text in texts:
            for w in self._bpe_pre_tokenize(text):
                counter[w] += 1

        # 2) 初始词表 = 高频单字（含 CJK 与非 CJK 字符），过滤低频噪声
        #    仅用 symbol_cap 的 80% 作为单字额度，预留 20% 给 BPE 合并出的子词，
        #    剩余额度已被 special(5) + 256 byte token 占去。
        letters = [w for w, c in counter.items() if c >= min_freq]
        letter_cap = self._symbol_cap - int(self._symbol_cap * 0.2)
        for ch in letters:
            if ch not in self.word2idx:
                self.word2idx[ch] = len(self.word2idx)
            if len(self.word2idx) >= letter_cap:
                break
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        # 3) 字节对合并：统计相邻符号对频次，反复合并最高频对
        # 语料以单字序列表示，逐步合并。预分词单元（连续中文串/英文词）拆成
        # 字符后，仅保留已在词表中的字符参与统计（中文串的字符都是单字，故
        # 中文相邻字对能正确进入 pair_freq 并被合并）。
        sym_freq: Counter = Counter()
        pair_freq: Counter = Counter()
        word_syms = {}  # word -> 词表内字符序列
        for word, cnt in counter.items():
            syms = [c for c in word if c in self.word2idx]
            if len(syms) < 2:
                continue
            word_syms[word] = syms
            for _ in range(cnt):
                for i in range(len(syms) - 1):
                    pair_freq[(syms[i], syms[i + 1])] += 1
                for s in syms:
                    sym_freq[s] += 1

        while len(self.word2idx) < self.vocab_size and pair_freq:
            # 取加权最高频对：中文相邻字对给予 cjk_boost 加成，使其优先合并
            best = max(pair_freq.items(),
                       key=lambda kv: (self._pair_score(kv[0]) * kv[1], kv[0]))
            pair, freq = best
            if self._pair_score(pair) * freq < min_freq:
                break
            a, b = pair
            new_sym = a + b
            if new_sym in self.word2idx:
                # 已存在则仅更新统计，不再合并
                pair_freq.pop(pair, None)
                continue
            self.word2idx[new_sym] = len(self.word2idx)
            self.idx2word[len(self.word2idx) - 1] = new_sym
            self.merges.append(pair)
            # 增量更新：在 word_syms 中把 a b 合并为 new_sym，并同步修正 pair_freq
            # 仅处理受影响位置（a b 处及其左右邻接对），避免每轮全量重算
            for word, cnt in counter.items():
                if word not in word_syms:
                    continue
                syms = word_syms[word]
                if a not in syms or b not in syms:
                    continue
                new_syms = []
                i = 0
                changed = False
                while i < len(syms):
                    if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                        # 移除旧邻接对 (left,a) (a,b) (b,right)，加入 (left,new) (new,right)
                        left = syms[i - 1] if i > 0 else None
                        right = syms[i + 2] if i + 2 < len(syms) else None
                        if left is not None:
                            pair_freq[(left, a)] -= cnt
                        pair_freq[(a, b)] -= cnt
                        if right is not None:
                            pair_freq[(b, right)] -= cnt
                        new_syms.append(new_sym)
                        # 加入新邻接对
                        if left is not None:
                            pair_freq[(left, new_sym)] += cnt
                        if right is not None:
                            pair_freq[(new_sym, right)] += cnt
                        i += 2
                        changed = True
                    else:
                        new_syms.append(syms[i])
                        i += 1
                if changed:
                    word_syms[word] = new_syms
            # 清理计数为 0 的项，避免 max() 取到噪声
            pair_freq = Counter({k: v for k, v in pair_freq.items() if v > 0})
            if len(self.word2idx) >= self.vocab_size:
                break

        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def _truncate_to_vocab(self) -> None:
        """单字阶段若超过 vocab_size，按出现频次保留高频单字。"""
        if len(self.word2idx) <= self.vocab_size:
            return
        # special tokens 固定在前，截断普通单字
        extras = sorted(
            [(w, i) for w, i in self.word2idx.items() if i >= len(self.special_tokens)],
            key=lambda wi: wi[1],
        )
        keep = dict(list(self.word2idx.items())[:self.vocab_size])
        self.word2idx = keep

    @staticmethod
    def _is_cjk(ch: str) -> bool:
        # symbol 可能已是合并出的子词（长度>1）；只要含 CJK 字符即视为中文侧
        return any(0x3400 <= ord(c) <= 0x9fff or 0xf900 <= ord(c) <= 0xfaff
                   or 0xff00 <= ord(c) <= 0xffef for c in ch)

    def _pair_score(self, pair: Tuple[str, str]) -> float:
        """合并优先级权重：中文相邻字对给予 cjk_boost 加成，其余为 1。"""
        a, b = pair
        if self._is_cjk(a) and self._is_cjk(b):
            return getattr(self, '_cjk_boost', 5.0)
        return 1.0

    @staticmethod
    def _is_valid_char(ch: str) -> bool:
        """剔除无效/噪声字符：替换符、私用区、未定义码点、控制字符、CJK 扩展/增补区生僻字。"""
        if ch == '\ufffd':
            return False
        o = ord(ch)
        if o < 0x20 or 0x7f <= o < 0xa0:
            return False  # 控制字符 / DEL / C1
        if 0xe000 <= o <= 0xf8ff:
            return False  # 私用区
        if 0xf0000 <= o <= 0xfffff or 0x100000 <= o <= 0x10ffff:
            return False  # 增补私用区
        # 剔除 CJK 扩展区（U+3400 基本区之外）生僻字：扩展 A(U+3400-4DBF)/B+(U+20000+)
        # 这些字符在干净语料中极少出现，混入词表会导致生成偏向生僻乱码汉字。
        if 0x3400 <= o <= 0x4dbf:
            return False  # CJK 扩展 A
        if 0x20000 <= o <= 0x2fffd:
            return False  # CJK 扩展 B/C/D/E/F/G/H 等增补汉字
        try:
            unicodedata.name(ch)
        except ValueError:
            return False  # 未定义码点（如 U+2EC9）
        return True

    def _pre_tokenize(self, text: str) -> List[str]:
        """与 Vocabulary.tokenize 一致的预分词：CJK 逐字，其余按空格/标点；
        并过滤含无效字符的预分词单元（乱码/替换符/私用区）。"""
        text = ' '.join(text.split()).lower()
        text = re.sub(r'([\u3400-\u9fff\uf900-\ufaff\uff00-\uffef])', r' \1 ', text)
        text = re.sub(r'([.!?,;:-])', r' \1 ', text)
        out = []
        for w in text.split():
            if w and all(self._is_valid_char(c) for c in w):
                out.append(w)
        return out

    def _bpe_pre_tokenize(self, text: str) -> List[str]:
        """BPE 专用预分词：CJK 字符保持连续（不逐字插空格），使中文相邻字对
        能被 BPE 合并为词组子词；非 CJK 仍按空格/标点切分。过滤无效字符。"""
        text = ' '.join(text.split()).lower()
        # 标点单独成 token（与训练统计一致）
        text = re.sub(r'([.!?,;:-])', r' \1 ', text)
        out = []
        for w in text.split():
            if w and all(self._is_valid_char(c) for c in w):
                out.append(w)
        return out

    # ---------- 编码/解码 ----------
    def _merge_word(self, word: str) -> List[str]:
        """把单个预分词单元按 merges 规则贪心合并为子词序列。"""
        # 初始为单字
        syms = list(word)
        for a, b in self.merges:
            new_syms = []
            i = 0
            while i < len(syms):
                if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                    new_syms.append(a + b)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            syms = new_syms
        return syms

    def tokenize(self, text: str) -> List[str]:
        out: List[str] = []
        for w in self._bpe_pre_tokenize(text):
            out.extend(self._merge_word(w))
        return out

    def _sym_to_id(self, sym: str) -> int:
        """符号查表；查不到则回退到 UTF-8 字节级 token，确保任何字符都能编码（OOV=0）。"""
        if sym in self.word2idx:
            return self.word2idx[sym]
        # fallback：拆成 UTF-8 字节，用 bytes:N token 表示
        try:
            bs = sym.encode('utf-8')
        except UnicodeEncodeError:
            bs = sym.encode('utf-8', errors='replace')
        ids = []
        for b in bs:
            tok = f'{self.BYTE_PREFIX}{b}'
            ids.append(self.word2idx.get(tok, self.unk_idx))
        return ids

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.bos_idx)
        for sym in self.tokenize(text):
            r = self._sym_to_id(sym)
            if isinstance(r, list):
                tokens.extend(r)
            else:
                tokens.append(r)
        if add_special_tokens:
            tokens.append(self.eos_idx)
        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        words: List[str] = []
        byte_buf: List[int] = []  # 收集连续的字节级 token，整段还原为字符
        prefix = self.BYTE_PREFIX

        def flush() -> None:
            if byte_buf:
                try:
                    words.append(bytes(byte_buf).decode('utf-8', errors='replace'))
                except Exception:
                    words.append(''.join(chr(b) for b in byte_buf))
                byte_buf.clear()

        for tid in token_ids:
            if skip_special and tid in {self.pad_idx, self.bos_idx, self.eos_idx, self.sep_idx}:
                continue
            w = self.idx2word.get(tid, self.idx2word.get(str(tid), '<?>'))
            if isinstance(w, str) and w.startswith(prefix):
                byte_buf.append(int(w[len(prefix):]))
            else:
                flush()
                words.append(w)
        flush()
        text = ''.join(words)  # BPE 子词直接拼接，无需空格
        text = text.replace('[SEP]', '').replace('[sep]', '')
        return text

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, path: str) -> None:
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': {str(k): v for k, v in self.idx2word.items()},
                'merges': [list(m) for m in self.merges],
                'special_tokens': self.special_tokens,
                'bpe': True,
            }, f, ensure_ascii=False, indent=2)


class BPETokenizer(BaseTokenizer):
    """BPE（Byte-pair Encoding）子词分词器。

    继承 BaseTokenizer 的共享词表/编码基础设施，BPE 特有的训练、字节对合并、
    合并后 tokenize 逻辑直接复用基类实现（train / _merge_word / tokenize 定义在基类）。
    单独成类仅为语义清晰：字符级 CharTokenizer 不再"误继承"BPE 分词器。
    """


class CharTokenizer(BaseTokenizer):
    """字符级分词器（学习型分词的词表基座）。

    与 BPETokenizer 差异：不做 BPE 合并，词表 = 单字 + 256 byte token；
    相邻字符的"合并成词"完全交给模型侧的 CharMergeLayer 学习（受 LM loss
    监督），分词器本身只负责字符→索引的零 OOV 映射。导出标记 char:true。
    """

    def train(self, texts: List[str], min_freq: int = 1) -> None:
        # 1) 统计字符频次（CJK 逐字 + 非 CJK 按空格切分后的字符/词字符）
        counter: Counter = Counter()
        for text in texts:
            for w in self._bpe_pre_tokenize(text):
                for ch in w:
                    if self._is_valid_char(ch):
                        counter[ch] += 1
        # 2) 取最高频单字填满 symbol 额度（special + 256 byte 已占，其余给单字）
        letters = [w for w, c in counter.most_common() if c >= min_freq]
        for ch in letters:
            if ch not in self.word2idx:
                self.word2idx[ch] = len(self.word2idx)
            if len(self.word2idx) >= self.vocab_size:
                break
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def tokenize(self, text: str) -> List[str]:
        # 字符级：每个合法字符单独成 token（未知字符走 byte fallback 由 encode 处理）
        out = []
        for w in self._bpe_pre_tokenize(text):
            out.extend(list(w))
        return out

    def save(self, path: str) -> None:
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': {str(k): v for k, v in self.idx2word.items()},
                'merges': [],
                'special_tokens': self.special_tokens,
                'bpe': True,
                'char': True,
            }, f, ensure_ascii=False, indent=2)