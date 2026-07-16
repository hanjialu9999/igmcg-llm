from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Vocabulary:
    """Vocabulary class with improved tokenization and filtering"""
    
    def __init__(self, vocab_size: int = 5000, min_freq: int = 1, special_tokens: Optional[List[str]] = None):
        """
        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum word frequency to include
            special_tokens: Custom special tokens
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq = Counter()
        
        # Define special tokens (including [SEP] for question-answer separation)
        if special_tokens is None:
            special_tokens = ['<pad>', '<???>', '<bos>', '<eos>', '[SEP]']
        
        self.special_tokens = special_tokens
        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3
        self.sep_idx = 4
        
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts with better filtering"""
        # Count word frequencies with improved tokenization
        for text in texts:
            words = self.tokenize(text)
            self.word_freq.update(words)
        
        # Create word2idx mapping with special tokens
        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
        
        # Add most frequent words with minimum frequency threshold
        # Only add words that appear at least min_freq times
        added = 0
        for word, freq in self.word_freq.most_common():
            if freq < self.min_freq:
                break  # Stop if frequency too low
            if added >= self.vocab_size - len(self.special_tokens):
                break  # Stop if vocab size reached
            
            self.word2idx[word] = len(self.word2idx)
            added += 1
        
        # Create idx2word mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Print statistics
        coverage = len(self.word2idx) / len(self.word_freq) * 100 if self.word_freq else 0
        print(f"\nVocabulary Statistics:")
        print(f"  Special tokens: {len(self.special_tokens)}")
        print(f"  Regular words: {len(self.word2idx) - len(self.special_tokens)}")
        print(f"  Total vocab size: {len(self.word2idx)}")
        print(f"  Unique words in corpus: {len(self.word_freq)}")
        print(f"  Coverage: {coverage:.2f}%")
    
    def tokenize(self, text: str) -> List[str]:
        """支持中英文混合的分词：
        - 中日韩(CJK)字符逐字切分（中文无词边界，按字建模）
        - 其它文本按空格/标点切分
        """
        # Clean text: remove extra spaces
        text = ' '.join(text.split())
        text = text.lower()

        # 先把每个 CJK 字符用空格隔开，非 CJK 片段保持原样
        # 这样后续统一按空格切分时，CJK 会逐字成 token
        text = re.sub(r'([\u3400-\u9fff\uf900-\ufaff\uff00-\uffef])', r' \1 ', text)

        # 常见标点也单独成 token
        text = re.sub(r'([.!?,;:-])', r' \1 ', text)
        words = text.split()

        # Filter empty strings
        words = [w for w in words if w]
        return words
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token indices"""
        words = self.tokenize(text)
        tokens: List[int] = []
        
        if add_special_tokens:
            tokens.append(self.bos_idx)
        
        for word in words:
            token = self.word2idx.get(word, self.unk_idx)
            tokens.append(token)
        
        if add_special_tokens:
            tokens.append(self.eos_idx)
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Convert token indices to text"""
        words: List[str] = []
        for token_id in token_ids:
            if skip_special:
                # Skip special tokens: pad(0), bos(2), eos(3), and sep(4)
                if token_id in [0, 2, 3, 4]:
                    continue
            
            # idx2word 以 int 为键；直接按 int 查找（兼容 vocab.json 中字符串键的情况）
            word = self.idx2word.get(token_id, self.idx2word.get(str(token_id), '<?>'))
            words.append(word)
        
        text = ' '.join(words)
        
        # Remove literal [SEP] or [sep] text markers (from training data)
        text = text.replace('[SEP]', '').replace('[sep]', '')
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def get_vocab_stats(self) -> Dict[str, Union[int, float]]:
        """Get vocabulary statistics"""
        return {
            'vocab_size': len(self.word2idx),
            'special_tokens': len(self.special_tokens),
            'regular_words': len(self.word2idx) - len(self.special_tokens),
            'unique_words_in_corpus': len(self.word_freq),
            'coverage': (len(self.word2idx) / len(self.word_freq) * 100) if self.word_freq else 0
        }
    
    def __len__(self) -> int:
        return len(self.word2idx)


class TextDataset(Dataset):
    """Improved dataset with lazy loading and better preprocessing"""
    
    def __init__(self, texts: List[str], vocab: Vocabulary, max_seq_length: int = 32, preprocess: bool = True):
        """
        Args:
            texts: List of text strings
            vocab: Vocabulary object
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


def load_data(data_file: str, vocab_size: int = 5000, max_seq_length: int = 32, min_freq: int = 1) -> Tuple[TextDataset, Vocabulary]:
    """Load data from file with validation"""
    print(f"Loading data from {data_file}...")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(texts)} lines")
    
    # Build vocabulary
    vocab = Vocabulary(vocab_size=vocab_size, min_freq=min_freq)
    vocab.build_vocab(texts)
    
    # Create dataset
    dataset = TextDataset(texts, vocab, max_seq_length, preprocess=True)
    del texts  # 释放原始字符串列表，避免 3.6M 条字符串常驻内存
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Max sequence length: {max_seq_length}")
    
    vocab_stats = vocab.get_vocab_stats()
    print(f"\nVocabulary Statistics:")
    for key, value in vocab_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
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


class BPETokenizer:
    """自研轻量 BPE（Byte-Pair Encoding）分词器，接口对齐 Vocabulary。

    与 Vocabulary 的差异：
    - 词表由训练语料的字节对合并生成（子词），而非纯频率截断；
    - 中文以单字为最小单元起步，高频相邻字/词逐步合并为子词，OOV 趋近 0；
    - encode/decode 走 BPE 合并规则，导出格式兼容 load_vocab（含 merges 键）。
    """

    def __init__(self, vocab_size: int = 8000, special_tokens: Optional[List[str]] = None):
        if special_tokens is None:
            special_tokens = ['<pad>', '<???>', '<bos>', '<eos>', '[SEP]']
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3
        self.sep_idx = 4
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        for idx, tok in enumerate(self.special_tokens):
            self.word2idx[tok] = idx
            self.idx2word[idx] = tok

    # ---------- 训练 ----------
    def train(self, texts: List[str], min_freq: int = 2) -> None:
        """对训练语料跑 BPE，生成子词词表。

        min_freq: 单字/子词进入词表的最低出现次数，过滤一次性噪声。
        """
        # 1) 预分词：CJK 逐字，非 CJK 按空格/标点切（与 Vocabulary 一致）
        counter: Counter = Counter()
        for text in texts:
            for w in self._pre_tokenize(text):
                counter[w] += 1

        # 2) 初始词表 = 高频单字（含 CJK 与非 CJK 字符），过滤低频噪声
        #    只取前 80% 额度，预留 20% 给后续 BPE 合并出的子词，否则单字直接填满
        #    vocab_size 会导致合并循环无空间执行（退化为纯单字截断词表）。
        letters = [w for w, c in counter.items() if c >= min_freq]
        letter_cap = int(self.vocab_size * 0.8)
        for ch in letters:
            if ch not in self.word2idx:
                self.word2idx[ch] = len(self.word2idx)
            if len(self.word2idx) >= letter_cap:
                break
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        # 3) 字节对合并：统计相邻符号对频次，反复合并最高频对
        # 语料以单字序列表示，逐步合并
        sym_freq: Counter = Counter()
        pair_freq: Counter = Counter()
        # 用 token 字符串列表表示每个词，初始为单字
        word_syms = {}  # word -> sym list（仅对词表内单字）
        for word, cnt in counter.items():
            if word not in self.word2idx:
                continue
            syms = list(word)
            word_syms[word] = syms
            for _ in range(cnt):
                for i in range(len(syms) - 1):
                    pair_freq[(syms[i], syms[i + 1])] += 1
                for s in syms:
                    sym_freq[s] += 1

        while len(self.word2idx) < self.vocab_size and pair_freq:
            # 取最高频对（频次相同时按字典序稳定）
            best = max(pair_freq.items(), key=lambda kv: (kv[1], kv[0]))
            pair, freq = best
            if freq < min_freq:
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

    def _pre_tokenize(self, text: str) -> List[str]:
        """与 Vocabulary.tokenize 一致的预分词：CJK 逐字，其余按空格/标点。"""
        text = ' '.join(text.split()).lower()
        text = re.sub(r'([\u3400-\u9fff\uf900-\ufaff\uff00-\uffef])', r' \1 ', text)
        text = re.sub(r'([.!?,;:-])', r' \1 ', text)
        return [w for w in text.split() if w]

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
        for w in self._pre_tokenize(text):
            out.extend(self._merge_word(w))
        return out

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.bos_idx)
        for sym in self.tokenize(text):
            tokens.append(self.word2idx.get(sym, self.unk_idx))
        if add_special_tokens:
            tokens.append(self.eos_idx)
        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        words: List[str] = []
        for tid in token_ids:
            if skip_special and tid in (0, 2, 3, 4):
                continue
            words.append(self.idx2word.get(tid, self.idx2word.get(str(tid), '<?>')))
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