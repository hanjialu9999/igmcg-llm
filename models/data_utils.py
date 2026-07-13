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
        
        self.texts = texts
        self.encoded_cache: Dict[int, Dict[str, torch.Tensor]] = {}  # Cache for encoded sequences
        
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
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回预填充并转为 long tensor 的样本；结果按 idx 缓存，
        避免每个 epoch 重复做 tokenize / 截padding / 张量分配（训练热路径上的主要主机开销）。"""
        if idx in self.encoded_cache:
            return self.encoded_cache[idx]

        tokens = self.vocab.encode(self.texts[idx])
        L = self.max_seq_length + 1
        if len(tokens) > L:
            tokens = tokens[:L]
        elif len(tokens) < L:
            tokens = tokens + [self.vocab.pad_idx] * (L - len(tokens))

        item = {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'target_ids': torch.tensor(tokens[1:], dtype=torch.long),
            'length': torch.tensor(L - 1, dtype=torch.long),
        }
        self.encoded_cache[idx] = item
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