#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPE (Byte Pair Encoding) Tokenizer implementation
improved tokenization for better vocabulary coverage
"""

import re
from collections import Counter
import json
from pathlib import Path

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer
    More efficient than character-level or word-level tokenization
    """
    
    def __init__(self, vocab_size=10000, num_merges=None):
        self.vocab_size = vocab_size
        self.num_merges = num_merges or (vocab_size - 256)  # Reserve 256 for bytes
        self.word_tokenizer = None
        self.bpe = {}  # Stores merge operations
        self.word_freq = Counter()
        
    def _get_words(self, text):
        """Split text into words (with frequency info)"""
        # Clean and split text
        text = text.lower()
        # Split into words, keeping punctuation
        words = re.findall(r'\w+|[.,!?;:-]', text)
        return words
    
    def _get_stats(self, vocab):
        """Count frequency of adjacent pairs"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def _merge_vocab(self, pair, vocab):
        """Merge the most frequent pair"""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        
        return new_vocab
    
    def train(self, texts, vocab_size=None):
        """Train BPE tokenizer on texts"""
        if vocab_size:
            self.vocab_size = vocab_size
            self.num_merges = vocab_size - 256
        
        print(f"Training BPE tokenizer (vocab_size={self.vocab_size})...")
        
        # Get word frequencies
        words = []
        for text in texts:
            words.extend(self._get_words(text))
        
        self.word_freq = Counter(words)
        
        # Initialize vocab with characters
        vocab = {}
        for word, freq in self.word_freq.items():
            vocab[' '.join(list(word)) + ' </w>'] = freq
        
        print(f"Initial vocab from {len(self.word_freq)} unique words")
        
        # BPE iterations
        for i in range(self.num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                print(f"Stopped at iteration {i}: no more pairs to merge")
                break
            
            best_pair = pairs.most_common(1)[0][0]
            vocab = self._merge_vocab(best_pair, vocab)
            self.bpe[best_pair] = i
            
            if (i + 1) % 100 == 0:
                print(f"  Iteration {i + 1}/{self.num_merges}: merged {best_pair}")
        
        print(f"BPE training completed!")
        print(f"  Total merges: {len(self.bpe)}")
        print(f"  Final vocab size will be: ~{256 + len(self.bpe)}")
    
    def encode(self, word):
        """Encode a word using learned BPE"""
        # Start with character-level
        word = word.lower() + '</w>'
        tokens = list(word)
        
        # Apply learned merges in order
        while len(tokens) > 1:
            # Find the best merge to apply
            best_merge = None
            best_priority = float('inf')
            best_pos = -1
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.bpe:
                    priority = self.bpe[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_merge = pair
                        best_pos = i
            
            if best_merge is None:
                break  # No merges left
            
            # Apply merge
            tokens = tokens[:best_pos] + [''.join(best_merge)] + tokens[best_pos + 2:]
        
        return tokens
    
    def save(self, path):
        """Save BPE model"""
        bpe_dict = {str(k): v for k, v in self.bpe.items()}
        data = {
            'vocab_size': self.vocab_size,
            'num_merges': self.num_merges,
            'bpe': bpe_dict,
            'word_freq': dict(self.word_freq)
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"BPE model saved to {path}")
    
    def load(self, path):
        """Load BPE model"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.num_merges = data['num_merges']
        # Reconstruct bpe dict with tuple keys
        self.bpe = {}
        for k, v in data['bpe'].items():
            # Parse string representation of tuple
            k_tuple = eval(k)  # Safe here since we control the input
            self.bpe[k_tuple] = v
        self.word_freq = Counter(data['word_freq'])
        print(f"BPE model loaded from {path}")


class ImprovedVocabulary:
    """Enhanced vocabulary with BPE support"""
    
    def __init__(self, vocab_size=10000, use_bpe=True):
        self.vocab_size = vocab_size
        self.use_bpe = use_bpe
        self.word2idx = {}
        self.idx2word = {}
        self.bpe_tokenizer = BPETokenizer(vocab_size=vocab_size) if use_bpe else None
        
        # Special tokens
        self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '[SEP]']
        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3
        self.sep_idx = 4
    
    def build_vocab(self, texts):
        """Build vocabulary from texts with BPE"""
        print(f"\nBuilding vocabulary (size={self.vocab_size}, use_bpe={self.use_bpe})...")
        
        # Train BPE if enabled
        if self.use_bpe:
            self.bpe_tokenizer.train(texts, vocab_size=self.vocab_size)
        
        # Add special tokens
        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
        
        # Get BPE vocabulary
        if self.use_bpe:
            bpe_vocab = set()
            for token_list in [self.bpe_tokenizer.encode(t) for t in texts[:1000]]:
                bpe_vocab.update(token_list)
            
            # Add most common BPE tokens
            for idx, token in enumerate(sorted(bpe_vocab)[:self.vocab_size - len(self.special_tokens)]):
                self.word2idx[token] = len(self.word2idx)
        
        # Create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"✅ Vocabulary built!")
        print(f"  Special tokens: {len(self.special_tokens)}")
        print(f"  Total vocab size: {len(self.word2idx)}")
    
    def encode(self, text, add_special_tokens=True):
        """Encode text to token IDs"""
        text = text.lower()
        
        if self.use_bpe:
            # Use BPE tokenization
            words = re.findall(r'\w+|[.,!?;:-]', text)
            tokens = []
            for word in words:
                bpe_tokens = self.bpe_tokenizer.encode(word)
                for bpe_token in bpe_tokens:
                    if bpe_token in self.word2idx:
                        tokens.append(self.word2idx[bpe_token])
                    else:
                        tokens.append(self.unk_idx)
        else:
            # Fallback to character-level or word-level
            words = re.findall(r'\w+|[.,!?;:-]', text)
            tokens = [self.word2idx.get(w, self.unk_idx) for w in words]
        
        if add_special_tokens:
            tokens = [self.bos_idx] + tokens + [self.eos_idx]
        
        return tokens
    
    def decode(self, token_ids, skip_special=True):
        """Decode token IDs to text"""
        words = []
        for token_id in token_ids:
            if skip_special:
                if token_id in [self.pad_idx, self.bos_idx, self.eos_idx]:
                    continue
            
            word = self.idx2word.get(token_id, '<unk>')
            # Remove BPE markers
            word = word.replace('</w>', ' ').strip()
            words.append(word)
        
        return ' '.join(words)
    
    def save(self, path):
        """Save vocabulary and BPE model"""
        data = {
            'vocab_size': self.vocab_size,
            'use_bpe': self.use_bpe,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        if self.use_bpe:
            self.bpe_tokenizer.save(str(path.parent / 'bpe_model.json'))
        
        print(f"Vocabulary saved to {path}")
    
    def load(self, path):
        """Load vocabulary and BPE model"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.use_bpe = data['use_bpe']
        self.word2idx = data['word2idx']
        self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        
        if self.use_bpe:
            bpe_path = Path(path).parent / 'bpe_model.json'
            if bpe_path.exists():
                self.bpe_tokenizer.load(str(bpe_path))
        
        print(f"Vocabulary loaded from {path}")


if __name__ == '__main__':
    # Test BPE tokenizer
    sample_texts = [
        "what is artificial intelligence",
        "machine learning is a subset of artificial intelligence",
        "neural networks are computing systems inspired by biological neural networks"
    ]
    
    # Create and train
    vocab = ImprovedVocabulary(vocab_size=100, use_bpe=True)
    vocab.build_vocab(sample_texts)
    
    # Test encoding/decoding
    test_text = "artificial intelligence and machine learning"
    tokens = vocab.encode(test_text)
    decoded = vocab.decode(tokens)
    
    print(f"\nTest:")
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
