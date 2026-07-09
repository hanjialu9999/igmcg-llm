import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint  # For gradient checkpointing

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, 
                 hidden_dim, max_seq_length, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_head = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        """
        Args:
            src: Input tensor (batch_size, seq_length)
            src_mask: Attention mask
        Returns:
            logits: (batch_size, seq_length, vocab_size)
        """
        # Embedding
        embedded = self.embedding(src) * math.sqrt(self.embedding_dim)
        
        # Positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Transformer encoder with gradient checkpointing (saves memory)
        # Use checkpointing to trade compute for memory during training
        if self.training:
            # Create a wrapper function for checkpointing
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, src_mask)
                return custom_forward
            
            # Apply gradient checkpointing
            try:
                encoded = checkpoint(
                    create_custom_forward(self.transformer_encoder),
                    embedded,
                    use_reentrant=False
                )
            except:
                # Fallback if checkpointing fails
                encoded = self.transformer_encoder(embedded, src_mask)
        else:
            # No checkpointing during evaluation
            encoded = self.transformer_encoder(embedded, src_mask)
        
        # Output layer
        logits = self.output_head(encoded)
        
        return logits
    
    def generate(self, token_ids, max_length=50, temperature=1.0, top_k=50, device='cpu', penalty_alpha=0.6, repetition_penalty=1.2):
        """
        Generate text given initial token ids.
        
        Args:
            token_ids: List of initial token ids
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling
            device: Device to use
            penalty_alpha: Contrastive search penalty (0-1)
            repetition_penalty: Penalty for repeated tokens (>1.0)
        """
        self.eval()
        
        generated = token_ids.copy()
        max_seq_length = self.max_seq_length
        eos_token_id = 3
        pad_token_id = 0
        sep_token_id = 4  # [SEP] special token
        
        with torch.no_grad():
            for i in range(max_length):
                # Ensure input doesn't exceed max sequence length
                if len(generated) > max_seq_length:
                    input_ids = torch.tensor([generated[-max_seq_length:]], dtype=torch.long).to(device)
                else:
                    input_ids = torch.tensor([generated], dtype=torch.long).to(device)
                
                # Forward pass
                logits = self.forward(input_ids)
                
                # Get last token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply repetition penalty for all previously generated tokens
                for prev_token in set(generated):
                    if prev_token >= 0 and prev_token < next_token_logits.shape[0]:
                        next_token_logits[prev_token] = next_token_logits[prev_token] / repetition_penalty
                
                # Suppress special tokens
                next_token_logits[pad_token_id] = float('-inf')   # pad
                next_token_logits[sep_token_id] = float('-inf')   # [SEP]
                
                # Suppress EOS token until we've generated enough tokens
                min_length = max(3, len(token_ids) + 2)  # At least 2-3 new tokens
                if len(generated) < min_length:
                    next_token_logits[eos_token_id] = float('-inf')
                else:
                    # Apply penalty to EOS to discourage early stopping
                    next_token_logits[eos_token_id] = next_token_logits[eos_token_id] - 5.0
                
                # Top-k filtering
                if top_k > 0 and top_k < next_token_logits.shape[0]:
                    try:
                        top_k_vals = torch.topk(next_token_logits, min(top_k, next_token_logits.shape[0]))[0]
                        threshold = top_k_vals[..., -1]
                        indices_to_remove = next_token_logits < threshold
                        next_token_logits[indices_to_remove] = float('-inf')
                    except:
                        pass
                
                # Check for all -inf case
                if torch.isinf(next_token_logits).all():
                    next_token_logits = logits[0, -1, :] / temperature
                    next_token_logits[pad_token_id] = float('-inf')
                
                # Softmax and sampling
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Check probability threshold
                if probs.max() < 0.01:
                    break
                    
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated.append(next_token)
                
                # Stop if we generate end token (only after min_length)
                if next_token == eos_token_id and len(generated) >= min_length:
                    break
        
        return generated

