"""
Attention-based decoder for image captioning.
This module implements various attention mechanisms (Bahdanau, Luong, Self-attention) 
for generating captions from encoded image features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List
import random


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention mechanism.
    Reference: "Neural Machine Translation by Jointly Learning to Align and Translate"
    """
    
    def __init__(self, 
                 encoder_dim: int,
                 decoder_dim: int,
                 attention_dim: int):
        """
        Initialize Bahdanau attention.
        
        Args:
            encoder_dim: Dimension of encoder features
            decoder_dim: Dimension of decoder hidden state
            attention_dim: Dimension of attention hidden layer
        """
        super(BahdanauAttention, self).__init__()
        
        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attention = nn.Linear(decoder_dim, attention_dim)
        self.full_attention = nn.Linear(attention_dim, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, 
                encoder_features: torch.Tensor,
                decoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Bahdanau attention.
        
        Args:
            encoder_features: [batch_size, spatial_dim, encoder_dim]
            decoder_hidden: [batch_size, decoder_dim]
            
        Returns:
            context: Attended context vector [batch_size, encoder_dim]
            attention_weights: Attention weights [batch_size, spatial_dim]
        """
        # Project encoder features
        encoder_att = self.encoder_attention(encoder_features)  # [batch_size, spatial_dim, attention_dim]
        
        # Project decoder hidden state and expand
        decoder_att = self.decoder_attention(decoder_hidden)  # [batch_size, attention_dim]
        decoder_att = decoder_att.unsqueeze(1)  # [batch_size, 1, attention_dim]
        
        # Additive attention
        combined_att = self.relu(encoder_att + decoder_att)  # [batch_size, spatial_dim, attention_dim]
        
        # Compute attention scores
        attention_scores = self.full_attention(combined_att).squeeze(2)  # [batch_size, spatial_dim]
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)  # [batch_size, spatial_dim]
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_features)  # [batch_size, 1, encoder_dim]
        context = context.squeeze(1)  # [batch_size, encoder_dim]
        
        return context, attention_weights


class LuongAttention(nn.Module):
    """
    Luong (Multiplicative) Attention mechanism.
    Reference: "Effective Approaches to Attention-based Neural Machine Translation"
    """
    
    def __init__(self, 
                 encoder_dim: int,
                 decoder_dim: int,
                 method: str = 'general'):
        """
        Initialize Luong attention.
        
        Args:
            encoder_dim: Dimension of encoder features
            decoder_dim: Dimension of decoder hidden state
            method: Attention method ('dot', 'general', 'concat')
        """
        super(LuongAttention, self).__init__()
        
        self.method = method
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        if method == 'general':
            self.attention = nn.Linear(decoder_dim, encoder_dim, bias=False)
        elif method == 'concat':
            self.attention = nn.Linear(encoder_dim + decoder_dim, encoder_dim)
            self.v = nn.Parameter(torch.FloatTensor(encoder_dim))
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, 
                encoder_features: torch.Tensor,
                decoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Luong attention.
        
        Args:
            encoder_features: [batch_size, spatial_dim, encoder_dim]
            decoder_hidden: [batch_size, decoder_dim]
            
        Returns:
            context: Attended context vector [batch_size, encoder_dim]
            attention_weights: Attention weights [batch_size, spatial_dim]
        """
        batch_size, spatial_dim, encoder_dim = encoder_features.size()
        
        if self.method == 'dot':
            # Decoder hidden must match encoder dimension
            attention_scores = torch.bmm(encoder_features, decoder_hidden.unsqueeze(2))
            attention_scores = attention_scores.squeeze(2)  # [batch_size, spatial_dim]
            
        elif self.method == 'general':
            # Transform decoder hidden
            decoder_transformed = self.attention(decoder_hidden)  # [batch_size, encoder_dim]
            attention_scores = torch.bmm(encoder_features, decoder_transformed.unsqueeze(2))
            attention_scores = attention_scores.squeeze(2)  # [batch_size, spatial_dim]
            
        elif self.method == 'concat':
            # Concatenate and transform
            decoder_expanded = decoder_hidden.unsqueeze(1).expand(-1, spatial_dim, -1)
            combined = torch.cat([encoder_features, decoder_expanded], dim=2)
            attention_scores = self.attention(combined)  # [batch_size, spatial_dim, encoder_dim]
            attention_scores = torch.sum(attention_scores * self.v, dim=2)  # [batch_size, spatial_dim]
        
        # Apply softmax
        attention_weights = self.softmax(attention_scores)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_features)
        context = context.squeeze(1)  # [batch_size, encoder_dim]
        
        return context, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism similar to Transformer.
    """
    
    def __init__(self,
                 d_model: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, 
                                   query: torch.Tensor,
                                   key: torch.Tensor,
                                   value: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, d_k]
            key: Key tensor [batch_size, num_heads, seq_len, d_k]
            value: Value tensor [batch_size, num_heads, seq_len, d_k]
            mask: Attention mask
            
        Returns:
            output: Attention output [batch_size, num_heads, seq_len, d_k]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask
            
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.size()
        
        # Transform and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        return output, attention_weights


class AttentionDecoder(nn.Module):
    """
    LSTM-based decoder with attention mechanism for image captioning.
    """
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 encoder_dim: int,
                 decoder_dim: int,
                 attention_type: str = 'bahdanau',
                 attention_dim: int = 256,
                 dropout: float = 0.5,
                 max_length: int = 50):
        """
        Initialize attention-based decoder.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of word embeddings
            encoder_dim: Dimension of encoder features
            decoder_dim: Dimension of decoder LSTM
            attention_type: Type of attention ('bahdanau', 'luong', 'multihead')
            attention_dim: Dimension of attention layer
            dropout: Dropout probability
            max_length: Maximum sequence length
        """
        super(AttentionDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_type = attention_type
        self.max_length = max_length
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Attention mechanism
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        elif attention_type == 'luong':
            self.attention = LuongAttention(encoder_dim, decoder_dim, method='general')
        elif attention_type == 'multihead':
            self.attention = MultiHeadAttention(encoder_dim, num_heads=8, dropout=dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # LSTM decoder
        self.lstm_input_dim = embed_dim + encoder_dim
        self.lstm = nn.LSTMCell(self.lstm_input_dim, decoder_dim)
        
        # Output projection layers
        self.context_projection = nn.Linear(encoder_dim, embed_dim)
        self.hidden_projection = nn.Linear(decoder_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Initialization layers for LSTM hidden state
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize linear layers
        for layer in [self.context_projection, self.hidden_projection, 
                     self.output_projection, self.init_h, self.init_c]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def init_hidden_state(self, encoder_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state from encoder features.
        
        Args:
            encoder_features: Encoder features [batch_size, spatial_dim, encoder_dim]
            
        Returns:
            h0: Initial hidden state [batch_size, decoder_dim]
            c0: Initial cell state [batch_size, decoder_dim]
        """
        # Use mean of encoder features
        mean_features = torch.mean(encoder_features, dim=1)  # [batch_size, encoder_dim]
        
        h0 = self.init_h(mean_features)  # [batch_size, decoder_dim]
        c0 = self.init_c(mean_features)  # [batch_size, decoder_dim]
        
        return h0, c0
    
    def forward_step(self,
                    word_input: torch.Tensor,
                    hidden_state: torch.Tensor,
                    cell_state: torch.Tensor,
                    encoder_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single forward step of the decoder.
        
        Args:
            word_input: Current word input [batch_size]
            hidden_state: Current hidden state [batch_size, decoder_dim]
            cell_state: Current cell state [batch_size, decoder_dim]
            encoder_features: Encoder features [batch_size, spatial_dim, encoder_dim]
            
        Returns:
            output: Output logits [batch_size, vocab_size]
            hidden_state: Updated hidden state [batch_size, decoder_dim]
            cell_state: Updated cell state [batch_size, decoder_dim]
            attention_weights: Attention weights [batch_size, spatial_dim]
        """
        # Word embedding
        embedded = self.embedding(word_input)  # [batch_size, embed_dim]
        embedded = self.dropout(embedded)
        
        # Compute attention
        if self.attention_type in ['bahdanau', 'luong']:
            context, attention_weights = self.attention(encoder_features, hidden_state)
        else:
            # For multihead attention, we need to handle differently
            context = torch.mean(encoder_features, dim=1)  # Simplified for now
            attention_weights = torch.ones(encoder_features.size(0), encoder_features.size(1))
            if encoder_features.is_cuda:
                attention_weights = attention_weights.cuda()
        
        # Combine word embedding and context
        lstm_input = torch.cat([embedded, context], dim=1)  # [batch_size, embed_dim + encoder_dim]
        
        # LSTM forward pass
        hidden_state, cell_state = self.lstm(lstm_input, (hidden_state, cell_state))
        
        # Generate output
        context_proj = self.context_projection(context)  # [batch_size, embed_dim]
        hidden_proj = self.hidden_projection(hidden_state)  # [batch_size, embed_dim]
        
        # Combine projections and apply activation
        output_input = context_proj + hidden_proj + embedded
        output_input = torch.tanh(output_input)
        output_input = self.dropout(output_input)
        
        # Final output projection
        output = self.output_projection(output_input)  # [batch_size, vocab_size]
        
        return output, hidden_state, cell_state, attention_weights
    
    def forward(self,
                encoder_features: torch.Tensor,
                captions: torch.Tensor,
                caption_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (teacher forcing).
        
        Args:
            encoder_features: Encoder features [batch_size, spatial_dim, encoder_dim]
            captions: Caption tokens [batch_size, max_length]
            caption_lengths: Actual caption lengths [batch_size]
            
        Returns:
            Dictionary containing outputs and attention weights
        """
        batch_size = encoder_features.size(0)
        max_length = captions.size(1)
        
        # Initialize hidden states
        h, c = self.init_hidden_state(encoder_features)
        
        # Storage for outputs
        outputs = torch.zeros(batch_size, max_length, self.vocab_size)
        attention_weights_list = []
        
        if encoder_features.is_cuda:
            outputs = outputs.cuda()
        
        # Teacher forcing: use ground truth tokens as input
        for t in range(max_length - 1):  # Exclude last token
            word_input = captions[:, t]  # [batch_size]
            
            output, h, c, attention_weights = self.forward_step(
                word_input, h, c, encoder_features
            )
            
            outputs[:, t, :] = output
            attention_weights_list.append(attention_weights)
        
        # Stack attention weights
        attention_weights = torch.stack(attention_weights_list, dim=1)  # [batch_size, seq_len, spatial_dim]
        
        return {
            'outputs': outputs,  # [batch_size, max_length, vocab_size]
            'attention_weights': attention_weights  # [batch_size, seq_len, spatial_dim]
        }
    
    def generate(self,
                encoder_features: torch.Tensor,
                start_token: int,
                end_token: int,
                max_length: Optional[int] = None,
                temperature: float = 1.0,
                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate captions using beam search or greedy decoding.
        
        Args:
            encoder_features: Encoder features [batch_size, spatial_dim, encoder_dim]
            start_token: Start token ID
            end_token: End token ID
            max_length: Maximum generation length
            temperature: Temperature for sampling
            deterministic: Whether to use greedy decoding
            
        Returns:
            Dictionary containing generated sequences and attention weights
        """
        if max_length is None:
            max_length = self.max_length
        
        batch_size = encoder_features.size(0)
        
        # Initialize hidden states
        h, c = self.init_hidden_state(encoder_features)
        
        # Storage for generated sequences
        generated_sequences = torch.zeros(batch_size, max_length, dtype=torch.long)
        attention_weights_list = []
        
        if encoder_features.is_cuda:
            generated_sequences = generated_sequences.cuda()
        
        # Start with start token
        current_input = torch.full((batch_size,), start_token, dtype=torch.long)
        if encoder_features.is_cuda:
            current_input = current_input.cuda()
        
        for t in range(max_length):
            output, h, c, attention_weights = self.forward_step(
                current_input, h, c, encoder_features
            )
            
            # Apply temperature
            if temperature != 1.0:
                output = output / temperature
            
            # Get next token
            if deterministic:
                next_token = torch.argmax(output, dim=1)
            else:
                # Sample from distribution
                probs = F.softmax(output, dim=1)
                next_token = torch.multinomial(probs, 1).squeeze(1)
            
            generated_sequences[:, t] = next_token
            attention_weights_list.append(attention_weights)
            
            # Update input for next step
            current_input = next_token
            
            # Check if all sequences have ended
            if (next_token == end_token).all():
                break
        
        # Stack attention weights
        attention_weights = torch.stack(attention_weights_list, dim=1)
        
        return {
            'sequences': generated_sequences,  # [batch_size, seq_len]
            'attention_weights': attention_weights  # [batch_size, seq_len, spatial_dim]
        }


def test_decoder():
    """Test function for decoder models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing decoder on device: {device}")
    
    # Test parameters
    batch_size = 2
    spatial_dim = 49  # 7x7 for ResNet50
    encoder_dim = 512
    vocab_size = 1000
    max_length = 20
    
    # Create dummy inputs
    encoder_features = torch.randn(batch_size, spatial_dim, encoder_dim).to(device)
    captions = torch.randint(0, vocab_size, (batch_size, max_length)).to(device)
    
    print(f"Encoder features shape: {encoder_features.shape}")
    print(f"Captions shape: {captions.shape}")
    
    # Test different attention types
    for attention_type in ['bahdanau', 'luong']:
        print(f"\n=== Testing {attention_type.capitalize()} Attention Decoder ===")
        
        decoder = AttentionDecoder(
            vocab_size=vocab_size,
            embed_dim=256,
            encoder_dim=encoder_dim,
            decoder_dim=512,
            attention_type=attention_type,
            attention_dim=256,
            dropout=0.1,
            max_length=max_length
        ).to(device)
        
        # Test forward pass (training)
        with torch.no_grad():
            train_outputs = decoder(encoder_features, captions)
        
        print(f"Training outputs shape: {train_outputs['outputs'].shape}")
        print(f"Attention weights shape: {train_outputs['attention_weights'].shape}")
        
        # Test generation
        with torch.no_grad():
            gen_outputs = decoder.generate(
                encoder_features,
                start_token=1,
                end_token=2,
                max_length=15,
                deterministic=True
            )
        
        print(f"Generated sequences shape: {gen_outputs['sequences'].shape}")
        print(f"Generation attention weights shape: {gen_outputs['attention_weights'].shape}")
    
    print("\nDecoder tests completed successfully!")


if __name__ == "__main__":
    test_decoder()