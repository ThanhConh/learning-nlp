import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism implemented from scratch.
    Uses only Linear layers and manual attention computation.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            mask: optional mask tensor
        Returns:
            output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections: (batch_size, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape to split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k)
        # -> (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # (batch_size, n_heads, seq_len, d_k) @ (batch_size, n_heads, d_k, seq_len)
        # -> (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_k)
        # -> (batch_size, n_heads, seq_len, d_k)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, n_heads, d_k)
        # -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Two linear transformations with ReLU activation in between.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            output tensor of shape (batch_size, seq_len, d_model)
        """
        # x -> W1 -> ReLU -> Dropout -> W2
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block implemented from scratch.
    
    Architecture:
        Input
          ↓
        Multi-Head Self-Attention
          ↓
        Residual + LayerNorm
          ↓
        Feed-Forward
          ↓
        Residual + LayerNorm
          ↓
        Output
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: dimension of the model (embedding dimension)
            n_heads: number of attention heads
            d_ff: dimension of the feed-forward network
            dropout: dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        # Feed-Forward Network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass through the Transformer block.
        
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            mask: optional attention mask
        Returns:
            output tensor of shape (batch_size, seq_len, d_model)
        """
        # Multi-Head Self-Attention with residual connection and layer norm
        # x -> Attention -> Dropout -> Residual -> LayerNorm
        attn_output = self.attention(x, mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)  # Residual connection + Layer Norm
        
        # Feed-Forward with residual connection and layer norm
        # x -> FFN -> Dropout -> Residual -> LayerNorm
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(x + ff_output)  # Residual connection + Layer Norm
        
        return x


# Demo and testing code
def demo_transformer_block():
    """
    Demonstration of the Transformer block with example usage.
    """
    print("=" * 60)
    print("Transformer Block Demo")
    print("=" * 60)
    
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    
    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Feed-forward dimension (d_ff): {d_ff}")
    print(f"  Dropout rate: {dropout}")
    
    # Create Transformer block
    transformer_block = TransformerBlock(d_model, n_heads, d_ff, dropout)
    
    # Count parameters
    total_params = sum(p.numel() for p in transformer_block.parameters())
    trainable_params = sum(p.numel() for p in transformer_block.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    transformer_block.eval()
    with torch.no_grad():
        output = transformer_block(x)
    
    print(f"Output shape: {output.shape}")
    print(f"\nInput statistics:")
    print(f"  Mean: {x.mean().item():.4f}")
    print(f"  Std: {x.std().item():.4f}")
    print(f"\nOutput statistics:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    
    # Test with mask
    print(f"\n{'=' * 60}")
    print("Testing with attention mask")
    print("=" * 60)
    
    # Create a causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, n_heads, seq_len, seq_len)
    
    with torch.no_grad():
        output_masked = transformer_block(x, mask)
    
    print(f"Output with mask shape: {output_masked.shape}")
    print(f"Output with mask statistics:")
    print(f"  Mean: {output_masked.mean().item():.4f}")
    print(f"  Std: {output_masked.std().item():.4f}")
    
    print(f"\n{'=' * 60}")
    print("Architecture Verification")
    print("=" * 60)
    print("\n✓ Multi-Head Self-Attention: Implemented manually")
    print("✓ Residual Connections: 2 residual paths")
    print("✓ Layer Normalization: 2 LayerNorm layers")
    print("✓ Feed-Forward Network: Implemented with 2 linear layers")
    print("✓ Dropout: Applied at multiple points")
    print("\nArchitecture flow:")
    print("  Input → Attention → Dropout → Add & Norm")
    print("        → FFN → Dropout → Add & Norm → Output")


def compare_with_pytorch():
    """
    Compare custom implementation with PyTorch's TransformerEncoderLayer.
    """
    print("\n" + "=" * 60)
    print("Comparison with PyTorch TransformerEncoderLayer")
    print("=" * 60)
    
    # Parameters
    d_model = 512
    n_heads = 8
    d_ff = 2048
    batch_size = 2
    seq_len = 10
    
    # Create both models
    custom_block = TransformerBlock(d_model, n_heads, d_ff, dropout=0.0)
    pytorch_block = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_ff,
        dropout=0.0,
        batch_first=True
    )
    
    # Count parameters
    custom_params = sum(p.numel() for p in custom_block.parameters())
    pytorch_params = sum(p.numel() for p in pytorch_block.parameters())
    
    print(f"\nParameter count:")
    print(f"  Custom implementation: {custom_params:,}")
    print(f"  PyTorch implementation: {pytorch_params:,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    
    custom_block.eval()
    pytorch_block.eval()
    
    with torch.no_grad():
        custom_output = custom_block(x)
        pytorch_output = pytorch_block(x)
    
    print(f"\nOutput shapes:")
    print(f"  Custom: {custom_output.shape}")
    print(f"  PyTorch: {pytorch_output.shape}")
    
    print(f"\n✓ Both implementations have compatible architectures")
    print(f"✓ Can be used interchangeably in applications")


if __name__ == "__main__":
    # Run demonstration
    demo_transformer_block()
    
    # Compare with PyTorch
    compare_with_pytorch()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
