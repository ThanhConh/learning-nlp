import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        """
        Initialize RNN Classifier.
        
        Args:
            vocab_size (int): Size of vocabulary
            embed_dim (int): Embedding dimension
            hidden_dim (int): RNN hidden state dimension
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            nonlinearity='tanh'
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Forward pass of the RNN classifier.
        
        Args:
            x (torch.Tensor): Input tensor, shape [batch_size, seq_len]
        
        Returns:
            output (torch.Tensor): Logits, shape [batch_size, 1]
        """
        x = self.embedding(x)           # [batch, seq_len, embed_dim]
        output, hidden = self.rnn(x)    # output: [batch, seq_len, hidden_dim]
                                        # hidden: [1, batch, hidden_dim]
        h_last = hidden[-1]              # [batch, hidden_dim]
        logits = self.fc(h_last)         # [batch, 1]
        
        return logits


# ============================================================================
# TEST 1: Basic Model Test
# ============================================================================

def test_basic_model():
    """Test basic model forward pass."""
    print("\n" + "="*70)
    print("TEST 1: Basic Model Forward Pass")
    print("="*70)
    
    # Create model
    vocab_size = 1000
    embed_dim = 100
    hidden_dim = 50
    
    model = RNNClassifier(vocab_size, embed_dim, hidden_dim)
    
    # Show model structure
    print("\n📋 Model Structure:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total Parameters: {total_params:,}")
    
    # Create dummy input
    batch_size = 4
    seq_len = 20
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\n📥 Input:")
    print(f"   Shape: {input_ids.shape}")
    print(f"   Values (first batch): {input_ids[0][:10]}...")
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"\n📤 Output:")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Logits values:\n{logits}")
    
    # Probabilities
    probs = torch.sigmoid(logits)
    print(f"\n📊 Probabilities (after sigmoid):")
    print(f"{probs}")
    
    return model


# ============================================================================
# TEST 2: Different Input Sizes
# ============================================================================

def test_different_sizes():
    """Test model with different batch sizes and sequence lengths."""
    print("\n" + "="*70)
    print("TEST 2: Different Input Sizes")
    print("="*70)
    
    model = RNNClassifier(vocab_size=500, embed_dim=50, hidden_dim=30)
    
    test_cases = [
        (2, 10),
        (8, 20),
        (32, 50),
    ]
    
    print("\n🔄 Testing different configurations:")
    for batch_size, seq_len in test_cases:
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        
        with torch.no_grad():
            logits = model(input_ids)
        
        print(f"\n   Batch: {batch_size}, Seq Len: {seq_len}")
        print(f"   Input shape:  {input_ids.shape}")
        print(f"   Output shape: {logits.shape}")
        print(f"   Logits range: [{logits.min():.4f}, {logits.max():.4f}]")


# ============================================================================
# TEST 3: Training with Loss and Gradients
# ============================================================================

def test_training():
    """Test forward pass, loss computation, and backward pass."""
    print("\n" + "="*70)
    print("TEST 3: Training with Loss and Gradients")
    print("="*70)
    
    # Create model
    model = RNNClassifier(vocab_size=100, embed_dim=32, hidden_dim=16)
    
    # Create data
    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    targets = torch.randint(0, 2, (batch_size, 1)).float()
    
    print(f"\n📥 Input:")
    print(f"   Shape: {input_ids.shape}")
    print(f"   Targets: {targets.squeeze().tolist()}")
    
    # Forward pass
    logits = model(input_ids)
    print(f"\n📤 Forward Pass:")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Logits: {logits.squeeze().tolist()}")
    
    # Compute loss
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, targets)
    print(f"\n🎯 Loss:")
    print(f"   {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print(f"\n⬅️  Gradients computed!")
    
    # Show gradient info
    print(f"\n📊 Gradient Information:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.mean().item()
            grad_norm = param.grad.norm().item()
            print(f"   {name}:")
            print(f"      - Shape: {param.shape}")
            print(f"      - Grad mean: {grad_mean:.6f}")
            print(f"      - Grad norm: {grad_norm:.6f}")


# ============================================================================
# TEST 4: Model Evaluation Mode
# ============================================================================

def test_eval_mode():
    """Test model in evaluation mode."""
    print("\n" + "="*70)
    print("TEST 4: Evaluation Mode")
    print("="*70)
    
    model = RNNClassifier(vocab_size=200, embed_dim=64, hidden_dim=32)
    
    # Create batch
    batch_size = 8
    seq_len = 15
    input_ids = torch.randint(0, 200, (batch_size, seq_len))
    
    print(f"\n📥 Input batch shape: {input_ids.shape}")
    
    # Inference mode (no gradients)
    print(f"\n🔍 Evaluation mode:")
    model.eval()
    
    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.sigmoid(logits)
    
    print(f"   Logits shape: {logits.shape}")
    print(f"   Probs shape: {probs.shape}")
    print(f"\n   First 5 predictions:")
    for i in range(min(5, batch_size)):
        prob = probs[i].item()
        pred = "Positive" if prob > 0.5 else "Negative"
        print(f"   Sample {i}: {prob:.4f} → {pred}")


# ============================================================================
# TEST 5: GPU Support
# ============================================================================

def test_gpu_support():
    """Test model with GPU if available."""
    print("\n" + "="*70)
    print("TEST 5: GPU Support")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    model = RNNClassifier(vocab_size=300, embed_dim=80, hidden_dim=40)
    model = model.to(device)
    
    # Create data on device
    batch_size = 4
    seq_len = 20
    input_ids = torch.randint(0, 300, (batch_size, seq_len)).to(device)
    
    print(f"\n📊 Model and data on device: {device}")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Input device: {input_ids.device}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_ids)
    
    print(f"\n📤 Output:")
    print(f"   Shape: {output.shape}")
    print(f"   Device: {output.device}")


# ============================================================================
# MAIN: Run all tests
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# PyTorch RNN Classifier - Simple Tests")
    print("#"*70)
    
    # Test 1: Basic model
    model = test_basic_model()
    
    # Test 2: Different sizes
    test_different_sizes()
    
    # Test 3: Training with loss
    test_training()
    
    # Test 4: Evaluation
    test_eval_mode()
    
    # Test 5: GPU
    test_gpu_support()
    
    print("\n" + "#"*70)
    print("# ✅ All tests completed!")
    print("#"*70)
