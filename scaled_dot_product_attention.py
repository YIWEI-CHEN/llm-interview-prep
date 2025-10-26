import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: [batch_size, seq_len, d_k]
        # Compute attention scores: QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch_size, seq_len, seq_len]
        
        # Apply mask (optional, e.g., for padding or causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # Compute weighted sum of values
        output = torch.matmul(attn_weights, V)  # [batch_size, seq_len, d_v]
        
        return output, attn_weights

# Example usage
batch_size, seq_len, d_k, d_v = 2, 4, 64, 64
Q = torch.rand(batch_size, seq_len, d_k)
K = torch.rand(batch_size, seq_len, d_k)
V = torch.rand(batch_size, seq_len, d_v)

attention = ScaledDotProductAttention(d_k)
output, attn_weights = attention(Q, K, V)

print("Output shape:", output.shape)  # [batch_size, seq_len, d_v]
print("Attention weights shape:", attn_weights.shape)  # [batch_size, seq_len, seq_len]