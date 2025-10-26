class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: [batch_size, num_heads, seq_len, d_k]
        
        # Apply scaled dot-product attention
        output, attn_weights = self.attention(Q, K, V, mask)
        # output: [batch_size, num_heads, seq_len, d_k]
        
        # Concatenate heads and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)  # [batch_size, seq_len, d_model]
        
        return output, attn_weights

# Example usage
d_model, num_heads = 512, 8
mha = MultiHeadAttention(d_model, num_heads)
Q = torch.rand(batch_size, seq_len, d_model)
K = torch.rand(batch_size, seq_len, d_model)
V = torch.rand(batch_size, seq_len, d_model)

output, attn_weights = mha(Q, K, V)
print("Multi-Head Output shape:", output.shape)  # [batch_size, seq_len, d_model]
print("Multi-Head Attention weights shape:", attn_weights.shape)  # [batch_size, num_heads, seq_len, seq_len]