from SelfAttentions import SelfAttention
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_exp):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size=embed_size, heads=heads)
        self.heads = heads
        self.embed_size = embed_size
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_exp * embed_size),
            nn.ReLU(),
            nn.Linear(forward_exp * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, v, k, q, mask):
        attention = self.attention(k, v, q, mask)
        add_norm1 = self.norm1(attention + q)
        drop = self.dropout(add_norm1)
        forward = self.feed_forward(drop)
        out = self.norm2(forward + drop)
        
        return out