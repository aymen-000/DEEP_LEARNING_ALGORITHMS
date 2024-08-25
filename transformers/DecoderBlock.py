import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock
from SelfAttentions import SelfAttention

class DecoderBlock(nn.Module):
    def __init__(self, heads, embed_size, dropout, device, forward_exp):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformerBlock = TransformerBlock(
            embed_size=embed_size, 
            heads=heads, 
            dropout=dropout, 
            forward_exp=forward_exp
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, v, k, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        q = self.dropout(self.norm(attention + x))
        out = self.transformerBlock(v, k, q, src_mask)
        return out