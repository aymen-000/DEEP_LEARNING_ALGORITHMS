import torch.nn as nn
import torch
from TransformerBlock import TransformerBlock

class Encoder(nn.Module):
    def __init__(self,
                 voc_size,
                 heads,
                 device,
                 embed_size, 
                 num_layers, 
                 dropout, 
                 max_len, 
                 forward_exp):
        super(Encoder, self).__init__()
        self.embed_size = embed_size 
        self.device = device 
        self.word_embedding = nn.Embedding(voc_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size=embed_size, 
                    heads=heads, 
                    dropout=dropout,
                    forward_exp=forward_exp
                )
                for _ in range(num_layers)  # Create multiple Transformer blocks
            ]
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor): 
        batch, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(batch, seq_len).to(self.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions)) 
        
        for layer in self.layers: 
            out = layer(out, out, out, mask)
            
        return out