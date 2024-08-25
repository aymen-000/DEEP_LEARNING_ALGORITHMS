import torch
import torch.nn as nn 
from DecoderBlock import DecoderBlock

class Decoder(nn.Module): 
    def __init__(self, 
                 heads,
                 layers,
                 embed_size,
                 forward_exp, 
                 dropout, 
                 device, 
                 vocab_size,
                 max_len):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positionEmbedding = nn.Embedding(max_len, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.decoderBlocks = nn.ModuleList(
            [
                DecoderBlock(embed_size=embed_size,
                             heads=heads, 
                             dropout=dropout, 
                             forward_exp=forward_exp, 
                             device=device) 
                for _ in range(layers)
            ]
        )
        self.fc = nn.Linear(embed_size, vocab_size)
        self.device = device
    
    def forward(self, x, enc_out, src_mask, targ_mask): 
        batch, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(batch, seq_len).to(self.device)
        
        out = self.dropout(self.embedding(x) + self.positionEmbedding(positions)) 
        
        for layer in self.decoderBlocks: 
            out = layer(
                x=out,
                v=enc_out,
                k=enc_out, 
                src_mask=src_mask, 
                target_mask=targ_mask
            )
        
        out = self.fc(out)
        
        return out
        