from Encoder import Encoder 
from Decoder import Decoder 
import torch
import torch.nn as nn

class Transformer(nn.Module): 
    def __init__(self,
                vocab_size,  
                 trg_pad_idx, 
                 src_pad_idx, 
                 device , 
                 embed_size=512, 
                 heads = 8, 
                 forward_exp=4, 
                 encoder_layers=6, 
                 decoder_layers=6, 
                 max_len = 100, 
                 dropout = 0,
                 ): 
        super(Transformer, self).__init__() 
        self.encoder = Encoder(
            voc_size=vocab_size, 
            heads=heads, 
            device=device, 
            embed_size=embed_size, 
            num_layers=encoder_layers, 
            dropout=dropout, 
            max_len=max_len,
            forward_exp=forward_exp
        )
        self.decoder = Decoder(
            vocab_size=vocab_size, 
            heads=heads, 
            device=device, 
            embed_size=embed_size, 
            layers=decoder_layers, 
            dropout=dropout, 
            max_len=max_len, 
            forward_exp=forward_exp
        )
        self.src_pad_idx = src_pad_idx 
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def get_src_mask(self, src: torch.Tensor): 
        # Create a mask for the source sequence to avoid attending to padding tokens
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def get_trg_mask(self, trg: torch.Tensor): 
        # Create a mask for the target sequence to avoid attending to future tokens
        n, trg_len  = trg.shape 
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(n, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg): 
        src_mask = self.get_src_mask(src)
        trg_mask = self.get_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        
        return out