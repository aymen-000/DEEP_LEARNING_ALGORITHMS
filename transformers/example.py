import torch
from SelfAttentions import SelfAttention
from main import Transformer
device = torch.device('cpu') 

x = torch.tensor([[1, 5, 6, 4, 3], [1, 8, 7, 3, 4]]).to(
        device
    )
y = torch.tensor([[1, 7, 4, 3, 5, 9], [1, 5, 6, 2, 4, 7]]).to(device)


src_pad_idx = 0 
trg_pad_idx = 0 
src_vocab_size = 10
trg_vocab_size = 10
model = Transformer(vocab_size=trg_vocab_size , trg_pad_idx=trg_pad_idx , src_pad_idx=src_pad_idx , device=device)
out = model(x , y[: , :-1])
print(out.shape)
