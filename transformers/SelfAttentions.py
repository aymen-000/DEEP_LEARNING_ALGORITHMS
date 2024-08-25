import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)


    def forward(self, k, v, q, mask):
        N = q.shape[0]
        k_len , v_len , q_len = k.shape[1] , v.shape[1] , q.shape[1]

        # Split embedding into multiple heads
        values = self.values(v).view(N, v_len , self.heads, self.head_dim)
        keys = self.keys(k).view(N, k_len, self.heads, self.head_dim)
        queries = self.queries(q).view(N, q_len, self.heads, self.head_dim)


        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, q_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out