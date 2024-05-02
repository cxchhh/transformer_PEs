import torch
from torch import nn

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000) -> None:
        super(SinusoidalPositionEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros([max_len, d_model], dtype=torch.float)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        freq = 1 / 10000 ** (torch.arange(0, d_model, 2) / d_model).unsqueeze(0)
        pe[:, 0::2] = torch.sin(torch.matmul(pos, freq))
        pe[:, 1::2] = torch.cos(torch.matmul(pos, freq))
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (B, L, D)
        # out: (B, L, D)
        x = x + self.pe[x.shape[1],:]
        return x

class RelativePositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(RelativePositionEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.embedding = nn.Embedding(2 * max_len - 1, d_model)

    def forward(self, q, k):
        # q: (B, H, Lq, d), k: (B, H, Lk, d)
        # out: (Lq, Lk, d)
        seq_len = max(q.shape[2], k.shape[2])
        positions = (torch.arange(seq_len).repeat(seq_len, 1) 
                     + torch.arange(seq_len).flip(dims=[0]).reshape(-1,1)).to(q.device)
        positions = torch.clamp(positions - seq_len + 1 + self.max_len, 0, 2 * self.max_len - 2)[:q.shape[2], :k.shape[2]]
        pos_enc = self.embedding(positions)
        return pos_enc