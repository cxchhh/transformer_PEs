import torch
from torch import nn

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000) -> None:
        super(SinusoidalPositionEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros([max_len, d_model], dtype=torch.float)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        freq = 10000 ** (-torch.arange(0, d_model, 2) / d_model).unsqueeze(0)
        pe[:, 0::2] = torch.sin(torch.matmul(pos, freq))
        pe[:, 1::2] = torch.cos(torch.matmul(pos, freq))
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (B, L, D)
        # out: (B, L, D)
        x = x + self.pe[:x.shape[-2]]
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

class RotaryPositionEncoding(nn.Module):
    def __init__(self, nheads, d_model, max_len=512) -> None:
        super(RotaryPositionEncoding, self).__init__()
        assert d_model % nheads == 0
        self.d_model = d_model
        self.nheads = nheads
        self.depth = d_model // nheads
        assert self.depth % 2 == 0
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        freq = 10000 ** (-torch.arange(0, self.depth, 2) / self.depth).unsqueeze(0)
        sin_pos = torch.sin(torch.matmul(pos, freq)).repeat_interleave(2, dim=-1)
        cos_pos = torch.cos(torch.matmul(pos, freq)).repeat_interleave(2, dim=-1)

        self.register_buffer('sin_pos', sin_pos)
        self.register_buffer('cos_pos', cos_pos)
    
    def forward(self, q, k):
        # q: (B, H, Lq, d), k: (B, H, Lk, d)
        # sin_pos, cos_pos: (max_len, d)
        batch_size = q.shape[0]
        seq_len_q = q.shape[-2]
        seq_len_k = k.shape[-2]

        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape)
        q = q * self.cos_pos[:seq_len_q] + q2 * self.sin_pos[:seq_len_q]

        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape)
        k = k * self.cos_pos[:seq_len_k] + k2 * self.sin_pos[:seq_len_k]

        return q, k
