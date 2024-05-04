import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from position_encodings import SinusoidalPositionEncoding, RelativePositionEncoding, RotaryPositionEncoding

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',device_map='cuda')

class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(RPEMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.batch_first = True

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        self.relative_position_embedding = RelativePositionEncoding(self.depth)
        self.out_relative_position_embedding = RelativePositionEncoding(self.depth)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size = query.shape[1]
        src_len = key.shape[0]
        tgt_len = query.shape[0]

        query = self.query_projection(query).transpose(0, 1).contiguous() # B, Lq, D
        key = self.key_projection(key).transpose(0, 1).contiguous() # B, Lk, D
        value = self.value_projection(value).transpose(0, 1).contiguous()

        query = query.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2) # B, H, Lq, d
        key = key.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2) # B, H, Lk, d
        value = value.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        
        sqrt_d = torch.sqrt(torch.tensor(self.depth, dtype=torch.float32).to(query.device))
        # (B, H, Lq, d) * (B, H, d, Lk) -> (B, H, Lq, Lk)
        scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt_d

        relative_position = self.relative_position_embedding(query, key) # (Lq, Lk, d)

        # (Lq, B*H, d) * (Lq, d, Lk) -> (Lq, B*H, Lk)
        m1 = query.permute(2, 0, 1, 3).reshape(tgt_len, batch_size * self.num_heads, self.depth)
        m2 = relative_position.transpose(-2, -1)
        relative_position_scores = torch.matmul(m1,m2) / sqrt_d # (Lq, B*H, Lk)
        relative_position_scores = relative_position_scores.reshape(tgt_len, batch_size, self.num_heads, -1).permute(1,2,0,3) # (B, H, Lq, Lk)
                                                
        scores = scores + relative_position_scores 

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size,1,1,src_len) \
                    .expand(-1, self.num_heads, -1, -1).reshape(batch_size * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask
        
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(batch_size, self.num_heads, -1, src_len)

            attn_mask = (attn_mask != 0).to(query.device)
            #import pdb; pdb.set_trace()
            scores = scores.masked_fill(attn_mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1) # (B, H, Lq, Lk)

        # (B, H, Lq, Lk) * (B, H, Lk, d) -> (B, H, Lq, d)
        attention_output = torch.matmul(attention_weights, value)

        out_relative_position = self.out_relative_position_embedding(query, value) # (Lq, Lk, d)
        out_relative_position_scores = (attention_weights.unsqueeze(-1) * out_relative_position).sum(dim=-2) # (B, H, Lq, d)
        attention_output = attention_output + out_relative_position_scores

        attention_output = attention_output.permute(2,0,1,3).contiguous().view(-1, batch_size, self.d_model) # (Lq, B, D)
        
        return attention_output
    
class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(RoPEMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.batch_first = True

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionEncoding(num_heads, d_model)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size = query.shape[1]
        src_len = query.shape[0]

        query = self.query_projection(query).transpose(0, 1).contiguous() # B, Lq, D
        key = self.key_projection(key).transpose(0, 1).contiguous() # B, Lk, D
        value = self.value_projection(value).transpose(0, 1).contiguous()

        query = query.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2) # B, H, Lq, d
        key = key.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2) # B, H, Lk, d
        value = value.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        
        query, key = self.rope(query, key)
        
        sqrt_d = torch.sqrt(torch.tensor(self.depth, dtype=torch.float32).to(query.device))
        # (B, H, Lq, d) * (B, H, d, Lk) -> (B, H, Lq, Lk)
        scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt_d

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size,1,1,src_len) \
                    .expand(-1, self.num_heads, -1, -1).reshape(batch_size * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask
        
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(batch_size, self.num_heads, -1, src_len)

            attn_mask = (attn_mask != 0).to(query.device)
            #import pdb; pdb.set_trace()
            scores = scores.masked_fill(attn_mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1) # (B, H, Lq, Lk)

        # (B, H, Lq, Lk) * (B, H, Lk, d) -> (B, H, Lq, d)
        attention_output = torch.matmul(attention_weights, value)

        attention_output = attention_output.permute(2,0,1,3).contiguous().view(-1, batch_size, self.d_model) # (Lq, B, D)
        
        return attention_output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        x = self.linear2(F.relu(self.linear1(x)))
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self,pe_type, d_model, num_heads, dim_feedforward, dropout):
        super(EncoderLayer, self).__init__()
        if pe_type == 'rpe':
            self.self_attn = RPEMultiHeadAttention(d_model, num_heads)
        elif pe_type == 'rope':
            self.self_attn = RoPEMultiHeadAttention(d_model, num_heads)
        else:
            raise NotImplementedError
        
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = self.layer_norm1(x + self.dropout1(self.self_attn(x, x, x, src_mask, src_key_padding_mask)))

        x = self.layer_norm2(x + self.dropout2(self.feed_forward(x)))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, pe_type, d_model, num_heads, dim_feedforward, dropout):
        super(DecoderLayer, self).__init__()
        if pe_type == 'rpe':
            self.self_attn = RPEMultiHeadAttention(d_model, num_heads)
            self.multihead_attn = RPEMultiHeadAttention(d_model, num_heads)
        elif pe_type == 'rope':
            self.self_attn = RoPEMultiHeadAttention(d_model, num_heads)
            self.multihead_attn = RoPEMultiHeadAttention(d_model, num_heads)
        else:
            raise NotImplementedError
        
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, memory_mask=None, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False):
        x = self.layer_norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask, tgt_key_padding_mask)))
        
        x = self.layer_norm2(x + self.dropout2(self.multihead_attn(x, memory, memory, memory_mask, memory_key_padding_mask)))
 
        x = self.layer_norm3(x + self.dropout3(self.feed_forward(x)))
        
        return x