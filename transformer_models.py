import torch
from torch import nn
from position_encodings import SinusoidalPositionEncoding
from modules import tokenizer, RPEDecoderLayer, RPEEncoderLayer

NUM_EMB = tokenizer.vocab_size

class VanillaTransformer(nn.Module):
    def __init__(self, d_model, num_layers=6, nhead=8, dim_feedforward=2048, dropout=0.1) -> None:
        super(VanillaTransformer, self).__init__()
        self.src_embedding = nn.Embedding(NUM_EMB, d_model)
        self.tgt_embedding = nn.Embedding(NUM_EMB, d_model)
        self.transformer = nn.Transformer(d_model=d_model,
                                        nhead=nhead,  
                                        num_encoder_layers=num_layers,
                                        num_decoder_layers=num_layers,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout,
                                        batch_first=True
                                        )
        self.pe = SinusoidalPositionEncoding(d_model)
            
        self.fc = nn.Linear(d_model, NUM_EMB)
    
    def forward(self, src_tokens, tgt_tokens):
        src_embedded = self.src_embedding(src_tokens) # B, L, D
        tgt_embedded = self.tgt_embedding(tgt_tokens)
        src_embedded = self.pe(src_embedded) # B, L, D
        tgt_embedded = self.pe(tgt_embedded)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.shape[1])
        output = self.transformer(src_embedded, tgt_embedded, tgt_mask=tgt_mask)
        output = self.fc(output)
        return output

class RPETransformer(nn.Module):
    def __init__(self, d_model, num_layers=6, nhead=8, dim_feedforward=2048, dropout=0.1) -> None:
        super(RPETransformer, self).__init__()
        self.src_embedding = nn.Embedding(NUM_EMB, d_model)
        self.tgt_embedding = nn.Embedding(NUM_EMB, d_model)

        encoder_layer = RPEEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        decoder_layer = RPEDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        self.transformer = nn.Transformer(d_model=d_model,
                                        nhead=nhead,  
                                        num_encoder_layers=num_layers,
                                        num_decoder_layers=num_layers,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout,
                                        batch_first=True,
                                        custom_encoder=self.encoder,
                                        custom_decoder=self.decoder
                                        )
            
        self.fc = nn.Linear(d_model, NUM_EMB)
    
    def forward(self, src_tokens, tgt_tokens):
        src_embedded = self.src_embedding(src_tokens) # B, L, D
        tgt_embedded = self.tgt_embedding(tgt_tokens)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.shape[1])
        output = self.transformer(src_embedded, tgt_embedded, tgt_mask=tgt_mask)
        output = self.fc(output)
        return output