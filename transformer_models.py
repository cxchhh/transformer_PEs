import torch
from torch import nn
from position_encodings import SinusoidalPositionEncoding
from modules import DecoderLayer, EncoderLayer, tokenizer

NUM_EMB = tokenizer.vocab_size

class MultiPETransformer(nn.Module):
    def __init__(self, pe_type, d_model, num_layers=6, nhead=8, dim_feedforward=2048, dropout=0.1) -> None:
        super(MultiPETransformer, self).__init__()
        self.src_embedding = nn.Embedding(NUM_EMB, d_model)
        self.tgt_embedding = nn.Embedding(NUM_EMB, d_model)
        self.pe_type = pe_type
        self.need_sinpe = False
        print(f"Transformer using {self.pe_type}")
        if pe_type == 'sinpe' or pe_type == 'nonepe':
            self.transformer = nn.Transformer(d_model=d_model,
                                            nhead=nhead,  
                                            num_encoder_layers=num_layers,
                                            num_decoder_layers=num_layers,
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout
                                            )
            if pe_type == 'sinpe':
                self.need_sinpe = True
                self.pe = SinusoidalPositionEncoding(d_model)
        elif pe_type == 'rpe' or pe_type == 'rope':
            encoder_layer = EncoderLayer(pe_type, d_model, nhead, dim_feedforward, dropout)
            encoder_norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

            decoder_layer = DecoderLayer(pe_type, d_model, nhead, dim_feedforward, dropout)
            decoder_norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)

            self.transformer = nn.Transformer(d_model=d_model,
                                            nhead=nhead,  
                                            num_encoder_layers=num_layers,
                                            num_decoder_layers=num_layers,
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout,
                                            custom_encoder=self.encoder,
                                            custom_decoder=self.decoder
                                            )
        else:
            raise NotImplementedError
            
        self.fc = nn.Linear(d_model, NUM_EMB)
    
    def forward(self, src_tokens, tgt_tokens):
        src_embedded = self.src_embedding(src_tokens) # B, L, D
        tgt_embedded = self.tgt_embedding(tgt_tokens)
        if self.need_sinpe:
            src_embedded = self.pe(src_embedded).permute(1,0,2) # L, B, D
            tgt_embedded = self.pe(tgt_embedded).permute(1,0,2)
        else:
            src_embedded = src_embedded.permute(1,0,2) # L, B, D
            tgt_embedded = tgt_embedded.permute(1,0,2)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.shape[1])
        output = self.transformer(src_embedded, tgt_embedded, tgt_mask=tgt_mask)
        output = self.fc(output)
        return output
