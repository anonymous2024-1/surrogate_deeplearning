import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_inverted


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, dropout=configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projection_1 = nn.Linear(configs.seq_len + configs.pred_len, configs.d_model)  # (B,L,d).T->(B,d,D)

        # Decoder
        self.dec_embedding = DataEmbedding_inverted(configs.seq_len + configs.pred_len, configs.d_model, dropout=configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.pred_len, bias=True)  # (B,N,pred_len)
        )
        self.projection_2 = nn.Linear(configs.dec_in, configs.c_out)  # (B,pred_len,1)


    def forecast(self, x_enc, x_dec):
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        dec_out = self.dec_embedding(x_dec)

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = self.projection_1(enc_out.permute(0, 2, 1))

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        dec_out = self.projection_2(dec_out.permute(0, 2, 1))
        return dec_out

    def forward(self, x_enc, x_dec):
        dec_out = self.forecast(x_enc, x_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

