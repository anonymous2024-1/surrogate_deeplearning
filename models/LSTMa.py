import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding


with_attention = True

# Define the LSTM encoder-decoder model with attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)   # [B,L,D]
        attn_weights = torch.tanh(self.attn(torch.cat((H, encoder_outputs), 2)))
        attn_weights = attn_weights.transpose(2, 1)  # [B,D,L]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,D]
        attn_weights = torch.bmm(v, attn_weights)  #[B,1,L]
        attn_weights = F.softmax(attn_weights.squeeze(1), dim=1)
        attn_weights = attn_weights.unsqueeze(1)   #[B,1,L]

        return attn_weights


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.with_attn = with_attention

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model)

        self.encoder = nn.LSTM(configs.d_model, configs.d_model, configs.e_layers, batch_first=True)
        self.decoder = nn.LSTM(configs.d_model, configs.d_model, configs.e_layers, batch_first=True)
        self.attention = Attention(configs.d_model)
        self.fc = nn.Linear(configs.d_model * 2, configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forecast(self, x_enc, x_dec):
        # Embedding, configs.label_len==0
        enc_in = self.enc_embedding(x_enc)  #[B,L,D]
        dec_in = self.dec_embedding(x_dec)

        encoder_outputs, (hidden, cell) = self.encoder(enc_in)  # [B,L,D] [1,B,D] [1,B,D]
        outputs = []
        attns = []
        if self.with_attn:
            for t in range(self.label_len, self.pred_len):
                decoder_input = dec_in[:, t:t+1, :]
                decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))  # [B, 1, D]

                attn_weights = self.attention(hidden, encoder_outputs)
                context = attn_weights.bmm(encoder_outputs) # [B,1,D]
                decoder_output = torch.cat((decoder_output.squeeze(1), context.squeeze(1)), 1)
                decoder_output = self.fc(decoder_output).unsqueeze(1)  # [B,1,D]

                # decoder_input = decoder_output
                outputs.append(decoder_output)
                attns.append(attn_weights)

            attns = torch.cat(attns, dim=1)  # [B, pred_len, seq_len]
            outputs = torch.cat(outputs, dim=1)  # [B, pred_len, D]

        else:
            decoder_output, _ = self.decoder(dec_in, (hidden, cell))  # [B, L, D]
            outputs = decoder_output[:, -self.pred_len:, :]

        outputs = self.projection(outputs)
        return outputs


    def forward(self, x_enc, x_dec):
        dec_out = self.forecast(x_enc, x_dec)

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

