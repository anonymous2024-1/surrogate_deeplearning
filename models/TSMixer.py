import torch
import torch.nn as nn


class FeatureMixing(nn.Module):
    def __init__(self, d_model, enc_in, dropout):
        super(FeatureMixing, self).__init__()
        self.channel = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, enc_in),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.channel(x)
        return x


class MixingLayer(nn.Module):
    def __init__(self, enc_len, d_model, enc_in, enc_out, dropout):
        super(MixingLayer, self).__init__()

        if enc_out is None:
            enc_out = enc_in

        self.temporal = nn.Sequential(
            nn.Linear(enc_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, enc_len),
            nn.Dropout(dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, enc_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class MixingLayer_cross(nn.Module):
    def __init__(self, enc_len, d_model, enc_in, enc_out, dropout):
        super(MixingLayer_cross, self).__init__()

        if enc_out is None:
            enc_out = enc_in

        self.temporal = nn.Sequential(
            nn.Linear(enc_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, enc_len),
            nn.Dropout(dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, enc_out),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(enc_in, enc_out)

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)  # [B, 37, 8+22]
        x1 = self.fc(x) # [B, 37, 8]
        x = x1 + self.channel(x)

        return x




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.params_in = self.dec_in - self.c_out  # 22
        self.layer = configs.e_layers

        self.mixlayer_h = MixingLayer(self.seq_len,
                                      configs.d_model,
                                      self.enc_in + self.c_out,
                                      self.enc_in + self.c_out,
                                      configs.dropout) # (B,30,7+1)

        self.mixlayer_f = MixingLayer(self.pred_len,
                                      configs.d_model,
                                      self.enc_in + self.c_out,
                                      self.enc_in + self.c_out,
                                      configs.dropout) # (B,7,7+1)

        self.fmix_1 = FeatureMixing(configs.d_model, self.params_in, configs.dropout)  # (B,37,22)

        self.mixlayer_1 = MixingLayer_cross(self.seq_len + self.pred_len,
                                      configs.d_model,
                                      self.enc_in + self.c_out + self.params_in,
                                      self.enc_in + self.c_out,
                                      configs.dropout)

        self.mixlayer = nn.ModuleList([MixingLayer_cross(self.seq_len + self.pred_len,
                                                   configs.d_model,
                                                   self.enc_in + self.c_out + self.params_in,
                                                   self.enc_in + self.c_out,
                                                   configs.dropout) for _ in range(configs.e_layers)])

        self.fmix = nn.ModuleList([FeatureMixing(configs.d_model,
                                                 self.params_in,
                                                 configs.dropout) for _ in range(configs.e_layers)])

        self.projection1 = nn.Linear(self.seq_len + self.pred_len, self.pred_len)
        self.projection2 = nn.Linear(self.enc_in + self.c_out, self.c_out)

    def forecast(self, x_enc, x_dec):
        x = torch.cat([x_enc, x_dec[:, :, 0:self.c_out]], dim=2)
        x_params = x_dec[:, :, self.c_out:]

        x_h = self.mixlayer_h(x[:, :self.seq_len, :])  # historical
        x_f = self.mixlayer_f(x[:, -self.pred_len:, :]) # future
        x = torch.cat([x_h, x_f], dim=1)  # (B,37,7+1)

        x_p = self.fmix_1(x_params)

        x = torch.cat([x, x_p], dim=2)  # (B,37,8+22)

        x = self.mixlayer_1(x)  # (B,37,8)

        # x: [B, L, D]
        for i in range(self.layer):
            x = torch.cat([x, self.fmix[i](x_params)], dim=2)  # (B,37,8+22)
            x = self.mixlayer[i](x)  # (B,37,8)

        x_out = self.projection1(x.transpose(1, 2)).transpose(1, 2)  # (B,7,8)
        x_out = self.projection2(x_out)  # (B,7,1)
        return x_out


    def forward(self, x_enc, x_dec):
        dec_out = self.forecast(x_enc, x_dec)

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

