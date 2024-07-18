import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.seq_len
        self.futureM = configs.futureM
        self.relu = nn.Tanh()
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.seriesCorss1 = nn.Linear(configs.seq_len,configs.seq_len*2)
        self.varCross1 = nn.Linear(configs.d_model,configs.d_model*2)

        #Decoder
        self.seriesCorss2 = nn.Linear(configs.seq_len*2,configs.futureM)
        self.varCross2 = nn.Linear(configs.d_model*2,configs.d_model)

        #Projection
        self.projection1 = nn.Linear(configs.d_model,1)
        self.projection2 = nn.Linear(configs.futureM,self.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        #[B, L, D]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        [b,l,d] = enc_out.shape
        enc_out = enc_out.permute(0,2,1)
        #[B, D, L]
        enc_out = enc_out.reshape(-1,l)
        enc_out = self.seriesCorss1(enc_out)
        #enc_out = self.relu(enc_out)
        enc_out = enc_out.reshape(b,d,2*l)
        enc_out = enc_out.permute(0,2,1)
        #[B, L, D]
        enc_out = enc_out.reshape(-1,d)
        enc_out = self.varCross1(enc_out)
        #enc_out = self.relu(enc_out)
        enc_out = enc_out.reshape(b,2*d,2*l)

        enc_out = enc_out.permute(0,2,1)
        #[B, D, L]
        enc_out = enc_out.reshape(-1,2*l)
        dec_out = self.seriesCorss2(enc_out)
        #dec_out = self.relu(dec_out)
        dec_out = dec_out.reshape(b,2*d,self.futureM)
        dec_out = dec_out.permute(0,2,1)
        #[B, L, D]
        dec_out = dec_out.reshape(-1,2*d)
        dec_out = self.varCross2(dec_out)
        #dec_out = self.relu(dec_out)
        #[batch_size, futureM, d_model]
        out = self.projection1(dec_out)
        #out = self.relu(out)
        out = out.reshape(b,self.futureM,1)
        out = out.permute(0,2,1)
        #[B, D, L]
        out = out.reshape(-1,self.futureM)
        out = self.projection2(out)
        #out = self.relu(out)
        out = out.reshape(b,1,self.pred_len)
        out = out.permute(0,2,1)
        #[B, L, D]
        return out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
