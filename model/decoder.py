import torch.nn.functional as F
from torch import nn

from model.attention.auto_correlation import AutoCorrelation
from model.attention.full_attention import FullAttention
from model.attention.prob_attention import ProbAttention
from model.embedding import DataEmbedding


class DecoderLayer(nn.Module):
    def __init__(self, attention, d_k, d_v, d_model, d_ff, n_heads, dropout):
        super(DecoderLayer, self).__init__()

        if attention == 'FullAttention':
            self.self_attention = FullAttention(d_k, d_v, d_model, n_heads, dropout)
            self.cross_attention = FullAttention(d_k, d_v, d_model, n_heads, dropout)
        elif attention == 'ProbAttention':
            self.self_attention = ProbAttention(d_k, d_v, d_model, n_heads, dropout)
            self.cross_attention = ProbAttention(d_k, d_v, d_model, n_heads, dropout)
        elif attention == 'AutoCorrelation':
            self.self_attention = AutoCorrelation(d_k, d_v, d_model, n_heads, dropout)
            self.cross_attention = AutoCorrelation(d_k, d_v, d_model, n_heads, dropout)
        else:
            print('attention must in [FullAttention, ProbAttention, AutoCorrelation]!')

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self, x, enc_outputs):
        x = self.self_attention(x, x, x)
        x = self.cross_attention(x, enc_outputs, enc_outputs)

        residual = x.clone()
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(x).transpose(-1, 1))
        return self.norm(residual + y)


class Decoder(nn.Module):
    def __init__(self, attention, d_k, d_v, d_model, d_ff, n_heads, n_layer, d_feature, d_mark, dropout):
        super(Decoder, self).__init__()

        self.embedding = DataEmbedding(d_feature, d_mark, d_model, dropout)

        self.decoder = nn.ModuleList()
        for _ in range(n_layer):
            self.decoder.append(
                DecoderLayer(attention, d_k, d_v, d_model, d_ff, n_heads, dropout)
            )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, dec_in, dec_mark, enc_outputs):
        y = self.embedding(dec_in, dec_mark)

        for layer in self.decoder:
            y = layer(y, enc_outputs)

        y = self.norm(y)
        return y
