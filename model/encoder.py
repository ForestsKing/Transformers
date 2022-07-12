import torch.nn.functional as F
from torch import nn

from model.attention.auto_correlation import AutoCorrelation
from model.attention.flow_attention import FlowAttention
from model.attention.full_attention import FullAttention
from model.attention.prob_attention import ProbAttention
from model.embedding import DataEmbedding


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.Conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=(3,),
            padding=(1,),
            stride=(1,),
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.Conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_k, d_v, d_model, d_ff, n_heads, dropout):
        super(EncoderLayer, self).__init__()

        if attention == 'FullAttention':
            self.attention = FullAttention(d_k, d_v, d_model, n_heads, dropout)
        elif attention == 'ProbAttention':
            self.attention = ProbAttention(d_k, d_v, d_model, n_heads, dropout)
        elif attention == 'AutoCorrelation':
            self.attention = AutoCorrelation(d_k, d_v, d_model, n_heads, dropout)
        elif attention == 'FlowAttention':
            self.attention = FlowAttention(d_k, d_v, d_model, n_heads, dropout)
        else:
            print('attention must in [FullAttention, ProbAttention, AutoCorrelation, FlowAttention]!')

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self, x):
        x = self.attention(x, x, x)

        residual = x.clone()
        x = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        y = self.dropout(self.conv2(x).permute(0, 2, 1))
        return self.norm(residual + y)


class Encoder(nn.Module):
    def __init__(self, attention, d_k, d_v, d_model, d_ff, n_heads, n_layer, d_feature, d_mark, dropout):
        super(Encoder, self).__init__()

        self.embedding = DataEmbedding(d_feature, d_mark, d_model, dropout)

        self.encoder = nn.ModuleList()
        for _ in range(n_layer):
            self.encoder.append(
                ConvLayer(d_model)
            )
            self.encoder.append(
                EncoderLayer(attention, d_k, d_v, d_model, d_ff, n_heads, dropout)
            )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_x, enc_mark):
        y = self.embedding(enc_x, enc_mark)

        for layer in self.encoder:
            y = layer(y)

        y = self.norm(y)
        return y
