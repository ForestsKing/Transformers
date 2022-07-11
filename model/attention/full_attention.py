import numpy as np
import torch
from torch import nn


class FullAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(FullAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_Q, input_K, input_V):
        residual = input_Q.clone()
        Q = self.W_Q(input_Q).view(input_Q.size(0), input_Q.size(1), self.n_heads, self.d_k)
        K = self.W_K(input_K).view(input_K.size(0), input_K.size(1), self.n_heads, self.d_k)
        V = self.W_V(input_V).view(input_V.size(0), input_V.size(1), self.n_heads, self.d_v)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2)
        context = context.reshape(input_Q.size(0), input_Q.size(1), self.n_heads * self.d_v)
        output = self.fc(context)

        return self.dropout(self.norm(output + residual))
