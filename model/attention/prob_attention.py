import numpy as np
import torch
from torch import nn


class ProbAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout, c=5):
        super(ProbAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.c = c

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_Q, D = Q.shape
        _, _, L_K, D = K.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D).clone()
        index_sample = torch.randint(0, L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        M = torch.max(Q_K_sample, dim=-1).values - torch.mean(Q_K_sample, dim=-1)
        M_top_index = torch.topk(M, n_top, dim=-1).indices

        Q_sample = Q[
                   torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top_index,
                   :,
                   ]

        return Q_sample, M_top_index

    def forward(self, input_Q, input_K, input_V):
        residual = input_Q.clone()
        Q = self.W_Q(input_Q).view(input_Q.size(0), input_Q.size(1), self.n_heads, self.d_k)
        K = self.W_K(input_K).view(input_K.size(0), input_K.size(1), self.n_heads, self.d_k)
        V = self.W_V(input_V).view(input_V.size(0), input_V.size(1), self.n_heads, self.d_v)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        u_k = int(self.c * np.log(input_K.size(1)))
        u_q = int(self.c * np.log(input_Q.size(1)))

        Q_sample, index = self._prob_QK(Q, K, sample_k=u_k, n_top=u_q)
        scores = torch.matmul(Q_sample, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = nn.Softmax(dim=-1)(scores)
        values = torch.matmul(attn, V)

        V_sum = V.mean(dim=-2)
        context = V_sum.unsqueeze(-2).expand(Q.shape).clone()
        context[
            torch.arange(Q.size(0))[:, None, None],
            torch.arange(Q.size(1))[None, :, None],
            index,
            :
        ] = values

        context = context.transpose(1, 2)
        context = context.reshape(input_Q.size(0), input_Q.size(1), self.n_heads * self.d_v)
        output = self.fc(context)
        return self.dropout(self.norm(output + residual))
