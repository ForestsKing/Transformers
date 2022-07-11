import math

import torch
import torch.nn as nn


class AutoCorrelation(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout, c=5):
        super(AutoCorrelation, self).__init__()
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

    def time_delay_agg(self, V, corr):
        top_k = int(self.c * math.log(V.shape[2]))
        weights, delays = torch.topk(corr, top_k, dim=-2)
        weights = torch.softmax(weights, dim=-2)

        init_index = torch.arange(V.shape[2]).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        init_index = init_index.expand(V.shape).to(V.device)
        delays_agg = torch.zeros_like(V).float()
        V = V.repeat(1, 1, 2, 1)

        for i in range(top_k):
            weight = weights[:, :, i, :].unsqueeze(2)
            delay = delays[:, :, i, :].unsqueeze(2)
            index = init_index + delay
            pattern = torch.gather(V, dim=2, index=index)
            delays_agg = delays_agg + pattern * weight
        return delays_agg

    def forward(self, input_Q, input_K, input_V):
        L_Q = input_Q.shape[1]
        L_K = input_K.shape[1]

        residual = input_Q.clone()
        Q = self.W_Q(input_Q).view(input_Q.size(0), input_Q.size(1), self.n_heads, self.d_k)
        K = self.W_K(input_K).view(input_K.size(0), input_K.size(1), self.n_heads, self.d_k)
        V = self.W_V(input_V).view(input_V.size(0), input_V.size(1), self.n_heads, self.d_v)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        if L_Q > L_K:
            zeros = torch.zeros_like(Q[:, :, :(L_Q - L_K), :], device=Q.device).float()
            K = torch.cat([K, zeros], dim=2)
            V = torch.cat([V, zeros], dim=2)
        else:
            V = V[:, :, :L_Q, :]
            K = K[:, :, :L_Q, :]

        q_fft = torch.fft.rfft(Q, dim=2)
        k_fft = torch.fft.rfft(K, dim=2)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=2)

        context = self.time_delay_agg(V, corr)

        context = context.transpose(1, 2)
        context = context.reshape(input_Q.size(0), input_Q.size(1), self.n_heads * self.d_v)
        output = self.fc(context)
        return self.dropout(self.norm(output + residual))
