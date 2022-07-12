import torch
from torch import nn


class FlowAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(FlowAttention, self).__init__()
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

        # 1. Linear projection
        Q = self.W_Q(input_Q).view(input_Q.size(0), input_Q.size(1), self.n_heads, self.d_k)
        K = self.W_K(input_K).view(input_K.size(0), input_K.size(1), self.n_heads, self.d_k)
        V = self.W_V(input_V).view(input_V.size(0), input_V.size(1), self.n_heads, self.d_v)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # 2. Non-negative projection
        Q, K = torch.sigmoid(Q), torch.sigmoid(K)

        # 3. Flow-Attention
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / torch.mul(Q, K.sum(dim=2).unsqueeze(2)).sum(-1)
        source_outgoing = 1.0 / torch.mul(K, Q.sum(dim=2).unsqueeze(2)).sum(-1)

        # (2) conservation refine for source and sink
        conserved_sink = torch.mul(Q, (K * source_outgoing.unsqueeze(-1)).sum(dim=2).unsqueeze(2)).sum(-1)
        conserved_source = torch.mul(K, (Q * sink_incoming.unsqueeze(-1)).sum(dim=2).unsqueeze(2)).sum(-1)

        # (3) Competition & Allocation
        kv = torch.matmul(K.transpose(-1, -2), torch.mul(V, torch.softmax(conserved_source, dim=-1).unsqueeze(-1)))
        qkv = torch.matmul(Q * sink_incoming.unsqueeze(-1), kv)
        context = torch.mul(qkv, torch.sigmoid(conserved_sink).unsqueeze(-1))

        # 4. Final projection
        context = context.transpose(1, 2)
        context = context.reshape(input_Q.size(0), input_Q.size(1), self.n_heads * self.d_v)
        output = self.fc(context)

        return self.dropout(self.norm(output + residual))
