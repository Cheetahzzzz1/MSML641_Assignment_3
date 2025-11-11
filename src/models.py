"""
models.py
- PyTorch models for RNN, LSTM, Bidirectional LSTM
- Embedding dim = 100, hidden = 64, num_layers = 2, dropout = 0.4
"""
import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_size=64, num_layers=2, dropout=0.4, arch='RNN', bidirectional=False, activation='tanh'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.arch = arch
        self.bidirectional = bidirectional
        if arch == 'RNN':
            nonlin = 'tanh' if activation == 'tanh' else 'relu'
            self.rnn = nn.RNN(embed_dim, hidden_size, num_layers=num_layers, nonlinearity=nonlin, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            out_dim = hidden_size * (2 if bidirectional else 1)
        elif arch == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            out_dim = hidden_size * (2 if bidirectional else 1)
        else:
            raise ValueError('Unknown arch')
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        self.fc = nn.Linear(out_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)               # (B, L, E)
        out, _ = self.rnn(emb)               # out: (B, L, H * dirs)
        last = out[:, -1, :]                 # last timestep
        h = self.dropout(last)
        h = self.activation(h)
        out = self.fc(h)
        return self.sig(out).squeeze(1)
