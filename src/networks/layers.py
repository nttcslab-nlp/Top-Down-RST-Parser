import torch
import torch.nn as nn
import numpy


class SelectiveGate(nn.Module):
    def __init__(self, lstm):
        super(SelectiveGate, self).__init__()
        self.lstm = lstm
        hidden_size = lstm.hidden_size
        self.Ws = nn.Linear(hidden_size*2, hidden_size*2, bias=False)  # bを含まない
        self.Us = nn.Linear(hidden_size*2, hidden_size*2, bias=True)  # bを含む
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        rnn_output, last_status = self.lstm(x, lengths, return_last_state=True)
        sGate = self.sigmoid(self.Ws(rnn_output) + self.Us(last_status).unsqueeze(1))  # BxN_WORDx2H
        gated_rnn_output = rnn_output * sGate
        return gated_rnn_output, sGate


class BiLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout):
        super(BiLSTM, self).__init__()
        self.bidirectional = True
        self.bilstm = nn.LSTM(embed_size, hidden_size,
                              num_layers=2, dropout=dropout,
                              bidirectional=True, batch_first=True)
        self.hidden_size = hidden_size

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, lengths, return_last_state=False):
        lengths, perm_idx = torch.sort(lengths, 0, descending=True)
        x = x[perm_idx]
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        y, (h, _) = self.bilstm(x)
        y, _ = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True)
        perm_idx_rev = torch.tensor(self._inverse_indices(perm_idx), device=perm_idx.device)
        y = y[perm_idx_rev, :]
        if return_last_state:
            L, B, H = h.size()
            h = h[-2:, perm_idx_rev, :]  # (L=2,B,H) forward, backword of top layer
            h = h.transpose(1, 0).contiguous().view(B, 2*H)  # (B,2xH)
            return y, h
        else:
            return y

    def _inverse_indices(self, indices):
        indices = indices.cpu().numpy()
        r = numpy.empty_like(indices)
        r[indices] = numpy.arange(len(indices))
        return r


class FeedForward(nn.Sequential):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i, (prev_dim, next_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(prev_dim, next_dim))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        super(FeedForward, self).__init__(*layers)


class DeepBiAffine(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(DeepBiAffine, self).__init__()
        self.W_left = FeedForward(hidden_size, [hidden_size], hidden_size, dropout)
        self.W_right = FeedForward(hidden_size, [hidden_size], hidden_size, dropout)
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.V_left = nn.Linear(hidden_size, 1)
        self.V_right = nn.Linear(hidden_size, 1)

    def forward(self, h_ik, h_kj):
        h_ik = self.W_left(h_ik)
        h_kj = self.W_right(h_kj)
        return (h_ik * self.W_s(h_kj)).sum(1, keepdim=True) + self.V_left(h_ik) + self.V_right(h_kj)
