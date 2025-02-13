import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# LSTM Model
# Model Definition
class CpGPredictor(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2):
        super(CpGPredictor, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=0)  # 0 is padding
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)  # Regression output

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)  # Pack padded sequences
        packed_out, _ = self.lstm(packed_x)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # Unpack output

        # Get the last valid output (not from padding)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, -1, lstm_out.size(2))  # Index last valid timesteps
        last_outputs = lstm_out.gather(1, idx).squeeze(1)

        out = self.classifier(last_outputs)
        return out.squeeze()