import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch_size, seq_len, n_features)
        # Initialize hidden state and conveyor
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # out: (batch_size, seq_len, hidden_size)
        out = self.dense(out[:, -1, :])
        # out: (batch_size, m_features)
        
        return out


seq_len = 100    # Input time step: (hours)
in_features = 10  # Input Features: Observations and Actions
hidden_size = 20 # Features in hidden state
num_layers = 12   # Stacked lstm layers
out_features = 5   # Output features: Observations Only
model = LSTM(in_features, hidden_size, num_layers, out_features)
