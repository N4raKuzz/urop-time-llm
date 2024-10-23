import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size, input_size=1, dropout=0.5):
        super(Model, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x = x.unsqueeze(-1)
        _, hidden = self.gru(x)
        out = hidden[-1]
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out.squeeze()

        return out
        