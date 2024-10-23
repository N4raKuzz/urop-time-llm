import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"input: {x.shape}")
        x = x.float()
        x = self.mlp(x) 
        x = x.squeeze()
        # print(f"output: {x.type()}")
        return x
