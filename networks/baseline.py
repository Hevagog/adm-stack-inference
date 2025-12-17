import torch.nn as nn


class BaselineNetwork(nn.Module):
    """A simple baseline neural network with fully connected layers."""

    def __init__(self, input_size=4096, output_size=100):
        super(BaselineNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(1024, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out
