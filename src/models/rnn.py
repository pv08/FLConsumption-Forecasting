import torch.nn as nn
import torch as T
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, device, n_features, n_hidden, n_layers, drop_out):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        # RNN layers
        self.rnn = nn.RNN(
            n_features, n_hidden, n_layers, batch_first=True, dropout=drop_out
        )
        # Fully connected layer
        self.fc1 = nn.Linear(n_hidden, n_hidden * 2)
        self.regressor = nn.Linear(n_hidden * 2, 1)

        self.device = device

        self.to(self.device)

    def forward(self, x):
        self.rnn.flatten_parameters()

        _, hidden = self.rnn(x)

        out = hidden[-1]


        layer1 = F.relu(self.fc1(out))

        return self.regressor(layer1)