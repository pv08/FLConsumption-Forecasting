import torch.nn as nn
from typing import List

class MLP(nn.Module):
    def __init__(self, device, input_dim: int, layer_units: List[int]=[128, 64], num_outputs: int=2, init_weights: bool=True):
        super(MLP, self).__init__()
        activation = nn.ReLU()

        layers = [nn.Linear(input_dim, layer_units[0]), activation]

        for i in range(len(layer_units) - 1):
            layers.append(nn.Linear(layer_units[i], layer_units[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(layer_units[len(layer_units) - 1], num_outputs))
        self.MLP_layers = nn.Sequential(*layers)
        if init_weights:
            self.MLP_layers.apply(self._init_weights)
        self.to(device)

    def forward(self, x, exogenous=None, device=None, y_hist=None):
        if len(x.shape) > 2:
            x = x.view(x.size(0), x.size(1) * x.size(2))
        x = self.MLP_layers(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()