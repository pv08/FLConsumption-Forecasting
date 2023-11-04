import torch.nn as nn
import torch.optim as optim
from src.models.rnn import RNNModel
from src.regressors.basic_regressor import BasicRegressor

class ConsumptionRNNRegressor(BasicRegressor):
    def __init__(self, device, n_features, lr, n_hidden, n_layers, dropout):
        super(ConsumptionRNNRegressor, self).__init__(lr=lr)
        self.model = RNNModel(device, n_features, n_hidden, n_layers, dropout)

    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
             loss = self.criterion(output, labels.unsqueeze(dim = 1))
        return loss, output

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr = self.lr)