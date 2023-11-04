import torch as T
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from flwr.client import NumPyClient
from collections import OrderedDict


class FlowerClient(NumPyClient):
    def __init__(self, regressor, train_loader, val_loader, test_loader, device, _id):
        super(FlowerClient, self).__init__()
        self.regressor = regressor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.logger_master = WandbLogger(project='FLEnergyConsumption',
                                         tags=['pecanstreet', 'train', 'FederatedLearning', 'RNN', str(_id)],
                                         offline=False,
                                         name=f'participant_{str(_id)}')

    def get_parameters(self, config):
        return [v.cpu().numpy() for _, v in self.regressor.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.regressor.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: T.Tensor(v) for k, v in params_dict})
        self.regressor.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainer = pl.Trainer(max_epochs=10, logger=self.logger_master)
        trainer.fit(self.regressor, self.train_loader, self.val_loader)

        return self.get_parameters({}), len(self.train_loader), {}
