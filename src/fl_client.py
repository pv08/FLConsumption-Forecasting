import torch as T
from flwr.client import NumPyClient
from collections import OrderedDict


class FlowerClient(NumPyClient):
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
