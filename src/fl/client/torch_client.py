import torch as T
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from src.utils.logger import log
from src.individual_training import Trainers
from src.fl.client.client import Client
from torch.utils.data import DataLoader
from typing import *
from logging import INFO, DEBUG
from src.models.model_serializer import ModelSerializer

class TorchRegressionClient(Client):
    def __init__(self, cid: Union[str, int],
                 net: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 local_train_params: Optional[Dict[str, Union[str, int, float, bool]]]):
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_train_params = local_train_params
        self.epochs = None
        self.optimizer = None
        self.lr = None
        self.criterion = None
        self.early_stopping = None
        self.patience = None
        self.device = None
        self.reg1 = None
        self.reg2 = None
        self.max_grad_norm =None
        self.fed_prox_mu = None
        self._init_local_train_params()


    def _init_local_train_params(self):
        self.epochs = self.local_train_params['epochs']
        self.optimizer = self.local_train_params['optimizer']
        self.lr = self.local_train_params['lr']
        self.criterion = self.local_train_params['criterion']
        self.early_stopping = self.local_train_params['early_stopping']
        self.patience = self.local_train_params['patience']
        self.device = self.local_train_params['device']

        try:
            self.reg1 = self.local_train_params['reg1']
        except KeyError:
            self.reg1 = 0

        try:
            self.reg2 = self.local_train_params['reg2']
        except KeyError:
            self.reg2 = 0

        try:
            self.max_grad_norm = self.local_train_params['max_grad_norm']
        except KeyError:
            self.max_grad_norm = 0

        try:
            self.fed_prox_mu = self.local_train_params['fed_prox_mu']
        except KeyError:
            self.fed_prox_mu = 0

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, params: Union[List[np.ndarray], nn.Module]):
        if not isinstance(params, nn.Module):
            params_dict = zip(self.net.state_dict().keys(), params)
            state_dict = OrderedDict({k: T.Tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)
        else:
            self.net.load_state_dict(params.state_dict(), strict=True)

    def evaluate(self, data: Optional[Union[np.ndarray, DataLoader]]=None,
                 model: Optional[Union[nn.Module, List[np.ndarray]]]=None,
                 params: Optional[Dict[str, Any]]=None,
                 method: Optional[str]=None,
                 verbose: bool=False):
        if not params or "criterion" not in params:
            params = dict()
            params['criterion'] = nn.MSELoss()

        if model:
            self.set_parameters(model)

        if data is None and method == 'test':
            data = self.val_loader
        if data is None and method == 'train':
            data = self.train_loader

        loss, mse, rmse, mae, r2, nrmse = Trainers(args=None).test(self.net, data, params["criterion"], device=self.device)
        metrics = {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae), 'R^2': float(r2), "NRMSE": float(nrmse)}

        if verbose:
            log(INFO, f"[Client {self.cid} Evaluation on {len(data.dataset)} samples] "
                      f"loss: {loss}, mse: {mse}, rmse: {rmse}, mae: {mae}, nrmse: {nrmse}")

        return len(data.dataset), loss, metrics

    def fit(self, model: Optional[Union[nn.Module, List[np.ndarray]]], cid: str):
        if model is not None:
            self.set_parameters(model)
            self.net: nn.Module = Trainers(args=None).train(model=self.net, train_loader=self.train_loader,
                                                            test_loader=self.val_loader, epochs=self.epochs,
                                                            optimizer=self.optimizer, lr=self.lr, criterion=self.criterion,
                                                            early_stopping=self.early_stopping, patience=self.patience,
                                                            reg1=self.reg1, reg2=self.reg2, max_grad_norm=self.max_grad_norm,
                                                            fedprox_mu=self.fed_prox_mu, log_per=10, cid=cid)
        _, train_loss, train_metrics = self.evaluate(self.train_loader)
        num_test, test_loss, test_metrics = self.evaluate(self.val_loader)


        client_model_serializer = ModelSerializer(model=self.net, path='etc/ckpt/client/')
        client_model_serializer.save(f'{cid}.h5')
        log(INFO, f"Client model saved on etc/ckpt/client/{cid}.h5")

        return self.get_parameters(), len(self.train_loader.dataset), train_loss, train_metrics, num_test, test_loss, test_metrics

    def set_train_parameters(self, params: Dict[str, Union[bool, str, int, float]], verbose: bool=False):
        self.epochs = params["epochs"] if "epochs" in params else self.epochs
        self.optimizer = params['optimizer'] if "optimizer" in params else self.optimizer
        self.lr = params['lr'] if "lr" in params else self.lr
        self.criterion = params['criterion'] if "criterion" in params else self.criterion
        self.early_stopping = params['early_stopping'] if "early_stopping" in params else self.early_stopping
        self.patience = params['patience'] if "patience" in params else self.patience
        self.device = params['device'] if "device" in params else self.device
        self.reg1 = params['reg1'] if "reg1" in params else self.reg1
        self.reg2 = params['reg2'] if "reg2" in params else self.reg2
        self.max_grad_norm = params['max_grad_norm'] if "max_grad_norm" in params else self.max_grad_norm
        self.fed_prox_mu = params['fed_prox_mu'] if "fed_prox_mu" in params else self.fed_prox_mu

        if verbose:
            log(DEBUG, f"Training parameters change for client {self.cid}: "
                       f"epochs={self.epochs}, optimizer={self.optimizer}, lr={self.lr}, "
                       f"criterion={self.criterion}, early_stopping={self.early_stopping}, patience={self.patience}, "
                       f"device={self.device}, reg1={self.reg1}, reg2={self.reg2}, max_grad_norm={self.max_grad_norm}")

