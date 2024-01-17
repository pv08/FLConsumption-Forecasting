import torch as T
import torch.nn as nn
import numpy as np
from typing import *
from torch.utils.data import DataLoader
from src.fl.server.proxy import ClientProxy
from src.fl.client.client import Client


class SimpleClientProxy(ClientProxy):
    def __init__(self, cid: Union[str, int], client):
        super(SimpleClientProxy, self).__init__(cid=cid)
        self.client: Client = client

    def get_parameters(self) -> List[np.ndarray]:
        """Returns the current local model weights"""
        return self.client.get_parameters()

    def set_train_parameters(self, params: Dict[str, Union[str, bool, int, float]], verbose: bool=False) -> None:
        return self.client.set_train_parameters(params, verbose)

    def fit(self, model: Optional[Union[nn.Module, List[np.ndarray]]], cid: int):
        """Local training"""
        return self.client.fit(model, cid=cid)

    def evaluate(self, data: Optional[DataLoader] = None,
                 model: Optional[Union[nn.Module, List[np.ndarray]]] = None,
                 params: Dict[str, Any] = None,
                 method: Optional[str] = None,
                 verbose: bool = False) -> Tuple[int, float, Dict[str, float]]:
        """Gloabl model evaluation"""
        return self.client.evaluate(data, model, params, method, verbose)