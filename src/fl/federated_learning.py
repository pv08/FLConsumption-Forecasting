import copy

import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import *
from src.fl.client.torch_client import TorchRegressionClient
class FederatedLearning:
    @staticmethod
    def create_regression_client(cid: str, model: nn.Module,
                                 train_loader: DataLoader,
                                 test_loader: DataLoader,
                                 local_params: Dict[str, Union[bool, str, int, float]]):
        net = copy.deepcopy(model)
        return TorchRegressionClient(cid=cid, net=net, train_loader=train_loader, val_loader=test_loader,
                                     local_train_params=local_params)

    @staticmethod
    def weighted_loss_avg(n_per_client: List[int], losses: List[float]) -> float:
        """Aggregates losses received from clients"""
        n = sum(n_per_client)
        weighted_losses = [n_k * loss for n_k, loss in zip(n_per_client, losses)]
        return sum(weighted_losses) / n

    @staticmethod
    def weighted_metrics_avg(n_per_client: List[int], metrics_per_client: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        n = sum(n_per_client)
        metrics = dict()
        for cid in metrics_per_client:
            for metric in metrics_per_client[cid]:
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric].append(metrics_per_client[cid][metric])
        weighted_metrics = dict()
        for metric in metrics:
            weighted_metric = [n_k * m for n_k, m in zip(n_per_client, metrics[metric])]
            weighted_metrics[metric] = sum(weighted_metric) / n
        return weighted_metrics

