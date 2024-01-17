import numpy as np
import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import *
from abc import ABC, abstractmethod


class ClientProxy(ABC):
    def __init__(self, cid: Union[str, int]):
        self.cid = cid

    @abstractmethod
    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local model weights"""

    @abstractmethod
    def set_train_parameters(self, params: Dict[str, Union[str, bool, int, float]], verbose: bool=False) -> None:
        """Set local parameters"""

    @abstractmethod
    def fit(self, model: Optional[Union[nn.Module, List[np.ndarray]]], cid: int):
        """Local training.
                    Returns:
                        1) a list of np.ndarray which corresponds to local learned weights
                        2) the number of local instances
                        3) the local train loss
                        4) the local train metrics
                        5) the local test loss
                        6) the local test metrics
                Note that clients may not own a local validation/test set, i.e. the validation/test set can be global.
                We need a validation/test set to perform evaluation at each local epoch.
                """

    @abstractmethod
    def evaluate(self, data: Optional[DataLoader] = None,
                 model: Optional[Union[nn.Module, List[np.ndarray]]] = None,
                 params: Dict[str, Any] = None,
                 method: Optional[str] = None,
                 verbose: bool = False) -> Tuple[int, float, Dict[str, float]]:
        """Global model evaluation.
            Returns:
                1) The number of evaluation instances.
                2) The evaluation loss
                3) the evaluation metrics
        Note that the evaluate method can correspond to the evaluation
        of the global model to either the local training instances or the (local) testing instances.
        """