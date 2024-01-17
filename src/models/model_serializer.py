import h5py
import torch as T
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from src.utils.functions import mkdir_if_not_exists

class ModelSerializer:
    def __init__(self, model, path: str):

        mkdir_if_not_exists(f"etc/")
        mkdir_if_not_exists(f"etc/ckpt/")
        mkdir_if_not_exists(f"etc/ckpt/federated/")
        mkdir_if_not_exists(f"etc/ckpt/client/")
        self.model = model
        self.path = path


    def save(self, name):
        np_weights = self.state_to_numpy(self.model.state_dict())
        with h5py.File(f"{self.path}/{name}", "w") as f:
            group = f.create_group("model_weights", track_order=True)
            for k, v in np_weights.items():
                group[k] = v
            # group = f.create_group("x_scaler", track_order=True)
            # for k, v in self.x_scaler.items():
            #     group[k] = v
            # group = f.create_group("y_scaler", track_order=True)
            # for k, v in self.y_scaler.items():
            #     group[k] = v

    @staticmethod
    def state_to_numpy(model_state):
        assert type(model_state) in (dict, OrderedDict), \
            f"Model state must be of type dictionary. Received {type(model_state)}"
        k = next(iter(model_state))
        assert type(model_state[k]) in (T.tensor, T.Tensor, np.ndarray), \
            f"Model weights must be of type torch.tensor or numpy.ndarray. Received {type(model_state[k])}"
        if type(model_state[k]) == np.ndarray:
            return model_state
        np_ordered_dict = OrderedDict()
        for k, v in model_state.items():
            np_ordered_dict[k] = v.cpu().numpy().astype(np.float64)
        return np_ordered_dict