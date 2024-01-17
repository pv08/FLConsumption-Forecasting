import torch as T
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional


class TimeSeriesLoader:
    def __init__(self, cid, X, y,  num_lags, num_features, indices, batch_size, shuffle, exogenous_data=None):
        self.X = T.tensor(X).float()
        self.cid = cid
        self.y = T.tensor(y).float()
        self.num_lags = num_lags
        self.num_features = num_features
        self.exogenous_data = exogenous_data
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.exogenous_data is not None:
            self.exogenous_data = T.tensor(self.exogenous_data).float()


    def get_dataloader(self):
        tensor_dataset = TimeSeriesDataset(self.X, self.y, self.num_lags, self.num_features,
                                           self.indices, self.exogenous_data)
        return DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=self.shuffle)





class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, num_lags: int=10, num_features: int=11,
                 indices: List[int]=[0], exogenous: Optional[np.ndarray]=None):
        if exogenous is not None:
            assert X.size(0) == y.size(0) == exogenous.size(0), "Size mismatch between tensors"
        else:
            assert X.size(0) == y.size(0), "Size mismatch between tensors"

        self.X = X
        self.y = y
        self.num_lags = num_lags
        self.num_features = num_features
        self.indices = indices
        self.exogenous = exogenous


    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if idx == 0:
            tmp_X = self.X[idx]
            if len(self.X.shape) < 3:
                tmp_X = tmp_X.view(self.num_lags, self.num_features, 1)
            y_hist = []
            for i, lag in enumerate(tmp_X):
                if i == 0:
                    pad = T.zeros_like(lag[self.indices])
                    y_hist.append(pad.reshape(1, -1))
                else:
                    y_hist.append((tmp_X[i - 1][self.indices].reshape(1, -1)))
            y_hist = T.cat(y_hist)
        elif idx < self.num_lags + 1:
            last_obs = self.X[idx - 1]
            if len(self.X.shape) < 3:
                last_obs = last_obs.view(self.num_lags, self.num_features, 1)
            y_hist = []
            for i, lag in enumerate(last_obs):
                y_hist.append(last_obs[i][self.indices].reshape(1, -1))
            y_hist = T.cat(y_hist)
        else:
            y_hist = self.y[idx - self.num_lags - 1: idx - 1]

        if self.exogenous is None:
            return self.X[idx], [], y_hist, self.y[idx]
        else:
            return self.X[idx], self.exogenous[idx], y_hist, self.y[idx]
