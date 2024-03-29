from typing import Dict, Tuple

import torch as T
import flwr as fl
from argparse import ArgumentParser
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

from src.base.trainers import Trainers
from src.dataset.processing import Processsing
from collections import OrderedDict
from src.data import TimeSeriesLoader


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: T.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)


parser = ArgumentParser(description='[Pecan Street Dataport] Forecasting the energy consumption of Pecan Street')
parser.add_argument("--data_path", type=str, default='dataset/pecanstreet/15min/')
# parser.add_argument("--data_path_test", type=list, default=['dataset/ElBorn_test.csv'])
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--targets", type=list, default=['consumption'])  # index 0
parser.add_argument("--num_lags", type=int, default=10)

parser.add_argument("--filter_bs", type=int, default=0)
parser.add_argument("--identifier", type=str, default='cid')

parser.add_argument("--nan_constant", type=int, default=0)
parser.add_argument("--x_scaler", type=str, default='minmax')
parser.add_argument("--y_scaler", type=str, default='minmax')
parser.add_argument("--outlier_detection", type=any, default=None)

parser.add_argument("--criterion", type=str, default='mse')
parser.add_argument("--fl_rounds", type=int, default=10)
parser.add_argument("--fraction", type=float, default=.10)
parser.add_argument("--aggregation", type=str, default="fedavg")
parser.add_argument("--model_name", type=str, default='cnn',
                    help='["mlp", "rnn" ,"lstm", "gru", "cnn", "da_encoder_decoder"]')
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--early_stopping", type=bool, default=False)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--max_grad_norm", type=float, default=0.0)
parser.add_argument("--reg1", type=float, default=0.0)  # l1 regularization
parser.add_argument("--reg2", type=float, default=0.0)  # l2 regularization

parser.add_argument("--cuda", type=bool, default=T.cuda.is_available())
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--assign_stats", type=any,
                    default=None)  # whether to use statistics as exogenous data, ["mean", "median", "std", "variance", "kurtosis", "skew"]
parser.add_argument("--use_time_features", type=bool, default=False)  # whether to use datetime features
args = parser.parse_args()
args.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

print(f"Script arguments {args}", end='\n')

local_train_params = {"epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr,
                      "criterion": args.criterion, "early_stopping": args.early_stopping,
                      "patience": args.patience, "device": args.device
                      }

trainers = Trainers(args=args)
trainers.seed_all(args.seed)

processing = Processsing(args=args, data_path=args.data_path)

X_train, X_val, X_test, y_train, y_val, y_test, exogenous_data_train, exogenous_data_val, x_scaler, y_scaler = processing.make_preprocessing(
    per_area=False, filter_bs=args.filter_bs)

X_train, X_val, X_test, y_train, y_val, y_test, client_X_train, client_X_val, client_X_test, client_y_train, client_y_val, client_y_test, _, _ = (
    processing.make_postprocessing(X_train, X_val, X_test, y_train, y_val, y_test, exogenous_data_train,
                                   exogenous_data_val, x_scaler, y_scaler))

print(f"\tX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"\tX_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"\tX_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

input_dim, exogenous_dim = processing.get_input_dims(X_train)
num_features = len(X_train[0][0])
print(input_dim, exogenous_dim)

net = trainers.get_model(model=args.model_name,
                           input_dim=input_dim,
                           out_dim=y_train.shape[1],
                           lags=args.num_lags,
                           exogenous_dim=exogenous_dim,
                           seed=args.seed)

train_loader = TimeSeriesLoader(X=X_train,
                                y=y_train,
                                num_lags=args.num_lags,
                                num_features=num_features, exogenous_data=exogenous_data_train,
                                indices=[0], batch_size=args.batch_size, shuffle=False).get_dataloader()

val_loader = TimeSeriesLoader(X=X_val,
                              y=y_val,
                              num_lags=args.num_lags,
                              num_features=num_features, exogenous_data=exogenous_data_val,
                              indices=[0], batch_size=args.batch_size, shuffle=False).get_dataloader()

test_loader = TimeSeriesLoader(X=X_test,
                              y=y_test,
                              num_lags=args.num_lags,
                              num_features=num_features, exogenous_data=exogenous_data_val,
                              indices=[0], batch_size=args.batch_size, shuffle=False).get_dataloader()


class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [v.cpu().numpy() for _, v in net.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        trainers.train(model=net,
                      train_loader=train_loader, test_loader=val_loader,
                      epochs=args.epochs,
                      optimizer=args.optimizer, lr=args.lr,
                      criterion=args.criterion,
                      early_stopping=args.early_stopping,
                      patience=args.patience,
                      device=args.device, cid=args.filter_bs)



        return self.get_parameters({}), len(train_loader), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        set_parameters(net, parameters)
        criterion = trainers.get_criterion(args.criterion)
        loss, mse, rmse, mae, r2, nrmse = trainers.test(model=net, data=test_loader, criterion=criterion)
        return float(loss), len(test_loader.dataset), {'mse': float(mse)}


fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
