from typing import Dict, Tuple
from logging import INFO, DEBUG
from flwr.common.logger import log
import torch as T
import flwr as fl
import pandas as pd
from argparse import ArgumentParser
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from src.fl.history.history import History
from src.utils.functions import plot_test_prediction
from src.base.trainers import Trainers
from src.dataset.processing import Processsing
from collections import OrderedDict
from src.data import TimeSeriesLoader



def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: T.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)
def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]




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
parser.add_argument("--fraction", type=float, default=.25)
parser.add_argument("--aggregation", type=str, default="fedavg")
parser.add_argument("--model_name", type=str, default='cnn',
                    help='["mlp", "rnn" ,"lstm", "gru", "cnn", "da_encoder_decoder"]')
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--batch_size", type=int, default=128)
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
history = History()
processing = Processsing(args=args, data_path=args.data_path)

X_train, X_val, X_test, y_train, y_val, y_test, exogenous_data_train, exogenous_data_val, x_scaler, y_scaler = processing.make_preprocessing(
    per_area=True)

X_train, X_val, X_test, y_train, y_val, y_test, client_X_train, client_X_val, client_X_test, client_y_train, client_y_val, client_y_test, _, _ = (
    processing.make_postprocessing(X_train, X_val, X_test, y_train, y_val, y_test, exogenous_data_train,
                                   exogenous_data_val, x_scaler, y_scaler))

for client in client_X_train:
    print(f"\nClient: {client}")
    print(f"\tX_train shape: {client_X_train[client].shape}, y_train shape: {client_y_train[client].shape}")
    print(f"\tX_val shape: {client_X_val[client].shape}, y_val shape: {client_y_val[client].shape}")

input_dim, exogenous_dim = processing.get_input_dims(X_train)
num_features = len(X_train[0][0])
print(input_dim, exogenous_dim)

train_loaders, val_loaders = [], []

for client in client_X_train:
    if client == "all":
        continue
    if exogenous_data_train is not None:
        tmp_exogenous_data_train = exogenous_data_train[client]
        tmp_exogenous_data_val = exogenous_data_val[client]
    else:
        tmp_exogenous_data_train = None
        tmp_exogenous_data_val = None
    num_features = len(client_X_train[client][0][0])

    train_loaders.append(
        (client, TimeSeriesLoader(cid=client,X=client_X_train[client],
                         y=client_y_train[client],
                         num_lags=args.num_lags,
                         num_features=num_features, exogenous_data=tmp_exogenous_data_train,
                         indices=[0], batch_size=args.batch_size, shuffle=False).get_dataloader())
    )
    val_loaders.append(
        (client, TimeSeriesLoader(cid=client,X=client_X_val[client],
                         y=client_y_val[client],
                         num_lags=args.num_lags,
                         num_features=num_features, exogenous_data=tmp_exogenous_data_val,
                         indices=[0], batch_size=args.batch_size, shuffle=False).get_dataloader())
    )

cids = [k for k in client_X_train.keys() if k != 'all']
log(INFO, f"Trainable clients: {cids}")


model = trainers.get_model(model=args.model_name,
                             input_dim=input_dim,
                             out_dim=y_train.shape[1],
                             lags=args.num_lags,
                             exogenous_dim=exogenous_dim,
                             seed=args.seed)

state_dict = T.load('model_round_10.pth')
model.load_state_dict(state_dict)
state_dict_ndarrays = [v.cpu().numpy() for v in model.state_dict().values()]
parameters = fl.common.ndarrays_to_parameters(state_dict_ndarrays)


evaluate_metrics = []
for client in cids:
    num_features = len(client_X_test[client][0][0])
    test_loader = TimeSeriesLoader(X=client_X_test[client],
                                   y=client_y_test[client],
                                   num_lags=args.num_lags,
                                   num_features=num_features, exogenous_data=None,
                                   indices=[0], batch_size=1, shuffle=False, cid=client).get_dataloader()
    test_mse, test_rmse, test_mae, test_r2, test_nrmse, test_pinball, y_pred_test = Trainers(args=args).test(
        model, test_loader, None, device=args.device)
    evaluate_metrics.append({'cid': client, 'mse': test_mse, 'mae': test_mae, 'r2': test_r2, 'pinball': test_pinball})
    log(INFO, f"Client: {client} | MSE: {test_mse} | MAE: {test_mae} | pinball loss: {test_pinball}")
    plot_test_prediction(y_true=client_y_test[client], y_pred=y_pred_test, cid=client, model_name=args.model_name)
evaluate_metrics_df = pd.DataFrame(evaluate_metrics)
evaluate_metrics_df.to_csv('etc/results/eval_metrics.csv')

# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
