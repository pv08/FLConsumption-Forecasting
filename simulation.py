from typing import Dict, Tuple
from logging import INFO, DEBUG
from flwr.common.logger import log
import torch as T
import flwr as fl
from argparse import ArgumentParser
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from src.fl.history.history import History
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


class FlowerClient(NumPyClient):
    def __init__(self, _idx, cid, net, trainloader, valloader):
        self._idx = _idx
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)


    def fit(self, parameters, config):
        log(INFO, f"Fitting client {self.cid}...")
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        trainers.train(model=self.net,
                      train_loader=self.trainloader, test_loader=self.valloader,
                      epochs=args.epochs,
                      optimizer=args.optimizer, lr=args.lr,
                      criterion=args.criterion,
                      early_stopping=args.early_stopping,
                      patience=args.patience,
                      device=args.device, cid=self.cid)



        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        print(f"[Client {self.cid}] evaluate")
        set_parameters(self.net, parameters)
        criterion = trainers.get_criterion(args.criterion)
        loss, mse, rmse, mae, r2, nrmse = trainers.test(model=self.net, data=self.valloader, criterion=criterion)
        log(INFO, f'MSE: {float(mse)}')
        return float(loss), len(self.valloader), {'mse': float(mse)}

def numpyclient_fn(_idx):

    net = trainers.get_model(model=args.model_name,
                             input_dim=input_dim,
                             out_dim=y_train.shape[1],
                             lags=args.num_lags,
                             exogenous_dim=exogenous_dim,
                             seed=args.seed)

    cid, trainloader = train_loaders[int(_idx)]
    cid, valloader = val_loaders[int(_idx)]
    log(INFO, f'Assigning client {_idx} representing {cid}...')
    return FlowerClient(_idx=_idx, cid=cid, net=net, trainloader=trainloader, valloader=valloader)


def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    mses = [num_examples * m["mse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    weighted_mse = sum(mses) / sum(examples)
    # Aggregate and return custom metric (weighted average)
    history.add_global_test_metrics({'MSE': weighted_mse})
    return {"mse": weighted_mse}



strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average
    )

fl_history = fl.simulation.start_simulation(
    client_fn=numpyclient_fn,
    num_clients=25,
    config=fl.server.ServerConfig(num_rounds=3),
    client_resources={"num_gpus": 1},
    strategy=strategy
)
print(fl_history)
print(history)

# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
