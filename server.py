import flwr as fl
import torch as T
from argparse import ArgumentParser
def main(args):


    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description='[Pecan Street Dataport] Forecasting the energy consumption of Pecan Street')
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--num_clients_per_round_fit', type=int, default=10)
    parser.add_argument('--num_clients_per_round_eval', type=int, default=25)
    parser.add_argument('--strategy', type=str, default='fedavg')

    args = parser.parse_args()
    args.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    args.gpu = T.cuda.device_count()


    main(args = args)