import torch as T
from argparse import ArgumentParser
from src.preprocessing import ParticipantData
def main():
    parser = ArgumentParser(description='[Pecan Street Dataport] Forecasting the energy consumption of Pecan Street')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--debug_percent', type=float, default=0.2378)
    parser.add_argument('--sequence_length', type=int, default=60)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--root_path', type=str, default='data/')

    # Recorrent neural networks hyperparameters
    parser.add_argument('--n_hidden', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=201)
    parser.add_argument('--early_stopping', type=bool, default=False)
    parser.add_argument('--patience', type=int, default=2)

    #FL parameters
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--num_clients_per_round_fit', type=int, default=10)
    parser.add_argument('--num_clients_per_round_eval', type=int, default=25)
    parser.add_argument('--strategy', type=str, default='fedavg')


    args = parser.parse_args()
    args.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    args.gpu = T.cuda.device_count()

    ParticipantData(path=args.root_path)
if __name__ == "__main__":
    main()