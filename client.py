import torch as T
import flwr as fl
from argparse import ArgumentParser
from src.preprocessing import Participant
from src.dataset import PecanDataModule
from src.regressors.rnn_regressor import ConsumptionRNNRegressor
from src.fl_client import FlowerClient

def main(args):
    participant = Participant(path=args.root_path, _id=args._id, sequence_length=args.sequence_length)
    train_sequence, validation_sequence, test_sequence = (participant.readings['train_sequence'],
                                                          participant.readings['val_sequence'],
                                                          participant.readings['test_sequence'])
    print(f"[!] - Training model using _id: {participant.readings['_id']}")
    data_module = PecanDataModule(device=args.device,
                    train_sequences=train_sequence,
                    test_sequences=test_sequence,
                    val_sequences=validation_sequence,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers)

    data_module.setup()

    regressor = ConsumptionRNNRegressor(device=args.device,
                                        n_features=participant.readings['n_features'],
                                        lr=args.lr,
                                        n_hidden=args.n_hidden,
                                        n_layers=args.n_layers,
                                        dropout=args.dropout)

    client = FlowerClient(regressor=regressor,
                          train_loader=data_module.train_dataloader(),
                          val_loader=data_module.val_dataloader(),
                          test_loader=data_module.test_dataloader(),
                          device=args.device, _id=args._id)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)




if __name__ == "__main__":
    parser = ArgumentParser(description='[Pecan Street Dataport] Forecasting the energy consumption of Pecan Street')
    parser.add_argument('--sequence_length', type=int, default=60)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--root_path', type=str, default='data/')
    parser.add_argument('--_id', type=int, default=661)

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

    args = parser.parse_args()
    args.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    args.gpu = T.cuda.device_count()

    main(args=args)