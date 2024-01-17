import copy
import torch as T
import torch.nn as nn
import wandb
import time
import numpy as np
from collections import OrderedDict
from logging import DEBUG, INFO
from src.utils.logger import log
from typing import *
from src.fl.server.proxy import ClientProxy
from src.fl.server.manager import SimpleClientManager, ClientManager
from torch.utils.data import DataLoader
from src.fl.federated_learning import FederatedLearning
from src.fl.aggregation.aggregator import Aggregator
from src.fl.history.history import History
from src.models.model_serializer import ModelSerializer
from tqdm import tqdm

class Server:
    def __init__(self, client_proxies: List[ClientProxy],
                 client_manager: Optional[ClientManager] = None,
                 aggregation: Optional[str]=None,
                 aggregation_params: Optional[Dict[str, Union[str, int, float, bool]]]=None,
                 weighted_loss_fn: Optional[Callable]=None,
                 weighted_metrics_fn: Optional[Callable]=None,
                 val_loader: Optional[DataLoader]=None,
                 local_params_fn: Optional[Callable]=None,
                 server_model=None,
                 server_config: Optional[OrderedDict[str, any]]=None, logger=None):

        self.global_model = None
        self.logger = logger
        self.best_model = None
        self.server_model = server_model
        self.best_loss, self.best_epoch = np.inf, -1
        self.server_config = server_config
        self.client_proxies = client_proxies
        self._initialize_client_manager(client_manager)

        self.weighted_loss = weighted_loss_fn if weighted_metrics_fn is not None else FederatedLearning.weighted_loss_avg
        self.weighted_metrics = weighted_metrics_fn if weighted_metrics_fn is not None else FederatedLearning.weighted_metrics_avg

        if aggregation is None:
            aggregation = "fedavg"
        self.aggregator = Aggregator(aggregation_alg=aggregation, params=aggregation_params)
        log(INFO, f"Aggregation algorithm: {repr(self.aggregator)}")
        self.val_loader = val_loader
        self.local_params_fn = local_params_fn

    def _initialize_client_manager(self, client_manager) -> None:
        """Init. client manager"""
        log(INFO, f"Initializing client manager...")
        if client_manager is None:
            self.client_manager: ClientManager = SimpleClientManager()
        else:
            self.client_manager = client_manager

        log(INFO, "Registering clients...")
        for client in self.client_proxies:
            self.client_manager.register(client)

        log(INFO, "Client manager initialized")


    def fit(self, num_rounds: int, fraction: float, fraction_args: Optional[Callable]=None,
            use_carbotracker: bool=False) -> Tuple[List[np.ndarray], History]:
        history = History()

        self.evaluate_round(fl_round=0, history=history)
        log(INFO, "Starting FL rounds")

        start_time = time.time()
        for fl_round in range(1, num_rounds + 1):

            self.fit_round(fl_round=fl_round, fraction=fraction, fraction_args=fraction_args, history=history)

            self.evaluate_round(fl_round=fl_round, history=history)
        end_time = time.time()
        log(INFO, f"Time passed: {end_time - start_time} seconds.")
        log(INFO, f"Best global model found on fl_round={self.best_epoch} with loss={self.best_loss}")

        return self.best_model, history

    def fit_round(self, fl_round: int, fraction: float, fraction_args: Optional[Callable], history: History) -> None:
        """Perform a federated round, i.e.,
                    1) Select a fraction of available clients.
                    2) Instruct selected clients to execute local training.
                    3) Receive updated parameters from clients and their corresponding evaluation
                    4) Aggregate the local learned weights.
        """
        if self.local_params_fn:
            for client in self.client_proxies:
                client.set_train_parameters(self.local_params_fn(fl_round), verbose=True)

        # Step1: selection a fraction of available clients
        selected_clients = self.sample_clients(fl_round, fraction, fraction_args)

        # Step2-3: perform local training and receive updated parameters
        num_train_examples: List[int] = []
        num_test_examples: List[int] = []
        train_losses: Dict[str, float] = dict()
        test_losses: Dict[str, float] = dict()
        all_train_metrics: Dict[str, Dict[str, float]] = dict()
        all_test_metrics: Dict[str, Dict[str, float]] = dict()
        results: List[Tuple[List[np.ndarray], int]] = []

        for client in selected_clients:
            res = self.fit_client(fl_round, client)
            model_params, num_train, train_loss, train_metrics, num_test, test_loss, test_metrics = res
            num_train_examples.append(num_train)
            num_test_examples.append(num_test)
            train_losses[client.cid] = train_loss
            test_losses[client.cid] = test_loss
            all_train_metrics[client.cid] = train_metrics
            all_test_metrics[client.cid] = test_metrics
            results.append((model_params, num_train))

        history.add_local_train_loss(train_losses, fl_round)
        history.add_local_train_metrics(all_train_metrics, fl_round)
        history.add_local_test_loss(test_losses, fl_round)
        history.add_local_test_metrics(all_test_metrics, fl_round)

        # Step4: aggregate local models
        self.global_model = self.aggregate_models(fl_round, results)
        if self.best_model is None:
            self.best_model = copy.deepcopy(self.global_model)


    def aggregate_models(self, fl_round: int, results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        log(INFO, f"[Global round: {fl_round}] Aggregating local models...")
        aggregated_params = self.aggregator.aggregate(results, self.global_model)
        return aggregated_params



    def fit_client(self, fl_round: int, client: ClientProxy):
        if fl_round == 1:
            fit_res = client.fit(None, int(client.cid))
        else:
            fit_res = client.fit(model=self.global_model, cid=int(client.cid))
        return fit_res

    def sample_clients(self, fl_round: int, fraction: float, fraction_args: Optional[Callable]=None) -> List[ClientProxy]:
        """Sample available clients"""
        if fraction_args is not None:
            fraction: float = fraction_args(fl_round)
        selected_clients: List[ClientProxy] = self.client_manager.sample(fraction)
        return selected_clients

    def evaluate_round(self, fl_round: int, history: History):
        """Evaluate the global model"""
        num_train_examples: List[int] = []
        train_losses: Dict[str, float] = dict()
        train_metrics: Dict[str, Dict[str, float]] = dict()
        num_test_examples: List[int] = []
        test_losses: Dict[str, float] = dict()
        test_metrics: Dict[str, Dict[str, float]] = dict()

        if fl_round == 0:
            self.global_model: List[np.ndarray] = self._get_initial_model()

        if self.val_loader:
            random_client = self.client_manager.sample(0.)[0]
            num_instances, loss, eval_metrics = random_client.evaluate(data=self.val_loader, model=self.global_model)
            num_test_examples = [num_instances]
            test_metrics["Server"] = eval_metrics
            test_losses['Server'] = loss

        else:
            log(INFO, f"Evaluating global model in round: {fl_round}")

            for cid, client_proxy in (pbar := tqdm(self.client_manager.all().items(), total=len(self.client_manager))):
                num_train_instances, train_loss, train_eval_metrics = client_proxy.evaluate(model=self.global_model, method="train")
                num_train_examples.append(num_train_instances)
                train_losses[cid] = train_loss
                train_metrics[cid] = train_eval_metrics

                num_test_instances, test_loss, test_eval_metrics = client_proxy.evaluate(model=self.global_model, method="test")
                num_test_examples.append(num_test_instances)
                test_losses[cid] = test_loss
                test_metrics[cid] = test_eval_metrics
                self.logger.log({'fl_round': fl_round, 'cid': cid,'global_MSE': test_metrics[cid]['MSE']})
                pbar.set_postfix({'cid': cid, 'metrics': test_metrics[cid]})

        history.add_global_train_losses(self.weighted_loss(num_train_examples, list(train_losses.values())))
        history.add_global_train_metrics(self.weighted_metrics(num_train_examples, train_metrics))

        history.add_global_test_losses(self.weighted_loss(num_test_examples, list(test_losses.values())))

        if history.global_test_losses[-1] <= self.best_loss:
            self.best_loss = history.global_test_losses[-1]
            self.best_epoch = fl_round
            self.best_model = copy.deepcopy(self.global_model)

            params_dict = zip(self.server_model.state_dict(), self.best_model)
            state_dict = OrderedDict({k: T.Tensor(v) for k, v in params_dict})
            temp_model = copy.deepcopy(self.server_model)
            temp_model.load_state_dict(state_dict, strict=True)
            fl_model_serializer = ModelSerializer(model=temp_model, path=f"etc/ckpt/federated/")
            fl_model_serializer.save('federated_model.h5')

        history.add_global_test_metrics(self.weighted_metrics(num_test_examples, test_metrics))



    def _get_initial_model(self) -> List[np.ndarray]:
        """Get initial params from a random client"""
        random_client = self.client_manager.sample(0.)[0]
        client_model = random_client.get_parameters()
        return client_model
