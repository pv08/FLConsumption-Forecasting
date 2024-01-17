import random
from typing import *
from logging import INFO
from src.utils.logger import log
from abc import ABC, abstractmethod
from src.fl.server.proxy import ClientProxy
class ClientManager(ABC):
    @abstractmethod
    def num_available(self, verbose: bool=True) -> int:
        """Returns the number of available clients"""

    @abstractmethod
    def register(self, client: ClientProxy) -> bool:
        """Register client to the pool of clients"""

    @abstractmethod
    def unregister(self, client: ClientProxy) -> None:
        """Unregister ClientProxy instance"""

    @abstractmethod
    def all(self) -> Dict[Union[str, int], ClientProxy]:
        """Return all available clients"""

    @abstractmethod
    def sample(self, c: float) -> List[ClientProxy]:
        """Sample a number of ClientProxy instances"""


class SimpleClientManager(ClientManager):
    def __init__(self):
        self.clients: Dict[str, ClientProxy] = {}

    def __len__(self):
        return len(self.clients)

    def num_available(self, verbose: bool=True) -> int:
        if verbose:
            log(INFO, f"Number of available clients: {len(self)}")
        return len(self)

    def register(self, client: ClientProxy) -> bool:
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        log(INFO, f"Registered client with id: {client.cid}")
        return True

    def unregister(self, client: ClientProxy) -> None:
        if client.cid in self.clients:
            del self.clients[client.cid]
            log(INFO, f"Unregistered client with id: {client.cid}")

    def all(self) -> Dict[Union[str, int], ClientProxy]:
        return self.clients

    def sample(self, c: float) -> List[ClientProxy]:
        available_clients = list(self.clients.keys())
        if len(available_clients) == 0:
            log(INFO, f"Cannot sample clients. The number of available clients is 0")
            return []
        num_selection = int(c * self.num_available(verbose=True))
        if num_selection == 0:
            num_selection = 1
        if num_selection > self.num_available(verbose=True):
            num_selection = self.num_available(verbose=True)
        sampled_clients = random.sample(available_clients, num_selection)
        log(INFO, f"Parameter c={c}. Sampled {num_selection} client(s): {sampled_clients}")
        return [self.clients[cid] for cid in sampled_clients]
