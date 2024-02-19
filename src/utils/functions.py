import matplotlib.pyplot as plt
import os
import json
import numpy as np
import torch
from typing import Dict, List
from logging import INFO
import torch as T
from collections import OrderedDict
from src.utils.logger import log




def mkdir_if_not_exists(default_save_path: str):
    if not os.path.exists(default_save_path):
        os.mkdir(default_save_path)

def save_json_file(save_path: str, values):
    try:
        json_obj = json.dumps(values)
    except:
        values = {str(key): {str(key_i): value_i for key_i, value_i in value.items()}  for key, value in values.items()}
        json_obj = json.dumps(values)
    with open(save_path, 'w') as file:
        file.write(json_obj)
    log(INFO, f"json file created on {save_path}")



def get_params(alg):
    if alg == "fedprox":
        return {"mu": 0.01}
    elif alg == "fednova":
        return {"rho": 0.}
    elif alg == "fedadagrad":
        return {"beta_1": 0., "eta": 0.1, "tau": 1e-2}
    elif alg == "fedyogi":
        return {"beta_1": 0.9, "beta_2": 0.99, "eta": 0.01, "tau": 1e-3}
    elif alg == "fedadam":
        return {"beta_1": 0.9, "beta_2": 0.99, "eta": 0.01, "tau": 1e-3}
    elif alg == "fedavgm":
        return {"server_momentum": 0., "server_lr": 1.}
    else:
        return None

def plot_train_curve(train_history, test_history, title, fig_name):
    plt.title(title)
    plt.plot(train_history, label='Train')
    plt.plot(test_history, label='Test')
    plt.legend()
    plt.show()
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/results/')
    plt.savefig(f'etc/results/{fig_name}.png')
    plt.close()

def plot_global_losses(values: List[float]):
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/results/')
    mkdir_if_not_exists('etc/results/imgs')

    plt.title(f'Loss per round')
    plt.plot(values, marker='^')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig(f'etc/results/imgs/global_loss.png')
    plt.close()


def plot_global_metrics(history: Dict[str, List[np.float64]]):
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/results/')
    mkdir_if_not_exists('etc/results/imgs')

    for metric, value in history.items():
        plt.title(f'{metric} evaluation per round')
        plt.plot(value, marker='^')
        plt.xlabel('Round')
        plt.ylabel('Error')
        plt.show()
        plt.savefig(f'etc/results/imgs/{metric}.png')
        plt.close()

def plot_local_train_rounds(history):
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/results/')
    mkdir_if_not_exists('etc/results/imgs')

    cids = [participant for participant in history.keys()]
    counts = [len(participant) for participant in history.values()]
    plt.title('Count of local trainings')
    plt.bar(cids, counts)
    plt.xlabel('Participants')
    plt.ylabel('Training times')
    plt.xticks(rotation=45)
    plt.show()
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/results/')
    plt.savefig(f'etc/results/imgs/training_times.png')
    plt.close()

def plot_test_prediction(y_true, y_pred, cid, model_name):
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/results/')
    mkdir_if_not_exists('etc/results/imgs/')
    mkdir_if_not_exists(f'etc/results/imgs/{model_name}')
    mkdir_if_not_exists(f'etc/results/imgs/{model_name}/preds/')
    plt.title(f"Prediction of {cid}")
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.savefig(f'etc/results/imgs/{model_name}/preds/{cid}.png')
    plt.close()

