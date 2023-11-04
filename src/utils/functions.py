import pandas as pd
import os
from tqdm import tqdm

def create_sequences(input_data: pd.DataFrame, target_column, sequence_lenght):
    sequences = []
    data_size = len(input_data)

    for i in tqdm(range(data_size - sequence_lenght)):
        sequence = input_data[i:i+sequence_lenght]
        label_position = i + sequence_lenght
        label = input_data.iloc[label_position][target_column]
        # del sequence[target_column]
        sequences.append((sequence, label))

    return sequences


def mkdir_if_not_exists(default_save_path: str):
    if not os.path.exists(default_save_path):
        os.mkdir(default_save_path)
