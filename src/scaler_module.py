# scaler_module.py

import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def fit_scaler(data, feature_range=(-1, 1)):
    """
    Ajusta um MinMaxScaler com base nos dados fornecidos.

    Parâmetros:
    - data (array-like): Dados para ajustar o scaler.
    - feature_range (tuple): Intervalo para o escalonamento.

    Retorna:
    - scaler (MinMaxScaler): O scaler ajustado.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    data = np.array(data).reshape(-1, 1)
    scaler.fit(data)
    return scaler


def transform_data(scaler, data):
    """
    Transforma os dados usando o scaler fornecido.

    Parâmetros:
    - scaler (MinMaxScaler): O scaler a ser usado.
    - data (list ou array-like): Dados a serem transformados.

    Retorna:
    - list: Dados transformados.
    """
    data = np.array(data).reshape(-1, 1)
    data_normalized = scaler.transform(data)
    return data_normalized.flatten().tolist()


def save_scaler(scaler, filepath):
    """
    Salva o scaler em um arquivo usando pickle.

    Parâmetros:
    - scaler (MinMaxScaler): O scaler a ser salvo.
    - filepath (str): Caminho do arquivo onde o scaler será salvo.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(filepath):
    """
    Carrega um scaler de um arquivo usando pickle.

    Parâmetros:
    - filepath (str): Caminho do arquivo de onde o scaler será carregado.

    Retorna:
    - scaler (MinMaxScaler): O scaler carregado.
    """
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    return scaler
