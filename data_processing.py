import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import timeit
import pickle
import os

import pandas as pd
import numpy as np


def save_model(model, scaler, model_path='model.pkl', scaler_path='scaler.pkl'):
    with open(model_path, 'wb') as model_file, open(scaler_path, 'wb') as scaler_file:
        pickle.dump(model, model_file)
        pickle.dump(scaler, scaler_file)

def load_model(model_path='model.pkl', scaler_path='scaler.pkl'):
    model = None
    scaler = None

    if os.path.exists(model_path):
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
    else:
        print(f"Arquivo do modelo '{model_path}' não encontrado.")

    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
    else:
        print(f"Arquivo do escalador '{scaler_path}' não encontrado.")

    if model is None or scaler is None:
        raise FileNotFoundError("Um ou ambos os arquivos não foram encontrados.")

    return model, scaler

def clean_csv(df):

    # Verifica se o DataFrame tem pelo menos 2 linhas e 2 colunas
    if len(df) < 2 or df.shape[1] < 2:
        print("O DataFrame não tem linhas ou colunas suficientes.")
        return df

    while pd.isna(df.iloc[0, 1]):  # Verifica se a célula na segunda linha e segunda coluna é NaN
        print(f"Valor na segunda linha e segunda coluna: {df.iloc[1, 1]}")
        df.drop(index=0, inplace=True)  # Exclui a segunda linha
        df.reset_index(drop=True, inplace=True)  # Reseta os índices do DataFrame

        # Verifica se ainda há pelo menos 2 linhas no DataFrame após a exclusão
        if len(df) < 2:
            break

def load_data(file_path):

    # Read the CSV file
    df = pd.read_csv(file_path, sep=';', quotechar='"', skiprows=6)

    # df = clean_csv(df)
    # df.columns = df.iloc[0]
    columns_to_drop = ['Format', 'Flags']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    # df = df[1:].reset_index(drop=True)
    # Renomear colunas específicas usando um dicionário
    df = df.rename(columns={
        'Time': 'event_time',
        'Identifier (hex)': 'pgn',
        'Data (hex)': 'data'
    })
    df = df.dropna(subset=['data', 'event_time'])
    # Defina o horário base
    base_time = datetime(2024, 8, 8)  # Data arbitrária
    # Converter 'event_time' para timedelta
    df['event_time'] = pd.to_timedelta(df['event_time'])
    # Adicionar o timedelta ao horário base
    df['event_time'] = df['event_time'].apply(lambda x: base_time + x)
    # Converter 'event_time' para Unix Timestamp
    df['event_time'] = df['event_time'].astype(np.int64) // 10 ** 9
    return df

def process_data(data):
    if isinstance(data, str):
        return data.split(' ')
    return []

def generate_limited_square_wave_data(num_points, start_time, cycles):
    event_time = [start_time + timedelta(seconds=i) for i in range(num_points)]
    wave = []
    for i in range(num_points):
        cycle_position = (i % (cycles * 100))
        if cycle_position < (100 * cycles):
            wave.append(1 if (cycle_position // 100) % 2 == 0 else 2)
        else:
            wave.append(0)
    return pd.DataFrame({'event_time': event_time, 'data': wave})


def generate_limited_sawtooth_wave_data(num_points, start_time, cycles, period=80):
    event_time = [start_time + timedelta(seconds=i) for i in range(num_points)]
    wave = []
    for i in range(num_points):
        cycle_position = (i % (cycles * period))
        if cycle_position < (period * cycles):
            wave.append((cycle_position % period) + 3)
        else:
            wave.append(0)
    return pd.DataFrame({'event_time': event_time, 'data': wave})


def process_real_data(df):
    df['data'] = df['data'].apply(process_data)
    df_processed = df[df['pgn'] == '32D'].copy()
    dados_indice_2 = []
    for idx, row in df_processed.iterrows():
        event_time = row['event_time']
        partes = row['data']
        if len(partes) > 2:
            try:
                decimal_value = int(partes[2], 16)
                dados_indice_2.append((event_time, decimal_value))
            except ValueError:
                print(f"Valor inválido para conversão em decimal: {partes[2]}")
        else:
            print(f"Dados insuficientes para a linha com event_time: {event_time}")

    df_final = pd.DataFrame(dados_indice_2, columns=['event_time', 'data'])
    #df_final.set_index('event_time', inplace=True)
    #df_final['event_time'] = df_final['event_time'].astype(np.int64) // 10 ** 9
    df_final['dummy_feature'] = 0
    df_final['pgn'] = '32D'
    print(df_final)
    return df_final


def combine_data(df_wave, df_processed):
    df_train = pd.concat([df_wave, df_processed])
    return df_train

def train_model(df_train):
    X = df_train[['event_time', 'dummy_feature']]
    y = df_train['data']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier()
    start = timeit.default_timer()
    model.fit(X_train_scaled, y_train)
    end = timeit.default_timer()

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy = {accuracy:.2f}, Time = {end - start:.4f} seconds")
    print("Confusion Matrix:\n", conf_matrix)
    print(f'Importância das características: {model.feature_importances_}')

    return model, scaler

def plot_data(df, y_pred, title):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['data'], linestyle='-', color='b', linewidth=0.5, label='Real Data')
    plt.scatter(df.index[:len(y_pred)], y_pred, color='r', marker='x', label='Predicted Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def count_peaks_and_troughs(data):
    peaks = []
    troughs = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            peaks.append(data[i])
        elif data[i] < data[i - 1] and data[i] < data[i + 1]:
            troughs.append(data[i])
    return peaks, troughs

def analyze_pgn(df, cycles, model, scaler):
    def process_data(data):
        # Remove espaços extras entre os bytes e divide em uma lista
        data = data.split()

        # Processa cada byte, substituindo os espaços vazios por '00'
        processed_data = []
        for byte in data:
            #print(data)
            #print(byte)
            if byte.strip() == '':
                processed_data.append('00')
            else:
                processed_data.append(byte)

        # Adiciona '00' ao final da lista até ter 8 elementos
        while len(processed_data) < 8:
            processed_data.append('00')

        # Garante que a lista tenha exatamente 8 bytes
        return processed_data[:8]

    print(df['data'])
    df['data'] = df['data'].apply(process_data)
    unique_pgns = sorted(df['pgn'].unique())
    best_pgns = []
    print(df['data'])

    for pgn in unique_pgns:
        df_pgn = df[df['pgn'] == pgn].copy()

        # Verificar se todos os dados têm 8 bytes
        df_pgn = df_pgn[df_pgn['data'].apply(lambda x: isinstance(x, list) and len(x) == 8)]

        if df_pgn.empty:
            continue

        # Adiciona a coluna 'dummy_feature' para treinamento
        df_pgn['dummy_feature'] = 0

        for i in range(8):  # Assumindo que cada dado tem 8 bytes
            df_pgn[f'byte_{i}'] = df_pgn['data'].apply(lambda x: int(x[i], 16))

            byte_values = df_pgn[f'byte_{i}'].values
            peaks, troughs = count_peaks_and_troughs(byte_values)

            if len(peaks) <= 6 and len(troughs) <= 6:
                X = df_pgn[['event_time', 'dummy_feature']]
                y = df_pgn[f'byte_{i}'].values

                if len(X) > cycles and len(set(y)) > 1:  # Garantir que temos dados suficientes e mais de uma classe
                    X_real_scaled = scaler.transform(X)
                    y_pred = model.predict(X_real_scaled)
                    accuracy = accuracy_score(y, y_pred)

                    best_pgns.append((pgn, i, accuracy, y_pred, X, y))  # Adiciona as informações dos melhores pgns

    # Ordena por acurácia e retorna as 10 melhores séries
    best_pgns.sort(key=lambda x: x[2], reverse=True)
    return best_pgns[:10]


def plot_best_pgns(best_pgns, df):
    for pgn, i, accuracy, y_pred, X, y in best_pgns:
        df_pgn = df[df['pgn'] == pgn].copy()
        df_pgn = df_pgn[df_pgn['data'].apply(lambda x: isinstance(x, list) and len(x) == 8)]
        df_pgn[f'byte_{i}'] = df_pgn['data'].apply(lambda x: int(x[i], 16))

        # Verifica se a quantidade de y_pred corresponde ao tamanho de df_pgn
        if len(y_pred) != len(df_pgn):
            print(f"Warning: Length mismatch for PGN {pgn}, byte {i}")
            continue

        # Assegura que 'event_time' e 'y_pred' tenham o mesmo comprimento
        event_times = df_pgn['event_time'].iloc[:len(y_pred)]

        # Plota os dados
        plt.figure(figsize=(10, 6))
        plt.plot(df_pgn['event_time'], df_pgn[f'byte_{i}'], linestyle='-', color='b', linewidth=0.5, label='Real Data')
        plt.scatter(event_times, y_pred, color='r', marker='x', label='Predicted Data')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'PGN {pgn}, Byte {i} - Accuracy: {accuracy:.2f}')
        plt.legend()
        plt.show()
# Função para plotar os dados reais e as previsões

