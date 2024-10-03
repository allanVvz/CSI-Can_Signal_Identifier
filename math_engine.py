from datetime import datetime
import numpy as np
import pandas as pd
from numpy import cumsum

def fileToDataframe(file_path):
    # Exclusivamente .CSV exportado pelo XXAT
    df = pd.read_csv(file_path, sep=';', quotechar='"', skiprows=6)
    columns_to_drop = ['Format', 'Flags']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    df = df.dropna(subset=['Data (hex)', 'Time'])
    df['Time'] = process_time(df['Time'])

    # Converte 'Identifier (hex)' de hexadecimal para inteiro
    df['Identifier (hex)'] = df['Identifier (hex)'].apply(lambda x: int(x, 16))

    # Expande os bytes em colunas separadas
    byte_columns = df['Data (hex)'].apply(process_data)
    byte_columns = pd.DataFrame(byte_columns.tolist(), index=df.index, columns=[f'byte_{i + 1}' for i in range(8)])
    print(byte_columns)

    # Junta os novos dados com o DataFrame original
    df = df.drop(columns=['Data (hex)'])
    df = pd.concat([df, byte_columns], axis=1)

    df = df.rename(columns={
        'Time': 'event_time',
        'Identifier (hex)': 'pgn'
    })

    return df


# Event_time To Timestamp
def process_time(time):
        # Defina o horário base
        base_time = datetime(2024, 8, 8)  # Data arbitrária
        # Converter 'event_time' para timedelta
        time = pd.to_timedelta(time)
        # Adicionar o timedelta ao horário base
        time = time.apply(lambda x: base_time + x)
        # Converter 'event_time' para Unix Timestamp
        time = time.astype(np.int64) // 10 ** 9

        return time

def process_data(hex_data, type = 'df'):
    if type == 'row':

        if isinstance(hex_data, str):
            return hex_data.split(' ')
        return []

    elif type == 'df':
        # Verifica se a entrada é uma string
        if isinstance(hex_data, str):
            # Divide a string em bytes separados por espaços
            byte_list = hex_data.split(' ')

            # Verifica se a lista contém exatamente 8 itens
            if len(byte_list) < 8:
                byte_list.extend(['00'] * (8 - len(byte_list)))

            # Converte os valores de hexadecimal para decimal
            byte_list = [int(byte, 16) for byte in byte_list]

            # Converte a lista em um DataFrame com uma linha e 8 colunas
            df = pd.DataFrame([byte_list])

            return byte_list  # Retorna a lista para ser expandida em colunas
        else:
            raise ValueError("A entrada deve ser uma string hexadecimal.")

def normalized_df(df):
    df.iloc[:, 2:10] = df.iloc[:, 2:10].astype(np.float64) / 255.0
    return df


def filter_and_label_dataframe(df, pgn, byte, label):

    byte_column = f'byte_{byte}'

    if pgn not in df['pgn'].unique():
        raise ValueError(f"O PGN '{pgn}' não existe no DataFrame.")

    if byte_column not in df.columns:
        raise ValueError(f"A coluna '{byte_column}' não existe no DataFrame.")

    # Filtrar o DataFrame pelo PGN e byte
    filtered_df = df[df['pgn'] == pgn].copy()

    # Adicionar uma nova coluna de rótulo com o valor especificado
    filtered_df = filtered_df[[byte_column]].copy()
    filtered_df['label'] = label

    return filtered_df


def get_Xsequence(n_timesteps, df, byte_index, start_index):
    X = df.iloc[start_index:start_index + n_timesteps][byte_index].values
    X = X.reshape(1, n_timesteps, 1)  # Redimensionar para a entrada do modelo LSTM
    return X

def get_sequence(n_timesteps, df, byte_index, i):

    # Concatene "byte_" com o valor do índice fornecido para formar o nome da coluna
    column_name = byte_index

    if df is None or column_name not in df.columns:
        raise ValueError(f"A coluna '{column_name}' não existe no DataFrame.")

    # Extrair os valores de todas as linhas da coluna especificada
    X = df[column_name].values.flatten()
    y = df[column_name].values.flatten()
    # Certifique-se de que o tamanho do array X seja compatível com n_timesteps
    if len(X) < i + n_timesteps:
        raise ValueError(
            f"The length of X ({len(X)}) is less than the required range (i + n_timesteps = {i + n_timesteps}).")

    # Truncate X to have exactly n_timesteps elements
    X = X[i:i + n_timesteps]
    y = y[i:i + n_timesteps]
    # reshape input and output data to be suitable for LSTMs
    X = X.reshape(1, n_timesteps, 1)
    y = y.reshape(1, n_timesteps, 1)

    return X, y

def count_peaks_and_troughs(data):
    peaks = []
    troughs = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            peaks.append(data[i])
        elif data[i] < data[i - 1] and data[i] < data[i + 1]:
            troughs.append(data[i])
    return peaks, troughs






