import os
from tkinter import filedialog, messagebox

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Input
from sklearn.metrics import accuracy_score, confusion_matrix
import timeit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np

def carregar_modelo(nome_arquivo):
    try:
        with open(nome_arquivo, 'rb') as arquivo:
            model = pickle.load(arquivo)
        print(f"Modelo carregado com sucesso de {nome_arquivo}.")
        return model
    except FileNotFoundError:
        print(f"Erro: O arquivo {nome_arquivo} não foi encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

def process_data(hex_data, scaler_filename='scaler.pkl'):
    # Verifica se a entrada é uma string
    if isinstance(hex_data, str):
        byte_list = hex_data.split(' ')

        # Verifica se a lista contém exatamente 8 itens
        if len(byte_list) < 8:
            byte_list.extend(['00'] * (8 - len(byte_list)))

        # Converte os valores de hexadecimal para decimal
        byte_list = [int(byte, 16) for byte in byte_list]

        # Procura o scaler na pasta raiz
        scaler_path = os.path.join(os.getcwd(), scaler_filename)
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"O arquivo {scaler_filename} não foi encontrado na pasta raiz.")

        # Carrega o scaler salvo
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Normaliza os valores usando o scaler carregado
        byte_array = np.array(byte_list).reshape(-1, 1)  # Ajusta o formato para o scaler
        byte_list_scaled = scaler.transform(byte_array).flatten()  # Aplica normalização e achata para 1D

        return byte_list_scaled  # Retorna a lista para ser expandida em colunas
    else:
        raise ValueError("A entrada deve ser uma string hexadecimal.")


# Função para verificar se uma coluna tem valores constantes
def coluna_eh_constante(coluna):
    return len(np.unique(coluna)) == 1


def analyse_archive(self):
    data_list = []
    df = self.can_data

    # Exibir as primeiras linhas do DataFrame para verificar a estrutura
    print("Primeiras linhas do DataFrame:")
    print(self.can_data.head())

    # Iterar sobre cada PGN único no DataFrame
    for pgn in df['pgn'].unique():
        df_pgn = df[df['pgn'] == pgn].copy()  # Filtrar o DataFrame para o PGN atual

        print(f"\nAnalisando PGN: {pgn}")
        print("Primeiras linhas do DataFrame para este PGN:")
        print(df_pgn.head())  # Exibir o DataFrame filtrado por PGN

        # Iterar sobre as colunas de bytes (de 'byte_1' até 'byte_8')
        for i in range(1, 9):  # Para os bytes de 1 a 8
            byte_column = f'byte_{i}'  # Supondo que os bytes estão nomeados como byte_1, byte_2, ..., byte_8

            # Verificar se a coluna existe no DataFrame
            if byte_column not in df_pgn.columns:
                print(f"{byte_column} não encontrada no DataFrame.")
                continue

            # Verificar se a coluna tem valores constantes e descartar se tiver
            if coluna_eh_constante(df_pgn[byte_column]):
                print(f"{byte_column} descartada por ser constante.")
                continue

            print(f"Processando {byte_column}")

            # Obter os valores de cada byte
            X = df_pgn[byte_column].values

            if len(X) > 1:  # Garantir que temos dados suficientes
                # Armazenar os dados para processamento posterior
                data_list.append({'pgn': pgn, 'byte_column': byte_column, 'data': X})
            else:
                print(f"Dados insuficientes em {byte_column} para PGN {pgn}.")

    # Criar um DataFrame com os dados coletados
    data_df = pd.DataFrame(data_list)

    return data_df

def prepare_sequences(data_df, target_size=900):
    validated_pgns = []

    for index, row in data_df.iterrows():
        pgn = row['pgn']
        byte_column = row['byte_column']
        sequence = row['data']

        # Ajustar o tamanho da sequência usando duplicação ou decimação
        if len(sequence) != target_size:
            if len(sequence) > target_size:
                # Aplicar decimação dinâmica
                adjusted_sequence = decimacao_dinamica(sequence, target_size)
            else:
                # Aplicar duplicação de valores individualmente
                adjusted_sequence = duplicar_valores_individualmente(sequence, target_size)
        else:
            adjusted_sequence = sequence

        # Converter a sequência ajustada para 2D se necessário
        adjusted_sequence = adjusted_sequence.reshape(-1, 1)

        validated_pgns.append({'pgn': pgn, 'byte_column': byte_column, 'sequence': adjusted_sequence})

    # Converter para DataFrame
    validated_df = pd.DataFrame(validated_pgns)

    return validated_df

def decimacao_dinamica(array, target_size):
    array = np.array(array).flatten()
    current_size = len(array)

    if current_size <= target_size:
        return array  # Não precisa decimar

    step = max(current_size // target_size, 1)
    decimated_array = array[::step][:target_size]
    return decimated_array

def classify_sequences(model, validated_df):
    classified_results = []

    for index, row in validated_df.iterrows():
        pgn = row['pgn']
        byte_column = row['byte_column']
        sequence = row['sequence']

        # Fazer predições para a sequência
        y_pred = model.predict(sequence)

        # Calcular a acurácia (ou outro critério relevante)
        accuracy = np.mean(y_pred)

        classified_results.append({'pgn': pgn, 'byte_column': byte_column, 'accuracy': accuracy, 'predictions': y_pred})

    # Converter para DataFrame
    classified_df = pd.DataFrame(classified_results)

    return classified_df

def validate_peaks(classified_df):
    processed_results = []

    for index, row in classified_df.iterrows():
        pgn = row['pgn']
        byte_column = row['byte_column']
        accuracy = row['accuracy']
        sequence = row['predictions']

        # Garantir que a sequência seja um array 1D
        seq = sequence.flatten()

        # Realizar análise de picos
        peaks, _ = find_peaks(seq)
        num_peaks = len(peaks)

        # Verificar o número de picos
        if 3 <= num_peaks <= 8:
            # Manter este PGN
            processed_results.append({'pgn': pgn, 'byte_column': byte_column, 'accuracy': accuracy, 'num_peaks': num_peaks, 'sequence': sequence})
        else:
            # Descartar este PGN
            print(f"PGN {pgn}, {byte_column} descartado por ter {num_peaks} picos.")

    # Converter para DataFrame
    processed_df = pd.DataFrame(processed_results)

    return processed_df

def duplicar_valores_individualmente(array, target_size):
    array = np.array(array).flatten()  # Achata a array para 1D
    current_size = len(array)

    if current_size >= target_size:
        return array[:target_size]  # Se já tiver o tamanho ou maior, trunca a array

    # Calcula o número de repetições por valor como um número de ponto flutuante
    repeats_per_value = target_size / current_size

    # Lista para armazenar o número de repetições para cada valor
    repeats = []
    cumulative_error = 0.0

    for i in range(current_size):
        # Calcula o número de repetições para este valor
        repeat = int(repeats_per_value)
        cumulative_error += (repeats_per_value - repeat)

        # Ajusta a repetição quando o erro acumulado excede 1
        if cumulative_error >= 1.0:
            repeat += 1
            cumulative_error -= 1.0

        repeats.append(repeat)

    # Ajusta o total de repetições para corresponder ao tamanho alvo
    total_repeats = sum(repeats)
    index = 0
    while total_repeats < target_size:
        repeats[index % current_size] += 1
        total_repeats += 1
        index += 1
    while total_repeats > target_size:
        repeats[index % current_size] -= 1
        total_repeats -= 1
        index += 1

    # Repete cada valor de acordo com o número calculado
    repeated_array = np.repeat(array, repeats)

    return repeated_array


def plot_best_pgns(best_results_df):
    for index, row in best_results_df.iterrows():
        pgn = row['pgn']
        byte_column = row['byte_column']
        accuracy = row['accuracy']
        sequence = row['sequence'].flatten()  # Ensure sequence is 1D

        plt.figure(figsize=(10, 6))
        plt.plot(sequence, label=f"PGN {pgn}, {byte_column} (Acurácia: {accuracy:.2f})")
        plt.xlabel('Timestep')
        plt.ylabel('Valor')
        plt.title(f"Melhor sequência para PGN {pgn}, {byte_column}")
        plt.legend()
        plt.show()


def process_best_pgns(best_pgns):
    # Organizar best_pgns por ordem decrescente de acurácia
    best_pgns_sorted = sorted(best_pgns, key=lambda x: x[2], reverse=True)

    processed_pgns = []

    for pgn in best_pgns_sorted:
        pgn_number, byte_column, accuracy, sequence = pgn

        # Garantir que a sequência seja um array 1D
        seq = sequence.flatten()

        # Realizar análise de picos
        peaks, _ = find_peaks(seq)
        num_peaks = len(peaks)

        # Verificar o número de picos
        if 3 <= num_peaks <= 8:
            # Manter este pgn
            processed_pgns.append(pgn)
        else:
            # Descartar este pgn
            print(f"PGN {pgn_number}, {byte_column} descartado por ter {num_peaks} picos.")

    return processed_pgns


def aplicar_scaler_salvo(byte_list, scaler_filename='scaler.pkl'):
    # Procura o scaler na pasta raiz
    scaler_path = os.path.join(os.getcwd(), scaler_filename)

    if not os.path.exists(scaler_path):
        print(f"Erro: O arquivo {scaler_filename} não foi encontrado na pasta raiz.")
        return None

    # Carrega o scaler salvo
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Converte byte_list em um array NumPy e ajusta a forma
    byte_array = np.array(byte_list).reshape(-1, 1)

    # Aplica apenas o transform
    byte_list_scaled = scaler.transform(byte_array).flatten()

    return byte_list_scaled