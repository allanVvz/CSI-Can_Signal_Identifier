import os
import numpy as np
from numpy import array
from numpy import cumsum
from matplotlib import pyplot
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Input
from tkinter import filedialog
#from tensorflow.keras.models import load_model
import pickle
from math_engine import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import timeit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from random import random  # Importando o módulo random


# create a sequence classification instance
# def get_sequence(n_timesteps):
#     # create a sequence of random numbers in [0,1]
#     X = array([random() for _ in range(n_timesteps)])
#     print(X)
#     # calculate cut-off value to change class values
#     limit = n_timesteps / 4.0
#     # determinar o resultado da classe para cada item na sequência cumulativa
#     y = array([0 if x < limit else 1 for x in cumsum(X)])
#     # remodelar dados de entrada e saída para que sejam adequados para LSTMs
#     X = X.reshape(1, n_timesteps, 1)
#     y = y.reshape(1, n_timesteps, 1)
#     return X, y
# Funções para integrar com o Tkinter


def save_model(model, scaler, model_path='model.pkl', scaler_path='scaler.pkl'):
    """Salva o modelo no caminho especificado."""
    with open(model_path, 'wb') as model_file, open(scaler_path, 'wb') as scaler_file:
        pickle.dump(model, model_file)
        pickle.dump(scaler, scaler_file)
    print(f"Modelo salvo em {model_path}")

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

def get_lstm_model(n_timesteps, backwards):
    model = Sequential()
    model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True, go_backwards=backwards))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def get_bi_lstm_model(n_timesteps, mode):
    model = Sequential()
    model.add(Input(shape=(n_timesteps, 1)))
    model.add(Bidirectional(LSTM(20, return_sequences=True), merge_mode=mode))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def train_sequence_model(df, byte_index):
    n_timesteps = 250
    num_iterations = 250
    byte_index = f'byte_{byte_index}' # Supondo que 'byte_1' é a coluna relevante
    model_filename = 'model.pkl'
    scaler_filename = 'scaler.pkl'

    # Dividir o DataFrame em conjunto de treino e teste
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

    df_train_sorted = df_train.sort_index().copy()
    df_test_sorted = df_test.sort_index().copy()

    # Escalonamento dos dados
    scaler = StandardScaler()
    #
    # # Escalonando a coluna relevante e convertendo para float
    # df_train_sorted[byte_index] = scaler.fit_transform(df_train_sorted[[byte_index]].astype(float)).flatten()
    # df_test_sorted[byte_index] = scaler.transform(df_test_sorted[[byte_index]].astype(float)).flatten()
    # Escalonando os valores do DataFrame diretamente, ignorando os nomes das colunas
    df_train_sorted[byte_index] = scaler.fit_transform(df_train_sorted[byte_index].values.reshape(-1, 1)).flatten()
    df_test_sorted[byte_index] = scaler.transform(df_test_sorted[byte_index].values.reshape(-1, 1)).flatten()

    model = get_bi_lstm_model(n_timesteps, 'concat')

    start_time = timeit.default_timer()

    # Treinamento iterativo
    for i in range(len(df_train_sorted) - n_timesteps + 1):
        X, y = get_sequence(n_timesteps, df_train_sorted, byte_index, i)
        model.fit(X, y, epochs=1, batch_size=1, verbose=0)

    end_time = timeit.default_timer()

    all_predictions = []
    all_y_true = []

    for i in range(len(df_test_sorted) - n_timesteps + 1):
        # Avaliação do modelo usando o conjunto de teste
        X_test_seq, y_test_seq = get_sequence(n_timesteps, df_test_sorted, byte_index, i)

        y_pred = model.predict(X_test_seq)

        # Converter previsões contínuas para rótulos binários (0 ou 1) usando limiar de 0.5
        y_pred_labels = (y_pred > 0.5).astype(int).flatten()
        y_test_labels = (y_test_seq > 0.5).astype(int).flatten()

        # Salvar a previsão e o valor real
        all_predictions.extend(y_pred_labels)
        all_y_true.extend(y_test_labels)

    # Converte as listas em arrays para cálculo de métricas
    all_predictions = np.array(all_predictions)
    all_y_true = np.array(all_y_true)

    # Cálculo da acurácia e matriz de confusão para todas as previsões
    accuracy = accuracy_score(all_y_true, all_predictions)
    conf_matrix = confusion_matrix(all_y_true, all_predictions)

    print(f"Accuracy = {accuracy:.2f}, Time = {end_time - start_time:.4f} seconds")
    print("Confusion Matrix:\n", conf_matrix)

    # Salvar o modelo e o scaler
    save_model(model, scaler, model_filename, scaler_filename)

    # Avaliação final (se necessária)
    accuracy_real, conf_matrix_real = evaluate_model(df, all_predictions)

    return model


def plot_data(df, y_pred, title):
    """
    Função para plotar os dados reais vs previstos.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['data'], color='blue', label='Dados Reais')
    plt.plot(df.index, y_pred, color='red', linestyle='--', label='Dados Previstos')
    plt.title(title)
    plt.xlabel('Índice')
    plt.ylabel('Valores')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_model(df_processed, y_real_pred):
    # Assume que a primeira coluna do DataFrame é a que você deseja usar
    first_column = df_processed.iloc[:, 0]

    # Ajustar y_real_pred para ter o mesmo tamanho que first_column
    if len(y_real_pred) > len(first_column):
        y_real_pred = y_real_pred[:len(first_column)]
    elif len(y_real_pred) < len(first_column):
        first_column = first_column[:len(y_real_pred)]

    # Cálculo da acurácia
    accuracy_real = accuracy_score(first_column, y_real_pred)
    print(f"Accuracy for real data = {accuracy_real:.2f}")

    # Cálculo da matriz de confusão
    conf_matrix_real = confusion_matrix(first_column, y_real_pred)
    print("Confusion Matrix for real data:\n", conf_matrix_real)

    return accuracy_real, conf_matrix_real
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
        fig = plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(df_pgn['event_time'], df_pgn[f'byte_{i}'], linestyle='-', color='b', linewidth=0.5, label='Real Data')
        plt.scatter(event_times, y_pred, color='r', marker='x', label='Predicted Data')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'PGN {pgn}, Byte {i} - Accuracy: {accuracy:.2f}')
        plt.legend()
        plt.show()



def find_patterns(df, n_timesteps, byte_filter, model):
    # Prepara os dados no mesmo formato usado para treinamento
    X, _ = get_sequence(n_timesteps, df, byte_filter)

    # Fazer a previsão usando o modelo treinado
    yhat = model.predict(X, verbose=0)

    # Interpretação das previsões
    predictions = yhat.flatten()

    # Exibir resultados ou realizar alguma ação com base nas previsões
    for i in range(len(predictions)):
        print(f"Timestep {i + 1}: {predictions[i]}")

    # Plote o gráfico de pontos
    plt.figure(figsize=(10, 6))
    plt.scatter(range(1, len(predictions) + 1), predictions, color='blue')
    plt.title('Previsões do Modelo')
    plt.xlabel('Timesteps')
    plt.ylabel('Valor Predito')
    plt.grid(True)
    plt.show()

    return predictions

def discard_constant_pgn(df, n_timesteps):
    # Identifica os pgn únicos
    unique_pgns = df['pgn'].unique()

    # Lista para armazenar os índices dos PGNs a serem removidos
    indices_to_remove = []

    # Itera sobre cada pgn único
    for pgn in unique_pgns:
        # Filtra o DataFrame para o pgn atual
        df_pgn = df[df['pgn'] == pgn]

        # Verifica se o DataFrame filtrado tem dados suficientes para n_timesteps
        if len(df_pgn) >= n_timesteps:
            for i in range(len(df_pgn) - n_timesteps + 1):
                # Extrair a subsequência
                subsequence = df_pgn.iloc[i:i + n_timesteps]
                print(subsequence)

                # Verifica se todos os valores em todas as colunas da subsequência são iguais
                if (subsequence.iloc[:, 2:] == subsequence.iloc[0, 2:]).all().all():
                    indices_to_remove.extend(subsequence.index.tolist())

    # Remove as sequências com valores constantes
    df_cleaned = df.drop(indices_to_remove).reset_index(drop=True)

    return df_cleaned

def find_most_similar_sequence(df, n_timesteps, byte_filter, model):
    # Concatena o byte_filter com 'byte_' para identificar a coluna correta
    column_name = f'byte_{byte_filter}'

    # Verifica se o DataFrame tem dados suficientes para extrair sequências
    if len(df) < n_timesteps:
        raise ValueError("O DataFrame não tem linhas suficientes para o número de timesteps especificado.")

    # Obter todos os pgn únicos
    unique_pgns = df['pgn'].unique()

    overall_best_sequence = None
    overall_best_score = -np.inf
    overall_best_index = -1
    overall_best_pgn = None

    # Itera sobre cada pgn único
    for pgn in unique_pgns:
        # Filtra o DataFrame para o pgn atual
        df_pgn = df[df['pgn'] == pgn]
        print(df_pgn)

        best_sequence = None
        best_score = -np.inf
        best_index = -1

        # Verifica se o DataFrame filtrado tem dados suficientes para extrair sequências
        if len(df_pgn) < n_timesteps:
            continue  # Pula para o próximo pgn se não houver dados suficientes

        for i in range(len(df_pgn) - n_timesteps + 1):
            # Extrair a subsequência
            subsequence = df_pgn[column_name].iloc[i:i + n_timesteps].values.reshape(1, n_timesteps, 1)
            print(subsequence)
            # Fazer a previsão para a subsequência
            score = model.predict(subsequence, verbose=0).flatten()
            # Avaliar a subsequência com base na soma das classificações preditivas (ou outro critério)
            sum_score = np.sum(score)

            if sum_score > best_score:
                best_score = sum_score
                best_sequence = subsequence.flatten()
                best_index = df_pgn.index[i]  # Captura o índice original no DataFrame

        # Atualiza o melhor global se o atual for superior
        if best_score > overall_best_score:
            overall_best_score = best_score
            overall_best_sequence = best_sequence
            overall_best_index = best_index
            overall_best_pgn = pgn

    # Plote a melhor subsequência encontrada
    if overall_best_sequence is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(range(overall_best_index, overall_best_index + n_timesteps), overall_best_sequence, marker='o', linestyle='-',
                 color='blue', label=f'Melhor Sequência para PGN {overall_best_pgn}')
        plt.title('Sequência Numérica que Mais se Assemelha ao Dado de Treino')
        plt.xlabel('Índice do DataFrame')
        plt.ylabel('Valor da Sequência')
        plt.legend()
        plt.grid(True)
        plt.show()

    return overall_best_sequence, overall_best_index, overall_best_pgn

def find_sequences(df, model, scaler, n_timesteps=1000, label_options=[0, 1]):
    similar_sequences = []

    # Iterar sobre cada PGN único no DataFrame
    unique_pgns = df['pgn'].unique()
    for pgn in unique_pgns:
        df_pgn = df[df['pgn'] == pgn].copy()

        # Iterar sobre cada byte
        for byte_index in range(1, 9):  # byte_1 a byte_8
            byte_column = f'byte_{byte_index}'

            # Verificar se todos os valores da coluna são iguais
            if df_pgn[byte_column].nunique() == 1:
                print(f"Coluna {byte_column} ignorada: todos os valores são iguais.")
                continue  # Ignora esta coluna e passa para o próximo byte

            # Escalonar os valores da coluna individualmente
            byte_values = df_pgn[byte_column].values.reshape(-1, 1)
            byte_values_scaled = scaler.fit_transform(byte_values)
            df_pgn[byte_column] = byte_values_scaled.flatten()

            best_score = -np.inf
            best_sequence_info = None

            # Iterar sobre o DataFrame em blocos de n_timesteps
            for i in range(len(df_pgn) - n_timesteps + 1):
                X_seq = get_Xsequence(n_timesteps, df_pgn, byte_column, i)
                y_pred = model.predict(X_seq).flatten()

                # Comparar a previsão com os rótulos possíveis
                for label in label_options:
                    similarity = np.sum(np.abs(y_pred - label))

                    if similarity > best_score:
                        best_score = similarity
                        best_sequence_info = (pgn, byte_index, i, similarity)

            if best_sequence_info:
                similar_sequences.append(best_sequence_info)

    # Ordenar as sequências mais similares
    similar_sequences.sort(key=lambda x: x[3], reverse=True)

    return similar_sequences