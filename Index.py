import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import timeit


def generate_limited_square_wave_data(num_points, start_time, cycles):
    timestamps = [start_time + timedelta(seconds=i) for i in range(num_points)]
    wave = []
    for i in range(num_points):
        cycle_position = (i % (cycles * 100))
        if cycle_position < (100 * cycles):
            wave.append(1 if (cycle_position // 100) % 2 == 0 else 2)
        else:
            wave.append(0)
    return pd.DataFrame({'event_time': timestamps, 'data': wave})


def generate_limited_sawtooth_wave_data(num_points, start_time, cycles, period=100):
    timestamps = [start_time + timedelta(seconds=i) for i in range(num_points)]
    wave = []
    for i in range(num_points):
        cycle_position = (i % (cycles * period))
        if cycle_position < (period * cycles):
            wave.append((cycle_position % period) + 1)
        else:
            wave.append(0)
    return pd.DataFrame({'event_time': timestamps, 'data': wave})


def load_and_process_real_data(file_path):
    df_real = pd.read_csv(file_path, names=['event_time', 'pgn', 'data'])
    df_real = df_real.dropna(subset=['data', 'event_time'])
    df_real['event_time'] = pd.to_datetime(df_real['event_time'], errors='coerce')
    df_real['data'] = df_real['data'].apply(process_data)

    df_processed = df_real[df_real['pgn'] == '32D'].copy()

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
    df_final.set_index('event_time', inplace=True)
    df_final['timestamp'] = df_final.index.astype(np.int64) // 10 ** 9
    df_final['dummy_feature'] = 0
    df_final['pgn'] = '32D'  # Adicionar a coluna 'pgn' de volta

    return df_final


def process_data(data):
    if isinstance(data, str):
        return data.split(' ')
    return []


def combine_and_train_model(df_wave, df_processed):
    df_combined = pd.concat([df_wave, df_processed])
    X = df_combined[['timestamp', 'dummy_feature']]
    y = df_combined['data']
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
    plt.scatter(df.index, y_pred, color='r', marker='x', label='Predicted Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_pgn(df, cycles=10):
    df['data'] = df['data'].apply(process_data)
    unique_pgns = sorted(df['pgn'].unique())
    best_pgns = []

    for pgn in unique_pgns:
        df_pgn = df[df['pgn'] == pgn].copy()

        # Verificar se todos os dados têm 8 bytes
        df_pgn = df_pgn[df_pgn['data'].apply(lambda x: isinstance(x, list) and len(x) == 8)]

        if df_pgn.empty:
            continue

        for i in range(8):  # Assumindo que cada dado tem 8 bytes
            df_pgn[f'byte_{i}'] = df_pgn['data'].apply(lambda x: int(x[i], 16))

            byte_values = df_pgn[f'byte_{i}'].values
            peaks, troughs = count_peaks_and_troughs(byte_values)

            if len(peaks) <= 10 and len(troughs) <= 10:
                X = df_pgn[[f'byte_{i}']].values
                y = df_pgn[f'byte_{i}'].values

                if len(X) > cycles and len(set(y)) > 1:  # Garantir que temos dados suficientes e mais de uma classe
                    X_train, X_test, y_train, y_test = train_test_split(X[:cycles], y[:cycles], test_size=0.3, random_state=42)
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    best_pgns.append((pgn, i, accuracy, y_test, y_pred, X_test))

    best_pgns.sort(key=lambda x: x[2], reverse=True)  # Ordenar pelos melhores
    return best_pgns[:10]

def count_peaks_and_troughs(data):
    peaks = []
    troughs = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            peaks.append(data[i])
        elif data[i] < data[i - 1] and data[i] < data[i + 1]:
            troughs.append(data[i])
    return peaks, troughs


# Função para plotar os dados reais e as previsões
def plot_best_pgns(best_pgns, df):
    for pgn, i, accuracy, y_test, y_pred, X_test in best_pgns:
        df_pgn = df[df['pgn'] == pgn].copy()
        df_pgn = df_pgn[df_pgn['data'].apply(lambda x: isinstance(x, list) and len(x) == 8)]
        df_pgn[f'byte_{i}'] = df_pgn['data'].apply(lambda x: int(x[i], 16))

        plt.figure(figsize=(12, 6))
        plt.plot(df_pgn['event_time'], df_pgn[f'byte_{i}'], linestyle='-', color='b', linewidth=0.5, label='Real Data')
        plt.scatter(df_pgn['event_time'].iloc[:len(y_pred)], y_pred, color='r', marker='x', label='Predicted Data')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'PGN {pgn}, Byte {i}, Accuracy: {accuracy:.2f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    num_points = 1000
    start_time = datetime.now()
    cycles = 5

    df_square_wave = generate_limited_square_wave_data(num_points, start_time, cycles)
    df_square_wave['timestamp'] = df_square_wave['event_time'].astype(np.int64) // 10 ** 9
    df_square_wave['dummy_feature'] = 0

    df_sawtooth_wave = generate_limited_sawtooth_wave_data(num_points, start_time, cycles)
    df_sawtooth_wave['timestamp'] = df_sawtooth_wave['event_time'].astype(np.int64) // 10 ** 9
    df_sawtooth_wave['dummy_feature'] = 0

    df_processed = load_and_process_real_data('velocidade_0_20_5x_jumpy.csv')

    model, scaler = combine_and_train_model(df_square_wave, df_processed)

    X_real = df_processed[['timestamp', 'dummy_feature']]
    X_real_scaled = scaler.transform(X_real)
    y_real_pred = model.predict(X_real_scaled)

    X_saw = df_sawtooth_wave[['timestamp', 'dummy_feature']]
    X_saw_scaled = scaler.transform(X_saw)
    y_saw_pred = model.predict(X_saw_scaled)

    plot_data(df_processed, y_real_pred, 'Comparison of Real and Predicted Data')
    plot_data(df_sawtooth_wave, y_saw_pred, 'Comparison of Sawtooth Data and Predicted Data')

    accuracy_real = accuracy_score(df_processed['data'], y_real_pred)
    conf_matrix_real = confusion_matrix(df_processed['data'], y_real_pred)
    print(f"Accuracy for real data = {accuracy_real:.2f}")
    print("Confusion Matrix for real data:\n", conf_matrix_real)

    accuracy_saw = accuracy_score(df_sawtooth_wave['data'], y_saw_pred)
    conf_matrix_saw = confusion_matrix(df_sawtooth_wave['data'], y_saw_pred)
    print(f"Accuracy for sawtooth data = {accuracy_saw:.2f}")
    print("Confusion Matrix for sawtooth data:\n", conf_matrix_saw)

    df = pd.read_csv('velocidade_0_20_5x_jumpy.csv', names=['event_time', 'pgn', 'data'])
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    df = df.dropna(subset=['event_time', 'pgn', 'data'])

    cycles = 10
    best_pgns = analyze_pgn(df, cycles)

    for pgn, i, accuracy, y_test, y_pred, X_test in best_pgns:
        print(f'Melhor PGN: {pgn}, Melhor Índice: {i}, Acurácia: {accuracy:.2f}')

    plot_best_pgns(best_pgns, df)


if __name__ == "__main__":    main()
