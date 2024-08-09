
from data_processing import generate_limited_sawtooth_wave_data
from data_processing import generate_limited_square_wave_data
from data_processing import load_model
from data_processing import process_real_data
from data_processing import load_data
from data_processing import train_model
from data_processing import combine_data
from data_processing import save_model
from data_processing import plot_data
from data_processing import analyze_pgn
from data_processing import plot_best_pgns
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog


def select_file():
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal
    file_path = filedialog.askopenfilename(
        title="Selecione o arquivo CSV",
        filetypes=(("CSV files", "*.CSV"), ("All files", "*.*"))
    )
    return file_path


def main():
    num_points = 1000
    start_time = datetime.now()
    cycles = 5

    model_filename = 'model.pkl'
    scaler_filename = 'scaler.pkl'

    df_square_wave = generate_limited_square_wave_data(num_points, start_time, cycles)
    df_square_wave['event_time'] = df_square_wave['event_time'].astype(np.int64) // 10 ** 9
    df_square_wave['dummy_feature'] = 0

    df_sawtooth_wave = generate_limited_sawtooth_wave_data(num_points, start_time, cycles)
    df_sawtooth_wave['event_time'] = df_sawtooth_wave['event_time'].astype(np.int64) // 10 ** 9
    df_sawtooth_wave['dummy_feature'] = 0

    # Verificar se o modelo salvo já existe
    if os.path.exists(model_filename):
        print("Modelo encontrado. Carregando...")
        model, scaler = load_model(model_filename, scaler_filename)
        print(f",model: {model} \n ")
    else:
        print("Modelo não encontrado. Treinando novo modelo...")
        df_processed = process_real_data(load_data('velocidade_0_20_5x_jumpy.csv'))
        model, scaler = train_model(combine_data(df_sawtooth_wave, df_processed))
        save_model(model, scaler, model_filename, scaler_filename)

        X_real = df_processed[['event_time', 'dummy_feature']]
        X_real_scaled = scaler.transform(X_real)
        y_real_pred = model.predict(X_real_scaled)

        X_saw = df_sawtooth_wave[['event_time', 'dummy_feature']]
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

    file_path = select_file()
    if file_path:
        df = load_data(file_path)
        print(df.head())  # Exibe as primeiras linhas do DataFrame
    else:
        print("Nenhum arquivo foi selecionado.")

    cycles = 10
    print(df['data'])
    best_pgns = analyze_pgn(df, cycles, model, scaler)

    for pgn, i, accuracy, y_pred, X, y in best_pgns:
        print(f'Melhor PGN: {pgn}, Melhor Índice: {i}, Acurácia: {accuracy:.2f}')

    plot_best_pgns(best_pgns, df)


if __name__ == "__main__":
    main()
