from datetime import datetime
from tkinter import filedialog
from tkinter import messagebox
from tkinter import filedialog
import tkinter as tk
from math_engine import *
import os
from lstm_engine import *
import pandas as pd
from matplotlib import pyplot as plt


# Nome do arquivo padrão
DEFAULT_FILE = "CAN_DataChunks/JUMPY_0_20_5x.csv"
global df, flag
global_filtered_df = None


# def transform_column_to_single_row(df, byte_index):
#     if df is None or df.empty:
#         messagebox.showwarning("Aviso", "DataFrame está vazio ou não foi fornecido.")
#         return None
#
#     try:
#         # Construir o nome da coluna com base no índice fornecido
#         byte_column = f'byte_{byte_index}'
#
#         # Verificar se a coluna existe no DataFrame
#         if byte_column in df.columns:
#             # Somar os valores da coluna especificada e transformar em uma única linha
#             single_row = pd.DataFrame({byte_column: [df[byte_column].sum()]})
#
#             # Retornar a linha única resultante
#             return single_row
#         else:
#             messagebox.showwarning("Aviso", f"A coluna {byte_column} não existe no DataFrame.")
#             return None
#     except Exception as e:
#         messagebox.showerror("Erro", f"Erro ao transformar a coluna em uma única linha: {str(e)}")
#         return None

#
# def load_default_dataframe():
#     if os.path.exists(DEFAULT_FILE):
#         try:
#             global df
#             df = pd.read_csv(DEFAULT_FILE)
#             messagebox.showinfo("Informação", f"Arquivo {DEFAULT_FILE} carregado automaticamente.")
#             return True
#         except Exception as e:
#             messagebox.showerror("Erro", f"Erro ao carregar o arquivo padrão: {str(e)}")
#             return False
#     return False

#
# def load_dataframe():
#     file_path = filedialog.askopenfilename(
#         title="Selecione o arquivo CSV para carregar",
#         filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
#     )
#
#     if file_path:
#         try:
#             global df
#             df = pd.read_csv(file_path)
#             status_label.config(text="Dataframe filtrado carregado")
#             messagebox.showinfo("Sucesso", "DataFrame carregado com sucesso!")
#         except Exception as e:
#             messagebox.showerror("Erro", f"Erro ao carregar o arquivo: {str(e)}")
#     else:
#         messagebox.showwarning("Aviso", "Nenhum arquivo selecionado.")

#
# def analyse_archive(df, model):
#     best_pgns = []
#
#     for pgn in df['pgn'].unique():  # Assumindo que 'pgn' é uma coluna que identifica cada sequência
#         df_pgn = df[df['pgn'] == pgn].copy()  # Filtra o DataFrame para o pgn atual
#
#         for i in range(8):  # Assumindo que cada dado tem 8 bytes
#             byte_column = f'byte_{i}'
#
#             # Transformar a coluna de bytes em lista
#             df_pgn[byte_column] = df_pgn['data'].apply(lambda x: int(x[i], 16))
#
#             # Obter valores de byte e processá-los
#             X = df_pgn[byte_column].values
#             # peaks, troughs = count_peaks_and_troughs(byte_values)
#             #
#             # if len(peaks) <= 6 and len(troughs) <= 6:
#             #     # Preparar X e y para treinamento
#             #     X = df_pgn.drop(columns=['event_time', 'dummy_feature', 'data', 'pgn'])  # Ignora as colunas não relevantes
#             #     y = df_pgn[byte_column].values
#
#             if len(X) > 1:  # Garantir que temos dados suficientes e mais de uma classe
#                 # Treinar o modelo para este byte específico
#                 #model = train_sequence_model(df_pgn, i)
#                 y_pred = model.predict(X)
#
#                 # Calcular acurácia
#                 accuracy = accuracy_score(y, y_pred)
#
#                 # Armazenar os melhores pgns
#                 best_pgns.append((pgn, i, accuracy, y_pred, X, y))
#
#     # return best_pgns

def use_model():
    global df
    global model

    model_filename = 'model1.pkl'
    scaler_filename = 'scaler.pkl'

    # Verificar se o modelo salvo já existe
    if os.path.exists(model_filename):
        print("Modelo encontrado. Carregando...")
        model, scaler = load_model(model_filename, scaler_filename)
        messagebox.showinfo("Ação", "Usando o modelo de Machine Learning...")

        similar_sequences = find_sequences(df, model, scaler)

        # Exibindo as sequências mais similares
        for sequence in similar_sequences:
            print(f"PGN: {sequence[0]}, Byte: {sequence[1]}, Start Index: {sequence[2]}, Similarity: {sequence[3]}")

        plot_best_sequences(labeled_df, similar_sequences, n_timesteps=250, top_n=5)
    else:
        print("Modelo não encontrado. Treinando novo modelo...")



def load_selected_file(selected_file):
    directory = "CAN_DataChunks"  # Diretório onde os arquivos estão localizados
    global df
    file_path = os.path.join(directory, selected_file)  # Concatena o diretório com o nome do arquivo
    if selected_file:

        try:
            df = pd.read_csv(file_path)
            messagebox.showinfo("Informação", f"Arquivo {selected_file} carregado com sucesso!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar o arquivo: {str(e)}")
    else:
        messagebox.showwarning("Aviso", "Nenhum arquivo selecionado.")


def save_model_via_button():
    global model  # Declarando que estamos usando a variável global `model`
    if model is None:
        print("Erro: O modelo não foi carregado ou inicializado.")
        return

    model_path = filedialog.asksaveasfilename(defaultextension=".pkl",
                                              filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                                              title="Salvar Modelo")

    if model_path:
        save_model(model, model_path)
        print(f"Arquivo salvo: {model_path}")
    else:
        print("O salvamento foi cancelado ou ocorreu um erro.")


def load_model_via_button():
    global model
    model_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                                            title="Carregar Modelo")

    if model_path:
        model = load_model(model_path)
        print(f"Modelo carregado de {model_path}")
        status_label.config(text="Modelo carregado")
        root.update_idletasks()  # Atualiza a tela para mostrar a mensagem de carregamento
    else:
        status_label.config(text="O carregamento foi cancelado ou ocorreu um erro.")
        root.update_idletasks()  # Atualiza a tela para mostrar a mensagem de carregamento
        print("O carregamento foi cancelado ou ocorreu um erro.")


def save_filter_train_dataframe():
    global df
    global global_filtered_df

    if df is None:
        messagebox.showwarning("Aviso", "Carregue um arquivo primeiro.")
        return None

    byte_filter = byte_entry.get()
    pgn_filter = pgn_entry.get()

    if pgn_filter and byte_filter:
        try:
            # Filtrar pelo PGN
            filtered_df = df[df['pgn'] == int(pgn_filter, 16)]  # Se pgn for uma string hexadecimal

            # Verificar se o byte filtrado existe
            byte_filter_column = f'byte_{byte_filter}'
            if byte_filter_column in filtered_df.columns:
                # Filtrar os valores do byte especificado que não são 'FF' (255 em decimal)
                filtered_df = filtered_df[filtered_df[byte_filter_column] != 255]

                if not filtered_df.empty:
                    # Armazena o DataFrame filtrado na variável global
                    global_filtered_df = filtered_df
                    #messagebox.showinfo("Sucesso", "DataFrame filtrado com sucesso!")
                    status_label.config(text="DataFrame filtrado e armazenado com sucesso.")
                    return global_filtered_df
                else:
                    messagebox.showinfo("Informação", "Nenhum valor diferente de FF encontrado após filtrar.")
                    return None
            else:
                messagebox.showinfo("Informação", f"A coluna {byte_filter_column} não existe.")
                return None
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao aplicar filtros: {str(e)}")
            return None
    else:
        messagebox.showwarning("Aviso", "Preencha os campos de filtro antes de filtrar.")
        return None


def save_raw_dataframe():
    if df is None:
        messagebox.showwarning("Aviso", "Carregue um arquivo primeiro.")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        title="Salvar o DataFrame completo"
    )

    if save_path:
        df.to_csv(save_path, index=False)
        messagebox.showinfo("Sucesso", "DataFrame salvo com sucesso!")


def ui_load_ixxt_archive():
    file_path = filedialog.askopenfilename(
        title="Selecione o arquivo CSV",
        filetypes=(("CSV files", "*.CSV"), ("all files", "*.*"))
    )
    if file_path:
        global df
        status_label.config(text="Carregando arquivo, por favor, aguarde...")
        root.update_idletasks()  # Atualiza a tela para mostrar a mensagem de carregamento

        df = fileToDataframe(file_path)  # Simulação do carregamento de dados
        status_label.config(text="Arquivo ixxt carregado com sucesso!")
        root.update()  # Atualiza a tela para refletir a mudança de status

        messagebox.showinfo("Informação", "Arquivo carregado com sucesso!")

# Função para visualizar os dados

def plot_best_sequences(df, similar_sequences, n_timesteps=250, top_n=5):
    """
    Plota as melhores sequências com base nas similaridades calculadas.

    Parâmetros:
    - df: DataFrame original que contém os dados.
    - similar_sequences: Lista de tuplas com informações das sequências mais similares (pgn, byte_index, start_index, similarity).
    - n_timesteps: Número de timesteps em cada sequência.
    - top_n: Número de melhores sequências para plotar.
    """
    # Seleciona as top_n sequências mais similares
    top_sequences = similar_sequences[:top_n]

    plt.figure(figsize=(15, 10))

    for i, (pgn, byte_index, start_index, similarity) in enumerate(top_sequences):
        byte_column = f'byte_{byte_index}'
        sequence = df[df['pgn'] == pgn].iloc[start_index:start_index + n_timesteps][byte_column].values

        plt.subplot(top_n, 1, i + 1)
        plt.plot(range(start_index, start_index + n_timesteps), sequence, marker='o', linestyle='-', color='blue')
        plt.title(f'Sequence for PGN {pgn} - Byte {byte_index} (Similarity: {similarity:.2f})')
        plt.xlabel('Index')
        plt.ylabel('Byte Value')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def vizData():
    global df
    if df is None:
        messagebox.showwarning("Aviso", "Carregue um arquivo primeiro.")
        return

    filtered_df = save_filter_train_dataframe()
    byte_filter = byte_entry.get()
    pgn_filter = pgn_entry.get()
    byte_filter = f'byte_{byte_filter}'

    print(filtered_df.head)

    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(range(len(filtered_df)), filtered_df[byte_filter], linestyle='-', color='b',
             linewidth=0.7,
             label=f'{byte_filter}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f"Visualização de Dados - PGN: {pgn_filter}, Byte: {byte_filter}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6), dpi=100)
    for i in range(1, 9):
        byte_column = f'byte_{i}'
        if byte_column in filtered_df.columns:
            plt.plot(filtered_df['event_time'], filtered_df[byte_column], linestyle='-', linewidth=0.5,
                     label=f'{byte_column}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f"Visualização de Dados - PGN: {pgn_filter}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_dropdown_menu(root, file_list):
    selected_file = tk.StringVar(root)
    selected_file.set(file_list[0])  # Definir o primeiro arquivo como padrão

    dropdown = tk.OptionMenu(root, selected_file, *file_list)
    dropdown.pack(pady=10)

    load_button = tk.Button(root, text="Carregar Arquivo", command=lambda: load_selected_file(selected_file.get()))
    load_button.pack(pady=10)


def train_model_button():
    global labeled_df
    byte = int(byte_entry.get())

    # Verificar se o DataFrame filtrado global existe e não está vazio
    if global_filtered_df is None or global_filtered_df.empty:
        messagebox.showwarning("Verificação", "Nenhum dado filtrado disponível. Por favor, aplique os filtros primeiro.")
        return

    # Definir parâmetros para o treinamento do modelo
    n_timesteps = 500

    train_sequence_model(labeled_df, byte)
    # Treinar o modelo com base nos dados filtrados
    #model = train_sequence_model(filtered_df, byte_filter)
    #loss = list()

        #loss.append(hist.history['loss'][0])
        #print(global_filtered_df.head(), len(global_filtered_df))

    return


def on_filter_and_label():
    global labeled_df
    try:
        pgn = int(pgn_entry.get(), 16)
        byte = int(byte_entry.get())
        label = int(label_entry.get())

        if label not in [0, 1, 2, 3]:
            raise ValueError("O rótulo deve ser 0, 1, 2 ou 3.")

        labeled_df = filter_and_label_dataframe(df, pgn, byte, label)
        messagebox.showinfo("Sucesso", f"DataFrame filtrado e rotulado com sucesso:\n{labeled_df}")
        status_label.config(text="Rotulado / Pronto pro treinamento")
    except Exception as e:
        messagebox.showerror("Erro", str(e))



def main():
    global pgn_entry, byte_value_entry, byte_entry, df, root, status_label, label_entry, labeled_df
    df = None  # Inicializar o DataFrame como None

    # Criando a janela principal
    # Define o tamanho dos botões
    button_width = 20
    button_height = 2
    root = tk.Tk()
    root.title("ML CAN ML SCANNER")
    root.geometry("450x650")  # Define o tamanho da janela
    tk.Label(root, text="").pack(pady=10)
    button_bg_color = "#178b76"  # Cor de fundo dos botões

    directory = "CAN_DataChunks"  # Substitua pelo caminho do seu diretório
    # Estilizando a Label
    status_label = tk.Label(root, text="Aguardando ação...",
                            pady=5,
                            padx=9,
                            font=("Helvetica", 12, "bold"),
                            bg="#d3d3d3",
                            fg="#178b76",
                            relief="solid")

    status_label.pack(pady=20)
    status_label.pack()

    # Criando um Frame para agrupar os botões lado a lado
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    # Botão "Carregar dados"
    btn_load = tk.Button(button_frame, text="Carregar ixxt csv", command=ui_load_ixxt_archive, width=button_width, height=button_height, bg=button_bg_color,
                         fg="white")
    btn_load.pack(side=tk.LEFT, padx=5)

    # Botão "Salvar Dados brutos"
    btn_save = tk.Button(button_frame, text="Salvar dataframe", command=save_raw_dataframe, width=button_width, height=button_height,
                         bg=button_bg_color, fg="white")
    btn_save.pack(side=tk.LEFT, padx=5)

    # Filtra apenas arquivos CSV no diretório
    file_list = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not file_list:
        messagebox.showwarning("Aviso", "Nenhum arquivo CSV encontrado.")
        return
    create_dropdown_menu(root, file_list)

    # Frame para os campos de entrada
    filtered_frame = tk.Frame(root)
    filtered_frame.pack(pady=15)

    # Estilo das entradas
    entry_style = {'bd': 0, 'relief': 'solid', 'highlightthickness': 1, 'highlightbackground': '#FFFFFF'}

    # Frame para o Filtro de Byte
    byte_frame = tk.Frame(filtered_frame)
    byte_frame.pack(side=tk.LEFT, padx=10)

    # Label e Entry para o Número do Byte
    byte_label = tk.Label(byte_frame, text="Número do Byte (1 a 8):")
    byte_label.pack()
    byte_entry = tk.Entry(byte_frame, width=15, **entry_style)
    byte_entry.pack()

    # Frame para o Filtro de PGN
    pgn_frame = tk.Frame(filtered_frame)
    pgn_frame.pack(side=tk.LEFT, padx=10)

    # Label e Entry para o Filtro de PGN
    pgn_label = tk.Label(pgn_frame, text="Filtro de PGN (em hex):")
    pgn_label.pack()
    pgn_entry = tk.Entry(pgn_frame, width=15, **entry_style)
    pgn_entry.pack()

    # Frame para os botões de filtragem
    filtered_button_frame = tk.Frame(root)
    filtered_button_frame.pack(pady=15)

    # Visualizar dados
    btn_viz = tk.Button(filtered_button_frame, text="Visualizar dados", command=vizData, width=button_width, height=button_height, bg=button_bg_color, fg="white")
    btn_viz.pack(side=tk.LEFT, padx=5)

    # Salvar DataFrame filtrado
    btn_save = tk.Button(filtered_button_frame, text="Salvar Dados", command=save_filter_train_dataframe, width=button_width, height=button_height, bg=button_bg_color, fg="white")
    btn_save.pack(side=tk.LEFT, padx=5)

    # Label e Entry para o Label
    label_label = tk.Label(root, text="Label:")
    label_label.pack()
    label_entry = tk.Entry(root)
    label_entry.pack()

    # Botão para chamar a função de filtro e rótulo
    button_label = tk.Button(root, text="Aplicar o Rótulo", command=on_filter_and_label)
    button_label.pack(pady=10)

    # Criando um Frame para os botões
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)

    # Adicionando o botão para salvar o modelo
    train_button = tk.Button(button_frame, text="Treinar Modelo", command=train_model_button, width=button_width, height=button_height, bg=button_bg_color, fg="white")
    train_button.pack(side=tk.LEFT, padx=5)

    # Colocando o botão na janela
    #btn.pack(pady=20)

    # Criando um Frame para os botões
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)

    # Adicionando o botão para salvar o modelo
    save_button = tk.Button(button_frame, text="Salvar Modelo", command=save_model_via_button)
    save_button.pack(side=tk.LEFT, padx=10)

    # Adicionando o botão para carregar o modelo
    load_button = tk.Button(button_frame, text="Carregar Modelo", command=load_model_via_button)
    load_button.pack(side=tk.LEFT, padx=10)

    train_button = tk.Button(button_frame, text="BUSCAR SEQUENCIAS", command=use_model, width=button_width, height=button_height, bg=button_bg_color, fg="white")
    train_button.pack(side=tk.LEFT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    main()