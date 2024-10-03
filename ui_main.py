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


def use_model():
    global df
    global model

    model_filename = 'meu_modelo.pkl'

    # Verificar se o modelo salvo já existe
    df = ui_load_ixxt_archive('JUMPY 2024/limpador_vel1_5x_jumpy.CSV')

    model = carregar_modelo("meu_modelo.pkl")

    # Etapa 1: Analisar o DataFrame e coletar dados após descartar PGNs constantes
    data_df = analyse_archive(df)

    # Etapa 2: Ajustar as sequências para o tamanho desejado
    validated_df = prepare_sequences(data_df, target_size=1200)

    # Etapa 3: Classificar as sequências usando o modelo
    classified_df = classify_sequences(model, validated_df)

    # Etapa 4: Aplicar a validação de picos
    final_results_df = validate_peaks(classified_df)

    # Opcional: Exibir os resultados finais
    print("\nResultados finais após validação de picos:")
    for index, row in final_results_df.iterrows():
        print(
            f"PGN: {row['pgn']}, Byte: {row['byte_column']}, Acurácia: {row['accuracy']}, Número de Picos: {row['num_peaks']}")

    plot_best_pgns(final_results_df)



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

    return None


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

    btn_load = tk.Button(
        button_frame,
        text="Carregar ixxt csv",
        command=ui_load_ixxt_archive,
        width=button_width,
        height=button_height,
        bg=button_bg_color,
        fg="white"
    )

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

    train_button = tk.Button(button_frame, text="BUSCAR SEQUENCIAS", command=use_model, width=button_width, height=button_height, bg=button_bg_color, fg="white")
    train_button.pack(side=tk.LEFT, padx=5)

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

    root.mainloop()

if __name__ == "__main__":
    main()