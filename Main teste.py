import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
from data_engine import *
from matplotlib import pyplot as plt


class AplicacaoCSV:
    def __init__(self, master):
        self.master = master
        self.master.title("Aplicação CSV")  # Título inicial da janela
        self.master.geometry("800x600")

        self.can_data = None
        self.best_results = []  # Lista para armazenar os DataFrames de resultados finais
        self.current_plot_index = 0  # Índice do gráfico atual
        self.nome_arquivo = None  # Variável para armazenar o nome do arquivo analisado

        # Chamar método para criar o menu drop-down
        self.create_menu_dropdown()

        # Frame para os botões
        self.button_frame = ttk.Frame(self.master)
        self.button_frame.pack(pady=10)

        # Frame para exibir informações ou gráficos
        self.content_frame = ttk.Frame(self.master)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Frame para navegação de gráficos
        self.navigation_frame = ttk.Frame(self.master)
        self.navigation_frame.pack(pady=5)

        # Botão "Carregar dados"
        self.btn_load = tk.Button(
            self.button_frame,
            text="Carregar ixxt csv",
            command=lambda: ui_load_ixxt_archive(self),
            width=20,
            height=2,
            bg="#4CAF50",
            fg="white"
        )
        self.btn_load.pack(side=tk.LEFT, padx=5)

        # Botão "Plotar Gráfico"
        self.btn_plot = tk.Button(
            self.button_frame,
            text="Plotar Gráfico",
            command=self.plotar_grafico,
            width=20,
            height=2,
            bg="#2196F3",
            fg="white"
        )
        self.btn_plot.pack(side=tk.LEFT, padx=5)

        # Botão "Usar Modelo"
        self.btn_model = tk.Button(
            self.button_frame,
            text="Usar Modelo",
            command=self.use_model,
            width=20,
            height=2,
            bg="#FF9800",
            fg="white"
        )
        self.btn_model.pack(side=tk.LEFT, padx=5)

        # Botões de navegação de gráficos
        self.btn_prev = tk.Button(
            self.navigation_frame,
            text="Anterior",
            command=self.show_prev_plot,
            width=10,
            height=1,
            bg="#9E9E9E",
            fg="white"
        )
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = tk.Button(
            self.navigation_frame,
            text="Próximo",
            command=self.show_next_plot,
            width=10,
            height=1,
            bg="#9E9E9E",
            fg="white"
        )
        self.btn_next.pack(side=tk.LEFT, padx=5)

        # Label para mostrar o título do gráfico atual
        self.plot_title = tk.Label(
            self.navigation_frame,
            text="Nenhum gráfico carregado",
            font=("Helvetica", 12)
        )
        self.plot_title.pack(side=tk.LEFT, padx=10)



    def create_menu_dropdown(self):
        directory = "CAN_DataChunks"  # Diretório onde os arquivos CSV estão armazenados

        # Filtra apenas arquivos CSV no diretório
        file_list = [f for f in os.listdir(directory) if f.endswith('.csv')]
        if not file_list:
            messagebox.showwarning("Aviso", "Nenhum arquivo CSV encontrado.")
            return

        # Cria o menu drop-down com a lista de arquivos e a função de callback para carregar o arquivo
        self.create_dropdown_menu(file_list)

    def create_dropdown_menu(self, file_list):
        selected_file = tk.StringVar(self.master)
        selected_file.set(file_list[0])  # Definir o primeiro arquivo como padrão

        # Menu suspenso para seleção de arquivos
        dropdown = tk.OptionMenu(self.master, selected_file, *file_list)
        dropdown.pack(pady=10)

        # Botão para carregar o arquivo selecionado
        load_button = tk.Button(self.master, text="Carregar Arquivo", command=lambda: self.load_file_callback(selected_file.get()))
        load_button.pack(pady=10)

    def load_file_callback(self, selected_file):
        df = load_selected_file(selected_file)
        if df is not None:
            self.can_data = df  # Armazena o DataFrame carregado
            self.nome_arquivo = selected_file  # Atualiza o nome do arquivo
            self.master.title(f"Aplicação CSV - {self.nome_arquivo}")  # Atualiza o título da janela

    def exibir_dataframe(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        text = tk.Text(self.content_frame, wrap=tk.NONE)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, self.can_data.head(10).to_string())

        scrollbar_y = ttk.Scrollbar(self.content_frame, orient=tk.VERTICAL, command=text.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        text.config(yscrollcommand=scrollbar_y.set)

        scrollbar_x = ttk.Scrollbar(self.content_frame, orient=tk.HORIZONTAL, command=text.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        text.config(xscrollcommand=scrollbar_x.set)

    def plotar_grafico(self):
        if self.can_data is None:
            messagebox.showwarning("Aviso", "Nenhum dado carregado para plotar.")
            return

        # Plotar os melhores PGNs dentro do Tkinter
        self.plot_best_pgns()


    def use_model(self):
        if self.can_data is None:
            messagebox.showwarning("Aviso", "Nenhum dado carregado para usar o modelo.")
            return

        model_filename = 'meu_modelo.pkl'

        # Carregar o modelo
        model = self.carregar_modelo(model_filename)
        if model is None:
            return

        # Etapa 1: Analisar o DataFrame e coletar dados após descartar PGNs constantes
        data_df = analyse_archive(self)
        if data_df is None:
            return

        # Etapa 2: Ajustar as sequências para o tamanho desejado
        validated_df = prepare_sequences(data_df, target_size=1200)
        if validated_df is None:
            return

        # Etapa 3: Classificar as sequências usando o modelo
        classified_df = classify_sequences(model, validated_df)
        if classified_df is None:
            return

        # Etapa 4: Aplicar a validação de picos
        final_results_df = validate_peaks(classified_df)
        if final_results_df is None:
            return

        # Armazenar os resultados finais
        self.best_results = final_results_df

        # Exibir os resultados finais no terminal
        self.exibir_resultados_finais(final_results_df)

    def exibir_resultados_finais(self, final_results_df):

        if final_results_df is None or final_results_df.empty:
            print("Nenhum resultado para exibir.")
            return

        print("\nResultados finais após validação de picos:")
        for index, row in final_results_df.iterrows():
            print(
                f"PGN: {row['pgn']}, Byte: {row['byte_column']}, Acurácia: {row['accuracy']}, Número de Picos: {row['num_peaks']}"
            )

    def carregar_modelo(self, nome_arquivo):
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

    def plot_best_pgns(self):
        self.current_plot_index = 0
        for index, row in self.best_results.iterrows():
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

    def show_prev_plot(self):
        if not self.best_results:
            return
        if self.current_plot_index > 0:
            self.current_plot_index -= 1
            self.display_current_plot()
        else:
            messagebox.showinfo("Informação", "Este é o primeiro gráfico.")

    def show_next_plot(self):
        if not self.best_results:
            return
        if self.current_plot_index < len(self.best_results) - 1:
            self.current_plot_index += 1
            self.display_current_plot()
        else:
            messagebox.showinfo("Informação", "Este é o último gráfico.")


# Iniciar a aplicação
if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacaoCSV(root)
    root.mainloop()
