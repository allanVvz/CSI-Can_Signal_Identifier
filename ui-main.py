import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
from data_engine import *
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Adicionando para integração de gráficos no Tkinter


class AplicacaoCSV:
    def __init__(self, master):
        self.master = master
        self.master.title("Aplicação CSV")  # Título inicial da janela
        self.master.geometry("700x450")  # Definindo formato vertical (600x800)

        self.can_data = None
        self.best_results = []  # Lista para armazenar os DataFrames de resultados finais
        self.current_plot_index = 0  # Índice do gráfico atual
        self.nome_arquivo = None  # Variável para armazenar o nome do arquivo analisado
        self.canvas = None  # Variável para o canvas do gráfico

        # Definir cores
        dark_gray = "#333333"
        light_gray = "#CCCCCC"

        # Configurar fundo da tela
        self.master.configure(bg=light_gray)

        # Frame para o cabeçalho fixo
        self.header_frame = tk.Frame(self.master, width=500, height=100, bg=dark_gray)
        self.header_frame.pack_propagate(0)  # Manter a altura fixa
        self.header_frame.pack(side=tk.TOP, fill=tk.X)

        # Frame para exibir informações ou gráficos
        self.content_frame = tk.Frame(self.master, bg=light_gray)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame para navegação de gráficos
        self.navigation_frame = tk.Frame(self.master, width=500, height=100, bg=dark_gray)
        self.navigation_frame.pack_propagate(0)  # Manter a altura fixa
        self.navigation_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Botão "Carregar dados"
        self.btn_load = tk.Button(
            self.header_frame,
            text="Carregar ixxt csv",
            command=lambda: ui_load_ixxt_archive(self),
            width=15,
            height=2,
            bg="#4CAF50",
            fg="white"
        )
        self.btn_load.pack(side=tk.LEFT, padx=5)

        # Chamar método para criar o menu drop-down e botão Carregar Arquivo
        self.create_dropdown_menu_in_header()

        # Botão "Carregar dados"
        self.btn_load = tk.Button(
            self.header_frame,
            text="Carregar Arquivo",
            command=lambda: self.load_file_callback(self.selected_file.get()),
            width=15,
            height=2,
            bg="#4CAF50",
            fg="white"
        )
        self.btn_load.pack(side=tk.LEFT, padx=5)


        # Botão "Usar Modelo"
        self.btn_model = tk.Button(
            self.header_frame,
            text="Usar Modelo",
            command=self.use_model,
            width=15,
            height=2,
            bg="#FF9800",
            fg="white"
        )
        self.btn_model.pack(side=tk.LEFT, padx=5)

        # Botão "Plotar Gráfico"
        self.btn_plot = tk.Button(
            self.header_frame,
            text="Plotar Gráfico",
            command=self.plotar_grafico,
            width=15,
            height=2,
            bg="#2196F3",
            fg="white"
        )
        self.btn_plot.pack(side=tk.RIGHT, padx=5)

        # Botões de navegação de gráficos (no rodapé)
        self.btn_prev = tk.Button(
            self.navigation_frame,
            text="Anterior",
            command=self.show_prev_plot,
            width=12,
            height=2,
            bg="#9E9E9E",
            fg="white"
        )
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = tk.Button(
            self.navigation_frame,
            text="Próximo",
            command=self.show_next_plot,
            width=12,
            height=2,
            bg="#9E9E9E",
            fg="white"
        )
        self.btn_next.pack(side=tk.LEFT, padx=5)

        # Label para mostrar o título do gráfico atual
        self.plot_title = tk.Label(
            self.navigation_frame,
            text="Nenhum gráfico carregado",
            font=("Helvetica", 12),
            bg=light_gray
        )
        self.plot_title.pack(side=tk.LEFT, padx=10)

#-----------------------------------------------------------------------#

    def create_dropdown_menu_in_header(self):
        directory = "CAN_DataChunks"  # Diretório onde os arquivos CSV estão armazenados

        # Filtra apenas arquivos CSV no diretório
        file_list = [f for f in os.listdir(directory) if f.endswith('.csv')]
        if not file_list:
            messagebox.showwarning("Aviso", "Nenhum arquivo CSV encontrado.")
            file_list = [""]  # Evitar erro se não houver arquivos

        # Variável para armazenar o arquivo selecionado
        self.selected_file = tk.StringVar(self.master)
        self.selected_file.set(file_list[0])  # Definir o primeiro arquivo como padrão

        # Menu suspenso para seleção de arquivos
        dropdown = tk.OptionMenu(self.header_frame, self.selected_file, *file_list)
        dropdown.config(width=20)
        dropdown.pack(side=tk.LEFT, padx=5, pady=20)


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
                f"PGN: {row['pgn']}, {row['byte_column']}, Acurácia: {row['accuracy']}, Número de Picos: {row['num_peaks']}"
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
        """
        Exibe o primeiro gráfico da lista de best_results dentro do Tkinter e no terminal para debug.
        """
        # --- Verificar se o DataFrame está vazio ---
        if self.best_results.empty:
            messagebox.showwarning("Aviso", "Nenhum resultado disponível para plotar.")
            return

        # # --- Limpar o conteúdo anterior ---
        # for widget in self.content_frame.winfo_children():
        #     widget.destroy()

        # self.current_plot_index = 0  # Começar com o primeiro gráfico
        self.display_current_plot()  # Exibe o primeiro gráfico


    def display_current_plot(self):
        """
        Exibe o gráfico atual da lista de best_results dentro da interface Tkinter.
        """
        # --- Verificar se o DataFrame está vazio ---
        if self.best_results.empty or self.current_plot_index < 0 or self.current_plot_index >= len(self.best_results):
            messagebox.showwarning("Aviso", "Nenhum gráfico para exibir.")
            return

        # --- Limpar o frame de conteúdo ---
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Obtém os dados do gráfico atual
        row = self.best_results.iloc[self.current_plot_index]
        pgn = row['pgn']
        byte_column = row['byte_column']
        accuracy = row['accuracy']
        sequence = row['sequence'].flatten()  # Certifique-se de que a sequência é 1D

        # --- Plotar o gráfico no terminal para debug ---
        print(f"PGN {pgn}, Byte {byte_column}, Acurácia {accuracy}")
        plt.figure(figsize=(5, 3))
        plt.plot(sequence, label=f"PGN {pgn}, Byte {byte_column} (Acurácia: {accuracy:.2f})")
        plt.xlabel('Timestep')
        plt.ylabel('Valor')
        plt.title(f"Melhor sequência para PGN {pgn}, Byte {byte_column}")
        plt.legend()
        plt.show()  # Mantém a plotagem no terminal para debug

        # --- Criar a figura para o Tkinter ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sequence, label=f"PGN {pgn}, Byte {byte_column} (Acurácia: {accuracy:.2f})")
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Valor')
        ax.set_title(f"Melhor sequência para PGN {pgn}, Byte {byte_column}")
        ax.legend()

        # --- Exibir o gráfico no Tkinter ---
        self.canvas = FigureCanvasTkAgg(fig, master=self.content_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Atualizar o título do gráfico
        self.plot_title.config(text=f"Gráfico {self.current_plot_index + 1} de {len(self.best_results)}")

    def show_prev_plot(self):
        if self.best_results.empty:  # Use .empty para verificar se o DataFrame está vazio
            return
        if self.current_plot_index > 0:
            self.current_plot_index -= 1
            self.display_current_plot()
        else:
            messagebox.showinfo("Informação", "Este é o primeiro gráfico.")


    def show_next_plot(self):
        if self.best_results.empty:  # Use .empty para verificar se o DataFrame está vazio
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
