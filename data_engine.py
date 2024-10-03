from tkinter import ttk, filedialog, messagebox
import pandas as pd
import tkinter as tk
from lstm_engine import *

def ui_load_ixxt_archive(self):
    file_path = filedialog.askopenfilename(
        title="Selecione o arquivo CSV",
        filetypes=(("CSV files", "*.CSV;*.csv"), ("All files", "*.*"))
    )

    if file_path:
        self.nome_arquivo = file_path  # Armazena o nome do arquivo analisado
        self.can_data = ixxt_filetodataframe(file_path)
        messagebox.showinfo("Sucesso", f"Arquivo '{file_path}' carregado com sucesso!")
        self.exibir_dataframe()
    else:
        messagebox.showwarning("Aviso", "Nenhum arquivo foi selecionado.")


def ixxt_filetodataframe(file_path):
    df = pd.read_csv(file_path, sep=';', quotechar='"', skiprows=6)
    df = df.dropna(subset=['Data (hex)'])
    df['Identifier (hex)'] = df['Identifier (hex)'].apply(lambda x: int(x, 16))

    byte_columns = df['Data (hex)'].apply(process_data)

    byte_columns = pd.DataFrame(byte_columns.tolist(), index=df.index, columns=[f'byte_{i + 1}' for i in range(8)])
    print(byte_columns)
    columns_to_drop = ['Format', 'Flags', 'Time', 'Data (hex)']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df = pd.concat([df, byte_columns], axis=1)
    df = df.rename(columns={'Identifier (hex)': 'pgn'})
    return df


def load_selected_file(selected_file, directory = "CAN_DataChunks"):
  # Diretório onde os arquivos estão localizados
    file_path = os.path.join(directory, selected_file)  # Concatena o diretório com o nome do arquivo
    if selected_file:
        try:
            df = pd.read_csv(file_path)
            messagebox.showinfo("Informação", f"Arquivo {selected_file} carregado com sucesso!")
            return df
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar o arquivo: {str(e)}")
    else:
        messagebox.showwarning("Aviso", "Nenhum arquivo selecionado.")



def save_raw_dataframe(df):
    if df is not None:
        save_path = "raw_data_output.csv"  # Nome do arquivo salvo
        df.to_csv(save_path, index=False)
        messagebox.showinfo("Sucesso", f"DataFrame salvo com sucesso em {save_path}")
    else:
        messagebox.showwarning("Aviso", "Nenhum DataFrame para salvar.")



def create_dropdown_menu(root, file_list, callback):
    selected_file = tk.StringVar()
    dropdown = ttk.Combobox(root, textvariable=selected_file, values=file_list)
    dropdown.pack(pady=10)

    # Adicionar botão para carregar o arquivo selecionado
    load_button = tk.Button(root, text="Carregar arquivo", command=lambda: callback(selected_file.get()))
    load_button.pack(pady=5)

    return dropdown