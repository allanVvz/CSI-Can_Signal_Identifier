from tkinter import ttk, filedialog, messagebox
import pandas as pd
import tkinter as tk
from lstm_engine import *
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from reportlab.lib.units import inch
import tempfile  # Para criar arquivos temporários


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


def ui_load_ms_archive(self):
    # Se o caminho não for fornecido, abrir a caixa de diálogo para selecionar o arquivo
    file_path = filedialog.askopenfilename(
        title="Selecione o arquivo .CAN ou .TXT",
        filetypes=(("CAN files", "*.can"), ("TXT files", "*.txt"), ("all files", "*.*"))
    )
    if file_path:
        self.nome_arquivo = file_path
        self.can_data = ms_filetodataframe(file_path)  # Carrega os dados do arquivo
        messagebox.showinfo("Sucesso", f"Arquivo '{file_path}' carregado com sucesso!")
        self.exibir_dataframe()
    else:
        messagebox.showwarning("Aviso", "Nenhum arquivo foi selecionado.")


def ui_load_can_archive(self):
    # Se o caminho não for fornecido, abrir a caixa de diálogo para selecionar o arquivo
    file_path = filedialog.askopenfilename(
        title="Selecione o arquivo .CAN ou .TXT",
        filetypes=(("CAN files", "*.can"), ("TXT files", "*.txt"), ("all files", "*.*"))
    )
    if file_path:
        self.nome_arquivo = file_path
        self.can_data = can_filetodataframe(file_path)  # Carrega os dados do arquivo
        if self.can_data is not None:
            messagebox.showinfo("Sucesso", f"Arquivo '{file_path}' carregado com sucesso!")
            self.exibir_dataframe()
        else:
            messagebox.showwarning("Erro", "Erro ao carregar o arquivo.")
    else:
        messagebox.showwarning("Aviso", "Nenhum arquivo foi selecionado.")

def is_valid_hex(value):
    """Função para verificar se o valor é hexadecimal válido"""
    try:
        int(value, 16)
        return True
    except ValueError:
        return False


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


def ms_filetodataframe(file_path):
    try:
        # Ler o arquivo .txt linha a linha
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Separar as colunas por ';' e montar a lista de dados
        data = []
        for line in lines:
            # Ignorar as linhas de cabeçalho e as que não têm a quantidade esperada de colunas
            if line.startswith("//") or "Logging" in line or "Microchip" in line:
                continue

            # Dividir os valores por ';'
            parts = line.strip().split(';')

            # Garantir que tenha o número mínimo de colunas para processar (pelo menos 5)
            if len(parts) < 5:
                continue

            # Extrair o ID e os bytes
            pgn = parts[2]  # PGN (ID)
            byte_count = int(parts[3])  # Número de bytes
            byte_values = parts[4:4 + byte_count]  # Extraindo os bytes reais

            # Preencher com '00' caso não tenha 8 bytes
            if len(byte_values) < 8:
                byte_values.extend(['00'] * (8 - len(byte_values)))

            # Adicionar a linha processada (PGN + 8 bytes)
            data.append([pgn] + byte_values[:8])  # Garantir que tenha no máximo 8 bytes

        # Converter a lista de dados em um DataFrame
        df = pd.DataFrame(data, columns=['pgn', 'byte_1', 'byte_2', 'byte_3', 'byte_4',
                                         'byte_5', 'byte_6', 'byte_7', 'byte_8'])

        # Aplicar a conversão de hexadecimal para decimal nas colunas de bytes e no PGN
        for i in range(1, 9):
            df[f'byte_{i}'] = df[f'byte_{i}'].apply(lambda x: int(x, 16) if is_valid_hex(x) else 0)

        # # Converter o PGN de hexadecimal para decimal
        # df['pgn'] = df['pgn'].apply(lambda x: int(x, 16) if is_valid_hex(x) else 0)

        return df

    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return None



def can_process_data(data_hex):
    bytes_list = data_hex.strip().split()

    # Preencher com '00' até que tenha 8 bytes
    if len(bytes_list) < 8:
        bytes_list.extend(['00'] * (8 - len(bytes_list)))

    return bytes_list[:8]  # Garantir que tenha no máximo 8 bytes

def can_filetodataframe(file_path):
    try:
        # Ler o arquivo .txt linha a linha
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Separar as colunas e montar a lista de dados
        data = []
        for line in lines:
            # Ignorar cabeçalhos ou linhas inválidas
            if line.startswith("Time") or line.startswith("ASCII") or "ID(hex)" in line:
                continue

            # Dividir os valores por espaços
            parts = line.strip().split()

            # Garantir que tenha o número mínimo de colunas para processar
            if len(parts) < 5:
                continue

            # # Extrair o ID (hexadecimal) e convertê-lo para decimal
            #pgn_hex = parts[1]  # ID(hex)
            # try:
            #     pgn_decimal = int(pgn_hex, 16)  # Converte para decimal
            # except ValueError:
            #     pgn_decimal = 0  # Definir como 0 se a conversão falhar

            # Extrair os bytes da coluna 'Data (hex)'
            byte_values = parts[4:12]  # Extraindo os 8 bytes reais

            # Preencher com '00' caso não tenha 8 bytes
            if len(byte_values) < 8:
                byte_values.extend(['00'] * (8 - len(byte_values)))

            # Adicionar a linha processada (PGN + 8 bytes)
            data.append([parts[1]] + byte_values[:8])  # Garantir que tenha no máximo 8 bytes

        # Converter a lista de dados em um DataFrame
        df = pd.DataFrame(data, columns=['pgn', 'byte_1', 'byte_2', 'byte_3', 'byte_4',
                                         'byte_5', 'byte_6', 'byte_7', 'byte_8'])

        # Aplicar a conversão de hexadecimal para decimal nas colunas de bytes
        for i in range(1, 9):
            df[f'byte_{i}'] = df[f'byte_{i}'].apply(lambda x: int(x, 16) if is_valid_hex(x) else 0)

        return df

    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return None



def load_selected_file(selected_file, directory = "CANDataChunks"):
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



# def create_dropdown_menu(root, file_list, callback):
#     selected_file = tk.StringVar()
#     dropdown = ttk.Combobox(root, textvariable=selected_file, values=file_list)
#     dropdown.pack(pady=10)
#
#     # Adicionar botão para carregar o arquivo selecionado
#     load_button = tk.Button(root, text="Carregar arquivo", command=lambda: callback(selected_file.get()))
#     load_button.pack(pady=5)
#
#     return dropdown

def gerar_pdf_com_graficos(best_results, pdf_filename="graficos_pgn.pdf"):
    """
    Gera um arquivo PDF contendo todos os gráficos a partir dos dados do DataFrame best_results.
    """
    # Verifica se há dados para plotar
    if best_results.empty:
        raise ValueError("Nenhum resultado disponível para gerar o PDF.")

    # Cria o canvas para o PDF
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    # Para cada gráfico, crie a figura e adicione ao PDF
    for index, row in best_results.iterrows():
        pgn = row['pgn']
        byte_column = row['byte_column']
        accuracy = row['accuracy']
        sequence = row['sequence'].flatten()  # Certifique-se de que a sequência é 1D

        # Cria o gráfico no Matplotlib
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sequence, color='red', label=f"PGN {pgn}, {byte_column} (Acurácia: {accuracy:.2f})")
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Valor')
        ax.set_title(f"Melhor sequência para PGN {pgn},{byte_column}")
        ax.legend()

        # Salva o gráfico em um buffer de memória (em vez de exibi-lo)
        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        buf.seek(0)
        plt.close(fig)

        # Usar PIL para converter o gráfico em uma imagem
        image = Image.open(buf)
        image_width, image_height = image.size

        # Redimensiona a imagem para caber no PDF
        aspect = image_height / float(image_width)
        new_width = width * 0.8
        new_height = new_width * aspect
        image = image.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)

        # Salva a imagem temporariamente em um arquivo PNG
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)  # Cria um arquivo temporário
        image.save(temp_file.name, 'PNG')

        # Desenha a imagem no PDF usando o arquivo temporário
        c.drawImage(temp_file.name, 0.5 * inch, height - new_height - 1 * inch, width=new_width, height=new_height)

        # Remove o arquivo temporário
        temp_file.close()
        os.remove(temp_file.name)

        # Adiciona uma nova página no PDF para o próximo gráfico
        c.showPage()

    # Salva o PDF
    c.save()
