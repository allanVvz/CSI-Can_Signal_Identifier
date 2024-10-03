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



def create_dropdown_menu(root, file_list, callback):
    selected_file = tk.StringVar()
    dropdown = ttk.Combobox(root, textvariable=selected_file, values=file_list)
    dropdown.pack(pady=10)

    # Adicionar botão para carregar o arquivo selecionado
    load_button = tk.Button(root, text="Carregar arquivo", command=lambda: callback(selected_file.get()))
    load_button.pack(pady=5)

    return dropdown

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
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sequence, color='red', label=f"PGN {pgn}, {byte_column} (Acurácia: {accuracy:.2f})")
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Valor')
        ax.set_title(f"Melhor sequência para PGN {pgn}, Byte {byte_column}")
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
        new_width = width * 0.9
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


def save_filter_train_dataframe():
    global df  # Certifique-se de que `df` está sendo carregado corretamente
    global global_filtered_df

    if df is None:
        messagebox.showwarning("Aviso", "Carregue um arquivo primeiro.")
        return None

    byte_filter = byte_entry.get()  # Pegando valor do campo de entrada de byte
    pgn_filter = pgn_entry.get()    # Pegando valor do campo de entrada de PGN

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
