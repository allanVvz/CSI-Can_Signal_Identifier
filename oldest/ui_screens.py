import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from data_processing import *
def select_file():
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal
    file_path = filedialog.askopenfilename(
        title="Selecione o arquivo CSV",
        filetypes=(("CSV files", "*.CSV"), ("All files", "*.*"))
    )
    return file_path

def train_model():

    num_points = 1000
    start_time = datetime.now()
    cycles = 5

    df_sawtooth_wave = generate_limited_sawtooth_wave_data(num_points, start_time, cycles)

    file_path = select_file()

    if file_path:
        df = load_data(file_path)
        print(df.head())  # Exibe as primeiras linhas do DataFrame

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

    else:
        print("Nenhum arquivo foi selecionado.")



    messagebox.showinfo("Ação", "Treinando o modelo de Machine Learning...")

# Função que será chamada quando o botão "Usar modelo ML" for clicado
def use_model():

    model_filename = '../model.pkl'
    scaler_filename = '../scaler.pkl'

    # Verificar se o modelo salvo já existe
    if os.path.exists(model_filename):
        print("Modelo encontrado. Carregando...")
        model, scaler = load_model(model_filename, scaler_filename)
        print(f",model: {model} \n ")
    else:
        print("Modelo não encontrado. Treinando novo modelo...")

    for pgn, i, accuracy, y_pred, X, y in best_pgns:
        print(f'Melhor PGN: {pgn}, Melhor Índice: {i}, Acurácia: {accuracy:.2f}')

    messagebox.showinfo("Ação", "Usando o modelo de Machine Learning...")

    plot_best_pgns(best_pgns, df)

def open_interactive_window():
    # Função que será chamada quando o botão "Treinar modelo ML" for clicado

    # Criando a janela principal
    root = tk.Tk()
    root.title("ML CAN ML SCANNER")
    root.geometry("500x250")  # Define o tamanho da janela

    tk.Label(root, text="").pack(pady=10)

    # Define o tamanho dos botões
    button_width = 20
    button_height = 2

    # Cor de fundo dos botões
    button_bg_color = "#4CAF50"  # Verde escuro (pode ser mudado para outra cor ou textura)

    # Criando e posicionando os botões na janela
    btn_train = tk.Button(root, text="Treinar modelo ML", command=train_model, width=button_width, height=button_height, bg=button_bg_color, fg="white")
    btn_train.pack(pady=10)

    btn_use = tk.Button(root, text="Usar modelo ML", command=use_model, width=button_width, height=button_height, bg=button_bg_color, fg="white")
    btn_use.pack(pady=10)

    # Mantém a janela aberta
    root.mainloop()