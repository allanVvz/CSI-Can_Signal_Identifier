def find_most_similar_sequence(df, model, n_timesteps, test_mode=False):
    # Se o modo de teste estiver ativado, ajusta n_timesteps para o tamanho exato da planilha
    if test_mode:
        n_timesteps = len(df)
        print(f"Modo de teste ativado: n_timesteps ajustado para {n_timesteps} (tamanho do DataFrame)")

    # Verifica se o DataFrame tem dados suficientes para extrair sequências
    if len(df) < n_timesteps:
        raise ValueError("O DataFrame não tem linhas suficientes para o número de timesteps especificado.")

    # Obter todos os pgn únicos
    unique_pgns = df['pgn'].unique()
    print(f"Unique PGNs: {unique_pgns}")

    overall_best_sequence = None
    overall_best_score = -np.inf
    overall_best_index = -1
    overall_best_pgn = None

    # Itera sobre cada pgn único
    for pgn in unique_pgns:
        # Filtra o DataFrame para o pgn atual
        df_pgn = df[df['pgn'] == pgn]
        print(f"DataFrame para PGN {pgn}:\n{df_pgn}")

        best_sequence = None
        best_score = -np.inf
        best_index = -1

        # Verifica se o DataFrame filtrado tem dados suficientes para extrair sequências
        if len(df_pgn) < n_timesteps:
            print("pulado para a proxima! dados insuficientes")
            continue  # Pula para o próximo pgn se não houver dados suficientes

        for i in range(len(df_pgn) - n_timesteps + 1):
            for byte_filter in range(7):
                # Extrair a subsequência
                column_name = f'byte_{byte_filter}'
                subsequence = df_pgn[column_name].iloc[i:i + n_timesteps].values.reshape(1, n_timesteps, 1)
                print(subsequence)
                # Fazer a previsão para a subsequência
                score = model.predict(subsequence, verbose=0).flatten()


                sum_score = np.sum(score)
                print(f"Score predito: {score}, Soma do Score: {sum_score}")

                if sum_score > best_score:
                    best_score = sum_score
                    best_sequence = subsequence.flatten()
                    best_index = df_pgn.index[i]  # Captura o índice original no DataFrame

        # Atualiza o melhor global se o atual for superior
        if best_score > overall_best_score:
            overall_best_score = best_score
            overall_best_sequence = best_sequence
            overall_best_index = best_index
            overall_best_pgn = pgn
            print(f"{overall_best_sequence}\n{overall_best_pgn}")

    # Plote a melhor subsequência encontrada
    if overall_best_sequence is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(range(overall_best_index, overall_best_index + n_timesteps), overall_best_sequence, marker='o', linestyle='-',
                 color='blue', label=f'Melhor Sequência para PGN {overall_best_pgn}')
        plt.title('Sequência Numérica que Mais se Assemelha ao Dado de Treino')
        plt.xlabel('Índice do DataFrame')
        plt.ylabel('Valor da Sequência')
        plt.legend()
        plt.grid(True)
        plt.show()

    return overall_best_sequence, overall_best_index, overall_best_pgn






def discard_constant_pgn(df, n_timesteps):
    # Identifica os pgn únicos
    unique_pgns = df['pgn'].unique()

    # Lista para armazenar os índices dos PGNs a serem removidos
    indices_to_remove = []

    # Itera sobre cada pgn único
    for pgn in unique_pgns:
        # Filtra o DataFrame para o pgn atual
        df_pgn = df[df['pgn'] == pgn]

        # Verifica se o DataFrame filtrado tem dados suficientes para n_timesteps
        if len(df_pgn) >= n_timesteps * 2:  # Verifica para um tamanho maior de n_timesteps
            # Aqui o passo é aumentado para diminuir a quantidade de ciclos
            for i in range(0, len(df_pgn) - n_timesteps + 1, n_timesteps):
                # Extrair a subsequência com o tamanho aumentado
                subsequence = df_pgn.iloc[i:i + n_timesteps]

                # Verifica se todos os valores em todas as colunas da subsequência são iguais
                if (subsequence.iloc[:, 2:] == subsequence.iloc[0, 2:]).all().all():
                    indices_to_remove.extend(subsequence.index.tolist())

    # Remove as sequências com valores constantes
    df_cleaned = df.drop(indices_to_remove).reset_index(drop=True)

    return df_cleaned





def find_patterns(df, n_timesteps, byte_filter, model):
    # Aumenta o tamanho do get_sequence (ou seja, considera mais dados por sequência)
    X, _ = get_sequence(n_timesteps * 2, df, byte_filter)  # Aqui o tamanho da sequência é aumentado

    # Fazer a previsão usando o modelo treinado
    yhat = model.predict(X, verbose=0)

    # Interpretação das previsões
    predictions = yhat.flatten()

    # Exibir resultados ou realizar alguma ação com base nas previsões
    for i in range(0, len(predictions), 2):  # Diminui o número de ciclos ao pular elementos
        print(f"Timestep {i + 1}: {predictions[i]}")

    # Plote o gráfico de pontos
    plt.figure(figsize=(10, 6))
    plt.scatter(range(1, len(predictions) + 1, 2), predictions[::2], color='blue')  # Plota apenas pontos selecionados
    plt.title('Previsões do Modelo')
    plt.xlabel('Timesteps')
    plt.ylabel('Valor Predito')
    plt.grid(True)
    plt.show()

    return predictions


def plot_best_pgns(best_pgns, df):
    for pgn, i, accuracy, y_pred, X, y in best_pgns:
        df_pgn = df[df['pgn'] == pgn].copy()
        df_pgn = df_pgn[df_pgn['data'].apply(lambda x: isinstance(x, list) and len(x) == 8)]
        df_pgn[f'byte_{i}'] = df_pgn['data'].apply(lambda x: int(x[i], 16))

        # Verifica se a quantidade de y_pred corresponde ao tamanho de df_pgn
        if len(y_pred) != len(df_pgn):
            print(f"Warning: Length mismatch for PGN {pgn}, byte {i}")
            continue

        # Assegura que 'event_time' e 'y_pred' tenham o mesmo comprimento
        event_times = df_pgn['event_time'].iloc[:len(y_pred)]

        # Plota os dados
        fig = plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(df_pgn['event_time'], df_pgn[f'byte_{i}'], linestyle='-', color='b', linewidth=0.5, label='Real Data')
        plt.scatter(event_times, y_pred, color='r', marker='x', label='Predicted Data')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'PGN {pgn}, Byte {i} - Accuracy: {accuracy:.2f}')
        plt.legend()
        plt.show()


def plot_data(df, y_pred, title):
    """
    Função para plotar os dados reais vs previstos.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['data'], color='blue', label='Dados Reais')
    plt.plot(df.index, y_pred, color='red', linestyle='--', label='Dados Previstos')
    plt.title(title)
    plt.xlabel('Índice')
    plt.ylabel('Valores')
    plt.legend()
    plt.grid(True)
    plt.show()

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



# def train_sequence_model(df_train):
#     n_timesteps = 100  # Tamanho da janela
#     num_iterations = 20
#     model_filename = 'model.pkl'
#     scaler_filename = 'scaler.pkl'
#     print(df_train)
#
#     # Dividir o DataFrame em conjunto de treino e teste
#     #df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
#
#     # Extrair os valores da coluna 'x', que são strings de valores concatenados
#     df_train_sorted = df_train.sort_index().copy()
#     df_test_sorted = df_test.sort_index().copy()
#
#     plot_line_for_each_row(df_train_sorted)
#     plot_line_for_each_row(df_test_sorted)
#
#     # Extrair e processar os valores de 'x'
#     train_sequences = []
#     test_sequences = []
#     train_labels = []
#     test_labels = []
#
#     # Processar os dados de treino
#     for index, row in df_train_sorted.iterrows():
#         x_values = list(row['x'])  # Transformar string em lista de floats
#         train_sequences.append(x_values)
#         label = row['label']  # Supomos que existe uma coluna 'label' com os rótulos corretos
#         train_labels.append(label)
#
#
#     # Processar os dados de teste
#     for index, row in df_test_sorted.iterrows():
#         x_values = list(row['x'])  # Transformar string em lista de floats
#         test_sequences.append(x_values)
#         label = row['label']  # Supomos que existe uma coluna 'label' com os rótulos corretos
#         test_labels.append(label)
#
#     print(test_labels)
#
#     # Normalizar os valores de treino e teste
#     train_sequences_flat = [item for sublist in train_sequences for item in sublist]
#     test_sequences_flat = [item for sublist in test_sequences for item in sublist]
#
#
#     # Obter sequências de treino e teste
#     X_train = []
#     y_train = []
#     X_test = []
#     y_test = []
#
#     # Obter as sequências e rótulos para o conjunto de treino
#     for i, seq in enumerate(train_sequences):
#         X_seq, y_seq = get_sequence(n_timesteps, train_sequences_flat[i * len(seq):(i + 1) * len(seq)], train_labels[i])
#         X_train.extend(X_seq)
#         y_train.extend(y_seq)
#
#     # Obter as sequências e rótulos para o conjunto de teste
#     for i, seq in enumerate(test_sequences):
#         X_seq, y_seq = get_sequence(n_timesteps, test_sequences_flat[i * len(seq):(i + 1) * len(seq)], test_labels[i])
#         X_test.extend(X_seq)
#         y_test.extend(y_seq)
#
#     # Converter para numpy arrays e reshape para entrada do modelo
#     X_train = np.array(X_train).reshape(-1, n_timesteps, 1)
#     y_train = np.array(y_train).reshape(-1, 1)
#     X_test = np.array(X_test).reshape(-1, n_timesteps, 1)
#     y_test = np.array(y_test).reshape(-1, 1)
#
#     # Treinar o modelo LSTM
#     model = get_bi_lstm_model(n_timesteps, 'concat')
#
#     start_time = timeit.default_timer()
#
#     # Treinamento iterativo
#     model.fit(X_train, y_train, epochs=num_iterations, batch_size=1, verbose=1)
#
#     end_time = timeit.default_timer()
#
#     # Fazer predições no conjunto de teste
#     y_pred = model.predict(X_test)
#
#     # Converter previsões contínuas para rótulos binários (0 ou 1) usando limiar de 0.5
#     y_pred_labels = (y_pred > 0.5).astype(int).flatten()
#     y_test_labels = (y_test > 0.5).astype(int).flatten()






# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, confusion_matrix
# import timeit
#
# # Função auxiliar para plotar
# def plot_sequences(X, y, title):
#     plt.figure(figsize=(10, 6))
#     for i in range(min(12, len(X))):  # Mostrar no máximo 12 sequências
#         plt.plot(X[i].flatten(), label=f"Sequência {i} - Label: {y[i][0]}")
#     plt.title(title)
#     plt.xlabel('Timestep')
#     plt.ylabel('Valor')
#     plt.legend()
#     plt.show()
#
# def train_sequence_model(df_train_augmented, df_test_augmented):
#     n_timesteps = 600  # Tamanho da janela
#     num_iterations = 1
#
#     # Processar os dados de treino e teste
#     train_sequences = []
#     test_sequences = []
#     train_labels = []
#     test_labels = []
#
#     model = get_bi_lstm_model(n_timesteps, 'concat')
#
#     y_train = np.array([label]).reshape(-1, 1)
#
#
#     # Processar os dados de treino
#     print("Processando dados de treino...")
#     for index, row in df_train_augmented.iterrows():
#         x_values = np.array(row['x'])  # Transformar lista de valores diretamente em np.array
#         label = row['label']
#         start_time = timeit.default_timer()
#         # Criação de janelas deslizantes para o conjunto de treino
#         for start in range(0, len(x_values) - n_timesteps + 1, 1):  # Janela deslizante
#             window = x_values[start:start + n_timesteps]
#
#             # Redefinir X_train e y_train para cada janela
#             X_train = np.array(window).reshape(-1, n_timesteps, 1)  # Ajustar o formato
#             print(X_train.shape)
#             y_train = np.array([label]).reshape(-1, 1)  # Um rótulo por janela
#             print(y_train.shape)
#             # Treinar o modelo LSTM com a janela atual
#             model.fit(X_train, y_train, epochs=num_iterations, batch_size=n_timesteps, verbose=2)
#
#         end_time = timeit.default_timer()
#
#
#     # Processar os dados de teste
#     print("Processando dados de teste...")
#     for index, row in df_test_augmented.iterrows():
#         x_values = np.array(row['x'])  # Transformar lista de valores diretamente em np.array
#         label = row['label']
#
#         # Criação de janelas deslizantes para o conjunto de teste
#         for start in range(0, len(x_values) - n_timesteps + 1, 1):  # Janela deslizante
#             window = x_values[start:start + n_timesteps]
#             test_sequences.append(window)
#             test_labels.append(label)
#
#
#     # Converter as sequências de teste em arrays
#     test_sequences_flat = np.array(test_sequences)
#     X_test = test_sequences_flat.reshape(-1, n_timesteps, 1)
#     y_test = np.array(test_labels).reshape(-1, 1)
#
#     # Fazer predições no conjunto de teste
#     y_pred = model.predict(X_test)
#
#     # Printar os dados
#     print("Últimas 5 sequências de X_train:")
#     print(X_train[-5:])
#     print("Últimos 5 rótulos de y_train:")
#     print(y_train[-5:])
#
#     # Plotar as últimas 20 sequências de treino e teste
#     plot_sequences(X_train[-20:], y_train[-20:], "Sequências de Treino (X_train)")
#     plot_sequences(X_test[-20:], y_test[-20:], "Sequências de Teste (X_test)")
#
#     # Converter previsões contínuas para rótulos 1 ou 2 usando um limiar de 1.5
#     y_pred_labels = (y_pred > 1.5).astype(int).flatten() + 1  # Ajustar para rótulos 1 ou 2
#     y_test_labels = y_test.flatten()  # Usar os rótulos reais diretamente de y_test
#
#     # Cálculo da acurácia e matriz de confusão
#     accuracy = accuracy_score(y_test_labels, y_pred_labels)
#     labels = [1, 2]
#     conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, labels=labels)
#
#     print(f"Accuracy = {accuracy:.2f}, Time = {end_time - start_time:.4f} seconds")
#     print("Confusion Matrix:\n", conf_matrix)
#
#     for i in range(len(y_pred_labels)):
#         print(f"Prediction: {y_pred_labels[i]}, Confidence: {y_pred[i][0]:.2f}")
#
#     return model
