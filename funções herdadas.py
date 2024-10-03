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



    def load_default_dataframe():
        if os.path.exists(DEFAULT_FILE):
            try:
                global df
                df = pd.read_csv(DEFAULT_FILE)
                messagebox.showinfo("Informação", f"Arquivo {DEFAULT_FILE} carregado automaticamente.")
                return True
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar o arquivo padrão: {str(e)}")
                return False
        return False


    def load_dataframe():
        file_path = filedialog.askopenfilename(
            title="Selecione o arquivo CSV para carregar",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )

        if file_path:
            try:
                global df
                df = pd.read_csv(file_path)
                status_label.config(text="Dataframe filtrado carregado")
                messagebox.showinfo("Sucesso", "DataFrame carregado com sucesso!")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar o arquivo: {str(e)}")
        else:
            messagebox.showwarning("Aviso", "Nenhum arquivo selecionado.")


def analyse_archive(df, model):
    best_pgns = []

    for pgn in df['pgn'].unique():  # Assumindo que 'pgn' é uma coluna que identifica cada sequência
        df_pgn = df[df['pgn'] == pgn].copy()  # Filtra o DataFrame para o pgn atual

        for i in range(8):  # Assumindo que cada dado tem 8 bytes
            byte_column = f'byte_{i}'

            # Transformar a coluna de bytes em lista
            df_pgn[byte_column] = df_pgn['data'].apply(lambda x: int(x[i], 16))

            # Obter valores de byte e processá-los
            X = df_pgn[byte_column].values
            # peaks, troughs = count_peaks_and_troughs(byte_values)
            #
            # if len(peaks) <= 6 and len(troughs) <= 6:
            #     # Preparar X e y para treinamento
            #     X = df_pgn.drop(columns=['event_time', 'dummy_feature', 'data', 'pgn'])  # Ignora as colunas não relevantes
            #     y = df_pgn[byte_column].values

            if len(X) > 1:  # Garantir que temos dados suficientes e mais de uma classe
                # Treinar o modelo para este byte específico
                #model = train_sequence_model(df_pgn, i)
                y_pred = model.predict(X)

                # Calcular acurácia
                accuracy = accuracy_score(y, y_pred)

                # Armazenar os melhores pgns
                best_pgns.append((pgn, i, accuracy, y_pred, X, y))

    # return best_pgns



def save_df_raw(df):
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





    # def ui_load_ixxt_archive(self):
    #     file_path = filedialog.askopenfilename(
    #         title="Selecione o arquivo CSV",
    #         filetypes=(("CSV files", "*.CSV;*.csv"), ("All files", "*.*"))
    #     )
    #
    #     if file_path:
    #         df = self.ixxt_filetodataframe(file_path)
    #         if df is not None:
    #             df_normalizado = self.normalizar_dataframe(df)
    #             if df_normalizado is not None:
    #                 messagebox.showinfo("Sucesso", f"Arquivo '{file_path}' carregado e normalizado com sucesso!")
    #                 self.exibir_dataframe(df_normalizado)
    #     else:
    #         messagebox.showwarning("Aviso", "Nenhum arquivo foi selecionado.")





    # def use_model(self):
    #     if self.ixxt_data is None:
    #         messagebox.showwarning("Aviso", "Nenhum dado carregado para usar o modelo.")
    #         return
    #
    #     model_filename = 'meu_modelo.pkl'
    #
    #     # Carregar o modelo
    #     model = self.carregar_modelo(self, model_filename)
    #     if model is None:
    #         return
    #
    #     # Etapa 1: Analisar o DataFrame e coletar dados após descartar PGNs constantes
    #     data_df = self.analyse_archive(self.ixxt_data)
    #
    #     # Etapa 2: Ajustar as sequências para o tamanho desejado
    #     validated_df = prepare_sequences(data_df, target_size=1200)
    #
    #     # Etapa 3: Classificar as sequências usando o modelo
    #     classified_df = classify_sequences(model, validated_df)
    #
    #     # Etapa 4: Aplicar a validação de picos
    #     final_results_df = validate_peaks(classified_df)
    #
    #     # Armazenar os resultados finais
    #     self.best_results = final_results_df
    #
    #     # Exibir os resultados finais no terminal
    #     print("\nResultados finais após validação de picos:")
    #     for index, row in final_results_df.iterrows():
    #         print(
    #             f"PGN: {row['pgn']}, Byte: {row['byte_column']}, Acurácia: {row['accuracy']}, Número de Picos: {row['num_peaks']}")
    #
    #     # Plotar os melhores PGNs dentro do Tkinter
    #     self.plot_best_pgns(final_results_df)
    #
    #     # Exibir o DataFrame final na interface
    #     self.exibir_dataframe(final_results_df)