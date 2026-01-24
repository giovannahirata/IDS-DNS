# inferenciaPickle.py

"""
1. implementar este script que:
    - carregará os arquivos .pkl dos modelos (binário e multiclasse)
    - carregará o pré-processamento e as features selecionadas
    - com o modelo e as features em mãos, realizar a predição tal qual feito
    no script de treinamento, a diferença é que não realizaremos o treinamento,
    apenas a inferência/predição e avaliação das métricas (comparando ao original)
Passar este programa e os dataset bruto juntamente com o diretório /modelos_salvos
para outra máquina (testar godzilla e raspberry8) 
(fazer o mesmo procedimento para o joblib)

"""

import pickle
import pandas as pd
import numpy as np
import os

print("Carregando os modelos para inferência....")

dir = "modelos_salvos"

try:
    with open(os.path.join(dir, "modelo_binario.pkl"), "rb") as f:
        modelo_bin = pickle.load(f)
    with open(os.path.join(dir, "modelo_multiclasse.pkl"), "rb") as f:
        modelo_multi = pickle.load(f)
    with open(os.path.join(dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(dir, "selected_columns.pkl"), "rb") as f:
        final_columns = pickle.load(f)
    with open(os.path.join(dir, "label_encoder_protocol.pkl"), "rb") as f:
        le_protocol = pickle.load(f)
    with open(os.path.join(dir, "categorical_columns.pkl"), "rb") as f:
        cat_cols = pickle.load(f)
except FileNotFoundError as e:
    print(f"Arquivo pickle não encontrado: {e}")
except Exception as e:
    print(f"Erro ao carregar arquivos pickle: {e}")


print("Modelos carregados.")

# função para predição:

def predicao(dados_dict):
    """
    recebe os dados de um fluxo dns com as features especificadas e
    padronizadas para o modelo e retorna a predição.

    obs.: esta função executa exatamente o mesmo pipeline de pré-processamento
    usado no treinamento do modelo.
    """

    # pipeline de pré-processamento:

    try:
        df = pd.DataFrame([dados_dict])
    except Exception as e:
        return f"Erro ao criar dataframe: {e}"

    df = df.fillna(0)

    try:
        if "protocol" in df.columns:
            protocol_val = df["protocol"].values[0]
            df["protocol"] = le_protocol.transform([protocol_val])[0]
        else:
            df["protocol"] = 0
    except ValueError:
        print(f"Protocolo '{protocol_val}' desconhecido. Trocando por 0.")
        df["protocol"] = 0

    df = pd.get_dummies(df, columns=cat_cols)

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    df = df.reindex(columns=final_columns, fill_value=0)

    try:
        scaled = scaler.transform(df)
    except Exception as e:
        return f"Erro ao aplicar o scaler: {e}."

    # inferência:

    # predição binária:
    try:
        pred_bin = modelo_bin.predict(scaled)
    except Exception as e:
        return f"Erro na predição binária {e}."

    if pred_bin[0] == 0:
        return "[Benigno]"
    else:
	# é maligno, então realizamos a predição multiclasse:
        try:
            pred_multi = modelo_multi.predict(scaled)
            return f"[Maligno] -> {pred_multi[0]}"
        except Exception as e:
            return f"Erro na predição multiclasse {e}."


if __name__ == "__main__":
    print("Testando a inferência com dados do csv...")

    # leitura dos CSVs:

    benigns = ['output-of-benign-pcap-0.csv', 'output-of-benign-pcap-1.csv', 'output-of-benign-pcap-2.csv', 'output-of-benign-pcap-3.csv']
    df_benigns = [pd.read_csv(f) for f in benigns]
    df_benign = pd.concat(df_benigns)

    df_mal = pd.read_csv('output-of-malware-pcap.csv')

    df_phi = pd.read_csv('output-of-phishing-pcap.csv')

    df_spam = pd.read_csv('output-of-spam-pcap.csv')

    # testes:

    nl = 3 # numero de linhas do csv para realizar a inferencia

    #amostras_benigno = df_benign.sample(n=nl, random_state=42)
    amostras_benigno = df_benign.sample(n=nl)
    esperado_ben = "[Benigno]"

    #amostras_mal = df_mal.sample(n=nl, random_state=42)
    amostras_mal = df_mal.sample(n=nl)
    esperado_mal = "[Maligno] -> Malware"

    #amostras_phi = df_phi.sample(n=nl, random_state=42)
    amostras_phi = df_phi.sample(n=nl)
    esperado_phi = "[Maligno] -> Phishing"

    #amostras_spam = df_spam.sample(n=nl, random_state=42)
    amostras_spam = df_spam.sample(n=nl)
    esperado_spam = "[Maligno] -> Spam"

    for i, row in amostras_benigno.iterrows():
        data_dict = row.to_dict()
        print(f"Testando amostra de índice {i} do csv.")
        #print("Dados de entrada: ", data_dict)
        prediction = predicao(data_dict)
        print(f"Resultado da predição: {prediction}")
        print(f"Resultdao esperado: {esperado_ben}")

        if prediction == esperado_ben:
            print("[✓] Teste passou!")
        else:
            print("[X] Teste falhou :(")

    for i, row in amostras_mal.iterrows():
        data_dict = row.to_dict()
        print(f"Testando amostra de índice {i} do csv.")
        #print("Dados de entrada: ", data_dict)
        prediction = predicao(data_dict) 
        print(f"Resultado da predição: {prediction}")
        print(f"Resultdao esperado: {esperado_mal}")
        if prediction == esperado_mal:
            print("[✓] Teste passou!")
        else:
            print("[X] Teste falhou :(")

    for i, row in amostras_phi.iterrows():
        data_dict = row.to_dict()
        print(f"Testanto amostra de índice {i} do csv.")
        prediction = predicao(data_dict)
        print(f"Resultado da predição: {prediction}")
        print(f"Resultdao esperado: {esperado_phi}")
        if prediction == esperado_phi:
            print("[✓] Teste passou!")
        else:
            print("[X] Teste falhou :(")

    for i, row in amostras_spam.iterrows():
        data_dict = row.to_dict()
        print(f"Testanto amostra de índice {i} do csv.")
        prediction = predicao(data_dict)
        print(f"Resultado da predição: {prediction}")
        print(f"Resultdao esperado: {esperado_spam}")
        if prediction == esperado_spam:
            print("[✓] Teste passou!")
        else:
            print("[X] Teste falhou :(")
