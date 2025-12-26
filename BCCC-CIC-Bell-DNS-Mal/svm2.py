# (re)começando do zero

"""
objetivo: dados os dataset BCCC-CIC-Bell-DNS-Mal, desenvolver um modelo SVM baseado em redes neurais sobre este dataset de forma a detectar 
e classifixar domínios DNS em maligno ou benigno e, se maligno, classificar nas seguintes categorias: malware, phishing e spam

planejamento:

1. Trabalhando com os dados:
- leitura do dataset
- entendimento de sua estrutura bem como suas informações que serão úteis para o modelo
- realizar uma EDA sobre os datasets e 'brincar' com os dados
(nesta fase é adequado entender cada coluna/feature de cada dataset para entender como poderão ser utilizados, e entender se correlações)

2. processmaneto dos dados
- definição da estrutura de quais e como usarei os dados disponíveis
- os alvos de classificação serão: benigno
                                   maligno
                                   |_ malware
                                   |_ phishing
                                   |_ spam

3. desenvolvimento da estrutura do modelo
- planejar camadas de processamento do modelo
- definir ferramentas e artifícios
- desenvolver o modelo

4. avaliação de desempenho
- treinar modelo com dados de treinamento
- avaliar desempenho segundo métricas (precisão, acurácia, F1 score, ...) sobre os dados de teste

5. escrever um relatório final
- descrever a arquiteura, o método/ a técnica utilizados, bem como os resultados obtidos
- procurar desenvolver algo flexível para aplicar para outros modelos e métodos de ML para fins de comparação e avaliação de efetividade

mãos à obra!
"""

# bibliotecas necessárias:
import pandas as pd # manipulação do dataset
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
sns.set(color_codes=True)


'''
1. Trabalhando com os dados:
- leitura do dataset
- entendimento de sua estrutura bem como suas informações que serão úteis para o modelo
- realizar uma EDA sobre os datasets e 'brincar' com os dados
(nesta fase é adequado entender cada coluna/feature de cada dataset para entender como poderão ser utilizados, e entender se correlações)
'''


# leitura do dataset
df_benigno0 = pd.read_csv("output-of-benign-pcap-0.csv", nrows=1000)
df_benigno1 = pd.read_csv("output-of-benign-pcap-1.csv", nrows=1000)
df_benigno2 = pd.read_csv("output-of-benign-pcap-2.csv", nrows=1000)
df_benigno3 = pd.read_csv("output-of-benign-pcap-3.csv", nrows=1000)
df_malware = pd.read_csv("output-of-malware-pcap.csv", nrows=1000)
df_phishing = pd.read_csv("output-of-phishing-pcap.csv", nrows=1000)
df_spam = pd.read_csv("output-of-spam-pcap.csv", nrows=1000)

# EDA e entendendo a estrutura do dataset

print(df_benigno0.head(5))
print(df_benigno0.columns)
print(df_benigno0.dtypes)
print(df_benigno0.corr())

# print(df_malware.head(5))
# print(df_malware.columns)
# print(df_malware.dtypes)

# print(df_phishing.head(5))
# print(df_phishing.columns)
# print(df_phishing.dtypes)

# print(df_spam.head(5))
# print(df_spam.columns)
# print(df_spam.dtypes)