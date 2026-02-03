import pandas as pd
import numpy as np

df = pd.read_csv('output-of-benign-pcap-3.csv')  # ou use encoding='utf-8' / 'latin1' se der erro

# print(df.head())             # Ver as primeiras linhas
# print(df.columns)           # Ver nomes das colunas
# print(df.info())            # Tipos de dados e nulls
# print(df.describe())        # Estatísticas de colunas numéricas

# print(df.dtypes.value_counts())

# print(df["src_port"].unique())
print(df["dst_port"].unique())
# print(df["character_distribution"].unique())
# features = list(df.columns)
# print (features)

"""ttl = df['ttl_values_median'].mean()
print(ttl)
ttl_neg = df['ttl_values_median']<0
print(ttl_neg.sum()/ttl_neg.count()*100 , "%")

# IDEIA: ADICIONAR UMA NOVA FEATURE QUE MEDE A PROPORÇÃO DE VALORES TTL
# PARA CADA CATEGORIA: BENIGN, MALWARE, PHISHING E SPAM"""

"""total_bytes = df['total_bytes'].mean()
print(total_bytes)
# os valores de total_bytes do tráfego maligno (malw, phi, spam) foram
# mais altos, em média, do que os valores total_bytes do tráfego benigno"""

"""
testar pegar 70% de cada categoria ao invés de 70% de todo o dataset com
as categorias concatenadas
"""