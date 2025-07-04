import pandas as pd

df = pd.read_csv('output-of-spam-pcap.csv')  # ou use encoding='utf-8' / 'latin1' se der erro

print(df.head())             # Ver as primeiras linhas
print(df.columns)           # Ver nomes das colunas
print(df.info())            # Tipos de dados e nulls
print(df.describe())        # Estatísticas de colunas numéricas
