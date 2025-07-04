import pandas as pd

# Caminho para seu CSV
caminho_csv = "dominios.csv"

# Lê o CSV — ajusta se a coluna tiver nome diferente
df = pd.read_csv(caminho_csv)

# Supondo que a coluna com os domínios se chame 'dominio'
# (substitua pelo nome real se for diferente)
coluna_dominio = "dominio"

# Filtra apenas os domínios que terminam com .br (insensível a maiúsculas)
df_br = df[df[coluna_dominio].str.lower().str.endswith('.br', na=False)]

# Salva em um novo CSV
df_br.to_csv("dominios_br.csv", index=False)

print(f"{len(df_br)} domínios .br encontrados e salvos em 'dominios_br.csv'")
