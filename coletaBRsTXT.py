# Nome do arquivo de entrada
entrada = "dominios.txt"

# Nome do arquivo de saída
saida = "dominios_br.txt"

# Abrir o arquivo, ler linhas e filtrar as que terminam com .br
with open(entrada, 'r', encoding='utf-8') as f:
    linhas = f.readlines()

# Filtrar linhas que terminam com .br (ignorando espaços e maiúsculas)
dominios_br = [linha.strip() for linha in linhas if linha.strip().lower().endswith('.br')]

# Escrever os domínios .br em outro arquivo
with open(saida, 'w', encoding='utf-8') as f:
    for dominio in dominios_br:
        f.write(dominio + '\n')

print(f"{len(dominios_br)} domínios .br encontrados e salvos em '{saida}'")
