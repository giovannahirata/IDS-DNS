import pandas as pd
import pycaret
from pycaret.classification import *
from pycaret.classification import ClassificationExperiment

# carregamento e rotulamento dos dados

data_dir = "../"

benigns = ["output-of-benign-pcap-0.csv",
           "output-of-benign-pcap-1.csv",
           "output-of-benign-pcap-2.csv",
           "output-of-benign-pcap-3.csv"]

df_benigns = [pd.read_csv(data_dir + f) for f in benigns]
df_benign = pd.concat(df_benigns)
df_benign["maligno"] = 0
# df_benign["tipo_maligno"] = "Benigno"

df_malware = pd.read_csv(data_dir + "output-of-malware-pcap.csv")
df_malware['maligno'] = 1
# df_malware['tipo_maligno'] = 'Malware'

df_phishing = pd.read_csv(data_dir + "output-of-phishing-pcap.csv")
df_phishing['maligno'] = 1
# df_phishing['tipo_maligno'] = 'Phishing'

df_spam = pd.read_csv(data_dir + "output-of-spam-pcap.csv")
df_spam['maligno'] = 1
# df_spam['tipo_maligno'] = 'Spam'

df = pd.concat([df_benign, df_malware, df_phishing, df_spam])

# reset do índice pra evitar duplicatas
df = df.reset_index(drop=True)

# reduz dataset para evitar OOM (Out of Memory)
print(f"Dataset original: {len(df)} linhas")
df = df.sample(n=min(200000, len(df)), random_state=42)
print(f"Dataset reduzido: {len(df)} linhas (para evitar crash por memória)")

# remove colunas desnecessárias
cols_to_drop = ['flow_id', 'timestamp', 'src_ip', 'dst_ip', 'label']
if 'Unnamed: 0' in df.columns:
    cols_to_drop.append('Unnamed: 0')
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

# setup
# função que inicializa o ambiente de treinamento e cria o pipeline
s = setup(df, target="maligno")

# inicializa a classe ClassificationExperiment
exp = ClassificationExperiment()

exp.setup(df, target="maligno")

# treina e avalia o desempenho de todos os estimadores disponíveis 
# na biblioteca de modelos usando a cross validation. A saída é uma
# tabela com os scores médios da cross validation.
# best = compare_models(turbo=False, budget_time=20, exclude=['ridge'])
best = compare_models(turbo=False)

# exp.compare_models(turbo=False, budget_time=20)
exp.compare_models(turbo=False)

"""
Analisando a performance do modelo treinado no conjunto de teste
"""

# plota a matriz de confusão
plot_model(best, plot='confusion_matrix', save=True)
print("Confusion matrix salva")

# plota AUC
try:
    if hasattr(best, 'predict_proba'):
        plot_model(best, plot='auc', save=True)
        print("AUC salva")
    else:
        print("AUC não disponível para este modelo")
except Exception as e:
    print(f"Erro ao gerar AUC: {e}")

# plota a importância das features
try:
    plot_model(best, plot='feature', save=True)
    print("Feature importance salva")
except Exception as e:
    print(f"Erro ao gerar feature importance: {e}")


"""
Predição
"""

holdout_pred = predict_model(best)
