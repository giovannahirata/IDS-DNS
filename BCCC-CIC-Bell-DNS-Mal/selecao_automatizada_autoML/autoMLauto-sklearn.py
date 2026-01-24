'''
import sklearn.datasets
import sklearn.model_selection
import autosklearn.classification
import sklearn.metrics

# 1. Preparação dos dados
X, y = sklearn.datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

# 2. Configuração do Modelo
# time_left_for_this_task: Tempo total (em segundos) que o modelo pode rodar.
# per_run_time_limit: Tempo máximo para testar cada modelo individual.
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=300, 
    per_run_time_limit=30,
    memory_limit=3072 # Limite de RAM em MB
)

# 3. Treinamento (A "mágica" acontece aqui)
automl.fit(X_train, y_train)

# 4. Avaliação
predictions = automl.predict(X_test)
print("Acurácia:", sklearn.metrics.accuracy_score(y_test, predictions))

# 5. Ver os modelos escolhidos
print(automl.show_models())
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import autosklearn.classification
import autosklearn.metrics
import sklearn.metrics

# carregamento dos dados

data_dir = "../"

benigns = ["output-of-benign-pcap-0.csv",
           "output-of-benign-pcap-1.csv",
           "output-of-benign-pcap-2.csv",
           "output-of-benign-pcap-3.csv"]

df_benigns = [pd.read_csv(data_dir + f, nrows=50) for f in benigns]
df_benign = pd.concat(df_benigns)
df_benign["maligno"] = 0
# df_benign["tipo_maligno"] = "Benigno"

df_malware = pd.read_csv(data_dir + "output-of-malware-pcap.csv", nrows=200)
df_malware['maligno'] = 1
# df_malware['tipo_maligno'] = 'Malware'

df_phishing = pd.read_csv(data_dir + "output-of-phishing-pcap.csv", nrows=200)
df_phishing['maligno'] = 1
# df_phishing['tipo_maligno'] = 'Phishing'

df_spam = pd.read_csv(data_dir + "output-of-spam-pcap.csv", nrows=200)
df_spam['maligno'] = 1
# df_spam['tipo_maligno'] = 'Spam'

df = pd.concat([df_benign, df_malware, df_phishing, df_spam])

# remover colunas não úteis
cols_remover = ['Unnamed: 0', 'flow_id', 'src_ip', 'dst_ip', 'label']
df = df.drop(columns=cols_remover)

# # preencher valores ausentes
# df = df.fillna(0)

# # codificar variáveis categóricas
# le = LabelEncoder()
# df['protocol'] = le.fit_transform(df['protocol'])

# separar features e rótulos
# X = df.drop(columns=['maligno', 'tipo_maligno'])
X = df.drop(columns=['maligno'])
y_bin = df['maligno']  # Binário (0=Benigno, 1=Maligno)
# y_multi = df['tipo_maligno']  # Multiclasse (Benigno, Malware, Phishing, Spam)
#X = pd.get_dummies(X)   
# cat_cols = [col for col in X.columns if X[col].dtype == 'object' and X[col].nunique() < 50]
# X = pd.get_dummies(X, columns=cat_cols)
# X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
# X = X.replace([np.inf, -np.inf], np.nan)
# X = X.fillna(0)
# X = X.astype(np.float64)

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X, y_bin
)

automl = autosklearn.classification.AutoSklearnClassifier(
    metric=[autosklearn.metrics.precision, autosklearn.metrics.recall],
    delete_tmp_folder_after_terminate=False
)

automl.fit(X_train_bin, y_train_bin)

predictions = automl.predict(X_test_bin)
print("Precision", sklearn.metrics.precision_score(y_test_bin, predictions))
print("Recall", sklearn.metrics.recall_score(y_test_bin, predictions))

print(automl.leaderboard())

print(automl.cv_results_)