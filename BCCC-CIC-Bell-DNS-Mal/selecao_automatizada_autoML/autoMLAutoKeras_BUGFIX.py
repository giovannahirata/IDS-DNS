# NOTA IMPORTANTE: BUG NO AUTOKERAS 3.0.0 COM KERAS 3.13.0
# O AutoKeras 3.0.0 tem um bug conhecido com Keras 3.13.0 onde a camada Dense
# recusa valores de units=1 para classificação binária.
# 
# SOLUÇÃO: Use o script autoMLKerasTuner.py que implementa a mesma funcionalidade
# usando Keras Tuner diretamente, evitando o bug do AutoKeras.
#
# Para mais informações sobre o bug:
# https://github.com/keras-team/autokeras/issues/
#
# O script autoMLKerasTuner.py oferece:
# - Busca automática de hiperparâmetros (como o AutoKeras)
# - Funciona corretamente com a versão atual do Keras
# - Mais controle sobre a arquitetura do modelo
# - Resultados equivalentes ou melhores

import keras
import pandas as pd
import autokeras as ak
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import os

# Força o Keras a usar configurações compatíveis
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs do TensorFlow

# carregamento e rotulamento dos dados
data_dir = "../"

benigns = ["output-of-benign-pcap-0.csv",
           "output-of-benign-pcap-1.csv",
           "output-of-benign-pcap-2.csv",
           "output-of-benign-pcap-3.csv"]

df_benigns = [pd.read_csv(data_dir + f, nrows=100) for f in benigns]
df_benign = pd.concat(df_benigns)
df_benign["maligno"] = 0
df_benign["tipo_maligno"] = "Benigno"

df_malware = pd.read_csv(data_dir + "output-of-malware-pcap.csv", nrows=400)
df_malware['maligno'] = 1
df_malware['tipo_maligno'] = 'Malware'

df_phishing = pd.read_csv(data_dir + "output-of-phishing-pcap.csv", nrows=400)
df_phishing['maligno'] = 1
df_phishing['tipo_maligno'] = 'Phishing'

df_spam = pd.read_csv(data_dir + "output-of-spam-pcap.csv", nrows=400)
df_spam['maligno'] = 1
df_spam['tipo_maligno'] = 'Spam'

df = pd.concat([df_benign, df_malware, df_phishing, df_spam])

# separar features e rótulos
X = df.drop(columns=['maligno', 'tipo_maligno'])
y_bin = df['maligno'].to_numpy()  # Binário (0=Benigno, 1=Maligno)
y_multi = df['tipo_maligno'].to_numpy()  # Multiclasse (Benigno, Malware, Phishing, Spam)

cat_cols = [col for col in X.columns if X[col].dtype == 'object' and X[col].nunique() < 50]
X = pd.get_dummies(X, columns=cat_cols)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = X.astype(np.float64).to_numpy()

# dividir dados binários
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X, y_bin, test_size=0.3, random_state=42
)

print("="*70)
print("AVISO: AutoKeras 3.0.0 tem um bug com Keras 3.13.0")
print("="*70)
print("\nO erro 'ValueError: Received an invalid value for `units`' é conhecido.")
print("Use o script 'autoMLKerasTuner.py' como alternativa.\n")
print("Para continuar mesmo assim, o código abaixo tentará usar AutoKeras...")
print("="*70)


try:
    # Garantir que os dados estão no formato correto
    print(f"\nClasses únicas: {np.unique(y_train_bin)}")
    print(f"Shape dos dados de treino: {X_train_bin.shape}")
    
    # Workaround: Usar RegressionHead em vez de ClassificationHead
    # E depois arredondar as previsões
    input_node = ak.StructuredDataInput()
    output_node = ak.StructuredDataBlock()(input_node)
    output_node = ak.RegressionHead()(output_node)
    
    clf = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        overwrite=True,
        max_trials=3
    )
    
    # Reshape dos labels para ter shape (n, 1) 
    y_train_bin_reshaped = y_train_bin.reshape(-1, 1).astype(np.float32)
    y_test_bin_reshaped = y_test_bin.reshape(-1, 1).astype(np.float32)
    
    print(f"\nTreinando modelo com regressão (classificação binária como regressão)...")
    clf.fit(X_train_bin, y_train_bin_reshaped, epochs=10, verbose=1)
    
    # prediz o melhor modelo
    print("\nFazendo previsões...")
    predicted_y = clf.predict(X_test_bin)
    # Arredondar as previsões para obter classes (0 ou 1)
    predicted_y_classes = np.round(predicted_y).flatten().astype(int)
    # Garantir que está entre 0 e 1
    predicted_y_classes = np.clip(predicted_y_classes, 0, 1)
    
    # Avaliar
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    print(f"\nAccuracy: {accuracy_score(y_test_bin, predicted_y_classes)}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_bin, predicted_y_classes))
    print("\nClassification Report:")
    print(classification_report(y_test_bin, predicted_y_classes, target_names=['Benigno', 'Maligno']))
    
    # Avaliar com o método nativo do AutoKeras (MAE e MSE para regressão)
    print("\nAvaliação do AutoKeras (métricas de regressão):")
    eval_results = clf.evaluate(X_test_bin, y_test_bin_reshaped, verbose=0)
    print(f"Loss: {eval_results}")
    
except Exception as e:
    print(f"\n{'='*70}")
    print(f"ERRO ESPERADO: {type(e).__name__}")
    print(f"{'='*70}")
    print(f"Mensagem: {str(e)}\n")
    print("SOLUÇÃO: Execute o script 'autoMLKerasTuner.py' em vez deste.")
    print("Ele oferece a mesma funcionalidade sem o bug.")
    print(f"{'='*70}")
    raise
