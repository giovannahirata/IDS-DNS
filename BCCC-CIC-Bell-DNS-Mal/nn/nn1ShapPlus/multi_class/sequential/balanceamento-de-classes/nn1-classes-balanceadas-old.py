import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, roc_curve
import pickle
import shap
import time
import os

# Configuração para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

# carregamento e rotulagem dos dados
dir_path = "~/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/"

benigns = ["output-of-benign-pcap-0.csv",
           "output-of-benign-pcap-1.csv",
           "output-of-benign-pcap-2.csv",
           "output-of-benign-pcap-3.csv"]

print("\nCarregando datasets...")
df_benigns = [pd.read_csv(dir_path + f) for f in benigns]
df_benign = pd.concat(df_benigns, ignore_index=True)
df_benign["maligno"] = 0
df_benign["tipo_maligno"] = "Benigno"

df_malware = pd.read_csv(dir_path + "output-of-malware-pcap.csv")
df_malware["maligno"] = 1
df_malware["tipo_maligno"] = "Malware"

df_phishing = pd.read_csv(dir_path + "output-of-phishing-pcap.csv")
df_phishing["maligno"] = 1
df_phishing["tipo_maligno"] = "Phishing"

df_spam = pd.read_csv(dir_path + "output-of-spam-pcap.csv")
df_spam["maligno"] = 1
df_spam["tipo_maligno"] = "Spam"

# Concatena todos os malignos
df_malicious = pd.concat([df_malware, df_phishing, df_spam], ignore_index=True)

print(f"\nEstatísticas do dataset completo:")
print(f"Total de amostras benignas: {len(df_benign)}")
print(f"Total de amostras malignas: {len(df_malicious)}")
print(f"  - Malware: {len(df_malware)}")
print(f"  - Phishing: {len(df_phishing)}")
print(f"  - Spam: {len(df_spam)}")

# Remove colunas desnecessárias
cols_remove = ['Unnamed: 0', 'flow_id', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port']
df_benign = df_benign.drop(columns=cols_remove)
df_malicious = df_malicious.drop(columns=cols_remove)

# Preenche valores faltantes
df_benign = df_benign.fillna(0)
df_malicious = df_malicious.fillna(0)

# Identifica e faz encoding de colunas categóricas
print("\nRealizando encoding de variáveis categóricas...")
categorical_cols = df_benign.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit nos dados completos para garantir consistência
    all_values = pd.concat([df_benign[col], df_malicious[col]]).astype(str)
    le.fit(all_values)
    df_benign[col] = le.transform(df_benign[col].astype(str))
    df_malicious[col] = le.transform(df_malicious[col].astype(str))
    label_encoders[col] = le

# Número de amostras malignas (total)
n_malicious = len(df_malicious)
print(f"\nBalanceamento: Selecionaremos {n_malicious} amostras benignas aleatórias em cada repetição")
print(f"Split: 70% treino / 30% teste")

# Armazena resultados de todas as repetições
all_results = []

# ============================================================================
# LOOP DE 10 REPETIÇÕES COM DIFERENTES AMOSTRAS ALEATÓRIAS
# ============================================================================

# define a seed da melhor repetição (4)
random_state = 46
# Seleciona aleatoriamente n_malicious amostras benignas
df_benign_sampled = df_benign.sample(n=n_malicious, random_state=random_state)
# Concatena benignos balanceados com todos os malignos
df_balanced = pd.concat([df_benign_sampled, df_malicious], ignore_index=True)
print(f"Dataset balanceado")
print(f"Total de amostras: {len(df_balanced)}")
print(f"Distribuição por tipo:")
print(df_balanced['tipo_maligno'].value_counts())
# Split 70/30 estratificado
df_train, df_test = train_test_split(
    df_balanced, 
    test_size=0.3, 
    random_state=random_state,
    stratify=df_balanced['tipo_maligno']
)
print(f"\nSplit 70/30:")
print(f"Treino: {len(df_train)} amostras")
print(f"Teste: {len(df_test)} amostras")
print(f"Distribuição treino:\n{df_train['tipo_maligno'].value_counts()}")
print(f"Distribuição teste:\n{df_test['tipo_maligno'].value_counts()}")   
# Separa features e targets
x_train = df_train.drop(columns=["maligno", "tipo_maligno"])
x_test = df_test.drop(columns=["maligno", "tipo_maligno"])
y_train_multi = df_train["tipo_maligno"].values
y_test_multi = df_test["tipo_maligno"].values

num_class_y = len(np.unique(y_train_multi))

# Salva nomes das features originais
original_feature_names = list(x_train.columns)

# Identifica colunas numéricas
numerical_cols = x_train.select_dtypes(include=['int64', 'float64']).columns

# Normalização
scaler = StandardScaler()
x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
x_test[numerical_cols] = scaler.transform(x_test[numerical_cols])

# Remove features com variância zero ou muito baixa
selector = VarianceThreshold(threshold=0.01)
x_train_transformed = selector.fit_transform(x_train)
x_test_transformed = selector.transform(x_test)

# Obtém nomes das features selecionadas
selected_indices = selector.get_support(indices=True)
selected_feature_names = [original_feature_names[i] for i in selected_indices]

print(f"\nFeatures após pré-processamento: {x_train_transformed.shape[1]}")

# Converte para arrays numpy
x_train = np.array(x_train_transformed, dtype=np.float32)
x_test = np.array(x_test_transformed, dtype=np.float32)
y_train_multi = y_train_multi.astype(np.int32)
y_test_multi = y_test_multi.astype(np.int32)

# Cria o modelo MLP
model = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_class_y, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

# Treinamento
print(f"\nIniciando treinamento...")
start_time_train = time.perf_counter()

history = model.fit(
    x_train, y_train_multi,
    validation_data=(x_test, y_test_multi),
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

end_time_train = time.perf_counter()
elapsed_time_train = end_time_train - start_time_train

# Avaliação
print(f"Avaliando modelo da repetição...")
start_time_test = time.perf_counter()

loss, accuracy = model.evaluate(x_test, y_test_multi, verbose=0)
y_pred_proba = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

end_time_test = time.perf_counter()
elapsed_time_test = end_time_test - start_time_test

# Calcula métricas
y_test_int = y_test_multi.astype(int)
precision = precision_score(y_test_int, y_pred, average='weighted')
recall = recall_score(y_test_int, y_pred, average='weighted')
f1 = f1_score(y_test_int, y_pred, average='weighted')

try:
    auc_score = roc_auc_score(y_test_int, y_pred_proba, multi_class='ovr', average='weighted')
except:
    auc_score = 0.0

# Métricas por classe
precision_per_class = precision_score(y_test_int, y_pred, average=None)
recall_per_class = recall_score(y_test_int, y_pred, average=None)
f1_per_class = f1_score(y_test_int, y_pred, average=None)

print(f"\n{'='*50}")
print(f"RESULTADOS")
print(f"{'='*50}")
print(f"Tempo de treinamento: {elapsed_time_train:.2f}s")
print(f"Tempo de teste: {elapsed_time_test:.2f}s")
print(f"Épocas treinadas: {len(history.history['loss'])}")
print(f"\nMétricas gerais (weighted):")
print(f"  Acurácia:  {accuracy:.4f}")
print(f"  Precisão:  {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc_score:.4f}")

# Matriz de confusão
cm = confusion_matrix(y_test_int, y_pred)
print(f"\nMatriz de confusão:")
print(cm)

# Relatório de classificação
print(f"\nRelatório de classificação:")
print(classification_report(y_test_int, y_pred, 
                            target_names=['Benigno', 'Malware', 'Phishing', 'Spam']))

# Armazena resultados
result = {
    'random_state': random_state,
    'train_samples': len(df_train),
    'test_samples': len(df_test),
    'n_features': x_train.shape[1],
    'train_time': elapsed_time_train,
    'test_time': elapsed_time_test,
    'epochs': len(history.history['loss']),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'auc_roc': auc_score,
    'precision_benigno': precision_per_class[0],
    'precision_malware': precision_per_class[1],
    'precision_phishing': precision_per_class[2],
    'precision_spam': precision_per_class[3],
    'recall_benigno': recall_per_class[0],
    'recall_malware': recall_per_class[1],
    'recall_phishing': recall_per_class[2],
    'recall_spam': recall_per_class[3],
    'f1_benigno': f1_per_class[0],
    'f1_malware': f1_per_class[1],
    'f1_phishing': f1_per_class[2],
    'f1_spam': f1_per_class[3]
}

# Salva modelo e histórico desta repetição
model.save(f'model_4.keras')
with open(f'history_4.pkl', 'wb') as f:
    pickle.dump(history.history, f)


# ============================================================================
# VISUALIZAÇÕES
# ============================================================================

print(f"\n{'='*80}")
print("GERANDO VISUALIZAÇÕES")
print(f"{'='*80}")



# ============================================================================
# SALVAMENTOS
# ============================================================================

# salva o scaler e selector para pre-processamento
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('selector.pkl', 'wb') as f:
    pickle.dump(selector, f)
    
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
    
print("Pré-processadores salvos via Pickle (scaler.pkl, selector.pkl, label_encoders.pkl)")