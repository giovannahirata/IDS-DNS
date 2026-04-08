import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, roc_curve
import pickle
import shap

# carregamento e rotulagem dos dados

dir_path = "../../../../"

benigns = ["output-of-benign-pcap-0.csv",
           "output-of-benign-pcap-1.csv",
           "output-of-benign-pcap-2.csv",
           "output-of-benign-pcap-3.csv"]

df_benigns = [pd.read_csv(dir_path + f) for f in benigns]
df_benign = pd.concat(df_benigns)
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

df = pd.concat([df_benign, df_malware, df_phishing, df_spam])

print(f"\nAlgumas estatísticas do dataset:")
print(f"Total de amostras: {len(df)}")
print(f"Distribuição de classes:\n{df['maligno'].value_counts()}")
print(f"Proporção: {df['maligno'].value_counts(normalize=True)}")
print(f"Proporção entre tipos malignos: {df['tipo_maligno'].value_counts(normalize=True)}")
print(f"\nFeatures originais: {df.shape[1]}")

cols_remove = ['Unnamed: 0', 'flow_id', 'timestamp', 'src_ip', 'dst_ip']

df = df.drop(columns=cols_remove)
df = df.fillna(0)

# identifica colunas categoricas:
categorical_cols = df.select_dtypes(include=['object']).columns

# encoding de variaveis categoricas
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

x_train = df_train.drop(columns=["maligno", "tipo_maligno", "label"])
x_valid = df_valid.drop(columns=["maligno", "tipo_maligno", "label"])


y_train_multi = df_train["tipo_maligno"].values  
y_valid_multi = df_valid["tipo_maligno"].values  

num_class_y = len(np.unique(y_train_multi))

print(f"\nClasses encontradas: {num_class_y}")
print(f"Distribuição y_train: {np.bincount(y_train_multi.astype(int))}")
print(f"Distribuição y_valid: {np.bincount(y_valid_multi.astype(int))}")

# print(f"\nFeatures originais: {x_train.shape[1]}")

# salva nomes das features
original_feature_names = list(x_train.columns)

# identifica coluna numericas:
numerical_cols = x_train.select_dtypes(include=['int64', 'float64']).columns
   
# normalização
scaler = StandardScaler()
x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
x_valid[numerical_cols] = scaler.transform(x_valid[numerical_cols])

# remove features com variancia zero ou muito baixa
selector = VarianceThreshold(threshold=0.01)
x_train_transformed = selector.fit_transform(x_train)
x_valid_transformed = selector.transform(x_valid)

# obtém nomes das features que foram mantidas após VarianceThreshold
selected_indices = selector.get_support(indices=True)
selected_feature_names = [original_feature_names[i] for i in selected_indices]

# verifica e corrige nomes duplicados (adiciona índice se houver duplicatas)
name_counts = {}
unique_feature_names = []
for name in selected_feature_names:
    if name in name_counts:
        name_counts[name] += 1
        unique_feature_names.append(f"{name}_{name_counts[name]}")
    else:
        name_counts[name] = 0
        unique_feature_names.append(name)

# verifica se há duplicatas
duplicates = [name for name, count in name_counts.items() if count > 0]
if duplicates:
    print(f"\nFeatures duplicadas encontradas e renomeadas: {duplicates}")
    selected_feature_names = unique_feature_names

print(f"Features após remoção: {x_train_transformed.shape[1]}")
print(f"Features selecionadas: {len(selected_feature_names)}")
print(f"Primeiras 10 features: {selected_feature_names[:10]}")

x_train = np.array(x_train_transformed, dtype=np.float32)
x_valid = np.array(x_valid_transformed, dtype=np.float32)
y_train_multi = y_train_multi.astype(np.int32)
y_valid_multi = y_valid_multi.astype(np.int32)

print(f"\nShape X_train: {x_train.shape}, y_train: {y_train_multi.shape}")
print(f"Shape X_valid: {x_valid.shape}, y_valid: {y_valid_multi.shape}")
print(f"Tipo y_train: {y_train_multi.dtype}, valores únicos: {np.unique(y_train_multi)}")

# modelo para classificação multiclasse

model = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_class_y, activation='softmax')
])

# compilação do modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# visao da arquitetura
model.summary()

# callbacks pra melhorar o treinamento
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

from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
plot_model(model,
    to_file='model.png',
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=True,
    dpi=200,
    show_layer_activations=True,
    show_trainable=True)
