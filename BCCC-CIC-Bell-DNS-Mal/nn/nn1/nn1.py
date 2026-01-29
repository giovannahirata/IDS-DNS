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

# carregamento e rotulagem dos dados

dir_path = "../"

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
y_train_bin = df_train["maligno"]
y_valid_bin = df_valid["maligno"]

print(f"\nFeatures originais: {x_train.shape[1]}")

# identifica coluna numericas:
numerical_cols = x_train.select_dtypes(include=['int64', 'float64']).columns
   
# normalização
scaler = StandardScaler()
x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
x_valid[numerical_cols] = scaler.transform(x_valid[numerical_cols])

# remove features com variancia zero ou muito baixa
selector = VarianceThreshold(threshold=0.01)
x_train = selector.fit_transform(x_train)
x_valid = selector.transform(x_valid)

print(f"Features após remoção: {x_train.shape[1]}")

x_train = np.array(x_train, dtype=np.float32)
x_valid = np.array(x_valid, dtype=np.float32)
y_train_bin = np.array(y_train_bin, dtype=np.float32)
y_valid_bin = np.array(y_valid_bin, dtype=np.float32)

# modelo para classificação binaria: benigno vs maligno

model = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# compilação do modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
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

# treinamento
history = model.fit(
    x_train, y_train_bin,
    validation_data=(x_valid, y_valid_bin),
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# avalia modelo
loss, accuracy, precision, recall = model.evaluate(x_valid, y_valid_bin)
print(f'\nAcurácia: {accuracy:.4f}')
print(f'\nPrecisão: {precision:.4f}')
print(f'\nRecall: {recall:.4f}')
y_pred_proba = model.predict(x_valid).ravel()
auc_score = roc_auc_score(y_valid_bin, y_pred_proba)
print(f'\nAUC-ROC Score: {auc_score:.4f}')


# predicoes
y_pred = model.predict(x_valid)
y_pred_class = (y_pred > 0.5).astype(int)

# matriz de confusao
print('\nMatriz de confusão:')
print(confusion_matrix(y_valid_bin, y_pred_class))
print('\nRelatório de classificação:')
print(classification_report(y_valid_bin, y_pred_class))

# validação cruzada com 5 folds
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train_bin)):
    print(f'\nFold {fold + 1}')
    
    X_fold_train, X_fold_val = x_train[train_idx], x_train[val_idx]
    y_fold_train, y_fold_val = y_train_bin[train_idx], y_train_bin[val_idx]
    
    # cria novo modelo pra cada fold
    model_fold = keras.Sequential([
        layers.Input(shape=(x_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model_fold.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    history_fold = model_fold.fit(X_fold_train, y_fold_train,
                             validation_data=(X_fold_val, y_fold_val),
                             epochs=50, batch_size=32, verbose=0)
    
    score = model_fold.evaluate(X_fold_val, y_fold_val, verbose=0)
    scores.append(score[1]) # accuracy
    print(f'Acurácia: {score[1]:.4f}')

print(f'\nAcurácia média: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})')

# análise das métricas do k-fold
print("\nResumo da validação cruzada:")
print(f"Melhor fold: {np.argmax(scores) + 1} com {np.max(scores):.4f}")
print(f"Pior fold: {np.argmin(scores) + 1} com {np.min(scores):.4f}")
print(f"Variação: {np.max(scores) - np.min(scores):.4f}")

# análise de features
feature_names = [col for col in df.columns if col not in ["maligno", "tipo_maligno", "label"]]

# remove features com desvio padrão zero antes de calcular correlação
df_features = df[feature_names]
non_constant_features = df_features.loc[:, df_features.std()>0].columns

# checa correlação com target
correlations = df[non_constant_features].corrwith(df["maligno"]).abs().sort_values(ascending=False)
print("\nAs 10 features mais correlacionadas com o target binário:")
print(correlations.head(10))
print(f'\nFeatures constantes removidas: {len(feature_names) - len(non_constant_features)}')

# visualizacao

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Loss durante o treinamento')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# plotar curva ROC
fpr, tpr, thresholds = roc_curve(y_valid_bin, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend()
plt.savefig('roc_curve.png')
plt.show()

# salva modelo
model.save('dns_intrusion_model_nn1.keras')
print("\nModelo salvo como 'dns_intrusion_model_nn1.keras'")

# salva o scaler e selector para pre-processamento
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('selector.pkl', 'wb') as f:
    pickle.dump(selector, f)
    
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
    
print("Pré-processadores salvos via Pickle (scaler.pkl, selector.pkl, label_encoders.pkl)")