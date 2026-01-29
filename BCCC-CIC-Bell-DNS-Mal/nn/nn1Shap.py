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
plt.close() 
print("Gráfico de treinamento salvo: training_history.png")

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
plt.close() 
print("Curva ROC salva: roc_curve.png")

# análise com o SHAP:
print("\nIniciando análise com SHAP")

# usa subset menor para SHAP (eh mais rápido)
x_train_sample = x_train[:1000]  # pega 1000 amostras
x_valid_sample = x_valid[:100]   # pega 100 amostras para análise

print(f"Usando {len(x_train_sample)} amostras de treino para background")
print(f"Calculando SHAP para {len(x_valid_sample)} amostras de validação")

# calcula valores shap
print("Criando explainer...")
explainer = shap.DeepExplainer(model, x_train_sample)

print("Calculando valores SHAP (pode demorar)...")
shap_values = explainer.shap_values(x_valid_sample)

print(f"\nNúmero de features no modelo: {x_train.shape[1]}")
print(f"Número de nomes de features salvos: {len(selected_feature_names)}")

assert x_train.shape[1] == len(selected_feature_names), \
    f"Erro: {x_train.shape[1]} features vs {len(selected_feature_names)} nomes"

# verifica formato dos shap_values
if isinstance(shap_values, list):
    print(f"SHAP values é uma lista com {len(shap_values)} elementos")
    shap_array = shap_values[0]  # para classificação binária, pegar primeiro elemento
else:
    shap_array = shap_values

print(f"Shape dos SHAP values: {shap_array.shape}")
print(f"Shape do x_valid_sample: {x_valid_sample.shape}")

# remove dimensão extra se existir (100, 104, 1) -> (100, 104)
if len(shap_array.shape) == 3 and shap_array.shape[2] == 1:
    shap_array = shap_array[:, :, 0]
    print(f"Shape após squeeze: {shap_array.shape}")

# visualização do summary plot (importância global)
print("\nGerando summary plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_array, x_valid_sample, 
                  feature_names=selected_feature_names,
                  max_display=20,  # mostra top 20 features
                  show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP summary plot salvo: shap_summary_plot.png")

# visualização do bar plot (importância média)
print("Gerando bar plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_array, x_valid_sample,
                  feature_names=selected_feature_names,
                  max_display=20,  # mostra top 20 features
                  plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP bar plot salvo: shap_bar_plot.png")

# visualização do plot waterfall para primeira predição
print("Gerando waterfall plot...")
plt.figure(figsize=(12, 10))

# converte expected_value para valor numérico Python
if isinstance(explainer.expected_value, (list, np.ndarray)):
    expected_val = float(explainer.expected_value[0])
else:
    # se é tensor TensorFlow, converter para numpy primeiro
    try:
        expected_val = float(explainer.expected_value.numpy())
    except:
        expected_val = float(explainer.expected_value)

print(f"Expected value: {expected_val}")
print(f"Shape para waterfall - shap_array[0]: {shap_array[0].shape}")

shap.waterfall_plot(
    shap.Explanation(
        values=shap_array[0],  # 1D: (104,)
        base_values=expected_val,
        data=x_valid_sample[0],
        feature_names=selected_feature_names
    ),
    max_display=20,  # mostra top 20 features
    show=False
)
plt.tight_layout()
plt.savefig('shap_waterfall_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP waterfall plot salvo: shap_waterfall_plot.png")

print("\nAnálise SHAP concluída")

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