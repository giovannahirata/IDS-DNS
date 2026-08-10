import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import pickle
import shap
import time
import os

# Configuração para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

repetition = 10
random_state = 42 + repetition  # Seed = 44 para a repetição 10

print("="*80)
print(f"Experimento MLP Binário - apenas repetição {repetition} (seed={random_state}) + análise SHAP")
print("="*80)

# ============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ============================================================================

dir_path = "/home/giovanna/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/"

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

# Colunas conhecidas para remoção + filtro dinâmico para qualquer coluna 'Unnamed'
cols_remove = ['flow_id', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']

def remove_unwanted_cols(df):
    cols_to_drop = [col for col in df.columns if col in cols_remove or col.startswith('Unnamed')]
    return df.drop(columns=cols_to_drop)

df_benign = remove_unwanted_cols(df_benign)
df_malicious = remove_unwanted_cols(df_malicious)

# Preenche valores faltantes
df_benign = df_benign.fillna(0)
df_malicious = df_malicious.fillna(0)

# Identifica e faz encoding de colunas categóricas
print("\nRealizando encoding de variáveis categóricas...")
categorical_cols = df_benign.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    all_values = pd.concat([df_benign[col], df_malicious[col]]).astype(str)
    le.fit(all_values)
    df_benign[col] = le.transform(df_benign[col].astype(str))
    df_malicious[col] = le.transform(df_malicious[col].astype(str))
    label_encoders[col] = le

output_dir = "resultados_mlp_binario_rep2"
os.makedirs(output_dir, exist_ok=True)
print(f"\n{'='*50}")
print(f"Diretório de saída: {output_dir}")
print(f"{'='*50}")

# Número de amostras malignas (total)
n_malicious = len(df_malicious)

# Seleciona aleatoriamente n_malicious amostras benignas com a seed da repetição 2
df_benign_sampled = df_benign.sample(n=n_malicious, random_state=random_state)
df_balanced = pd.concat([df_benign_sampled, df_malicious], ignore_index=True)

print(f"\nDataset balanceado para a Repetição 2:")
print(f"Total de amostras: {len(df_balanced)}")
print(f"Distribuição binária:")
print(df_balanced['maligno'].value_counts())

# Split 70/30 estratificado pela classe binária
df_train, df_test = train_test_split(
    df_balanced, 
    test_size=0.3, 
    random_state=random_state,
    stratify=df_balanced['maligno']
)

print(f"\nSplit 70/30:")
print(f"Treino: {len(df_train)} amostras")
print(f"Teste: {len(df_test)} amostras")

# Separa features e targets binários
x_train = df_train.drop(columns=["maligno", "tipo_maligno"])
x_test = df_test.drop(columns=["maligno", "tipo_maligno"])
y_train_bin = df_train["maligno"].values
y_test_bin = df_test["maligno"].values

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

selected_indices = selector.get_support(indices=True)
selected_feature_names = [original_feature_names[i] for i in selected_indices]

print(f"\nFeatures após pré-processamento: {x_train_transformed.shape[1]}")

# Converte para arrays numpy
x_train = np.array(x_train_transformed, dtype=np.float32)
x_test = np.array(x_test_transformed, dtype=np.float32)
y_train_bin = y_train_bin.astype(np.int32)
y_test_bin = y_test_bin.astype(np.int32)

# ============================================================================
# CRIAÇÃO E TREINAMENTO DO MODELO BINÁRIO
# ============================================================================

model = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

model.summary()

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

print(f"\nIniciando treinamento da Repetição {repetition}...")
start_time_train = time.perf_counter()

history = model.fit(
    x_train, y_train_bin,
    validation_data=(x_test, y_test_bin),
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

elapsed_time_train = time.perf_counter() - start_time_train

# ============================================================================
# AVALIAÇÃO DO MODELO
# ============================================================================

print(f"\nAvaliando modelo...")
start_time_test = time.perf_counter()

loss, accuracy, precision, recall = model.evaluate(x_test, y_test_bin, verbose=0)
y_pred_proba = model.predict(x_test, verbose=0).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)

elapsed_time_test = time.perf_counter() - start_time_test

y_test_int = y_test_bin.astype(int)
precision = precision_score(y_test_int, y_pred, average='binary')
recall = recall_score(y_test_int, y_pred, average='binary')
f1 = f1_score(y_test_int, y_pred, average='binary')

try:
    auc_score = roc_auc_score(y_test_int, y_pred_proba)
except:
    auc_score = 0.0

print(f"\n{'='*50}")
print(f"RESULTADOS DA REPETIÇÃO {repetition}")
print(f"{'='*50}")
print(f"Tempo de treinamento: {elapsed_time_train:.2f}s")
print(f"Tempo de teste: {elapsed_time_test:.2f}s")
print(f"Épocas treinadas: {len(history.history['loss'])}")
print(f"\nMétricas gerais:")
print(f"  Acurácia:  {accuracy:.4f}")
print(f"  Precisão:  {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc_score:.4f}")

print(f"\nMatriz de confusão:")
print(confusion_matrix(y_test_int, y_pred))

print(f"\nRelatório de classificação:")
print(classification_report(y_test_int, y_pred, target_names=['Benigno', 'Maligno']))

# Salva modelo e histórico
model.save(os.path.join(output_dir, f'model_{repetition}.keras'))
with open(os.path.join(output_dir, f'history_{repetition}.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

# Gráfico de histórico de treinamento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Loss durante o Treinamento')
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
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close()

# ============================================================================
# ANÁLISE SHAP (CLASSIFICAÇÃO BINÁRIA)
# ============================================================================

print(f"\n{'='*80}")
print("INICIANDO ANÁLISE SHAP (BINÁRIA)")
print(f"{'='*80}")

x_train_sample = x_train[:1000]  # 1000 amostras para background
x_valid_sample = x_test[:100]    # 100 amostras de teste para explicar
y_valid_sample_labels = y_test_bin[:100]

print(f"Usando {len(x_train_sample)} amostras de treino como background")
print(f"Calculando SHAP para {len(x_valid_sample)} amostras de teste")

explainer = shap.DeepExplainer(model, x_train_sample)
shap_values = explainer.shap_values(x_valid_sample)

# Trata adequadamente o retorno do SHAP para saída sigmoid (1 neurônio)
if isinstance(shap_values, list):
    shap_array = shap_values[0]
else:
    shap_array = shap_values

if len(shap_array.shape) == 3:
    shap_array = shap_array.squeeze(axis=-1)

print(f"Shape final dos SHAP values: {shap_array.shape}")

# Extrai o base value (expected value) para o objeto Explanation
if isinstance(explainer.expected_value, (list, np.ndarray)):
    expected_val = float(np.ravel(explainer.expected_value)[0])
else:
    expected_val = float(explainer.expected_value)

print(f"Expected value (Base Value): {expected_val:.4f}")

# Cria o objeto Explanation
shap_explanation = shap.Explanation(
    values=shap_array,
    base_values=np.full(len(shap_array), expected_val),
    data=x_valid_sample,
    feature_names=selected_feature_names
)

# 1. Summary Plot
print("\nGerando Summary Plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_array, x_valid_sample, feature_names=selected_feature_names, max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), dpi=150, bbox_inches='tight')
plt.close()

# 2. Bar Plot
print("Gerando Bar Plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_array, x_valid_sample, feature_names=selected_feature_names, max_display=20, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_bar_plot.png'), dpi=150, bbox_inches='tight')
plt.close()

# 3. Waterfall Plot (Primeira amostra)
print("Gerando Waterfall Plot...")
plt.figure(figsize=(12, 10))
shap.waterfall_plot(shap_explanation[0], max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_waterfall_plot.png'), dpi=150, bbox_inches='tight')
plt.close()

# 4. Heatmap
print("Gerando Heatmap...")
plt.figure(figsize=(14, 10))
shap.plots.heatmap(shap_explanation, max_display=15, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5. Bar plot com valores máximos
print("Gerando Bar Plot (Max)...")
plt.figure(figsize=(12, 10))
shap.plots.bar(shap_explanation.abs.max(0), max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_bar_max.png'), dpi=150, bbox_inches='tight')
plt.close()

# 6. Beeswarm Plot Absoluto
print("Gerando Beeswarm Plot Absoluto...")
plt.figure(figsize=(12, 10))
shap.plots.beeswarm(shap_explanation.abs, color="shap_red", max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_beeswarm_abs.png'), dpi=150, bbox_inches='tight')
plt.close()

# 7. Scatter plots para top 5 features
print("Gerando Scatter Plots para top features...")
mean_abs_shap = np.abs(shap_array).mean(axis=0)
top_features_idx = np.argsort(mean_abs_shap)[-5:][::-1]
top_features = [selected_feature_names[i] for i in top_features_idx]

for i, feature_name in enumerate(top_features, 1):
    plt.figure(figsize=(10, 6))
    shap.plots.scatter(shap_explanation[:, feature_name], show=False)
    plt.tight_layout()
    safe_name = feature_name.replace("/", "_").replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f'shap_scatter_{safe_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()

# 8. Dependence plots
print("Gerando Dependence Plots...")
for i, feature_name in enumerate(top_features[:3], 1):
    try:
        plt.figure(figsize=(10, 6))
        shap.plots.scatter(shap_explanation[:, feature_name], color=shap_explanation, show=False)
        plt.tight_layout()
        safe_name = feature_name.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f'shap_dependence_{safe_name}_colored.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Erro no dependence plot para {feature_name}: {e}")

# 9. Waterfall plots exemplares para cada classe (Benigno vs Maligno)
print("Gerando Waterfall plots para amostras benignas e malignas...")
class_names = ['Benigno', 'Maligno']
for class_id, class_name in enumerate(class_names):
    class_idx = np.where(y_valid_sample_labels == class_id)[0]
    if len(class_idx) > 0:
        sample_idx = class_idx[0]
        plt.figure(figsize=(12, 10))
        shap.plots.waterfall(shap_explanation[sample_idx], max_display=15, show=False)
        plt.tight_layout()
        filename = os.path.join(output_dir, f'shap_waterfall_{class_name.lower()}_example.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

# 10. Clustering de features
print("Análise de Clustering de Features...")
try:
    x_valid_df = pd.DataFrame(x_valid_sample, columns=selected_feature_names)
    non_constant = x_valid_df.std() > 0
    x_valid_filtered = x_valid_df.loc[:, non_constant]
    
    if len(x_valid_filtered.columns) > 1:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        
        corr = x_valid_filtered.corr().fillna(0)
        corr_dist = 1 - np.abs(corr)
        linkage_matrix = linkage(squareform(corr_dist), method='average')
        
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix, labels=x_valid_filtered.columns, leaf_rotation=90, leaf_font_size=8)
        plt.title('Dendrograma de Clustering de Features (por correlação)')
        plt.xlabel('Features')
        plt.ylabel('Distância (1 - |correlação|)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_clustering_dendrogram.png'), dpi=150, bbox_inches='tight')
        plt.close()
except Exception as e:
    print(f"Erro no clustering de features: {e}")

# 11. Salva Estatísticas dos Valores SHAP em CSV
print("Exportando estatísticas dos SHAP values...")
shap_stats = pd.DataFrame({
    'Feature': selected_feature_names,
    'Mean |SHAP|': np.abs(shap_array).mean(axis=0),
    'Std |SHAP|': np.abs(shap_array).std(axis=0),
    'Max |SHAP|': np.abs(shap_array).max(axis=0),
    'Min SHAP': shap_array.min(axis=0),
    'Max SHAP': shap_array.max(axis=0)
}).sort_values('Mean |SHAP|', ascending=False)

shap_stats.to_csv(os.path.join(output_dir, 'shap_feature_statistics.csv'), index=False)

print("\nTop 15 features por impacto médio absoluto:")
print(shap_stats.head(15).to_string(index=False))

# 12. Histogramas das Top 6 Features
print("\nGerando histogramas das top features...")
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Distribuição de SHAP Values - Top 6 Features', fontsize=14, y=1.00)

for idx, feature_name in enumerate(top_features[:6]):
    ax = axes[idx // 2, idx % 2]
    feature_idx = selected_feature_names.index(feature_name)
    ax.hist(shap_array[:, feature_idx], bins=50, edgecolor='black', alpha=0.7)
    ax.set_title(f'{feature_name}')
    ax.set_xlabel('SHAP value')
    ax.set_ylabel('Frequência')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_distribution_histograms.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# SALVA PRÉ-PROCESSADORES
# ============================================================================

with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(output_dir, 'selector.pkl'), 'wb') as f:
    pickle.dump(selector, f)

with open(os.path.join(output_dir, 'label_encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)

with open(os.path.join(output_dir, 'selected_feature_names.pkl'), 'wb') as f:
    pickle.dump(selected_feature_names, f)

print(f"\n{'='*80}")
print(f"PROCESSO CONCLUÍDO COM SUCESSO!")
print(f"Todos os artefatos foram salvos na pasta: {output_dir}/")
print(f"{'='*80}\n")