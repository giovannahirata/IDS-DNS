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

# cria objeto Explanation para os plots que necessitam
# converte expected_value para valor numérico Python
if isinstance(explainer.expected_value, (list, np.ndarray)):
    expected_val = float(explainer.expected_value[0])
else:
    try:
        # converte tensor para numpy e extrai escalar
        expected_val_array = explainer.expected_value.numpy()
        expected_val = float(expected_val_array.item() if expected_val_array.size == 1 else expected_val_array[0])
    except:
        expected_val = float(explainer.expected_value)

print(f"Expected value (base value): {expected_val}")

# cria objeto Explanation com todos os dados necessários
shap_explanation = shap.Explanation(
    values=shap_array,
    base_values=np.full(len(shap_array), expected_val),
    data=x_valid_sample,
    feature_names=selected_feature_names
)

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

# visualização do waterfall para primeira predição
print("Gerando waterfall plot...")
plt.figure(figsize=(12, 10))

# usa o objeto Explanation já criado
shap.waterfall_plot(shap_explanation[0], max_display=20, show=False)
plt.tight_layout()
plt.savefig('shap_waterfall_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP waterfall plot salvo: shap_waterfall_plot.png")

# heatmap de valores SHAP para visualizar padrões entre amostras
print("\nGerando heatmap de valores SHAP...")
plt.figure(figsize=(14, 10))
shap.plots.heatmap(shap_explanation, max_display=15, show=False)
plt.tight_layout()
plt.savefig('shap_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Heatmap salvo: shap_heatmap.png")
print("   (mostra padrões de SHAP values em 100 amostras)")

# bar plot com valores absolutos máximos (destaca features com alto impacto)
print("\nGerando bar plot com valores máximos...")
plt.figure(figsize=(12, 10))
shap.plots.bar(shap_explanation.abs.max(0), max_display=20, show=False)
plt.tight_layout()
plt.savefig('shap_bar_max.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Bar plot (max) salvo: shap_bar_max.png")
print("   (destaca features com impacto máximo em casos específicos)")

# beeswarm plot com valores absolutos e cor sólida
print("\nGerando beeswarm plot com valores absolutos...")
plt.figure(figsize=(12, 10))
shap.plots.beeswarm(shap_explanation.abs, color="shap_red", max_display=20, show=False)
plt.tight_layout()
plt.savefig('shap_beeswarm_abs.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Beeswarm absoluto salvo: shap_beeswarm_abs.png")
print("   (mostra magnitude do impacto sem considerar direção)")

# scatter plots para as top 5 features mais importantes
print("\nGerando scatter plots para top features...")
mean_abs_shap = np.abs(shap_array).mean(axis=0)
top_features_idx = np.argsort(mean_abs_shap)[-5:][::-1]
top_features = [selected_feature_names[i] for i in top_features_idx]

for i, feature_name in enumerate(top_features, 1):
    print(f"   {i}. {feature_name}")
    plt.figure(figsize=(10, 6))
    shap.plots.scatter(shap_explanation[:, feature_name], show=False)
    plt.tight_layout()
    plt.savefig(f'shap_scatter_{feature_name.replace("/", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"   Scatter plots salvos: shap_scatter_*.png")

# dependence plots - mostra interações entre features
print("\nGerando dependence plots (interações entre features)...")
# para as top 3 features, mostra interação com outras features
for i, feature_name in enumerate(top_features[:3], 1):
    try:
        plt.figure(figsize=(10, 6))
        shap.plots.scatter(shap_explanation[:, feature_name], 
                          color=shap_explanation, 
                          show=False)
        plt.tight_layout()
        plt.savefig(f'shap_dependence_{feature_name.replace("/", "_")}_colored.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   {i}. Dependence plot para {feature_name} salvo")
    except Exception as e:
        print(f"   {i}. Erro ao gerar dependence plot para {feature_name}: {e}")

# force plot para exemplos específicos (maligno e benigno)
print("\nGerando force plots para exemplos específicos...")
# usa as predições das 100 amostras
y_valid_sample = y_valid_bin[:100]
y_pred_sample = model.predict(x_valid_sample).ravel()

# encontra exemplos de tráfego maligno e benigno nas 100 amostras
malicious_idx = np.where(y_valid_sample == 1)[0]
benign_idx = np.where(y_valid_sample == 0)[0]

if len(malicious_idx) > 0 and len(benign_idx) > 0:
    # exemplo maligno
    mal_sample = malicious_idx[0]
    shap.plots.waterfall(shap_explanation[mal_sample], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig('shap_waterfall_malicious_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Waterfall plot (tráfego MALIGNO) salvo: shap_waterfall_malicious_example.png")
    print(f"   Predição: {y_pred_sample[mal_sample]:.4f}, Real: {y_valid_sample[mal_sample]}")
    
    # exemplo benigno
    ben_sample = benign_idx[0]
    shap.plots.waterfall(shap_explanation[ben_sample], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig('shap_waterfall_benign_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Waterfall plot (tráfego BENIGNO) salvo: shap_waterfall_benign_example.png")
    print(f"   Predição: {y_pred_sample[ben_sample]:.4f}, Real: {y_valid_sample[ben_sample]}")
else:
    print(f"   Aviso: Não há exemplos suficientes nas 100 amostras")
    print(f"   Malignos: {len(malicious_idx)}, Benignos: {len(benign_idx)}")

# análise de features agrupadas por correlação
print("\nAnálise de clustering de features...")
try:
    # calcula matriz de correlação para features selecionadas
    x_valid_df = pd.DataFrame(x_valid_transformed, columns=selected_feature_names)
    
    # remove features constantes
    non_constant = x_valid_df.std() > 0
    x_valid_filtered = x_valid_df.loc[:, non_constant]
    
    if len(x_valid_filtered.columns) > 1:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        
        # calcula correlação e distância
        corr = x_valid_filtered.corr().fillna(0)
        corr_dist = 1 - np.abs(corr)
        
        # hierarchical clustering
        linkage_matrix = linkage(squareform(corr_dist), method='average')
        
        # plot dendrograma
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix, labels=x_valid_filtered.columns, 
                  leaf_rotation=90, leaf_font_size=8)
        plt.title('Dendrograma de Clustering de Features (por correlação)')
        plt.xlabel('Features')
        plt.ylabel('Distância (1 - |correlação|)')
        plt.tight_layout()
        plt.savefig('feature_clustering_dendrogram.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   Dendrograma salvo: feature_clustering_dendrogram.png")
        print("   (mostra agrupamentos de features correlacionadas)")
except Exception as e:
    print(f"   Erro no clustering: {e}")

# resumo estatístico dos valores SHAP
print("\nEstatísticas dos valores SHAP:")
shap_stats = pd.DataFrame({
    'Feature': selected_feature_names,
    'Mean |SHAP|': np.abs(shap_array).mean(axis=0),
    'Std |SHAP|': np.abs(shap_array).std(axis=0),
    'Max |SHAP|': np.abs(shap_array).max(axis=0),
    'Min SHAP': shap_array.min(axis=0),
    'Max SHAP': shap_array.max(axis=0)
})
shap_stats = shap_stats.sort_values('Mean |SHAP|', ascending=False)

print("\n   Top 15 features por impacto médio absoluto:")
print(shap_stats.head(15).to_string(index=False))

# salva estatísticas completas
shap_stats.to_csv('shap_feature_statistics.csv', index=False)
print("\n   Estatísticas completas salvas: shap_feature_statistics.csv")

# análise de distribuição de SHAP values
print("\nGerando histogramas de distribuição SHAP...")
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
plt.savefig('shap_distribution_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Histogramas salvos: shap_distribution_histograms.png")

print("\nArquivos gerados:")
print("  - shap_summary_plot.png (beeswarm - distribuição completa)")
print("  - shap_bar_plot.png (importância média)")
print("  - shap_bar_max.png (importância por impacto máximo)")
print("  - shap_beeswarm_abs.png (magnitude absoluta)")
print("  - shap_waterfall_plot.png (exemplo geral)")
print("  - shap_waterfall_malicious_example.png (exemplo maligno)")
print("  - shap_waterfall_benign_example.png (exemplo benigno)")
print("  - shap_heatmap.png (padrões entre amostras)")
print("  - shap_scatter_*.png (dependência de features individuais)")
print("  - shap_dependence_*_colored.png (interações entre features)")
print("  - shap_distribution_histograms.png (distribuições SHAP)")
print("  - feature_clustering_dendrogram.png (agrupamento de features)")
print("  - shap_feature_statistics.csv (estatísticas detalhadas)")
print("  - training_history.png (histórico de treinamento)")
print("  - roc_curve.png (curva ROC)")

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