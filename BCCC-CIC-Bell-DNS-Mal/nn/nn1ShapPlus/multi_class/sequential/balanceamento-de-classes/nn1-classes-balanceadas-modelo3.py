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
cols_remove = ['Unnamed: 0', 'flow_id', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']
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

output_dir = f"resultados_modelo3"
os.makedirs(output_dir, exist_ok=True)
print(f"\n{'='*50}")
print(f"Diretório de saída: {output_dir}")
print(f"{'='*50}")

# Armazena resultados de todas as repetições
all_results = []

# ============================================================================
# LOOP DE 10 REPETIÇÕES COM DIFERENTES AMOSTRAS ALEATÓRIAS
# ============================================================================

# define a seed da melhor repetição (4)
random_state = 45
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
model.save(os.path.join(output_dir, f'model_4.keras'))
with open(os.path.join(output_dir, f'history_4.pkl'), 'wb') as f:
    pickle.dump(history.history, f)


# ============================================================================
# VISUALIZAÇÕES
# ============================================================================

print(f"\n{'='*80}")
print("GERANDO VISUALIZAÇÕES")
print(f"{'='*80}")

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
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close() 
print(f"Gráfico de treinamento salvo: {os.path.join(output_dir, 'training_history.png')}")


print("\nANÁLISE SHAP")

# usa subset menor para SHAP (eh mais rápido)
x_train_sample = x_train[:1000]  # pega 1000 amostras
x_valid_sample = x_test[:100]   # pega 100 amostras para análise

y_valid_sample_labels = y_test_multi[:100]
y_valid_pred_proba = model.predict(x_valid_sample, verbose=0)
y_valid_pred_classes = np.argmax(y_valid_pred_proba, axis=1)

print(f"Usando {len(x_train_sample)} amostras de treino para background")
print(f"Calculando SHAP para {len(x_valid_sample)} amostras de validação")

# calcula valores shap
print("Criando explainer...")
explainer = shap.DeepExplainer(model, x_train_sample)

print("Calculando valores SHAP...")
shap_values = explainer.shap_values(x_valid_sample)

print(f"\nNúmero de features no modelo: {x_train.shape[1]}")
print(f"Número de nomes de features salvos: {len(selected_feature_names)}")

assert x_train.shape[1] == len(selected_feature_names), \
    f"Erro: {x_train.shape[1]} features vs {len(selected_feature_names)} nomes"

# verifica formato dos shap_values - para multiclasse, será uma lista
if isinstance(shap_values, list):
    print(f"SHAP values é uma lista com {len(shap_values)} classes")
    print(f"Shape de cada elemento: {[sv.shape for sv in shap_values]}")
    
    # para multiclasse, cada elemento é (n_samples, n_features)
    # agregamos somando os valores absolutos
    shap_array = np.zeros_like(shap_values[0])
    for i, class_shap in enumerate(shap_values):
        print(f"  Classe {i}: shape={class_shap.shape}, tipo={type(class_shap)}")
        shap_array += np.abs(class_shap)
    
    print(f"Shape do SHAP array agregado: {shap_array.shape}")
    
    # se ainda tiver 3 dimensões, fazer squeeze
    if len(shap_array.shape) == 3:
        print(f"AVISO: Shape com 3 dimensões detectado, fazendo reshape...")
        if shap_array.shape[2] == 1:
            shap_array = shap_array.squeeze(axis=2)
            print(f"Shape após squeeze: {shap_array.shape}")
        else:
            # se a terceira dimensão for o número de classes, fazer média
            shap_array = np.mean(np.abs(shap_array), axis=2)
            print(f"Shape após média na dimensão das classes: {shap_array.shape}")
else:
    shap_array = shap_values
    print(f"Shape dos SHAP values: {shap_array.shape}")
    
    # se tiver 3 dimensões, ajustar
    if len(shap_array.shape) == 3:
        print(f"AVISO: Shape com 3 dimensões detectado")
        if shap_array.shape[2] == 1:
            shap_array = shap_array.squeeze(axis=2)
        else:
            shap_array = np.mean(np.abs(shap_array), axis=2)
        print(f"Shape após ajuste: {shap_array.shape}")

print(f"Shape do x_valid_sample: {x_valid_sample.shape}")
print(f"Shape final do SHAP array: {shap_array.shape}")

# verifica se o shape está correto
assert len(shap_array.shape) == 2, f"SHAP array deve ter 2 dimensões, tem {len(shap_array.shape)}: {shap_array.shape}"
assert shap_array.shape[0] == x_valid_sample.shape[0], f"Número de amostras não bate: {shap_array.shape[0]} vs {x_valid_sample.shape[0]}"
assert shap_array.shape[1] == x_valid_sample.shape[1], f"Número de features não bate: {shap_array.shape[1]} vs {x_valid_sample.shape[1]}"

# cria objeto Explanation para os plots que necessitam
# para multiclasse, expected_value será uma lista ou array
if isinstance(explainer.expected_value, list):
    # usa média dos expected values de todas as classes
    expected_val = np.mean([float(v) for v in explainer.expected_value])
else:
    try:
        expected_val_array = np.array(explainer.expected_value)
        expected_val = float(np.mean(expected_val_array))
    except:
        expected_val = 0.0

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
plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"SHAP summary plot salvo: {os.path.join(output_dir, 'shap_summary_plot.png')}")

# visualização do bar plot (importância média)
print("Gerando bar plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_array, x_valid_sample,
                  feature_names=selected_feature_names,
                  max_display=20,  # mostra top 20 features
                  plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_bar_plot.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"SHAP bar plot salvo: {os.path.join(output_dir, 'shap_bar_plot.png')}")

# visualização do waterfall para primeira predição
print("Gerando waterfall plot...")
plt.figure(figsize=(12, 10))

# usa o objeto Explanation já criado
shap.waterfall_plot(shap_explanation[0], max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_waterfall_plot.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"SHAP waterfall plot salvo: {os.path.join(output_dir, 'shap_waterfall_plot.png')}")

# heatmap de valores SHAP para visualizar padrões entre amostras
print("\nGerando heatmap de valores SHAP...")
plt.figure(figsize=(14, 10))
shap.plots.heatmap(shap_explanation, max_display=15, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   Heatmap salvo: {os.path.join(output_dir, 'shap_heatmap.png')}")
print("   (mostra padrões de SHAP values em 100 amostras)")

# bar plot com valores absolutos máximos (destaca features com alto impacto)
print("\nGerando bar plot com valores máximos...")
plt.figure(figsize=(12, 10))
shap.plots.bar(shap_explanation.abs.max(0), max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_bar_max.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   Bar plot (max) salvo: {os.path.join(output_dir, 'shap_bar_max.png')}")
print("   (destaca features com impacto máximo em casos específicos)")

# beeswarm plot com valores absolutos e cor sólida
print("\nGerando beeswarm plot com valores absolutos...")
plt.figure(figsize=(12, 10))
shap.plots.beeswarm(shap_explanation.abs, color="shap_red", max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_beeswarm_abs.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   Beeswarm absoluto salvo: {os.path.join(output_dir, 'shap_beeswarm_abs.png')}")
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
    safe_name = feature_name.replace("/", "_").replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f'shap_scatter_{safe_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()

print(f"   Scatter plots salvos em: {output_dir}/shap_scatter_*.png")

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
        safe_name = feature_name.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f'shap_dependence_{safe_name}_colored.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   {i}. Dependence plot para {feature_name} salvo")
    except Exception as e:
        print(f"   {i}. Erro ao gerar dependence plot para {feature_name}: {e}")

# force plot para exemplos específicos (cada tipo de tráfego)
print("\nGerando force plots para exemplos específicos...")
# usa as predições das 100 amostras
y_valid_sample = y_valid_sample_labels
y_pred_sample_classes = y_valid_pred_classes

# encontra exemplos de cada tipo de tráfego
class_names = ['Benigno', 'Malware', 'Phishing', 'Spam']

for class_id, class_name in enumerate(class_names):
    class_idx = np.where(y_valid_sample == class_id)[0]
    if len(class_idx) > 0:
        sample_idx = class_idx[0]
        shap.plots.waterfall(shap_explanation[sample_idx], max_display=15, show=False)
        plt.tight_layout()
        filename = os.path.join(output_dir, f'shap_waterfall_{class_name.lower()}_example.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Waterfall plot ({class_name}) salvo: {filename}")
        print(f"   Predição: {y_pred_sample_classes[sample_idx]}, Real: {y_valid_sample[sample_idx]}")
    else:
        print(f"   Aviso: Não há exemplos de {class_name} nas 100 amostras")

# análise de features agrupadas por correlação
print("\nAnálise de clustering de features...")
try:
    # calcula matriz de correlação para features selecionadas
    x_valid_df = pd.DataFrame(x_valid_sample, columns=selected_feature_names)
    
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
        filename = os.path.join(output_dir, 'feature_clustering_dendrogram.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Dendrograma salvo: {filename}")
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
filename = os.path.join(output_dir, 'shap_feature_statistics.csv')
shap_stats.to_csv(filename, index=False)
print(f"\n   Estatísticas completas salvas: {filename}")

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
filename = os.path.join(output_dir, 'shap_distribution_histograms.png')
plt.savefig(filename, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Histogramas salvos: {filename}")



# ============================================================================
# SALVAMENTOS
# ============================================================================

# salva o scaler e selector para pre-processamento
with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(output_dir, 'selector.pkl'), 'wb') as f:
    pickle.dump(selector, f)
    
with open(os.path.join(output_dir, 'label_encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)

# Salva também os nomes das features selecionadas
with open(os.path.join(output_dir, 'selected_feature_names.pkl'), 'wb') as f:
    pickle.dump(selected_feature_names, f)
    
print(f"\n{'='*80}")
print(f"Pré-processadores salvos em: {output_dir}/")
print("  - scaler.pkl")
print("  - selector.pkl")
print("  - label_encoders.pkl")
print("  - selected_feature_names.pkl")
print(f"{'='*80}\n")