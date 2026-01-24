import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 1. Carregar e combinar datasets -----------------------------------------------
print("Carregando e combinando datasets...")

# Carregar datasets benignos
benign_files = [
    'output-of-benign-pcap-0.csv',
    'output-of-benign-pcap-1.csv',
    'output-of-benign-pcap-2.csv',
    'output-of-benign-pcap-3.csv'
]
benign_dfs = [pd.read_csv(f) for f in benign_files]
benign_df = pd.concat(benign_dfs)
benign_df['maligno'] = 0  # Rótulo binário
benign_df['tipo_maligno'] = 'Benigno'  # Rótulo multiclasse

# Carregar datasets malignos
malware_df = pd.read_csv('output-of-malware-pcap.csv')
malware_df['maligno'] = 1
malware_df['tipo_maligno'] = 'Malware'

phishing_df = pd.read_csv('output-of-phishing-pcap.csv')
phishing_df['maligno'] = 1
phishing_df['tipo_maligno'] = 'Phishing'

spam_df = pd.read_csv('output-of-spam-pcap.csv')
spam_df['maligno'] = 1
spam_df['tipo_maligno'] = 'Spam'

# Combinar todos os dados
df = pd.concat([benign_df, malware_df, phishing_df, spam_df])

# 2. Pré-processamento ---------------------------------------------------------
print("\nPré-processamento de dados...")

# Remover colunas não úteis
cols_remover = ['Unnamed: 0', 'flow_id', 'timestamp', 'src_ip', 'dst_ip', 
                'query_resource_record_type', 'ans_resource_record_type',
                'query_resource_record_class', 'ans_resource_record_class', 'label']
df = df.drop(columns=cols_remover)

# Preencher valores ausentes
df = df.fillna(0)

# Codificar variáveis categóricas
le = LabelEncoder()
df['protocol'] = le.fit_transform(df['protocol'])

# Separar features e rótulos
X = df.drop(columns=['maligno', 'tipo_maligno'])
y_bin = df['maligno']  # Binário (0=Benigno, 1=Maligno)
y_multi = df['tipo_maligno']  # Multiclasse (Benigno, Malware, Phishing, Spam)
#X = pd.get_dummies(X)   
cat_cols = [col for col in X.columns if X[col].dtype == 'object' and X[col].nunique() < 50]
X = pd.get_dummies(X, columns=cat_cols)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = X.astype(np.float64)

# 3. Seleção de Features -------------------------------------------------------
print("\nSelecionando features relevantes...")

# Remover features com variância zero
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# Obter nomes das colunas selecionadas
selected_mask = selector.get_support()
selected_columns = X.columns[selected_mask]
print(f"Features originais: {X.shape[1]}, Features selecionadas: {len(selected_columns)}")

# Atualizar X com features selecionadas
X = pd.DataFrame(X_selected, columns=selected_columns)

# 4. Classificação Binária (Benigno vs Maligno) --------------------------------
print("\nIniciando classificação binária...")

# Dividir dados binários
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X, y_bin, test_size=0.3, random_state=42
)

# Padronizar features
scaler = StandardScaler()
X_train_bin_scaled = scaler.fit_transform(X_train_bin)
X_test_bin_scaled = scaler.transform(X_test_bin)

"""

# Treinar SVM com kernel RBF
svm_bin = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
svm_bin.fit(X_train_bin_scaled, y_train_bin)

# Avaliar
y_pred_bin = svm_bin.predict(X_test_bin_scaled)
print("\nRelatório Binário (Benigno vs Maligno):")
print(classification_report(y_test_bin, y_pred_bin, target_names=['Benigno', 'Maligno']))

# Matriz de confusão
cm = confusion_matrix(y_test_bin, y_pred_bin)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benigno', 'Maligno'], 
            yticklabels=['Benigno', 'Maligno'])
plt.title('Matriz de Confusão - Classificação Binária')
plt.show()

# 5. Classificação Multiclasse (Tipos de Ameaças) ------------------------------
print("\nIniciando classificação multiclasse...")

# Filtrar apenas amostras malignas
malign_df = df[df['maligno'] == 1]
# X_malign = malign_df[selected_columns]  # Usar mesmas features selecionadas
X_malign = pd.get_dummies(malign_df.drop(columns=['maligno', 'tipo_maligno']))
X_malign = X_malign.reindex(columns=selected_columns, fill_value=0)
y_malign = malign_df['tipo_maligno']  # Malware, Phishing, Spam

# Dividir dados multiclasse
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_malign, y_malign, test_size=0.3, random_state=42
)

# Padronizar (usar mesmo scaler do binário)
X_train_multi_scaled = scaler.transform(X_train_multi)
X_test_multi_scaled = scaler.transform(X_test_multi)

# Configurar pesos para classes desbalanceadas
class_weights = {
    'Malware': 1.0,
    'Phishing': 1.0,
    'Spam': 2.0  # Peso maior para classe minoritária
}

# Treinar SVM multiclasse
svm_multi = SVC(
    kernel='rbf', 
    C=10,
    gamma=0.01,
    decision_function_shape='ovr',
    class_weight=class_weights,
    random_state=42,
    probability=True
)
svm_multi.fit(X_train_multi_scaled, y_train_multi)

# Avaliar
y_pred_multi = svm_multi.predict(X_test_multi_scaled)
print("\nRelatório Multiclasse (Tipos de Malignos):")
print(classification_report(y_test_multi, y_pred_multi))

# Matriz de confusão
cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm_multi.classes_,
            yticklabels=svm_multi.classes_)
plt.title('Matriz de Confusão - Classificação Multiclasse')
plt.xticks(rotation=45)
plt.show()

# 6. Otimização de Hiperparâmetros (Opcional) ----------------------------------
print("\nOtimizando hiperparâmetros...")

# Configurar grade de parâmetros
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

# Usar subset de dados para otimização mais rápida
# sample_idx = np.random.choice(X_train_bin_scaled.shape[0], size=10000, replace=False)
sample_size = min(10000, X_train_bin_scaled.shape[0])
sample_idx = np.random.choice(X_train_bin_scaled.shape[0], size=sample_size, replace=False)
X_sample = X_train_bin_scaled[sample_idx]
y_sample = y_train_bin.iloc[sample_idx]

# Executar GridSearch
grid_bin = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, scoring='f1', verbose=1)
grid_bin.fit(X_sample, y_sample)
print("\nMelhores parâmetros (Binário):", grid_bin.best_params_)

# Treinar modelo final com melhores parâmetros
svm_bin_optimized = grid_bin.best_estimator_
svm_bin_optimized.fit(X_train_bin_scaled, y_train_bin)

# 7. Validação Cruzada --------------------------------------------------------
print("\nValidando modelo com validação cruzada...")

# Criar pipeline completo
pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),
    SVC(kernel='rbf', C=10, gamma=0.01)
)

# Executar validação cruzada para classificação binária
scores_bin = cross_val_score(pipeline, X, y_bin, cv=3, scoring='f1')
print("\nF1 médio (Binário - Validação Cruzada):", scores_bin.mean())

"""

# 8. Modelo para Grandes Volumes de Dados -------------------------------------
print("\nTreinando modelo para grandes volumes de dados...")

# Usar LinearSVC (mais eficiente)
linear_svc = LinearSVC(C=0.1, class_weight='balanced', max_iter=10000, random_state=42)
linear_svc.fit(X_train_bin_scaled, y_train_bin)

# Avaliar
y_pred_linear = linear_svc.predict(X_test_bin_scaled)
print("\nRelatório LinearSVC (Binário):")
print(classification_report(y_test_bin, y_pred_linear))

# 5'. Classificação Multiclasse (Tipos de Ameaças) ------------------------------
print("\nIniciando classificação multiclasse...")

# Filtrar apenas amostras malignas
malign_df = df[df['maligno'] == 1]
# X_malign = malign_df[selected_columns]  # Usar mesmas features selecionadas
X_malign = pd.get_dummies(malign_df.drop(columns=['maligno', 'tipo_maligno']))
X_malign = X_malign.reindex(columns=selected_columns, fill_value=0)
y_malign = malign_df['tipo_maligno']  # Malware, Phishing, Spam

# Dividir dados multiclasse
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_malign, y_malign, test_size=0.3, random_state=42
)

# Padronizar (usar mesmo scaler do binário)
X_train_multi_scaled = scaler.transform(X_train_multi)
X_test_multi_scaled = scaler.transform(X_test_multi)

# Configurar pesos para classes desbalanceadas
class_weights = {
    'Malware': 1.0,
    'Phishing': 1.0,
    'Spam': 2.0  # Peso maior para classe minoritária
}

# Treinar SVM multiclasse
# svm_multi = SVC(
#     kernel='rbf', 
#     C=10,
#     gamma=0.01,
#     decision_function_shape='ovr',
#     class_weight=class_weights,
#     random_state=42,
#     probability=True
# )
# svm_multi.fit(X_train_multi_scaled, y_train_multi)

svm_multi_linear = LinearSVC(C=0.1, class_weight=class_weights, max_iter=10000, random_state=42)
svm_multi_linear.fit(X_train_multi_scaled, y_train_multi)

# Avaliar
y_pred_multi = svm_multi_linear.predict(X_test_multi_scaled)
print("\nRelatório Multiclasse (Tipos de Malignos):")
print(classification_report(y_test_multi, y_pred_multi))

# Matriz de confusão
cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm_multi_linear.classes_,
            yticklabels=svm_multi_linear.classes_)
plt.title('Matriz de Confusão - Classificação Multiclasse')
plt.xticks(rotation=45)
plt.show()


# 9. Análise de Importância de Features (Opcional) ----------------------------
print("\nAnalisando importância das features...")

# Usar modelo linear para interpretabilidade
coef = linear_svc.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': selected_columns,
    'Importance': np.abs(coef)
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Top 10 features mais importantes
print("\nTop 10 features mais importantes:")
print(feature_importance.head(10))

# Plotar importância
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Features Mais Importantes (Modelo Linear)')
plt.tight_layout()
plt.show()

# 10. Medição do Tempo de Inferência -------------------------------------------

print("\nIniciando medição do tempo de inferência...")

# Número de amostras para testar
n_samples_inference = 1000

# Selecionar um subconjunto aleatório do conjunto de teste já padronizado
# Usamos .shape[0] para garantir que não tentemos pegar mais amostras do que existem
sample_indices = np.random.choice(X_test_bin_scaled.shape[0], size=n_samples_inference, replace=False)
inference_samples = X_test_bin_scaled[sample_indices]

# Lista para armazenar o tempo de cada inferência
inference_times = []

# Iterar sobre cada amostra para medir o tempo de predição individual
for sample in inference_samples:
    # A predição espera um formato 2D, então remodelamos a amostra
    sample_reshaped = sample.reshape(1, -1)
    
    start_time = time.perf_counter() # Inicia o cronômetro com alta precisão
    linear_svc.predict(sample_reshaped)
    end_time = time.perf_counter()   # Para o cronômetro
    
    # Calcula o tempo decorrido e adiciona à lista
    duration = end_time - start_time
    inference_times.append(duration)

# Converter a lista para um array NumPy para cálculos fáceis
inference_times = np.array(inference_times)

# Calcular média e desvio padrão
mean_time = np.mean(inference_times)
std_dev_time = np.std(inference_times)

print("\n--- Resultados do Teste de Inferência ---")
print(f"Número de amostras testadas: {n_samples_inference}")
# Convertendo para milissegundos (ms) para melhor legibilidade
print(f"Tempo médio de inferência: {mean_time * 1000:.6f} ms")
print(f"Desvio padrão do tempo: {std_dev_time * 1000:.6f} ms")

# Visualização da distribuição dos tempos de inferência
plt.figure(figsize=(10, 6))
sns.histplot(x=inference_times * 1000, kde=True)
plt.title('Distribuição dos Tempos de Inferência por Amostra')
plt.xlabel('Tempo de Inferência (ms)')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()

print("\nMedição de inferência concluída!")

# 11. Medição do Tempo de Inferência (MODELO MULTICLASSE) ----------------------

print("\nIniciando medição do tempo de inferência do modelo MULTICLASSE...")

# O conjunto de teste multiclasse (X_test_multi_scaled) é menor,
# então podemos ajustar o número de amostras ou usar todas se forem poucas.
n_samples_multi = min(500, X_test_multi_scaled.shape[0]) 

# Selecionar um subconjunto aleatório do conjunto de teste multiclasse
sample_indices_multi = np.random.choice(X_test_multi_scaled.shape[0], size=n_samples_multi, replace=False)
inference_samples_multi = X_test_multi_scaled[sample_indices_multi]

# Lista para armazenar o tempo de cada inferência
inference_times_multi = []

# Iterar sobre cada amostra para medir o tempo de predição individual
for sample in inference_samples_multi:
    sample_reshaped = sample.reshape(1, -1)
    
    start_time = time.perf_counter()
    # Usando o modelo multiclasse para a predição!
    svm_multi_linear.predict(sample_reshaped)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    inference_times_multi.append(duration)

# Converter para array NumPy
inference_times_multi = np.array(inference_times_multi)

# Calcular estatísticas
mean_time_multi = np.mean(inference_times_multi)
std_dev_time_multi = np.std(inference_times_multi)

print("\n--- Resultados do Teste de Inferência (Multiclasse) ---")
print(f"Número de amostras testadas: {n_samples_multi}")
print(f"Tempo médio de inferência: {mean_time_multi * 1000:.6f} ms")
print(f"Desvio padrão do tempo: {std_dev_time_multi * 1000:.6f} ms")

# Visualização da distribuição dos tempos
plt.figure(figsize=(10, 6))
sns.histplot(x=inference_times_multi * 1000, kde=True, color='orange')
plt.title('Distribuição dos Tempos de Inferência por Amostra (Multiclasse)')
plt.xlabel('Tempo de Inferência (ms)')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()

print("\nMedição de inferência multiclasse concluída!")

# 12. SALVANDO OS MODELOS E OBJETOS PARA PRODUÇÃO ---------------------------
import joblib
import os

print("\nSalvando o pacote de inferência...")

# Criar um diretório para salvar os modelos, se não existir
output_dir = 'modelos_salvos'
os.makedirs(output_dir, exist_ok=True)

# Salvar os dois modelos
joblib.dump(linear_svc, os.path.join(output_dir, 'modelo_binario.joblib'))
joblib.dump(svm_multi_linear, os.path.join(output_dir, 'modelo_multiclasse.joblib'))

# Salvar os objetos de pré-processamento
joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
joblib.dump(selected_columns, os.path.join(output_dir, 'selected_columns.joblib'))
joblib.dump(le, os.path.join(output_dir, 'label_encoder_protocol.joblib'))
joblib.dump(cat_cols, os.path.join(output_dir, 'categorical_columns.joblib'))

print(f"Modelos e objetos salvos com sucesso no diretório '{output_dir}'!")

print("\nProcesso completo concluído!")
