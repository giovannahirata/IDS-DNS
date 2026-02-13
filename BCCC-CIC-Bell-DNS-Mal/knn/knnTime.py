import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import time

print("Carregando e combinando datasets...")

dir = "../"

# Carregar datasets benignos
benign_files = [
    'output-of-benign-pcap-0.csv',
    'output-of-benign-pcap-1.csv',
    'output-of-benign-pcap-2.csv',
    'output-of-benign-pcap-3.csv'
]
benign_dfs = [pd.read_csv(dir+f) for f in benign_files]
benign_df = pd.concat(benign_dfs)
benign_df['maligno'] = 0  # Rótulo binário
benign_df['tipo_maligno'] = 'Benigno'  # Rótulo multiclasse

# Carregar datasets malignos
malware_df = pd.read_csv(dir+'output-of-malware-pcap.csv')
malware_df['maligno'] = 1
malware_df['tipo_maligno'] = 'Malware'

phishing_df = pd.read_csv(dir+'output-of-phishing-pcap.csv')
phishing_df['maligno'] = 1
phishing_df['tipo_maligno'] = 'Phishing'

spam_df = pd.read_csv(dir+'output-of-spam-pcap.csv')
spam_df['maligno'] = 1
spam_df['tipo_maligno'] = 'Spam'

# Combinar todos os dados
df = pd.concat([benign_df, malware_df, phishing_df, spam_df])

print("\nPré-processamento de dados...")

# Remover colunas não úteis
cols_remover = ['Unnamed: 0', 'flow_id', 'timestamp', 'src_ip', 'dst_ip', 'label']
# Remover apenas colunas que existem
cols_remover = [col for col in cols_remover if col in df.columns]
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

print("\nIniciando classificação binária...")

# Dividir dados binários
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X, y_bin, test_size=0.3, random_state=42
)

# Padronizar features
scaler = StandardScaler()
X_train_bin_scaled = scaler.fit_transform(X_train_bin)
X_test_bin_scaled = scaler.transform(X_test_bin)

print("\nTreinando modelo para grandes volumes de dados...")

train_times_bin = []
test_times_bin = []
metrics_bin = []
# treino e avaliação para classificação binária testando para 2 <= k <= 20 vizinhos
for i in range(2, 21):
    neigh = KNeighborsClassifier(n_neighbors=i)
    
    # treinar:
    start_time_bin_train = time.perf_counter()
    neigh.fit(X_train_bin_scaled, y_train_bin)
    end_time_bin_train = time.perf_counter()
    
    elapsed_time_bin_train = end_time_bin_train - start_time_bin_train
    train_times_bin.append(elapsed_time_bin_train)
    
    print(f"\n[KNN binário] Tempo de treinamento com k = {i} vizinhos: {elapsed_time_bin_train}")
    
    # avaliar:
    start_time_bin_test = time.perf_counter()
    y_pred = neigh.predict(X_test_bin_scaled)
    end_time_bin_test = time.perf_counter()
    
    elapsed_time_bin_test = end_time_bin_test - start_time_bin_test
    test_times_bin.append(elapsed_time_bin_test)
    
    print(f"[KNN binário] Tempo de teste com k = {i} vizinhos: {elapsed_time_bin_test}")
    
    accuracy = accuracy_score(y_test_bin, y_pred)
    precision = precision_score(y_test_bin, y_pred)
    recall = recall_score(y_test_bin, y_pred)
    f1 = f1_score(y_test_bin, y_pred)
    
    print(f"[KNN binário] Métricas de desempenho para {i} vizinhos:")
    print(f"Acurácia: {accuracy}, precisão: {precision}, recall: {recall}, f1: {f1}")
    
    metrics_bin.append([accuracy, precision, recall, f1])

# depois, talvez colocar o relatorio do modelo que obteve o melhor desempenho para certo parâmetro
# print("\nRelatório Binário:")
# print(classification_report(y_test_bin, y_pred))

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

train_times_multi = []
test_times_multi = []
metrics_multi = []
# treino e avaliação para classificação multiclasse testando para 2 <= k <= 20 vizinhos
for i in range(2, 21):
    neigh_multi = KNeighborsClassifier(n_neighbors=i, weights='distance')
    
    # treino
    start_time_multi_train = time.perf_counter()
    neigh_multi.fit(X_train_multi_scaled, y_train_multi)
    end_time_multi_train = time.perf_counter()
    
    elapsed_time_multi_train = end_time_multi_train - start_time_multi_train
    train_times_multi.append(elapsed_time_multi_train)
    
    print(f"\n[KNN multiclasse] Tempo de treinamento com k = {i} vizinhos: {elapsed_time_multi_train}")
    
    # avalia:
    start_time_multi_test = time.perf_counter()
    y_pred_multi = neigh_multi.predict(X_test_multi_scaled)
    end_time_multi_test = time.perf_counter()
    
    elapsed_time_multi_test = end_time_multi_test - start_time_multi_test
    test_times_multi.append(elapsed_time_multi_test)
    
    print(f"[KNN multiclasse] Tempo de teste com k = {i} vizinhos: {elapsed_time_multi_test}")
    
    accuracy = accuracy_score(y_test_multi, y_pred_multi)
    precision = precision_score(y_test_multi, y_pred_multi, average='weighted')
    recall = recall_score(y_test_multi, y_pred_multi, average='weighted')
    f1 = f1_score(y_test_multi, y_pred_multi, average='weighted')
    
    print(f"[KNN multiclasse] Métricas de desempenho para {i} vizinhos:")
    print(f"Acurácia: {accuracy}, precisão: {precision}, recall: {recall}, f1: {f1}")
    
    metrics_multi.append([accuracy, precision, recall, f1])

# depois, talvez colocar o relatorio do modelo que obteve o melhor desempenho para certo parâmetro    
# print("\nRelatório Multiclasse:")
# print(classification_report(y_test_multi, y_pred_multi))

# # Matriz de confusão
# cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
# sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
#             xticklabels=neigh_multi.classes_,
#             yticklabels=neigh_multi.classes_)
# plt.title('Matriz de Confusão - Classificação Multiclasse')
# plt.xticks(rotation=45)
# plt.show()

print("\nSalvando resultados em csv...")

# criação de dataframe com os tempos:
df_times = pd.DataFrame({
    'k_neighbors': range(2,21),
    'train_time_bin': train_times_bin,
    'test_time_bin': test_times_bin,
    'train_time_multi': train_times_multi,
    'test_time_multi': test_times_multi
})
df_times.to_csv('knn_times.csv', index=False)
print("Tempos salvos em knn_times.csv")

# criação de dataframe com as métricas:
df_metrics_bin = pd.DataFrame(
    metrics_bin,
    columns=["accuracy", "precision", "recall", "f1"]
)
df_metrics_bin["k_neighbors"] = range(2,21)
df_metrics_bin['type'] = 'binary'

df_metrics_multi = pd.DataFrame(
    metrics_multi,
    columns=["accuracy", "precision", "recall", "f1"]
)
df_metrics_multi["k_neighbors"] = range(2,21)
df_metrics_multi['type'] = 'multiclass'

df_metrics = pd.concat([df_metrics_bin, df_metrics_multi], ignore_index=True)
df_metrics.to_csv('knn_metrics.csv', index=False)
print("Métricas salvas em knn_metrics.csv")

print("\nAnalisando importância das features...")

# análise de features
feature_names = [col for col in df.columns if col not in ["maligno", "tipo_maligno", "label"]]

# remove features com desvio padrão zero antes de calcular correlação
df_features = df[feature_names]
# seleciona apenas colunas numéricas
df_features_numeric = df_features.select_dtypes(include=[np.number])
non_constant_features = df_features_numeric.loc[:, df_features_numeric.std()>0].columns

# checa correlação com target (converter tipo_maligno para numérico)
le_target = LabelEncoder()
df_target_encoded = le_target.fit_transform(df["tipo_maligno"])
correlations = df[non_constant_features].corrwith(pd.Series(df_target_encoded, index=df.index)).abs().sort_values(ascending=False)
print("\nAs 10 features mais correlacionadas com o target multiclasse:")
print(correlations.head(10))

# KNN não fornece coeficientes de importância como modelos lineares
# A correlação acima já mostra quais features têm maior relação com o target

print("\nPlotando métricas de tempo e de desempenho...")

# plot times and metrics:
k = [i for i in range(2,21)]

plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(k, train_times_bin, label="binário")
plt.plot(k, train_times_multi, label="multiclasse")
plt.title('Tempo de treinamento')
plt.xlabel('k neighbors')
plt.ylabel('Tempo de treinamento (segundos)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(k, test_times_bin, label='binário')
plt.plot(k, test_times_multi, label='multiclasse')
plt.title('Tempo de teste')
plt.xlabel('k neighbors')
plt.ylabel('Tempo de teste (segundos)')
plt.legend()

plt.tight_layout()
plt.savefig('tempos_treino_teste_bin_multi.png')
plt.close() 
print("Gráfico de tempo salvo: tempos_treino_teste_bin_multi.png")

plt.figure(figsize=(10, 6))
accuracy = [m[0] for m in metrics_bin]
precision = [m[1] for m in metrics_bin]
recall = [m[2] for m in metrics_bin]
f1 = [m[3] for m in metrics_bin]
plt.plot(k, accuracy, label="acurácia")
plt.plot(k, precision, label="precisão")
plt.plot(k, recall, label="recall")
plt.plot(k, f1, label="f1")
plt.title("Métricas de desempenho binário")
plt.xlabel("k neighbors")
plt.ylabel("Métricas de desempenho (acurácia, precisão, recall e f1)")
plt.legend()
plt.savefig('metricas_bin.png')
plt.close() 
print("Gráfico de métricas de desempenho salvo: metricas_bin.png")  

plt.figure(figsize=(10, 6))
accuracy = [m[0] for m in metrics_multi]
precision = [m[1] for m in metrics_multi]
recall = [m[2] for m in metrics_multi]
f1 = [m[3] for m in metrics_multi]
plt.plot(k, accuracy, label="acurácia")
plt.plot(k, precision, label="precisão")
plt.plot(k, recall, label="recall")
plt.plot(k, f1, label="f1")
plt.title("Métricas de desempenho multiclasse")
plt.xlabel("k neighbors")
plt.ylabel("Métricas de desempenho (acurácia, precisão, recall e f1)")
plt.legend()
plt.savefig('metricas_multi.png')
plt.close() 
print("Gráfico de métricas de desempenho salvo: metricas_multi.png")  

print("\nProcesso completo concluído!")
