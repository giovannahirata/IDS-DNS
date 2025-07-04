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
X_malign = malign_df[selected_columns]  # Usar mesmas features selecionadas
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
sample_idx = np.random.choice(X_train_bin_scaled.shape[0], size=10000, replace=False)
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

# 8. Modelo para Grandes Volumes de Dados -------------------------------------
print("\nTreinando modelo para grandes volumes de dados...")

# Usar LinearSVC (mais eficiente)
linear_svc = LinearSVC(C=0.1, class_weight='balanced', max_iter=10000, random_state=42)
linear_svc.fit(X_train_bin_scaled, y_train_bin)

# Avaliar
y_pred_linear = linear_svc.predict(X_test_bin_scaled)
print("\nRelatório LinearSVC (Binário):")
print(classification_report(y_test_bin, y_pred_linear))

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

print("\nProcesso completo concluído!")