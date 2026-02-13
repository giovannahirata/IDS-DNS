import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns

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

# treino de classificação binária
neigh = KNeighborsClassifier(n_neighbors=11)
neigh.fit(X_train_bin_scaled, y_train_bin)

# avaliar:
y_pred = neigh.predict(X_test_bin_scaled)
print("\nRelatório Binário:")
print(classification_report(y_test_bin, y_pred))

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

# treino multiclasse:

neigh_multi = KNeighborsClassifier(n_neighbors=11, weights='distance')
neigh_multi.fit(X_train_multi_scaled, y_train_multi)

# avalia:

y_pred_multi = neigh_multi.predict(X_test_multi_scaled)
print("\nRelatório Multiclasse:")
print(classification_report(y_test_multi, y_pred_multi))

# Matriz de confusão
cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=neigh_multi.classes_,
            yticklabels=neigh_multi.classes_)
plt.title('Matriz de Confusão - Classificação Multiclasse')
plt.xticks(rotation=45)
plt.show()


print("\nAnalisando importância das features...")

# Usar modelo linear para interpretabilidade
coef = neigh.coef_[0]
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
