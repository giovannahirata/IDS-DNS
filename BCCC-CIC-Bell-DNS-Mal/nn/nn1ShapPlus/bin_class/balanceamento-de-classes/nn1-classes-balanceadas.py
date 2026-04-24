import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import pickle
import time

np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("Experimento MLP binário com balanceamento de classes")
print("10 repetições com diferentes amostras aleatórias de dados benignos")
print("="*80)

# ============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ============================================================================

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

df_malicious = pd.concat([df_malware, df_phishing, df_spam], ignore_index=True)

print(f"\nEstatísticas do dataset completo:")
print(f"Total de amostras benignas: {len(df_benign)}")
print(f"Total de amostras malignas: {len(df_malicious)}")
print(f"  - Malware: {len(df_malware)}")
print(f"  - Phishing: {len(df_phishing)}")
print(f"  - Spam: {len(df_spam)}")

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
print(f"\nBalanceamento: selecionaremos {n_malicious} amostras benignas aleatórias em cada repetição")
print(f"Split: 70% treino / 30% teste")

# Armazena resultados de todas as repetições
all_results = []

# ============================================================================
# LOOP DE 10 REPETIÇÕES COM DIFERENTES AMOSTRAS ALEATÓRIAS
# ============================================================================

for repetition in range(1, 11):
    print(f"\n{'='*80}")
    print(f"REPETIÇÃO {repetition}/10")
    print(f"{'='*80}")
    
    # Define uma seed diferente para cada repetição
    random_state = 42 + repetition
    
    # Seleciona aleatoriamente n_malicious amostras benignas
    df_benign_sampled = df_benign.sample(n=n_malicious, random_state=random_state)
    
    # Concatena benignos balanceados com todos os malignos
    df_balanced = pd.concat([df_benign_sampled, df_malicious], ignore_index=True)
    
    print(f"\nDataset balanceado para repetição {repetition}:")
    print(f"Total de amostras: {len(df_balanced)}")
    print(f"Distribuição por tipo:")
    print(df_balanced['maligno'].value_counts())
    
    # Split 70/30 estratificado
    df_train, df_test = train_test_split(
        df_balanced, 
        test_size=0.3, 
        random_state=random_state,
        stratify=df_balanced['maligno']
    )
    
    print(f"\nSplit 70/30:")
    print(f"Treino: {len(df_train)} amostras")
    print(f"Teste: {len(df_test)} amostras")
    print(f"Distribuição treino:\n{df_train['maligno'].value_counts()}")
    print(f"Distribuição teste:\n{df_test['maligno'].value_counts()}")
    
    # Separa features e targets
    x_train = df_train.drop(columns=["maligno", "tipo_maligno"])
    x_test = df_test.drop(columns=["maligno", "tipo_maligno"])
    y_train_bin = df_train["maligno"].values
    y_test_bin = df_test["maligno"].values
    
    num_class_y = len(np.unique(y_train_bin))
    
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
    y_train_bin = y_train_bin.astype(np.int32)
    y_test_bin = y_test_bin.astype(np.int32)
    
    # Cria o modelo MLP
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
    
    if repetition == 1:
        print("\nArquitetura do modelo:")
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
    
    # Treinamento
    print(f"\nIniciando treinamento da repetição {repetition}...")
    start_time_train = time.perf_counter()
    
    history = model.fit(
        x_train, y_train_bin,
        validation_data=(x_test, y_test_bin),
        epochs=30,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    end_time_train = time.perf_counter()
    elapsed_time_train = end_time_train - start_time_train
    
    # Avaliação
    print(f"Avaliando modelo da repetição {repetition}...")
    start_time_test = time.perf_counter()
    
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test_bin, verbose=0)
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).ravel()
    y_pred_proba = y_pred_proba.ravel()
    
    end_time_test = time.perf_counter()
    elapsed_time_test = end_time_test - start_time_test
    
    # Calcula métricas
    y_test_int = y_test_bin.astype(int)
    precision = precision_score(y_test_int, y_pred, average='binary')
    recall = recall_score(y_test_int, y_pred, average='binary')
    f1 = f1_score(y_test_int, y_pred, average='binary')
    
    try:
        auc_score = roc_auc_score(y_test_int, y_pred_proba)
    except:
        auc_score = 0.0
    
    # Métricas por classe
    precision_per_class = precision_score(y_test_int, y_pred, average=None)
    recall_per_class = recall_score(y_test_int, y_pred, average=None)
    f1_per_class = f1_score(y_test_int, y_pred, average=None)
    
    print(f"\n{'='*50}")
    print(f"RESULTADOS DA REPETIÇÃO {repetition}")
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
                                target_names=['Benigno', 'Maligno']))
    
    # Armazena resultados
    result = {
        'repetition': repetition,
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
        'precision_maligno': precision_per_class[1],
        'recall_benigno': recall_per_class[0],
        'recall_maligno': recall_per_class[1],
        'f1_benigno': f1_per_class[0],
        'f1_maligno': f1_per_class[1],
    }
    
    all_results.append(result)
    
    # Salva modelo e histórico desta repetição
    model.save(f'model_repetition_{repetition}.keras')
    with open(f'history_repetition_{repetition}.pkl', 'wb') as f:
        pickle.dump(history.history, f)

# ============================================================================
# ANÁLISE AGREGADA DOS RESULTADOS
# ============================================================================

print(f"\n{'='*80}")
print("Análise agregada das 10 repetições")
print(f"{'='*80}")

df_results = pd.DataFrame(all_results)
df_results.to_csv('mlp_binary_10_repetitions_results.csv', index=False)
print(f"\nResultados salvos em: mlp_binary_10_repetitions_results.csv")

# Estatísticas agregadas
print(f"\n{'='*50}")
print("Métricas gerais (média ± desvio padrão)")
print(f"{'='*50}")
metrics_of_interest = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 
                       'train_time', 'test_time', 'epochs']

for metric in metrics_of_interest:
    mean = df_results[metric].mean()
    std = df_results[metric].std()
    min_val = df_results[metric].min()
    max_val = df_results[metric].max()
    print(f"{metric:15s}: {mean:.4f} ± {std:.4f} (min: {min_val:.4f}, max: {max_val:.4f})")

# Métricas por classe
print(f"\n{'='*50}")
print("MÉTRICAS POR CLASSE (média ± desvio padrão)")
print(f"{'='*50}")

classes = ['benigno', 'maligno']
for cls in classes:
    print(f"\n{cls.upper()}:")
    for metric_type in ['precision', 'recall', 'f1']:
        col_name = f'{metric_type}_{cls}'
        mean = df_results[col_name].mean()
        std = df_results[col_name].std()
        print(f"  {metric_type:10s}: {mean:.4f} ± {std:.4f}")

# Melhor e pior repetição
best_rep = df_results.loc[df_results['f1_score'].idxmax()]
worst_rep = df_results.loc[df_results['f1_score'].idxmin()]

print(f"\n{'='*50}")
print("MELHOR REPETIÇÃO (por F1-Score)")
print(f"{'='*50}")
print(f"Repetição: {int(best_rep['repetition'])}")
print(f"F1-Score: {best_rep['f1_score']:.4f}")
print(f"Acurácia: {best_rep['accuracy']:.4f}")
print(f"AUC-ROC: {best_rep['auc_roc']:.4f}")

print(f"\n{'='*50}")
print("PIOR REPETIÇÃO (por F1-Score)")
print(f"{'='*50}")
print(f"Repetição: {int(worst_rep['repetition'])}")
print(f"F1-Score: {worst_rep['f1_score']:.4f}")
print(f"Acurácia: {worst_rep['accuracy']:.4f}")
print(f"AUC-ROC: {worst_rep['auc_roc']:.4f}")

# ============================================================================
# VISUALIZAÇÕES
# ============================================================================

print(f"\n{'='*80}")
print("GERANDO VISUALIZAÇÕES")
print(f"{'='*80}")

# Gráfico 1: Distribuição de métricas gerais
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribuição de Métricas Gerais (10 Repetições)', fontsize=16)

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    ax.bar(range(1, 11), df_results[metric], color='steelblue', alpha=0.7)
    ax.axhline(df_results[metric].mean(), color='red', linestyle='--', label='Média')
    ax.set_xlabel('Repetição')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()}\n(μ={df_results[metric].mean():.4f}, σ={df_results[metric].std():.4f})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

# Tempos
ax = axes[1, 2]
x = np.arange(1, 11)
width = 0.35
ax.bar(x - width/2, df_results['train_time'], width, label='Treino', alpha=0.7)
ax.bar(x + width/2, df_results['test_time'], width, label='Teste', alpha=0.7)
ax.set_xlabel('Repetição')
ax.set_ylabel('Tempo (s)')
ax.set_title('Tempos de Execução')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('mlp_binary_10reps_overview.png', dpi=150)
plt.close()
print("Gráfico salvo: mlp_binary_10reps_overview.png")

# Gráfico 2: Métricas por classe
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('F1-Score por Classe (10 Repetições)', fontsize=16)

for idx, cls in enumerate(classes):
    ax = axes[idx // 2, idx % 2]
    col_name = f'f1_{cls}'
    ax.bar(range(1, 11), df_results[col_name], color='coral', alpha=0.7)
    ax.axhline(df_results[col_name].mean(), color='darkred', linestyle='--', label='Média')
    ax.set_xlabel('Repetição')
    ax.set_ylabel('F1-Score')
    ax.set_title(f'{cls.title()}\n(μ={df_results[col_name].mean():.4f}, σ={df_results[col_name].std():.4f})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('mlp_binary_10reps_f1_per_class.png', dpi=150)
plt.close()
print("Gráfico salvo: mlp_binary_10reps_f1_per_class.png")

# Gráfico 3: Boxplot comparativo
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Métricas gerais
ax1 = axes[0]
data_general = [df_results[m] for m in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']]
bp1 = ax1.boxplot(data_general, labels=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
ax1.set_ylabel('Valor')
ax1.set_title('Distribuição de Métricas Gerais')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1])

# F1-Score por classe
ax2 = axes[1]
data_classes = [df_results[f'f1_{cls}'] for cls in classes]
bp2 = ax2.boxplot(data_classes, labels=[cls.title() for cls in classes])
ax2.set_ylabel('F1-Score')
ax2.set_title('Distribuição de F1-Score por Classe')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('mlp_binary_10reps_boxplots.png', dpi=150)
plt.close()
print("Gráfico salvo: mlp_binary_10reps_boxplots.png")

print(f"\n{'='*80}")
print("EXPERIMENTO CONCLUÍDO!")
print(f"{'='*80}")
print(f"\nArquivos gerados:")
print(f"  - mlp_binary_10_repetitions_results.csv")
print(f"  - mlp_binary_10reps_overview.png")
print(f"  - mlp_binary_10reps_f1_per_class.png")
print(f"  - mlp_binary_10reps_boxplots.png")
print(f"  - model_repetition_[1-10].keras (10 modelos)")
print(f"  - history_repetition_[1-10].pkl (10 históricos)")
print(f"\n{'='*80}")
