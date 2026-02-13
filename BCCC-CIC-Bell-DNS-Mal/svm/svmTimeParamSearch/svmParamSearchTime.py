import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import VarianceThreshold
import matplotlib
matplotlib.use('Agg')  # backend sem display para ambientes servidor
import matplotlib.pyplot as plt
import seaborn as sns
import time
import itertools
from datetime import datetime

# Função de logging
def log_progress(message, log_file='svm_progress.log'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')
        f.flush()

log_progress("="*80)
log_progress("INÍCIO DA EXECUÇÃO")
log_progress("="*80)

print("Carregando e combinando datasets...")
log_progress("Carregando datasets...")

dir = "../../"

# Carregar datasets benignos
benign_files = [
    'output-of-benign-pcap-0.csv',
    'output-of-benign-pcap-1.csv',
    'output-of-benign-pcap-2.csv',
    'output-of-benign-pcap-3.csv'
]
benign_dfs = [pd.read_csv(dir+f) for f in benign_files]
benign_df = pd.concat(benign_dfs)
benign_df['maligno'] = 0
benign_df['tipo_maligno'] = 'Benigno'

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

df = pd.concat([benign_df, malware_df, phishing_df, spam_df])
log_progress(f"Dataset combinado: {len(df)} registros")

print("\nPré-processamento de dados...")
log_progress("Iniciando pré-processamento...")

# Remover colunas não úteis
cols_remover = ['Unnamed: 0', 'flow_id', 'timestamp', 'src_ip', 'dst_ip', 
                'query_resource_record_type', 'ans_resource_record_type',
                'query_resource_record_class', 'ans_resource_record_class', 'label']
cols_remover = [col for col in cols_remover if col in df.columns]
df = df.drop(columns=cols_remover)

df = df.fillna(0)

le = LabelEncoder()
df['protocol'] = le.fit_transform(df['protocol'])

# Separar features e rótulos
X = df.drop(columns=['maligno', 'tipo_maligno'])
y_bin = df['maligno']
y_multi = df['tipo_maligno']

cat_cols = [col for col in X.columns if X[col].dtype == 'object' and X[col].nunique() < 50]
X = pd.get_dummies(X, columns=cat_cols)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = X.astype(np.float64)

print("\nSelecionando features relevantes...")

selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
selected_mask = selector.get_support()
selected_columns = X.columns[selected_mask]
print(f"Features originais: {X.shape[1]}, Features selecionadas: {len(selected_columns)}")

X = pd.DataFrame(X_selected, columns=selected_columns)

# =============================================================================
# CLASSIFICAÇÃO BINÁRIA - TESTANDO PARÂMETROS
# =============================================================================

print("\n" + "="*80)
print("CLASSIFICAÇÃO BINÁRIA - TESTE DE PARÂMETROS")
print("="*80)

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X, y_bin, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_bin_scaled = scaler.fit_transform(X_train_bin)
X_test_bin_scaled = scaler.transform(X_test_bin)

# Definir grade de parâmetros para testar
# O kernel RBF usa amostra reduzida (3000 amostras) para tempo de treinamento razoável
# LinearSVC usa dataset completo

param_configs = [
    # LinearSVC com diferentes C
    {'model': 'LinearSVC', 'C': 0.001, 'max_iter': 5000},
    {'model': 'LinearSVC', 'C': 0.01, 'max_iter': 5000},
    {'model': 'LinearSVC', 'C': 0.1, 'max_iter': 5000},
    {'model': 'LinearSVC', 'C': 1, 'max_iter': 5000},
    {'model': 'LinearSVC', 'C': 10, 'max_iter': 5000},
    {'model': 'LinearSVC', 'C': 100, 'max_iter': 5000},
    {'model': 'LinearSVC', 'C': 1000, 'max_iter': 5000},
    
    # LinearSVC com diferentes max_iter
    {'model': 'LinearSVC', 'C': 1, 'max_iter': 10000},
    {'model': 'LinearSVC', 'C': 10, 'max_iter': 10000},
    
    # SVC com kernel RBF (usa amostra de 3000 - reduzido para evitar travamentos)
    {'model': 'SVC', 'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale', 'sample_size': 3000},
    {'model': 'SVC', 'kernel': 'rbf', 'C': 1, 'gamma': 'scale', 'sample_size': 3000},
    {'model': 'SVC', 'kernel': 'rbf', 'C': 10, 'gamma': 'scale', 'sample_size': 3000},
    {'model': 'SVC', 'kernel': 'rbf', 'C': 1, 'gamma': 'auto', 'sample_size': 3000},
    {'model': 'SVC', 'kernel': 'rbf', 'C': 10, 'gamma': 0.01, 'sample_size': 3000},
    
    # SVC com kernel linear (usa amostra de 10000)
    {'model': 'SVC', 'kernel': 'linear', 'C': 0.1, 'sample_size': 10000},
    {'model': 'SVC', 'kernel': 'linear', 'C': 1, 'sample_size': 10000},
    {'model': 'SVC', 'kernel': 'linear', 'C': 10, 'sample_size': 10000},
]

results_bin = []

for i, params in enumerate(param_configs, 1):
    print(f"\n[{i}/{len(param_configs)}] Testando: {params}")
    
    try:
        # Verificar se precisa usar amostra reduzida (para RBF)
        sample_size = params.get('sample_size', None)
        
        if sample_size and sample_size < len(X_train_bin_scaled):
            # Usar amostra para kernels lentos (RBF)
            print(f"Usando amostra de {sample_size} amostras (de {len(X_train_bin_scaled)})")
            sample_idx = np.random.choice(len(X_train_bin_scaled), size=sample_size, replace=False)
            X_train_sample = X_train_bin_scaled[sample_idx]
            y_train_sample = y_train_bin.iloc[sample_idx].reset_index(drop=True)
        else:
            # Usar dataset completo
            X_train_sample = X_train_bin_scaled
            y_train_sample = y_train_bin
        
        # Criar modelo baseado nos parâmetros
        if params['model'] == 'LinearSVC':
            model = LinearSVC(
                C=params['C'],
                max_iter=params['max_iter'],
                class_weight='balanced',
                random_state=42
            )
        else:  # SVC
            model_params = {
                'kernel': params['kernel'],
                'C': params['C'],
                'class_weight': 'balanced',
                'random_state': 42
            }
            if 'gamma' in params:
                model_params['gamma'] = params['gamma']
            
            model = SVC(**model_params)
        
        # Treinar
        log_progress(f"[BINÁRIO {i}/{len(param_configs)}] Iniciando treinamento...")
        start_train = time.perf_counter()
        model.fit(X_train_sample, y_train_sample)
        train_time = time.perf_counter() - start_train
        log_progress(f"[BINÁRIO {i}/{len(param_configs)}] Treino concluído em {train_time:.2f}s")
        
        # Testar (com o conjunto de teste completo)
        log_progress(f"[BINÁRIO {i}/{len(param_configs)}] Testando modelo...")
        start_test = time.perf_counter()
        y_pred = model.predict(X_test_bin_scaled)
        test_time = time.perf_counter() - start_test
        log_progress(f"[BINÁRIO {i}/{len(param_configs)}] Teste concluído em {test_time:.2f}s")
        
        # Métricas
        accuracy = accuracy_score(y_test_bin, y_pred)
        precision = precision_score(y_test_bin, y_pred)
        recall = recall_score(y_test_bin, y_pred)
        f1 = f1_score(y_test_bin, y_pred)
        
        # Criar string de identificação
        if params['model'] == 'LinearSVC':
            param_str = f"LinearSVC_C={params['C']}"
        else:
            gamma_str = f"_gamma={params.get('gamma', 'default')}"
            param_str = f"SVC_{params['kernel']}_C={params['C']}{gamma_str}"
        
        print(f"  Treino: {train_time:.2f}s | Teste: {test_time:.2f}s")
        print(f"  Acurácia: {accuracy:.4f} | F1: {f1:.4f}")
        log_progress(f"[BINÁRIO {i}/{len(param_configs)}] CONCLUÍDO - Acurácia: {accuracy:.4f}, F1: {f1:.4f}, Treino: {train_time:.2f}s")
        
        results_bin.append({
            'config': param_str,
            'model_type': params['model'],
            'kernel': params.get('kernel', 'linear'),
            'C': params['C'],
            'gamma': params.get('gamma', 'N/A'),
            'train_time': train_time,
            'test_time': test_time,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Salvar progressivamente a cada configuração
        df_results_bin = pd.DataFrame(results_bin)
        df_results_bin.to_csv('svm_param_search_binary.csv', index=False)
        
    except Exception as e:
        print(f"  ERRO: {e}")
        log_progress(f"[BINÁRIO {i}/{len(param_configs)}] ERRO: {e}")
        continue

# Salvar resultados
df_results_bin = pd.DataFrame(results_bin)
df_results_bin.to_csv('svm_param_search_binary.csv', index=False)
print(f"\nResultados binários salvos em: svm_param_search_binary.csv")
log_progress(f"Classificação binária concluída! {len(results_bin)} configurações testadas.")
log_progress("Arquivo salvo: svm_param_search_binary.csv")

# =============================================================================
# CLASSIFICAÇÃO MULTICLASSE - TESTANDO PARÂMETROS
# =============================================================================

print("\n" + "="*80)
print("CLASSIFICAÇÃO MULTICLASSE - TESTE DE PARÂMETROS")
print("="*80)
log_progress("\n" + "="*80)
log_progress("INICIANDO CLASSIFICAÇÃO MULTICLASSE")
log_progress("="*80)

malign_df = df[df['maligno'] == 1]
X_malign = pd.get_dummies(malign_df.drop(columns=['maligno', 'tipo_maligno']))
X_malign = X_malign.reindex(columns=selected_columns, fill_value=0)
y_malign = malign_df['tipo_maligno']

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_malign, y_malign, test_size=0.3, random_state=42
)

X_train_multi_scaled = scaler.transform(X_train_multi)
X_test_multi_scaled = scaler.transform(X_test_multi)

class_weights = {
    'Malware': 1.0,
    'Phishing': 1.0,
    'Spam': 2.0
}

results_multi = []

for i, params in enumerate(param_configs, 1):
    print(f"\n[{i}/{len(param_configs)}] Testando: {params}")
    log_progress(f"\n[MULTICLASSE {i}/{len(param_configs)}] INICIANDO: {params}")
    
    try:
        # Verificar se precisa usar amostra reduzida (para RBF)
        sample_size = params.get('sample_size', None)
        
        if sample_size and sample_size < len(X_train_multi_scaled):
            # Usar amostra para kernels lentos (RBF)
            print(f"Usando amostra de {sample_size} amostras (de {len(X_train_multi_scaled)})")
            sample_idx = np.random.choice(len(X_train_multi_scaled), size=sample_size, replace=False)
            X_train_sample = X_train_multi_scaled[sample_idx]
            y_train_sample = y_train_multi.iloc[sample_idx].reset_index(drop=True)
        else:
            # Usar dataset completo
            X_train_sample = X_train_multi_scaled
            y_train_sample = y_train_multi
        
        if params['model'] == 'LinearSVC':
            model = LinearSVC(
                C=params['C'],
                max_iter=params['max_iter'],
                class_weight=class_weights,
                random_state=42
            )
        else:
            model_params = {
                'kernel': params['kernel'],
                'C': params['C'],
                'class_weight': class_weights,
                'random_state': 42
            }
            if 'gamma' in params:
                model_params['gamma'] = params['gamma']
            
            model = SVC(**model_params)
        
        log_progress(f"[MULTICLASSE {i}/{len(param_configs)}] Iniciando treinamento...")
        start_train = time.perf_counter()
        model.fit(X_train_sample, y_train_sample)
        train_time = time.perf_counter() - start_train
        log_progress(f"[MULTICLASSE {i}/{len(param_configs)}] Treino concluído em {train_time:.2f}s")
        
        log_progress(f"[MULTICLASSE {i}/{len(param_configs)}] Testando modelo...")
        start_test = time.perf_counter()
        y_pred = model.predict(X_test_multi_scaled)
        test_time = time.perf_counter() - start_test
        log_progress(f"[MULTICLASSE {i}/{len(param_configs)}] Teste concluído em {test_time:.2f}s")
        
        accuracy = accuracy_score(y_test_multi, y_pred)
        precision = precision_score(y_test_multi, y_pred, average='weighted')
        recall = recall_score(y_test_multi, y_pred, average='weighted')
        f1 = f1_score(y_test_multi, y_pred, average='weighted')
        
        if params['model'] == 'LinearSVC':
            param_str = f"LinearSVC_C={params['C']}"
        else:
            gamma_str = f"_gamma={params.get('gamma', 'default')}"
            param_str = f"SVC_{params['kernel']}_C={params['C']}{gamma_str}"
        
        print(f"  Treino: {train_time:.2f}s | Teste: {test_time:.2f}s")
        print(f"  Acurácia: {accuracy:.4f} | F1: {f1:.4f}")
        log_progress(f"[MULTICLASSE {i}/{len(param_configs)}] CONCLUÍDO - Acurácia: {accuracy:.4f}, F1: {f1:.4f}, Treino: {train_time:.2f}s")
        
        results_multi.append({
            'config': param_str,
            'model_type': params['model'],
            'kernel': params.get('kernel', 'linear'),
            'C': params['C'],
            'gamma': params.get('gamma', 'N/A'),
            'train_time': train_time,
            'test_time': test_time,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Salvar progressivamente a cada configuração
        df_results_multi = pd.DataFrame(results_multi)
        df_results_multi.to_csv('svm_param_search_multiclass.csv', index=False)
        
    except Exception as e:
        print(f"  ERRO: {e}")
        log_progress(f"[MULTICLASSE {i}/{len(param_configs)}] ERRO: {e}")
        continue

df_results_multi = pd.DataFrame(results_multi)
df_results_multi.to_csv('svm_param_search_multiclass.csv', index=False)
print(f"\nResultados multiclasse salvos em: svm_param_search_multiclass.csv")
log_progress(f"Classificação multiclasse concluída! {len(results_multi)} configurações testadas.")
log_progress("Arquivo salvo: svm_param_search_multiclass.csv")

# =============================================================================
# PLOTAR RESULTADOS
# =============================================================================

print("\n" + "="*80)
print("GERANDO GRÁFICOS")
print("="*80)

# Gráfico 1: Comparação de métricas - Binário
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Desempenho SVM - Classificação Binária', fontsize=16, fontweight='bold')

metrics = ['accuracy', 'precision', 'recall', 'f1']
titles = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    data = df_results_bin.sort_values(metric, ascending=False).head(10)
    
    bars = ax.barh(range(len(data)), data[metric])
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data['config'], fontsize=8)
    ax.set_xlabel(title)
    ax.set_xlim([0, 1])
    ax.invert_yaxis()
    
    # Colorir barras
    for i, bar in enumerate(bars):
        if data.iloc[i]['model_type'] == 'LinearSVC':
            bar.set_color('steelblue')
        else:
            bar.set_color('coral')

plt.tight_layout()
plt.savefig('svm_metrics_binary.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo: svm_metrics_binary.png")
plt.close()

# Gráfico 2: Comparação de métricas - Multiclasse
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Desempenho SVM - Classificação Multiclasse', fontsize=16, fontweight='bold')

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    data = df_results_multi.sort_values(metric, ascending=False).head(10)
    
    bars = ax.barh(range(len(data)), data[metric])
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data['config'], fontsize=8)
    ax.set_xlabel(title)
    ax.set_xlim([0, 1])
    ax.invert_yaxis()
    
    for i, bar in enumerate(bars):
        if data.iloc[i]['model_type'] == 'LinearSVC':
            bar.set_color('steelblue')
        else:
            bar.set_color('coral')

plt.tight_layout()
plt.savefig('svm_metrics_multiclass.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo: svm_metrics_multiclass.png")
plt.close()

# Gráfico 3: Tempo vs F1 - Binário
plt.figure(figsize=(12, 6))
plt.scatter(df_results_bin['train_time'], df_results_bin['f1'], 
           s=100, alpha=0.6, c='steelblue', label='Treino')
plt.scatter(df_results_bin['test_time'], df_results_bin['f1'], 
           s=100, alpha=0.6, c='coral', label='Teste')

for idx, row in df_results_bin.iterrows():
    plt.annotate(row['config'], 
                (row['train_time'], row['f1']), 
                fontsize=7, alpha=0.7)

plt.xlabel('Tempo (segundos)')
plt.ylabel('F1-Score')
plt.title('Trade-off Tempo vs F1-Score - Binário')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('svm_time_vs_f1_binary.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo: svm_time_vs_f1_binary.png")
plt.close()

# Gráfico 4: Tempo vs F1 - Multiclasse
plt.figure(figsize=(12, 6))
plt.scatter(df_results_multi['train_time'], df_results_multi['f1'], 
           s=100, alpha=0.6, c='steelblue', label='Treino')
plt.scatter(df_results_multi['test_time'], df_results_multi['f1'], 
           s=100, alpha=0.6, c='coral', label='Teste')

for idx, row in df_results_multi.iterrows():
    plt.annotate(row['config'], 
                (row['train_time'], row['f1']), 
                fontsize=7, alpha=0.7)

plt.xlabel('Tempo (segundos)')
plt.ylabel('F1-Score')
plt.title('Trade-off Tempo vs F1-Score - Multiclasse')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('svm_time_vs_f1_multiclass.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo: svm_time_vs_f1_multiclass.png")
plt.close()

log_progress("\nGráficos gerados com sucesso!")

# Resumo dos melhores modelos
print("\n" + "="*80)
print("RESUMO - MELHORES MODELOS")
print("="*80)
log_progress("\n" + "="*80)
log_progress("RESUMO FINAL")
log_progress("="*80)

print("\nBINÁRIO - Top 3 por F1-Score:")
top3_bin = df_results_bin.nlargest(3, 'f1')[['config', 'f1', 'accuracy', 'train_time', 'test_time']]
print(top3_bin.to_string(index=False))

print("\nMULTICLASSE - Top 3 por F1-Score:")
top3_multi = df_results_multi.nlargest(3, 'f1')[['config', 'f1', 'accuracy', 'train_time', 'test_time']]
print(top3_multi.to_string(index=False))

print("\nProcesso completo concluído!")
log_progress("\n" + "="*80)
log_progress("PROCESSO COMPLETO CONCLUÍDO COM SUCESSO!")
log_progress("="*80)
