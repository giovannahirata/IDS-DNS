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
import shap
import warnings
warnings.filterwarnings('ignore')

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

dir = "~/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/"

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
                'src_port', 'dst_port', 'label']
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
# CLASSIFICAÇÃO BINÁRIA
# =============================================================================

print("\n" + "="*80)
print("CLASSIFICAÇÃO BINÁRIA")
print("="*80)

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X, y_bin, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_bin_scaled = scaler.fit_transform(X_train_bin)
X_test_bin_scaled = scaler.transform(X_test_bin)

# amostra de 10000 para ser mais rápido por conta do kernel usado
sample_idx = np.random.choice(len(X_train_bin_scaled), size=10000, replace=False)
X_train_sample = X_train_bin_scaled[sample_idx]
y_train_sample = y_train_bin.iloc[sample_idx].reset_index(drop=True)

X_train_bin_scaled = X_train_sample 
y_train_bin = y_train_sample

log_progress(f"Conjunto de treino binário: {X_train_bin_scaled.shape}, Teste: {X_test_bin_scaled.shape}")

model_bin = SVC(
        kernel='linear',
        C=1,
        class_weight='balanced',
        random_state=42)

log_progress("Treinando modelo SVM binário...")

# treina
start_time = time.time()
model_bin.fit(X_train_bin_scaled, y_train_bin)
train_time = time.time() - start_time
log_progress(f"Modelo binário treinado em {train_time:.2f} segundos")

# testa
y_pred_bin = model_bin.predict(X_test_bin_scaled)

accuracy = accuracy_score(y_test_bin, y_pred_bin)
precision = precision_score(y_test_bin, y_pred_bin)
recall = recall_score(y_test_bin, y_pred_bin)
f1 = f1_score(y_test_bin, y_pred_bin)

print("\nMétricas da classificação binária:")
print(f"  Acurácia: {accuracy:.4f} | Precisão: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

# =============================================================================
# ANÁLISE SHAP - CLASSIFICAÇÃO BINÁRIA
# =============================================================================

print("\n" + "="*80)
print("ANÁLISE SHAP - CLASSIFICAÇÃO BINÁRIA")
print("="*80)
log_progress("Iniciando análise SHAP para modelo binário...")

try:
    # amostra para acelerar o cálculo
    n_samples_shap = min(100, len(X_test_bin_scaled))
    X_test_sample_bin = X_test_bin_scaled[:n_samples_shap]
    
    # Cria o explainer SHAP para SVM
    # LinearExplainer para SVM linear
    print(f"\nCriando SHAP explainer com {n_samples_shap} amostras de teste...")
    
    explainer_bin = shap.LinearExplainer(model_bin, X_train_bin_scaled, feature_perturbation="interventional")
    shap_values_bin = explainer_bin.shap_values(X_test_sample_bin)
    
    log_progress(f"SHAP values calculados para modelo binário: shape={np.array(shap_values_bin).shape}")
    
    # Summary Plot - mostra importância geral das features
    print("Gerando summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_bin, X_test_sample_bin, 
                     feature_names=selected_columns.tolist(),
                     show=False, max_display=20)
    plt.title('SHAP Summary Plot - Classificação Binária (Benigno vs Malicioso)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('shap_summary_binary.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_progress("Summary plot salvo: shap_summary_binary.png")
    
    # Bar Plot - importância média absoluta
    print("Gerando bar plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_bin, X_test_sample_bin,
                     feature_names=selected_columns.tolist(),
                     plot_type="bar", show=False, max_display=20)
    plt.title('SHAP Feature Importance - Classificação Binária', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('shap_importance_binary.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_progress("Bar plot salvo: shap_importance_binary.png")
    
    # Dependence plots para top 3 features mais importantes
    print("Gerando dependence plots...")
    shap_abs_mean = np.abs(shap_values_bin).mean(axis=0)
    top_features_idx = np.argsort(shap_abs_mean)[-3:][::-1]
    
    for idx in top_features_idx:
        feature_name = selected_columns[idx]
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(idx, shap_values_bin, X_test_sample_bin,
                           feature_names=selected_columns.tolist(),
                           show=False)
        plt.title(f'SHAP Dependence Plot - {feature_name}', fontsize=12)
        plt.tight_layout()
        safe_name = feature_name.replace('/', '_').replace(' ', '_')
        plt.savefig(f'shap_dependence_binary_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    log_progress(f"Dependence plots salvos para top 3 features")
    
    # Force plot para algumas predições individuais
    print("Gerando force plots...")
    shap.force_plot(explainer_bin.expected_value, 
                   shap_values_bin[0,:], 
                   X_test_sample_bin[0,:],
                   feature_names=selected_columns.tolist(),
                   matplotlib=True, show=False)
    plt.savefig('shap_force_binary_sample0.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_progress("Force plot salvo: shap_force_binary_sample0.png")
    
    # Salvar estatísticas das SHAP values
    shap_stats_bin = pd.DataFrame({
        'feature': selected_columns,
        'mean_abs_shap': np.abs(shap_values_bin).mean(axis=0),
        'mean_shap': shap_values_bin.mean(axis=0),
        'std_shap': shap_values_bin.std(axis=0)
    })
    shap_stats_bin = shap_stats_bin.sort_values('mean_abs_shap', ascending=False)
    shap_stats_bin.to_csv('shap_statistics_binary.csv', index=False)
    log_progress("Estatísticas SHAP salvas: shap_statistics_binary.csv")
    
    print("\nTop 10 features mais importantes (SHAP):")
    print(shap_stats_bin.head(10).to_string())
    
    log_progress("Análise SHAP para modelo binário concluída com sucesso!")
    
except Exception as e:
    log_progress(f"ERRO na análise SHAP binária: {str(e)}")
    print(f"\nErro ao executar análise SHAP: {str(e)}")

# ----------------------------------------------------------------------------------

# =============================================================================
# CLASSIFICAÇÃO MULTICLASSE 
# =============================================================================

print("\n" + "="*80)
print("CLASSIFICAÇÃO MULTICLASSE - TESTE DE PARÂMETROS")
print("="*80)
log_progress("\n" + "="*80)
log_progress("INICIANDO CLASSIFICAÇÃO MULTICLASSE")
log_progress("="*80)

# classificação entre as amostras malignas (malware, phishing e spam)
malign_df = df[df['maligno'] == 1]
X_malign = pd.get_dummies(malign_df.drop(columns=['maligno', 'tipo_maligno']))
X_malign = X_malign.reindex(columns=selected_columns, fill_value=0)
y_malign = malign_df['tipo_maligno']

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_malign, y_malign, test_size=0.3, random_state=42
)

X_train_multi_scaled = scaler.transform(X_train_multi)
X_test_multi_scaled = scaler.transform(X_test_multi)

# # amostra de 3000 para ser mais rápido por conta do kernel usado
# sample_idx = np.random.choice(len(X_train_multi_scaled), size=3000, replace=False)
# X_train_sample = X_train_multi_scaled[sample_idx]
# y_train_sample = y_train_multi.iloc[sample_idx].reset_index(drop=True)

# X_train_multi_scaled = X_train_sample
# y_train_multi = y_train_sample

log_progress(f"Conjunto de treino multiclasse: {X_train_multi_scaled.shape}, Teste: {X_test_multi_scaled.shape}")
log_progress(f"Distribuição de classes: {y_train_multi.value_counts().to_dict()}")

class_weights = {
    'Malware': 1.0,
    'Phishing': 1.0,
    'Spam': 2.0
}

model_multi = SVC(
        kernel='rbf',
        C=10,
        gamma=0.01,
        class_weight=class_weights,
        probability=True,  
        random_state=42)

log_progress("Treinando modelo SVM multiclasse (RBF)...")

# treinar:
start_time_multi = time.time()
model_multi.fit(X_train_multi_scaled, y_train_multi)
train_time_multi = time.time() - start_time_multi
log_progress(f"Modelo multiclasse treinado em {train_time_multi:.2f} segundos")

# testar:
y_pred_multi = model_multi.predict(X_test_multi_scaled)

accuracy = accuracy_score(y_test_multi, y_pred_multi)
precision = precision_score(y_test_multi, y_pred_multi, average='weighted')
recall = recall_score(y_test_multi, y_pred_multi, average='weighted')
f1 = f1_score(y_test_multi, y_pred_multi, average='weighted')

print("\nMétricas da classificação multiclasse:")
print(f"  Acurácia: {accuracy:.4f} | Precisão: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")


# =============================================================================
# ANÁLISE SHAP - CLASSIFICAÇÃO MULTICLASSE
# =============================================================================

print("\n" + "="*80)
print("ANÁLISE SHAP - CLASSIFICAÇÃO MULTICLASSE")
print("="*80)
log_progress("Iniciando análise SHAP para modelo multiclasse...")

try:
    # amostra para acelerar o cálculo
    n_samples_shap_multi = min(100, len(X_test_multi_scaled))
    X_test_sample_multi = X_test_multi_scaled[:n_samples_shap_multi]
    if hasattr(y_test_multi, 'iloc'):
        y_test_sample_multi = y_test_multi.iloc[:n_samples_shap_multi].reset_index(drop=True)
    else:
        y_test_sample_multi = y_test_multi[:n_samples_shap_multi]
    
    print(f"\nCriando SHAP explainer para modelo multiclasse com {n_samples_shap_multi} amostras...")
    
    # Para SVM RBF multiclasse, usamos KernelExplainer
    # Nota: KernelExplainer pode ser lento, então usamos uma amostra de background menor
    n_background = min(100, len(X_train_multi_scaled))  
    background_multi = shap.kmeans(X_train_multi_scaled, n_background)  
    
    # Verificar se o modelo tem predict_proba
    if not hasattr(model_multi, 'predict_proba'):
        raise AttributeError("Modelo SVM precisa ter probability=True")
    
    explainer_multi = shap.KernelExplainer(model_multi.predict_proba, background_multi)
    shap_values_multi = explainer_multi.shap_values(X_test_sample_multi, nsamples=100)  
    
    log_progress(f"SHAP values calculados para modelo multiclasse: shape={np.array(shap_values_multi).shape}")
    
    # Para multiclasse, shap_values pode vir como lista ou array 3D
    # Se vier como array 3D (n_samples, n_features, n_classes), converter para lista de arrays
    if isinstance(shap_values_multi, np.ndarray) and shap_values_multi.ndim == 3:
        # Transpor de (n_samples, n_features, n_classes) para lista de (n_samples, n_features)
        shap_values_multi = [shap_values_multi[:, :, i] for i in range(shap_values_multi.shape[2])]
        log_progress(f"Convertido para lista de {len(shap_values_multi)} arrays")
    
    classes = model_multi.classes_
    print(f"Classes: {classes}")
    
    # Verificar compatibilidade
    if not isinstance(shap_values_multi, list):
        log_progress(f"AVISO: shap_values não é lista. Tipo: {type(shap_values_multi)}")
        shap_values_multi = [shap_values_multi]
    
    if len(shap_values_multi) != len(classes):
        log_progress(f"AVISO: Num SHAP arrays ({len(shap_values_multi)}) != num classes ({len(classes)})")
    
    # Summary plots para cada classe
    for i, class_name in enumerate(classes):
        print(f"\nGerando summary plot para classe: {class_name}")
        
        try:
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_multi[i], X_test_sample_multi,
                             feature_names=selected_columns.tolist(),
                             show=False, max_display=20)
            plt.title(f'SHAP Summary Plot - {class_name}', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(f'shap_summary_multi_{class_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            log_progress(f"Summary plot salvo: shap_summary_multi_{class_name}.png")
        except Exception as e:
            log_progress(f"ERRO ao gerar summary plot para {class_name}: {str(e)}")
            plt.close('all')
        
        try:
            # Bar plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_multi[i], X_test_sample_multi,
                             feature_names=selected_columns.tolist(),
                             plot_type="bar", show=False, max_display=20)
            plt.title(f'SHAP Feature Importance - {class_name}', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(f'shap_importance_multi_{class_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            log_progress(f"Bar plot salvo: shap_importance_multi_{class_name}.png")
        except Exception as e:
            log_progress(f"ERRO ao gerar bar plot para {class_name}: {str(e)}")
            plt.close('all')
    
    # Comparação de importância entre classes
    try:
        print("\nGerando comparação de importância entre classes...")
        fig, axes = plt.subplots(1, len(classes), figsize=(6*len(classes), 6))
        if len(classes) == 1:
            axes = [axes]
        elif not isinstance(axes, np.ndarray):
            axes = [axes]
        
        for i, (class_name, ax) in enumerate(zip(classes, axes)):
            shap_mean = np.abs(shap_values_multi[i]).mean(axis=0)
            n_top = min(10, len(shap_mean))
            top_n_idx = np.argsort(shap_mean)[-n_top:][::-1]
            
            ax.barh(range(n_top), shap_mean[top_n_idx])
            ax.set_yticks(range(n_top))
            ax.set_yticklabels([selected_columns[idx] for idx in top_n_idx], fontsize=8)
            ax.set_xlabel('Mean |SHAP value|', fontsize=10)
            ax.set_title(f'{class_name}', fontsize=11, fontweight='bold')
            ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('shap_comparison_multiclass.png', dpi=300, bbox_inches='tight')
        plt.close()
        log_progress("Gráfico de comparação salvo: shap_comparison_multiclass.png")
    except Exception as e:
        log_progress(f"ERRO ao gerar gráfico de comparação: {str(e)}")
        plt.close('all')
    
    # Salvar estatísticas das SHAP values para cada classe
    for i, class_name in enumerate(classes):
        if i >= len(shap_values_multi):
            log_progress(f"AVISO: Pulando estatísticas para {class_name} - SHAP values não disponíveis")
            continue
        
        try:
            # Verifica shape antes de calcular estatísticas
            shap_array = shap_values_multi[i]
            if shap_array.shape[1] != len(selected_columns):
                log_progress(f"AVISO: Shape incompativel para {class_name}: {shap_array.shape} vs {len(selected_columns)} features")
                continue
            
            shap_stats_multi = pd.DataFrame({
                'feature': selected_columns,
                'mean_abs_shap': np.abs(shap_array).mean(axis=0),
                'mean_shap': shap_array.mean(axis=0),
                'std_shap': shap_array.std(axis=0)
            })
            shap_stats_multi = shap_stats_multi.sort_values('mean_abs_shap', ascending=False)
            shap_stats_multi.to_csv(f'shap_statistics_multi_{class_name}.csv', index=False)
            
            print(f"\nTop 10 features para {class_name}:")
            print(shap_stats_multi.head(10).to_string())
            log_progress(f"Estatísticas salvas para {class_name}")
        except Exception as e:
            log_progress(f"ERRO ao salvar estatísticas para {class_name}: {str(e)}")
    
    # Force plot para uma predição de exemplo de cada classe
    print("\nGerando force plots para amostras de cada classe...")
    y_test_array = np.array(y_test_sample_multi) if hasattr(y_test_sample_multi, 'values') else y_test_sample_multi
    
    for class_name in classes:
        try:
            # Encontrar primeira amostra de cada classe
            class_idx = np.where(y_test_array == class_name)[0]
            if len(class_idx) > 0:
                sample_idx = class_idx[0]
                class_num = list(classes).index(class_name)
                
                shap.force_plot(explainer_multi.expected_value[class_num],
                              shap_values_multi[class_num][sample_idx,:],
                              X_test_sample_multi[sample_idx,:],
                              feature_names=selected_columns.tolist(),
                              matplotlib=True, show=False)
                plt.savefig(f'shap_force_multi_{class_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                log_progress(f"Force plot salvo: shap_force_multi_{class_name}.png")
            else:
                log_progress(f"AVISO: Nenhuma amostra da classe {class_name} encontrada no conjunto de teste")
        except Exception as e:
            log_progress(f"ERRO ao gerar force plot para {class_name}: {str(e)}")
            plt.close('all')
    
    log_progress("Análise SHAP para modelo multiclasse concluída com sucesso!")
    
except Exception as e:
    import traceback
    log_progress(f"ERRO na análise SHAP multiclasse: {str(e)}")
    log_progress(f"Traceback completo: {traceback.format_exc()}")
    print(f"\nErro ao executar análise SHAP multiclasse: {str(e)}")

print("\nProcesso completo concluído!")
log_progress("\n" + "="*80)
log_progress("PROCESSO COMPLETO CONCLUÍDO COM SUCESSO!")
log_progress("="*80)
