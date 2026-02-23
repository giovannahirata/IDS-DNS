import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import VarianceThreshold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Função de logging
def log_progress(message, log_file='knn_progress.log'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')
        f.flush()

log_progress("="*80)
log_progress("INÍCIO DA EXECUÇÃO - KNN")
log_progress("="*80)

print("Carregando e combinando datasets...")

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
cols_remover = ['Unnamed: 0', 'flow_id', 'timestamp', 'src_ip', 'dst_ip', 
                'src_port', 'dst_port', 'label']
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

print("\nTreinando modelo KNN binário (k=3)...")
log_progress(f"Conjunto de treino binário: {X_train_bin_scaled.shape}, Teste: {X_test_bin_scaled.shape}")

# treino de classificação binária
neigh = KNeighborsClassifier(n_neighbors=3)
start_time = time.time()
neigh.fit(X_train_bin_scaled, y_train_bin)
train_time = time.time() - start_time
log_progress(f"Modelo KNN binário treinado em {train_time:.2f} segundos")

# avaliar:
y_pred = neigh.predict(X_test_bin_scaled)
accuracy = accuracy_score(y_test_bin, y_pred)
precision = precision_score(y_test_bin, y_pred)
recall = recall_score(y_test_bin, y_pred)
f1 = f1_score(y_test_bin, y_pred)

print("\nMétricas da classificação binária:")
print(f"  Acurácia: {accuracy:.4f} | Precisão: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
print("\nRelatório Binário:")
print(classification_report(y_test_bin, y_pred))

# =============================================================================
# ANÁLISE SHAP - CLASSIFICAÇÃO BINÁRIA
# =============================================================================

print("\n" + "="*80)
print("ANÁLISE SHAP - CLASSIFICAÇÃO BINÁRIA (KNN)")
print("="*80)
log_progress("Iniciando análise SHAP para modelo KNN binário...")

try:
    n_samples_shap = min(100, len(X_test_bin_scaled))
    X_test_sample_bin = X_test_bin_scaled[:n_samples_shap]
    
    # usamos KernelExplainer
    print(f"\nCriando SHAP explainer com {n_samples_shap} amostras de teste...")
    n_background = min(100, len(X_train_bin_scaled))
    background_bin = shap.sample(X_train_bin_scaled, n_background)
    
    explainer_bin = shap.KernelExplainer(neigh.predict_proba, background_bin)
    shap_values_bin = explainer_bin.shap_values(X_test_sample_bin)
    
    log_progress(f"SHAP values calculados para modelo binário: shape={np.array(shap_values_bin).shape}")
    
    # Para classificação binária, converte array 3D para lista se necessário
    if isinstance(shap_values_bin, np.ndarray) and shap_values_bin.ndim == 3:
        # De (n_samples, n_features, 2) para lista de 2 arrays (n_samples, n_features)
        shap_values_bin = [shap_values_bin[:, :, i] for i in range(shap_values_bin.shape[2])]
        log_progress("Convertido array 3D para lista de arrays")
    
    # Pegar valores SHAP da classe positiva (maligno=1)
    if isinstance(shap_values_bin, list) and len(shap_values_bin) == 2:
        shap_values_bin_pos = shap_values_bin[1]  # classe 1 (maligno)
    else:
        shap_values_bin_pos = shap_values_bin
    
    # Summary Plot
    try:
        print("Gerando summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_bin_pos, X_test_sample_bin, 
                         feature_names=selected_columns.tolist(),
                         show=False, max_display=20)
        plt.title('SHAP Summary Plot - KNN Binário (Benigno vs Malicioso)', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig('shap_summary_knn_binary.png', dpi=300, bbox_inches='tight')
        plt.close()
        log_progress("Summary plot salvo: shap_summary_knn_binary.png")
    except Exception as e:
        log_progress(f"ERRO ao gerar summary plot binário: {str(e)}")
        plt.close('all')
    
    # Bar Plot
    try:
        print("Gerando bar plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_bin_pos, X_test_sample_bin,
                         feature_names=selected_columns.tolist(),
                         plot_type="bar", show=False, max_display=20)
        plt.title('SHAP Feature Importance - KNN Binário', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig('shap_importance_knn_binary.png', dpi=300, bbox_inches='tight')
        plt.close()
        log_progress("Bar plot salvo: shap_importance_knn_binary.png")
    except Exception as e:
        log_progress(f"ERRO ao gerar bar plot binário: {str(e)}")
        plt.close('all')
    
    # Dependence plots para top 3 features
    try:
        print("Gerando dependence plots...")
        shap_abs_mean = np.abs(shap_values_bin_pos).mean(axis=0)
        top_features_idx = np.argsort(shap_abs_mean)[-3:][::-1]
        
        for idx in top_features_idx:
            try:
                idx_scalar = int(idx) if hasattr(idx, '__iter__') else idx
                feature_name = selected_columns[idx_scalar]
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(idx_scalar, shap_values_bin_pos, X_test_sample_bin,
                                   feature_names=selected_columns.tolist(),
                                   show=False)
                plt.title(f'SHAP Dependence Plot - {feature_name}', fontsize=12)
                plt.tight_layout()
                safe_name = str(feature_name).replace('/', '_').replace(' ', '_')
                plt.savefig(f'shap_dependence_knn_binary_{safe_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                log_progress(f"ERRO ao gerar dependence plot para feature {idx}: {str(e)}")
                plt.close('all')
        log_progress(f"Dependence plots salvos para top 3 features")
    except Exception as e:
        log_progress(f"ERRO ao gerar dependence plots: {str(e)}")
        plt.close('all')
    
    # Force plot
    try:
        print("Gerando force plot...")
        shap.force_plot(explainer_bin.expected_value[1] if isinstance(explainer_bin.expected_value, list) else explainer_bin.expected_value, 
                       shap_values_bin_pos[0,:], 
                       X_test_sample_bin[0,:],
                       feature_names=selected_columns.tolist(),
                       matplotlib=True, show=False)
        plt.savefig('shap_force_knn_binary_sample0.png', dpi=300, bbox_inches='tight')
        plt.close()
        log_progress("Force plot salvo: shap_force_knn_binary_sample0.png")
    except Exception as e:
        log_progress(f"ERRO ao gerar force plot binário: {str(e)}")
        plt.close('all')
    
    # Salvar estatísticas
    try:
        # Garantir que shap_values_bin_pos é 2D
        if shap_values_bin_pos.ndim > 2:
            shap_values_bin_pos = shap_values_bin_pos.reshape(shap_values_bin_pos.shape[0], -1)
            log_progress(f"AVISO: Reshape de shap_values_bin_pos para {shap_values_bin_pos.shape}")
        
        # Validar compatibilidade de tamanhos
        if shap_values_bin_pos.shape[1] != len(selected_columns):
            log_progress(f"AVISO: Shape incompativel: {shap_values_bin_pos.shape[1]} features vs {len(selected_columns)} colunas")
            n_cols = min(shap_values_bin_pos.shape[1], len(selected_columns))
            shap_values_bin_pos = shap_values_bin_pos[:, :n_cols]
            selected_columns_subset = selected_columns[:n_cols]
        else:
            selected_columns_subset = selected_columns
        
        shap_stats_bin = pd.DataFrame({
            'feature': selected_columns_subset,
            'mean_abs_shap': np.abs(shap_values_bin_pos).mean(axis=0),
            'mean_shap': shap_values_bin_pos.mean(axis=0),
            'std_shap': shap_values_bin_pos.std(axis=0)
        })
        shap_stats_bin = shap_stats_bin.sort_values('mean_abs_shap', ascending=False)
        shap_stats_bin.to_csv('shap_statistics_knn_binary.csv', index=False)
        log_progress("Estatísticas SHAP salvas: shap_statistics_knn_binary.csv")
        
        print("\nTop 10 features mais importantes (SHAP):")
        print(shap_stats_bin.head(10).to_string())
    except Exception as e:
        log_progress(f"ERRO ao salvar estatísticas binárias: {str(e)}")
    
    log_progress("Análise SHAP para modelo KNN binário concluída com sucesso!")
    
except Exception as e:
    import traceback
    log_progress(f"ERRO na análise SHAP binária: {str(e)}")
    log_progress(f"Traceback completo: {traceback.format_exc()}")
    print(f"\nErro ao executar análise SHAP: {str(e)}")

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
log_progress(f"Conjunto de treino multiclasse: {X_train_multi_scaled.shape}, Teste: {X_test_multi_scaled.shape}")
log_progress(f"Distribuição de classes: {y_train_multi.value_counts().to_dict()}")
log_progress("Treinando modelo KNN multiclasse (k=11)...")

neigh_multi = KNeighborsClassifier(n_neighbors=11, weights='distance')
start_time_multi = time.time()
neigh_multi.fit(X_train_multi_scaled, y_train_multi)
train_time_multi = time.time() - start_time_multi
log_progress(f"Modelo KNN multiclasse treinado em {train_time_multi:.2f} segundos")

# avalia:
y_pred_multi = neigh_multi.predict(X_test_multi_scaled)
accuracy_multi = accuracy_score(y_test_multi, y_pred_multi)
precision_multi = precision_score(y_test_multi, y_pred_multi, average='weighted')
recall_multi = recall_score(y_test_multi, y_pred_multi, average='weighted')
f1_multi = f1_score(y_test_multi, y_pred_multi, average='weighted')

print("\nMétricas da classificação multiclasse:")
print(f"  Acurácia: {accuracy_multi:.4f} | Precisão: {precision_multi:.4f} | Recall: {recall_multi:.4f} | F1: {f1_multi:.4f}")
print("\nRelatório Multiclasse:")
print(classification_report(y_test_multi, y_pred_multi))

# Matriz de confusão
print("\nGerando matriz de confusão...")
cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=neigh_multi.classes_,
            yticklabels=neigh_multi.classes_)
plt.title('Matriz de Confusão - Classificação Multiclasse')
plt.xticks(rotation=45)
plt.savefig('confusion_matrix_knn_multiclass.png', dpi=300, bbox_inches='tight')
plt.close()
log_progress("Matriz de confusão salva: confusion_matrix_knn_multiclass.png")

# =============================================================================
# ANÁLISE SHAP - CLASSIFICAÇÃO MULTICLASSE
# =============================================================================

print("\n" + "="*80)
print("ANÁLISE SHAP - CLASSIFICAÇÃO MULTICLASSE (KNN)")
print("="*80)
log_progress("Iniciando análise SHAP para modelo KNN multiclasse...")

try:
    n_samples_shap_multi = min(300, len(X_test_multi_scaled))
    X_test_sample_multi = X_test_multi_scaled[:n_samples_shap_multi]
    if hasattr(y_test_multi, 'iloc'):
        y_test_sample_multi = y_test_multi.iloc[:n_samples_shap_multi].reset_index(drop=True)
    else:
        y_test_sample_multi = y_test_multi[:n_samples_shap_multi]
    
    print(f"\nCriando SHAP explainer para modelo multiclasse com {n_samples_shap_multi} amostras...")
    
    # Background menor para KNN multiclasse
    n_background_multi = min(50, len(X_train_multi_scaled))
    background_multi = shap.kmeans(X_train_multi_scaled, n_background_multi)  
    
    explainer_multi = shap.KernelExplainer(neigh_multi.predict_proba, background_multi)
    shap_values_multi = explainer_multi.shap_values(X_test_sample_multi, nsamples=100)  
    
    log_progress(f"SHAP values calculados para modelo multiclasse: shape={np.array(shap_values_multi).shape}")
    
    # Para multiclasse, shap_values pode vir como lista ou array 3D
    # Se vier como array 3D (n_samples, n_features, n_classes), converter para lista de arrays
    if isinstance(shap_values_multi, np.ndarray) and shap_values_multi.ndim == 3:
        # Transpor de (n_samples, n_features, n_classes) para lista de (n_samples, n_features)
        shap_values_multi = [shap_values_multi[:, :, i] for i in range(shap_values_multi.shape[2])]
        log_progress(f"Convertido para lista de {len(shap_values_multi)} arrays")
    
    classes = neigh_multi.classes_
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
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_multi[i], X_test_sample_multi,
                             feature_names=selected_columns.tolist(),
                             show=False, max_display=20)
            plt.title(f'SHAP Summary Plot - KNN {class_name}', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(f'shap_summary_knn_multi_{class_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            log_progress(f"Summary plot salvo: shap_summary_knn_multi_{class_name}.png")
        except Exception as e:
            log_progress(f"ERRO ao gerar summary plot para {class_name}: {str(e)}")
            plt.close('all')
        
        try:
            # Bar plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_multi[i], X_test_sample_multi,
                             feature_names=selected_columns.tolist(),
                             plot_type="bar", show=False, max_display=20)
            plt.title(f'SHAP Feature Importance - KNN {class_name}', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(f'shap_importance_knn_multi_{class_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            log_progress(f"Bar plot salvo: shap_importance_knn_multi_{class_name}.png")
        except Exception as e:
            log_progress(f"ERRO ao gerar bar plot para {class_name}: {str(e)}")
            plt.close('all')
    
    # Comparação entre classes
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
        plt.savefig('shap_comparison_knn_multiclass.png', dpi=300, bbox_inches='tight')
        plt.close()
        log_progress("Gráfico de comparação salvo: shap_comparison_knn_multiclass.png")
    except Exception as e:
        log_progress(f"ERRO ao gerar gráfico de comparação: {str(e)}")
        plt.close('all')
    
    # Salvar estatísticas para cada classe
    for i, class_name in enumerate(classes):
        if i >= len(shap_values_multi):
            log_progress(f"AVISO: Pulando estatísticas para {class_name} - SHAP values não disponíveis")
            continue
        
        try:
            # Verificar shape antes de calcular estatísticas
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
            shap_stats_multi.to_csv(f'shap_statistics_knn_multi_{class_name}.csv', index=False)
            
            print(f"\nTop 10 features para {class_name}:")
            print(shap_stats_multi.head(10).to_string())
            log_progress(f"Estatísticas salvas para {class_name}")
        except Exception as e:
            log_progress(f"ERRO ao salvar estatísticas para {class_name}: {str(e)}")
    
    log_progress("Estatísticas SHAP para modelo multiclasse salvas")
    
    # Force plots
    print("\nGerando force plots para amostras de cada classe...")
    y_test_array = np.array(y_test_sample_multi) if hasattr(y_test_sample_multi, 'values') else y_test_sample_multi
    
    for class_name in classes:
        try:
            class_idx = np.where(y_test_array == class_name)[0]
            if len(class_idx) > 0:
                sample_idx = class_idx[0]
                class_num = list(classes).index(class_name)
                
                shap.force_plot(explainer_multi.expected_value[class_num],
                              shap_values_multi[class_num][sample_idx,:],
                              X_test_sample_multi[sample_idx,:],
                              feature_names=selected_columns.tolist(),
                              matplotlib=True, show=False)
                plt.savefig(f'shap_force_knn_multi_{class_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                log_progress(f"Force plot salvo: shap_force_knn_multi_{class_name}.png")
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
log_progress("="*80)
log_progress("PROCESSO COMPLETO CONCLUÍDO COM SUCESSO!")
log_progress("="*80)
