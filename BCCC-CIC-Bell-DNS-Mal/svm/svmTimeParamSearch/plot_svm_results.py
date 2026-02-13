import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Carregando resultados dos modelos SVM...")

# Carregar dados binários e multiclasse
df_binary = pd.read_csv('svm_param_search_binary.csv')
df_multiclass = pd.read_csv('svm_param_search_multiclass.csv')

print(f"\nModelos binários: {len(df_binary)}")
print(f"Modelos multiclasse: {len(df_multiclass)}")

print("\nCriando visualizações...")

# =============================================================================
# Gráfico de tempos de treinamento e teste (binário e multiclasse)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Tempos de treinamento
ax = axes[0]
indices = np.arange(len(df_binary))
width = 0.35

ax.bar(indices - width/2, df_binary['train_time'], width, label='Binário', alpha=0.8)
ax.bar(indices + width/2, df_multiclass['train_time'], width, label='Multiclasse', alpha=0.8)

ax.set_xlabel('Configuração do Modelo')
ax.set_ylabel('Tempo de Treinamento (segundos)')
ax.set_title('Tempo de Treinamento - SVM Binário vs Multiclasse')
ax.set_xticks(indices)
ax.set_xticklabels(df_binary['config'], rotation=90, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Tempos de teste
ax = axes[1]
ax.bar(indices - width/2, df_binary['test_time'], width, label='Binário', alpha=0.8)
ax.bar(indices + width/2, df_multiclass['test_time'], width, label='Multiclasse', alpha=0.8)

ax.set_xlabel('Configuração do Modelo')
ax.set_ylabel('Tempo de Teste (segundos)')
ax.set_title('Tempo de Teste - SVM Binário vs Multiclasse')
ax.set_xticks(indices)
ax.set_xticklabels(df_binary['config'], rotation=90, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('svm_tempos_treino_teste_bin_multi.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico salvo: svm_tempos_treino_teste_bin_multi.png")

# =============================================================================
# Gráfico de métricas para classificação binária
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_names = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx // 2, idx % 2]
    indices = np.arange(len(df_binary))
    
    ax.plot(indices, df_binary[metric], marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Configuração do Modelo')
    ax.set_ylabel(name)
    ax.set_title(f'{name} - Classificação Binária')
    ax.set_xticks(indices)
    ax.set_xticklabels(df_binary['config'], rotation=90, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Destacar o melhor valor
    best_idx = df_binary[metric].idxmax()
    ax.scatter(best_idx, df_binary[metric].iloc[best_idx], 
              color='red', s=100, zorder=5, label=f'Melhor: {df_binary[metric].iloc[best_idx]:.4f}')
    ax.legend()

plt.tight_layout()
plt.savefig('svm_metricas_binario.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico salvo: svm_metricas_binario.png")

# =============================================================================
# Gráfico de métricas para classificação multiclasse
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx // 2, idx % 2]
    indices = np.arange(len(df_multiclass))
    
    ax.plot(indices, df_multiclass[metric], marker='s', linewidth=2, markersize=6, color='orange')
    ax.set_xlabel('Configuração do Modelo')
    ax.set_ylabel(name)
    ax.set_title(f'{name} - Classificação Multiclasse')
    ax.set_xticks(indices)
    ax.set_xticklabels(df_multiclass['config'], rotation=90, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Destacar o melhor valor
    best_idx = df_multiclass[metric].idxmax()
    ax.scatter(best_idx, df_multiclass[metric].iloc[best_idx], 
              color='red', s=100, zorder=5, label=f'Melhor: {df_multiclass[metric].iloc[best_idx]:.4f}')
    ax.legend()

plt.tight_layout()
plt.savefig('svm_metricas_multiclasse.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico salvo: svm_metricas_multiclasse.png")

# =============================================================================
# Gráfico comparativo de todas as métricas (binário e multiclasse juntos)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx // 2, idx % 2]
    indices_bin = np.arange(len(df_binary))
    indices_multi = np.arange(len(df_multiclass))
    
    ax.plot(indices_bin, df_binary[metric], marker='o', linewidth=2, markersize=5, 
           label='Binário', alpha=0.7)
    ax.plot(indices_multi, df_multiclass[metric], marker='s', linewidth=2, markersize=5, 
           label='Multiclasse', alpha=0.7)
    
    ax.set_xlabel('Configuração do Modelo')
    ax.set_ylabel(name)
    ax.set_title(f'{name} - Binário vs Multiclasse')
    ax.set_xticks(indices_bin)
    ax.set_xticklabels(df_binary['config'], rotation=90, ha='right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('svm_metricas_comparativo.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico salvo: svm_metricas_comparativo.png")

# =============================================================================
# Análise de trade-off: Tempo vs F1-Score
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Binário
ax = axes[0]
scatter = ax.scatter(df_binary['train_time'], df_binary['f1'], 
                    c=df_binary['test_time'], s=100, alpha=0.6, 
                    cmap='viridis', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Tempo de Treinamento (segundos)')
ax.set_ylabel('F1-Score')
ax.set_title('Trade-off: Tempo vs F1-Score (Binário)')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Tempo de Teste (s)')

# Adicionar labels para pontos específicos
for i, config in enumerate(df_binary['config']):
    if df_binary['f1'].iloc[i] > 0.85 or df_binary['train_time'].iloc[i] < 100:
        ax.annotate(config, (df_binary['train_time'].iloc[i], df_binary['f1'].iloc[i]),
                   fontsize=7, alpha=0.7, xytext=(5, 5), textcoords='offset points')

# Multiclasse
ax = axes[1]
scatter = ax.scatter(df_multiclass['train_time'], df_multiclass['f1'], 
                    c=df_multiclass['test_time'], s=100, alpha=0.6, 
                    cmap='viridis', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Tempo de Treinamento (segundos)')
ax.set_ylabel('F1-Score')
ax.set_title('Trade-off: Tempo vs F1-Score (Multiclasse)')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Tempo de Teste (s)')

# Adicionar labels para pontos específicos
for i, config in enumerate(df_multiclass['config']):
    if df_multiclass['f1'].iloc[i] > 0.79 or df_multiclass['train_time'].iloc[i] < 50:
        ax.annotate(config, (df_multiclass['train_time'].iloc[i], df_multiclass['f1'].iloc[i]),
                   fontsize=7, alpha=0.7, xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('svm_tempo_vs_f1.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico salvo: svm_tempo_vs_f1.png")

# =============================================================================
# Análise separada por tipo de kernel
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Binário - Kernel Linear
linear_bin = df_binary[df_binary['kernel'] == 'linear']
rbf_bin = df_binary[df_binary['kernel'] == 'rbf']

ax = axes[0, 0]
if len(linear_bin) > 0:
    indices = np.arange(len(linear_bin))
    ax.plot(indices, linear_bin['accuracy'], marker='o', label='Acurácia')
    ax.plot(indices, linear_bin['precision'], marker='s', label='Precisão')
    ax.plot(indices, linear_bin['recall'], marker='^', label='Recall')
    ax.plot(indices, linear_bin['f1'], marker='D', label='F1-Score')
    ax.set_xlabel('Configuração')
    ax.set_ylabel('Métrica')
    ax.set_title('Métricas - SVM Linear Binário')
    ax.set_xticks(indices)
    ax.set_xticklabels(linear_bin['config'], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Binário - Kernel RBF
ax = axes[0, 1]
if len(rbf_bin) > 0:
    indices = np.arange(len(rbf_bin))
    ax.plot(indices, rbf_bin['accuracy'], marker='o', label='Acurácia')
    ax.plot(indices, rbf_bin['precision'], marker='s', label='Precisão')
    ax.plot(indices, rbf_bin['recall'], marker='^', label='Recall')
    ax.plot(indices, rbf_bin['f1'], marker='D', label='F1-Score')
    ax.set_xlabel('Configuração')
    ax.set_ylabel('Métrica')
    ax.set_title('Métricas - SVM RBF Binário')
    ax.set_xticks(indices)
    ax.set_xticklabels(rbf_bin['config'], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Multiclasse - Kernel Linear
linear_multi = df_multiclass[df_multiclass['kernel'] == 'linear']
rbf_multi = df_multiclass[df_multiclass['kernel'] == 'rbf']

ax = axes[1, 0]
if len(linear_multi) > 0:
    indices = np.arange(len(linear_multi))
    ax.plot(indices, linear_multi['accuracy'], marker='o', label='Acurácia')
    ax.plot(indices, linear_multi['precision'], marker='s', label='Precisão')
    ax.plot(indices, linear_multi['recall'], marker='^', label='Recall')
    ax.plot(indices, linear_multi['f1'], marker='D', label='F1-Score')
    ax.set_xlabel('Configuração')
    ax.set_ylabel('Métrica')
    ax.set_title('Métricas - SVM Linear Multiclasse')
    ax.set_xticks(indices)
    ax.set_xticklabels(linear_multi['config'], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Multiclasse - Kernel RBF
ax = axes[1, 1]
if len(rbf_multi) > 0:
    indices = np.arange(len(rbf_multi))
    ax.plot(indices, rbf_multi['accuracy'], marker='o', label='Acurácia')
    ax.plot(indices, rbf_multi['precision'], marker='s', label='Precisão')
    ax.plot(indices, rbf_multi['recall'], marker='^', label='Recall')
    ax.plot(indices, rbf_multi['f1'], marker='D', label='F1-Score')
    ax.set_xlabel('Configuração')
    ax.set_ylabel('Métrica')
    ax.set_title('Métricas - SVM RBF Multiclasse')
    ax.set_xticks(indices)
    ax.set_xticklabels(rbf_multi['config'], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svm_metricas_por_kernel.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico salvo: svm_metricas_por_kernel.png")

# =============================================================================
# Resumo estatístico
# =============================================================================

print("\n" + "="*80)
print("RESUMO ESTATÍSTICO - CLASSIFICAÇÃO BINÁRIA")
print("="*80)

print("\nMelhores configurações por métrica:")
for metric in metrics:
    best_idx = df_binary[metric].idxmax()
    print(f"\n{metric.upper()}:")
    print(f"  Configuração: {df_binary['config'].iloc[best_idx]}")
    print(f"  Valor: {df_binary[metric].iloc[best_idx]:.4f}")
    print(f"  Tempo de treinamento: {df_binary['train_time'].iloc[best_idx]:.2f}s")
    print(f"  Tempo de teste: {df_binary['test_time'].iloc[best_idx]:.2f}s")

print("\n" + "="*80)
print("RESUMO ESTATÍSTICO - CLASSIFICAÇÃO MULTICLASSE")
print("="*80)

print("\nMelhores configurações por métrica:")
for metric in metrics:
    best_idx = df_multiclass[metric].idxmax()
    print(f"\n{metric.upper()}:")
    print(f"  Configuração: {df_multiclass['config'].iloc[best_idx]}")
    print(f"  Valor: {df_multiclass[metric].iloc[best_idx]:.4f}")
    print(f"  Tempo de treinamento: {df_multiclass['train_time'].iloc[best_idx]:.2f}s")
    print(f"  Tempo de teste: {df_multiclass['test_time'].iloc[best_idx]:.2f}s")

# =============================================================================
# Análise de eficiência (F1-Score por segundo de treinamento)
# =============================================================================

print("\n" + "="*80)
print("ANÁLISE DE EFICIÊNCIA")
print("="*80)

df_binary['efficiency'] = df_binary['f1'] / (df_binary['train_time'] + 1)  # +1 para evitar divisão por zero
df_multiclass['efficiency'] = df_multiclass['f1'] / (df_multiclass['train_time'] + 1)

print("\nModelos mais eficientes (maior F1-Score por segundo de treinamento):")

print("\nBINÁRIO:")
top5_bin = df_binary.nlargest(5, 'efficiency')[['config', 'f1', 'train_time', 'efficiency']]
print(top5_bin.to_string(index=False))

print("\nMULTICLASSE:")
top5_multi = df_multiclass.nlargest(5, 'efficiency')[['config', 'f1', 'train_time', 'efficiency']]
print(top5_multi.to_string(index=False))

# =============================================================================
# Salvar resumo em CSV
# =============================================================================

summary_data = {
    'Type': ['Binary', 'Multiclass'],
    'Best_Accuracy': [df_binary['accuracy'].max(), df_multiclass['accuracy'].max()],
    'Best_F1': [df_binary['f1'].max(), df_multiclass['f1'].max()],
    'Avg_Train_Time': [df_binary['train_time'].mean(), df_multiclass['train_time'].mean()],
    'Avg_Test_Time': [df_binary['test_time'].mean(), df_multiclass['test_time'].mean()],
    'Min_Train_Time': [df_binary['train_time'].min(), df_multiclass['train_time'].min()],
    'Max_Train_Time': [df_binary['train_time'].max(), df_multiclass['train_time'].max()]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('svm_summary_statistics.csv', index=False)
print("\n\nResumo estatístico salvo em: svm_summary_statistics.csv")

print("\n" + "="*80)
print("Processo completo concluído!")
print("="*80)
print("\nArquivos gerados:")
print("  - svm_tempos_treino_teste_bin_multi.png")
print("  - svm_metricas_binario.png")
print("  - svm_metricas_multiclasse.png")
print("  - svm_metricas_comparativo.png")
print("  - svm_tempo_vs_f1.png")
print("  - svm_metricas_por_kernel.png")
print("  - svm_summary_statistics.csv")
