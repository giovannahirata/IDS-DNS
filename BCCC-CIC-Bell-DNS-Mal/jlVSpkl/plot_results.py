import json
import numpy as np
import matplotlib.pyplot as plt

# Carregar dados do JSON
with open('resultados.json', 'r') as f:
    data = json.load(f)

# Preparar dados
machines = list(data["load_times"].keys())

# Converter para milissegundos e calcular estatísticas
save_pickle_mean = np.mean(data["save_times"]["pickle"]) * 1000
save_pickle_std = np.std(data["save_times"]["pickle"]) * 1000
save_joblib_mean = np.mean(data["save_times"]["joblib"]) * 1000
save_joblib_std = np.std(data["save_times"]["joblib"]) * 1000

# Preparar dados de carregamento por máquina
load_pickle_means = []
load_pickle_stds = []
load_joblib_means = []
load_joblib_stds = []

for machine in machines:
    load_pickle_means.append(np.mean(data["load_times"][machine]["pickle"]) * 1000)
    load_pickle_stds.append(np.std(data["load_times"][machine]["pickle"]) * 1000)
    load_joblib_means.append(np.mean(data["load_times"][machine]["joblib"]) * 1000)
    load_joblib_stds.append(np.std(data["load_times"][machine]["joblib"]) * 1000)

# ===== GRÁFICO 1: TEMPOS DE SALVAR E CARREGAR =====
fig, ax = plt.subplots(figsize=(14, 7))

# Labels para o eixo X
labels = ['Salvar\n(Lindor)'] + [f'Carregar\n({m.capitalize()})' for m in machines]
x = np.arange(len(labels))
width = 0.35

# Combinar dados de salvar e carregar
pickle_means = [save_pickle_mean] + load_pickle_means
pickle_stds = [save_pickle_std] + load_pickle_stds
joblib_means = [save_joblib_mean] + load_joblib_means
joblib_stds = [save_joblib_std] + load_joblib_stds

# Criar barras
bars_pickle = ax.bar(x - width/2, pickle_means, width, yerr=pickle_stds, 
                     label='Pickle', capsize=5, color='#2E86AB', alpha=0.9)
bars_joblib = ax.bar(x + width/2, joblib_means, width, yerr=joblib_stds,
                     label='Joblib', capsize=5, color='#A23B72', alpha=0.9)

# Configurar gráfico
ax.set_ylabel('Tempo (ms)', fontsize=12, fontweight='bold')
ax.set_title('Benchmark: Pickle vs Joblib - Serialização de Modelos LinearSVC\n(100 iterações por teste)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# Adicionar valores nas barras
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

autolabel(bars_pickle)
autolabel(bars_joblib)

plt.tight_layout()
plt.savefig('grafico_tempos_completo.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico de tempos salvo: grafico_tempos_completo.png")

# ===== GRÁFICO 2: TAMANHOS DOS ARQUIVOS =====
fig2, ax2 = plt.subplots(figsize=(8, 6))

sizes = [data["sizes"]["pickle"], data["sizes"]["joblib"]]
colors = ['#2E86AB', '#A23B72']
bars = ax2.bar(['Pickle', 'Joblib'], sizes, color=colors, alpha=0.9, edgecolor='black', linewidth=1.5)

ax2.set_ylabel('Tamanho (MB)', fontsize=12, fontweight='bold')
ax2.set_title('Comparação de Tamanho dos Arquivos Serializados', 
              fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='y', linestyle='--', alpha=0.3)
ax2.set_axisbelow(True)

# Adicionar valores
for i, (bar, size) in enumerate(zip(bars, sizes)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height,
            f'{size:.4f} MB',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('grafico_tamanhos.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico de tamanhos salvo: grafico_tamanhos.png")

# ===== GRÁFICO 3: COMPARAÇÃO APENAS DE CARREGAMENTO =====
fig3, ax3 = plt.subplots(figsize=(12, 7))

x3 = np.arange(len(machines))
bars3_pickle = ax3.bar(x3 - width/2, load_pickle_means, width, yerr=load_pickle_stds,
                       label='Pickle', capsize=5, color='#2E86AB', alpha=0.9)
bars3_joblib = ax3.bar(x3 + width/2, load_joblib_means, width, yerr=load_joblib_stds,
                       label='Joblib', capsize=5, color='#A23B72', alpha=0.9)

ax3.set_ylabel('Tempo (ms)', fontsize=12, fontweight='bold')
ax3.set_title('Comparação de Tempo de Carregamento por Máquina\n(100 iterações por teste)',
              fontsize=14, fontweight='bold', pad=20)
ax3.set_xticks(x3)
ax3.set_xticklabels([m.capitalize() for m in machines], fontsize=11)
ax3.legend(fontsize=11)
ax3.grid(axis='y', linestyle='--', alpha=0.3)
ax3.set_axisbelow(True)

autolabel(bars3_pickle)
autolabel(bars3_joblib)

plt.tight_layout()
plt.savefig('grafico_carregamento.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico de carregamento salvo: grafico_carregamento.png")

# ===== RESUMO ESTATÍSTICO =====
print("\n" + "="*60)
print("RESUMO DOS RESULTADOS (médias em ms)")
print("="*60)

print(f"\n📊 SALVAR MODELO (Lindor):")
print(f"  Pickle:  {save_pickle_mean:.3f} ± {save_pickle_std:.3f} ms")
print(f"  Joblib:  {save_joblib_mean:.3f} ± {save_joblib_std:.3f} ms")
print(f"  Vencedor: {'Pickle' if save_pickle_mean < save_joblib_mean else 'Joblib'} "
      f"({abs(save_pickle_mean - save_joblib_mean):.3f} ms mais rápido)")

print(f"\n📦 TAMANHO DO ARQUIVO:")
print(f"  Pickle:  {data['sizes']['pickle']:.4f} MB")
print(f"  Joblib:  {data['sizes']['joblib']:.4f} MB")
print(f"  Menor:   {'Pickle' if data['sizes']['pickle'] < data['sizes']['joblib'] else 'Joblib'} "
      f"({abs(data['sizes']['pickle'] - data['sizes']['joblib'])*1024:.2f} KB menor)")

print(f"\n📖 CARREGAR MODELO:")
for i, machine in enumerate(machines):
    print(f"\n  {machine.upper()}:")
    print(f"    Pickle:  {load_pickle_means[i]:.3f} ± {load_pickle_stds[i]:.3f} ms")
    print(f"    Joblib:  {load_joblib_means[i]:.3f} ± {load_joblib_stds[i]:.3f} ms")
    ratio = load_joblib_means[i] / load_pickle_means[i]
    print(f"    Joblib é {ratio:.2f}x {'mais lento' if ratio > 1 else 'mais rápido'} que Pickle")

# Calcular médias gerais de carregamento
avg_pickle_load = np.mean(load_pickle_means)
avg_joblib_load = np.mean(load_joblib_means)

print(f"\n🎯 MÉDIA GERAL DE CARREGAMENTO:")
print(f"  Pickle:  {avg_pickle_load:.3f} ms")
print(f"  Joblib:  {avg_joblib_load:.3f} ms")
print(f"  Joblib é {avg_joblib_load/avg_pickle_load:.2f}x mais lento que Pickle")

print("\n" + "="*60)
print("✅ Todos os gráficos foram gerados com sucesso!")
print("="*60)

plt.show()
