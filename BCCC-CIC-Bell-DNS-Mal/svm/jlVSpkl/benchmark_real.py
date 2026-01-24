import pandas as pd
import numpy as np
import time
import pickle
import joblib
import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

# --- CONFIGURAÇÕES ---
N_ITERATIONS = 100
FILE_PICKLE = "pacote_teste.pkl"
FILE_JOBLIB = "pacote_teste.joblib"
RESULTS_FILE = "resultados.json"

# --- 1. FUNÇÃO DE TREINAMENTO (O SEU PIPELINE) ---
def train_models_and_get_artifacts():
    print("--- Iniciando Treinamento Real (Isso vai rodar apenas uma vez) ---")
    print("Carregando e combinando datasets...")
    
    # Check se os arquivos existem
    required_files = [
        'output-of-benign-pcap-0.csv', 'output-of-benign-pcap-1.csv',
        'output-of-benign-pcap-2.csv', 'output-of-benign-pcap-3.csv',
        'output-of-malware-pcap.csv', 'output-of-phishing-pcap.csv',
        'output-of-spam-pcap.csv'
    ]
    for f in required_files:
        if not os.path.exists(f):
            print(f"ERRO CRÍTICO: Arquivo {f} não encontrado. O modo 'train' precisa dos CSVs.")
            sys.exit(1)

    # Carregar datasets
    benign_dfs = [pd.read_csv(f) for f in required_files[:4]]
    benign_df = pd.concat(benign_dfs)
    benign_df['maligno'] = 0; benign_df['tipo_maligno'] = 'Benigno'

    malware_df = pd.read_csv('output-of-malware-pcap.csv')
    malware_df['maligno'] = 1; malware_df['tipo_maligno'] = 'Malware'

    phishing_df = pd.read_csv('output-of-phishing-pcap.csv')
    phishing_df['maligno'] = 1; phishing_df['tipo_maligno'] = 'Phishing'

    spam_df = pd.read_csv('output-of-spam-pcap.csv')
    spam_df['maligno'] = 1; spam_df['tipo_maligno'] = 'Spam'

    df = pd.concat([benign_df, malware_df, phishing_df, spam_df])

    print("Pré-processamento...")
    cols_remover = ['Unnamed: 0', 'flow_id', 'timestamp', 'src_ip', 'dst_ip', 
                    'query_resource_record_type', 'ans_resource_record_type',
                    'query_resource_record_class', 'ans_resource_record_class', 'label']
    df = df.drop(columns=cols_remover, errors='ignore')
    df = df.fillna(0)

    le = LabelEncoder()
    df['protocol'] = le.fit_transform(df['protocol'])

    X = df.drop(columns=['maligno', 'tipo_maligno'])
    y_bin = df['maligno']
    y_multi = df['tipo_maligno']
    
    cat_cols = [col for col in X.columns if X[col].dtype == 'object' and X[col].nunique() < 50]
    X = pd.get_dummies(X, columns=cat_cols)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0) # CRÍTICO: Sua correção
    X = X.fillna(0).astype(np.float64)
    
    # Substituir valores infinitos por 0
    X = X.replace([np.inf, -np.inf], 0)

    print("Seleção de Features...")
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X)
    selected_mask = selector.get_support()
    selected_columns = X.columns[selected_mask]
    X = pd.DataFrame(X_selected, columns=selected_columns)
    
    # Limpar novamente após seleção de features
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    print("Treinando Modelo Binário (LinearSVC)...")
    X_train_bin, _, y_train_bin, _ = train_test_split(X, y_bin, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_bin_scaled = scaler.fit_transform(X_train_bin)
    X_train_bin_scaled = np.nan_to_num(X_train_bin_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    linear_svc = LinearSVC(C=0.1, class_weight='balanced', max_iter=10000, random_state=42, dual='auto')
    linear_svc.fit(X_train_bin_scaled, y_train_bin)

    print("Treinando Modelo Multiclasse (LinearSVC)...")
    malign_df = df[df['maligno'] == 1]
    X_malign = pd.get_dummies(malign_df.drop(columns=['maligno', 'tipo_maligno']))
    X_malign = X_malign.reindex(columns=selected_columns, fill_value=0) # Garante mesmas colunas
    y_malign = malign_df['tipo_maligno']

    X_train_multi, _, y_train_multi, _ = train_test_split(X_malign, y_malign, test_size=0.3, random_state=42)
    X_train_multi_scaled = scaler.transform(X_train_multi) # Usa o mesmo scaler
    X_train_multi_scaled = np.nan_to_num(X_train_multi_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    class_weights = {'Malware': 1.0, 'Phishing': 1.0, 'Spam': 2.0}
    svm_multi_linear = LinearSVC(C=0.1, class_weight=class_weights, max_iter=10000, random_state=42, dual='auto')
    svm_multi_linear.fit(X_train_multi_scaled, y_train_multi)

    # Empacotar tudo num dicionário
    artifacts = {
        "modelo_binario": linear_svc,
        "modelo_multiclasse": svm_multi_linear,
        "scaler": scaler,
        "selected_columns": selected_columns,
        "label_encoder": le,
        "cat_cols": cat_cols
    }
    
    print("Treinamento concluído.")
    return artifacts

# --- 2. FUNÇÕES DE BENCHMARK ---

def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)

def benchmark_save(artifacts):
    print(f"\n--- Medindo TEMPO DE SALVAR ({N_ITERATIONS} repetições) ---")
    
    # Pickle
    times_pickle = []
    print("Testando Pickle...", end="", flush=True)
    for _ in range(N_ITERATIONS):
        start = time.perf_counter()
        with open(FILE_PICKLE, 'wb') as f:
            pickle.dump(artifacts, f)
        times_pickle.append(time.perf_counter() - start)
    print(" OK")

    # Joblib
    times_joblib = []
    print("Testando Joblib...", end="", flush=True)
    for _ in range(N_ITERATIONS):
        start = time.perf_counter()
        joblib.dump(artifacts, FILE_JOBLIB, compress=3) # Compressão padrão do joblib ajuda no tamanho
        times_joblib.append(time.perf_counter() - start)
    print(" OK")

    return times_pickle, times_joblib

def benchmark_load():
    print(f"\n--- Medindo TEMPO DE CARREGAR ({N_ITERATIONS} repetições) ---")
    
    if not os.path.exists(FILE_PICKLE) or not os.path.exists(FILE_JOBLIB):
        print("Arquivos de modelo não encontrados! Rode o --mode train primeiro.")
        sys.exit(1)

    # Pickle
    times_pickle = []
    print("Testando leitura Pickle...", end="", flush=True)
    for _ in range(N_ITERATIONS):
        start = time.perf_counter()
        with open(FILE_PICKLE, 'rb') as f:
            _ = pickle.load(f)
        times_pickle.append(time.perf_counter() - start)
    print(" OK")

    # Joblib
    times_joblib = []
    print("Testando leitura Joblib...", end="", flush=True)
    for _ in range(N_ITERATIONS):
        start = time.perf_counter()
        _ = joblib.load(FILE_JOBLIB)
        times_joblib.append(time.perf_counter() - start)
    print(" OK")

    return times_pickle, times_joblib

# --- 3. GERENCIAMENTO DE RESULTADOS (JSON) ---

def load_json():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {"sizes": {}, "save_times": {}, "load_times": {}}

def save_json(data):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- 4. MODOS DE OPERAÇÃO ---

def run_train(machine):
    # 1. Treina e obtém os objetos reais
    artifacts = train_models_and_get_artifacts()
    
    # 2. Mede tempo de salvamento
    p_save, j_save = benchmark_save(artifacts)
    
    # 3. Mede tamanho
    size_p = get_file_size_mb(FILE_PICKLE)
    size_j = get_file_size_mb(FILE_JOBLIB)
    print(f"\nTamanho Pickle: {size_p:.4f} MB")
    print(f"Tamanho Joblib: {size_j:.4f} MB")

    # 4. Mede carregamento local
    p_load, j_load = benchmark_load()

    # 5. Salva dados
    data = load_json()
    data["sizes"] = {"pickle": size_p, "joblib": size_j}
    data["save_times"] = {"pickle": p_save, "joblib": j_save}
    if "load_times" not in data: data["load_times"] = {}
    data["load_times"][machine] = {"pickle": p_load, "joblib": j_load}
    
    save_json(data)
    print(f"\nDados salvos em {RESULTS_FILE}. Copie tudo para as outras máquinas!")

def run_load(machine):
    # Apenas mede o load dos arquivos existentes
    p_load, j_load = benchmark_load()
    
    data = load_json()
    if "load_times" not in data: data["load_times"] = {}
    data["load_times"][machine] = {"pickle": p_load, "joblib": j_load}
    
    save_json(data)
    print(f"\nResultados de {machine} adicionados ao JSON.")

def run_plot():
    data = load_json()
    if not data["save_times"]:
        print("Sem dados para plotar.")
        return

    machines = list(data["load_times"].keys())
    
    # Preparar médias e desvios (converter para ms)
    save_m = [np.mean(data["save_times"]["pickle"])*1000, np.mean(data["save_times"]["joblib"])*1000]
    save_std = [np.std(data["save_times"]["pickle"])*1000, np.std(data["save_times"]["joblib"])*1000]

    # Agrupar dados de Load
    load_means_p = []
    load_stds_p = []
    load_means_j = []
    load_stds_j = []
    
    for m in machines:
        load_means_p.append(np.mean(data["load_times"][m]["pickle"]) * 1000)
        load_stds_p.append(np.std(data["load_times"][m]["pickle"]) * 1000)
        load_means_j.append(np.mean(data["load_times"][m]["joblib"]) * 1000)
        load_stds_j.append(np.std(data["load_times"][m]["joblib"]) * 1000)

    # Plot
    labels = ['Gerar Modelo\n(Lindor)'] + [f'Carregar\n({m})' for m in machines]
    x = np.arange(len(labels))
    width = 0.35

    means_p = [save_m[0]] + load_means_p
    stds_p = [save_std[0]] + load_stds_p
    means_j = [save_m[1]] + load_means_j
    stds_j = [save_std[1]] + load_stds_j

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, means_p, width, yerr=stds_p, label='Pickle', capsize=5, color='#1f77b4')
    rects2 = ax.bar(x + width/2, means_j, width, yerr=stds_j, label='Joblib', capsize=5, color='#ff7f0e')

    ax.set_ylabel('Tempo (ms)')
    ax.set_title('Benchmark Real: Pickle vs Joblib (LinearSVC + Artifacts)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}ms', xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('grafico_benchmark_real.png')
    
    # Plot Tamanho
    plt.figure(figsize=(5, 5))
    sizes = [data["sizes"]["pickle"], data["sizes"]["joblib"]]
    plt.bar(['Pickle', 'Joblib'], sizes, color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Tamanho (MB)')
    plt.title('Tamanho do Arquivo')
    for i, v in enumerate(sizes):
        plt.text(i, v, f"{v:.2f} MB", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('grafico_tamanho_real.png')
    print("\nGráficos gerados: 'grafico_benchmark_real.png' e 'grafico_tamanho_real.png'")
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "load", "plot"], required=True)
    parser.add_argument("--machine", help="Nome da máquina (ex: lindor)")
    args = parser.parse_args()

    if args.mode in ["train", "load"] and not args.machine:
        print("Erro: --machine é obrigatório para train e load.")
    elif args.mode == "train":
        run_train(args.machine)
    elif args.mode == "load":
        run_load(args.machine)
    elif args.mode == "plot":
        run_plot()