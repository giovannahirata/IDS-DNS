import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# ============================================================================
# FUNÇÕES AUXILIARES DE PRÉ-PROCESSAMENTO E CARREGAMENTO
# ============================================================================

def load_artifacts(artifacts_dir):
    """Carrega o modelo .keras e os arquivos .pkl de um diretório de artefatos."""
    model_files = [f for f in os.listdir(artifacts_dir) if f.endswith('.keras')]
    if not model_files:
        raise FileNotFoundError(f"Nenhum modelo .keras encontrado em {artifacts_dir}")
    
    model_path = os.path.join(artifacts_dir, model_files[0])
    model = tf.keras.models.load_model(model_path)

    with open(os.path.join(artifacts_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    with open(os.path.join(artifacts_dir, 'selector.pkl'), 'rb') as f:
        selector = pickle.load(f)

    with open(os.path.join(artifacts_dir, 'label_encoders.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)

    return model, scaler, selector, label_encoders


def safe_label_transform(le, series):
    """Aplica o LabelEncoder tratando categorias desconhecidas."""
    known_classes = set(le.classes_)
    series_str = series.astype(str)
    fallback_value = le.classes_[0]
    series_cleaned = series_str.map(lambda x: x if x in known_classes else fallback_value)
    return le.transform(series_cleaned)


def preprocess_for_model(df_raw, label_encoders, scaler, selector, target_cols):
    """Aplica a pipeline específica de um modelo ao dataset de teste bruto."""
    df_clean = df_raw.copy()

    # 1. Remoção de metadados, identificadores e colunas Unnamed
    cols_remove = ['flow_id', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']
    cols_to_drop = [
        col for col in df_clean.columns 
        if col in cols_remove or col.startswith('Unnamed') or col in target_cols
    ]
    
    x = df_clean.drop(columns=cols_to_drop, errors='ignore')

    # 2. Preenchimento de NAs
    x = x.fillna(0)

    # 3. Encoding de variáveis categóricas
    categorical_cols = [col for col in x.select_dtypes(include=['object']).columns]
    for col in categorical_cols:
        if col in label_encoders:
            x[col] = safe_label_transform(label_encoders[col], x[col])

    # 4. Normalização (Escalonador específico do modelo)
    numerical_cols = x.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    x[numerical_cols] = scaler.transform(x[numerical_cols])

    # 5. Seleção de Features (Seletor específico do modelo)
    x_transformed = selector.transform(x)

    return np.array(x_transformed, dtype=np.float32)


def evaluate_single_model(model_name, artifacts_dir, df_test, target_col='maligno'):
    """Avalia um único modelo e retorna suas métricas e probabilidades."""
    print(f"\nAvaliando: {model_name}...")
    
    # Carrega artefatos do modelo
    model, scaler, selector, label_encoders = load_artifacts(artifacts_dir)
    
    # Extrai o ground truth
    y_true = df_test[target_col].values.astype(int)
    
    # Pré-processa os dados usando o próprio pipeline do modelo
    X_processed = preprocess_for_model(
        df_test, label_encoders, scaler, selector, 
        target_cols=[target_col, 'tipo_maligno']
    )

    # Predição e medição de latência
    start_time = time.perf_counter()
    y_pred_proba = model.predict(X_processed, verbose=0).ravel()
    latency_ms = ((time.perf_counter() - start_time) / len(df_test)) * 1000
    
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Métricas
    metrics = {
        'Modelo': model_name,
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1-Score': f1_score(y_true, y_pred, average='binary'),
        'AUC-ROC': roc_auc_score(y_true, y_pred_proba),
        'Latência (ms/amostra)': latency_ms
    }

    print(f"  ✓ {model_name} avaliado com sucesso. F1-Score: {metrics['F1-Score']:.4f} | AUC: {metrics['AUC-ROC']:.4f}")

    return metrics, y_true, y_pred_proba


# ============================================================================
# EXECUÇÃO PRINCIPAL E COMPARAÇÃO
# ============================================================================

def compare_two_models(dir_model_a, dir_model_b, test_csv_files, name_a="Modelo A", name_b="Modelo B", output_dir="comparacao_modelos"):
    """Compara dois modelos no mesmo conjunto de arquivos CSV de teste."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Carrega o dataset de teste comum e adiciona rótulos com base no arquivo
    print(f"Carregando dataset de teste comum ({len(test_csv_files)} arquivo(s))...")
    df_list = []
    for f in test_csv_files:
        df_temp = pd.read_csv(f)
        
        f_lower = f.lower()
        if 'benign' in f_lower:
            df_temp['maligno'] = 0
            df_temp['tipo_maligno'] = 'Benigno'
        elif 'malware' in f_lower:
            df_temp['maligno'] = 1
            df_temp['tipo_maligno'] = 'Malware'
        elif 'phishing' in f_lower:
            df_temp['maligno'] = 1
            df_temp['tipo_maligno'] = 'Phishing'
        elif 'spam' in f_lower:
            df_temp['maligno'] = 1
            df_temp['tipo_maligno'] = 'Spam'
        else:
            print(f"Aviso: Não foi possível inferir a classe do arquivo {f}. Assumindo Maligno.")
            df_temp['maligno'] = 1
            df_temp['tipo_maligno'] = 'Desconhecido'
            
        df_list.append(df_temp)
        
    df_test_common = pd.concat(df_list, ignore_index=True)
    print(f"Total de amostras de teste: {len(df_test_common)}")

    # 2. Avalia Modelo A
    metrics_a, y_true_a, proba_a = evaluate_single_model(name_a, dir_model_a, df_test_common)

    # 3. Avalia Modelo B
    metrics_b, y_true_b, proba_b = evaluate_single_model(name_b, dir_model_b, df_test_common)

    # 4. Tabela Comparativa em DataFrame
    df_comparison = pd.DataFrame([metrics_a, metrics_b])
    
    print("\n" + "="*80)
    print("TABELA COMPARATIVA DE DESEMPENHO")
    print("="*80)
    print(df_comparison.to_string(index=False))
    
    # Salva em CSV
    csv_out = os.path.join(output_dir, "resultado_comparativo.csv")
    df_comparison.to_csv(csv_out, index=False)
    print(f"\nTabela comparativa salva em: {csv_out}")

    # ============================================================================
    # GERAR VISUALIZAÇÕES
    # ============================================================================

    # Visualização 1: Gráfico de Barras Comparativo
    metrics_to_plot = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, [metrics_a[m] for m in metrics_to_plot], width, label=name_a, color='steelblue')
    rects2 = ax.bar(x + width/2, [metrics_b[m] for m in metrics_to_plot], width, label=name_b, color='darkorange')

    ax.set_ylabel('Pontuação (0.0 a 1.0)')
    ax.set_title('Comparação de Métricas de Desempenho no Mesmo Dataset de Teste')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Adiciona rótulos numéricos sobre as barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plot_bars_path = os.path.join(output_dir, "comparacao_metricas_barras.png")
    plt.savefig(plot_bars_path, dpi=150)
    plt.close()
    print(f"Gráfico de barras salvo em: {plot_bars_path}")

    # Visualização 2: Curvas ROC Sobrepostas
    plt.figure(figsize=(8, 6))
    
    fpr_a, tpr_a, _ = roc_curve(y_true_a, proba_a)
    fpr_b, tpr_b, _ = roc_curve(y_true_b, proba_b)

    plt.plot(fpr_a, tpr_a, color='steelblue', lw=2, label=f'{name_a} (AUC = {metrics_a["AUC-ROC"]:.4f})')
    plt.plot(fpr_b, tpr_b, color='darkorange', lw=2, label=f'{name_b} (AUC = {metrics_b["AUC-ROC"]:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Classificador Aleatório')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Comparação de Curvas ROC no Dataset de Teste')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_roc_path = os.path.join(output_dir, "comparacao_curvas_roc.png")
    plt.savefig(plot_roc_path, dpi=150)
    plt.close()
    print(f"Curvas ROC comparativas salvas em: {plot_roc_path}")


if __name__ == "__main__":
    dataset_dir = "/home/giovanna/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/datasets-br/"
    
    # Dataset comum de teste onde ambos os modelos serão avaliados
    test_files = [
        dataset_dir + "output-of-benign-br-pcap-0.csv",
        dataset_dir + "output-of-malware-br-pcap.csv",
        dataset_dir + "output-of-phishing-br-pcap.csv",
        dataset_dir + "output-of-spam-br-pcap.csv"
    ]

    # Pastas contendo cada um dos modelos treinados e seus respectivos .pkl
    folder_model_1 = "melhor-br"
    folder_model_2 = "melhor-geral"

    compare_two_models(
        dir_model_a=folder_model_1,
        dir_model_b=folder_model_2,
        test_csv_files=test_files,
        name_a="Modelo brasileiro treinado no dataset .br",
        name_b="Modelo internacional treinado no dataset geral",
        output_dir="comparacao_final_modelos"
    )