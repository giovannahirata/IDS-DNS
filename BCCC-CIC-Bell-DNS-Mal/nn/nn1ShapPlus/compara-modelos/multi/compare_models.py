import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ============================================================================
# FUNÇÕES AUXILIARES DE PRÉ-PROCESSAMENTO E CARREGAMENTO
# ============================================================================

def load_artifacts(artifacts_dir):
    """Carrega o modelo .keras e os arquivos .pkl"""
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
        
    target_encoder_path = os.path.join(artifacts_dir, 'target_encoder.pkl')
    if os.path.exists(target_encoder_path):
        with open(target_encoder_path, 'rb') as f:
            target_encoder = pickle.load(f)
    else:
        from sklearn.preprocessing import LabelEncoder
        target_encoder = LabelEncoder()
        target_encoder.fit(['Benigno', 'Malware', 'Phishing', 'Spam'])

    return model, scaler, selector, label_encoders, target_encoder


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
    x = x.fillna(0)

    # 2. Encoding de variáveis categóricas
    categorical_cols = [col for col in x.select_dtypes(include=['object']).columns]
    for col in categorical_cols:
        if col in label_encoders:
            x[col] = safe_label_transform(label_encoders[col], x[col])

    # 3. Normalização (Escalonador específico do modelo)
    numerical_cols = x.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    x[numerical_cols] = scaler.transform(x[numerical_cols])

    # 4. Seleção de Features (Seletor específico do modelo)
    x_transformed = selector.transform(x)

    return np.array(x_transformed, dtype=np.float32)


def evaluate_single_model(model_name, artifacts_dir, df_test, target_col='tipo_maligno'):
    """Avalia um modelo multiclasse e retorna métricas."""
    print(f"\nAvaliando: {model_name}...")
    
    model, scaler, selector, label_encoders, target_encoder = load_artifacts(artifacts_dir)
    
    # Extrai o ground truth numérico usando o encoder
    y_true_str = df_test[target_col].values
    y_true = target_encoder.transform(y_true_str)
    
    # Pré-processa as features
    X_processed = preprocess_for_model(
        df_test, label_encoders, scaler, selector, 
        target_cols=['maligno', 'tipo_maligno']
    )

    # Predição multiclasse e latência
    start_time = time.perf_counter()
    y_pred_proba = model.predict(X_processed, verbose=0)
    latency_ms = ((time.perf_counter() - start_time) / len(df_test)) * 1000
    
    # Extrai a classe com maior probabilidade
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Métricas multiclasse (weighted)
    metrics = {
        'Modelo': model_name,
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-Score': f1_score(y_true, y_pred, average='weighted'),
        'AUC-ROC': roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted'),
        'Latência (ms/amostra)': latency_ms
    }

    print(f"  ✓ {model_name} avaliado. F1-Score: {metrics['F1-Score']:.4f} | AUC: {metrics['AUC-ROC']:.4f}")

    return metrics


# ============================================================================
# EXECUÇÃO PRINCIPAL E COMPARAÇÃO
# ============================================================================

def compare_two_models(dir_model_a, dir_model_b, test_csv_files, name_a="Modelo A", name_b="Modelo B", output_dir="comparacao_modelos_multiclasse"):
    """Compara dois modelos multiclasse no mesmo conjunto de arquivos CSV de teste."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Carrega o dataset comum e rotula automaticamente
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
            print(f"Aviso: Não inferido. Assumindo Benigno por segurança.")
            df_temp['maligno'] = 0
            df_temp['tipo_maligno'] = 'Benigno'
            
        df_list.append(df_temp)
        
    df_test_common = pd.concat(df_list, ignore_index=True)
    print(f"Total de amostras de teste: {len(df_test_common)}")

    # 2. Avalia Modelo A
    metrics_a = evaluate_single_model(name_a, dir_model_a, df_test_common, target_col='tipo_maligno')

    # 3. Avalia Modelo B
    metrics_b = evaluate_single_model(name_b, dir_model_b, df_test_common, target_col='tipo_maligno')

    # 4. Tabela Comparativa em DataFrame
    df_comparison = pd.DataFrame([metrics_a, metrics_b])
    
    print("\n" + "="*80)
    print("TABELA COMPARATIVA DE DESEMPENHO (MULTICLASSE)")
    print("="*80)
    print(df_comparison.to_string(index=False))
    
    csv_out = os.path.join(output_dir, "resultado_comparativo_multiclasse.csv")
    df_comparison.to_csv(csv_out, index=False)
    print(f"\nTabela comparativa salva em: {csv_out}")

    # ============================================================================
    # GERAR VISUALIZAÇÕES
    # ============================================================================

    # Gráfico de Barras Comparativo
    metrics_to_plot = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, [metrics_a[m] for m in metrics_to_plot], width, label=name_a, color='steelblue')
    rects2 = ax.bar(x + width/2, [metrics_b[m] for m in metrics_to_plot], width, label=name_b, color='darkorange')

    ax.set_ylabel('Pontuação (0.0 a 1.0)')
    ax.set_title('Comparação de Desempenho (Métricas Multiclasse Ponderadas)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plot_bars_path = os.path.join(output_dir, "comparacao_metricas_barras.png")
    plt.savefig(plot_bars_path, dpi=150)
    plt.close()
    print(f"Gráfico de barras salvo em: {plot_bars_path}")


if __name__ == "__main__":
    dataset_dir = "/home/giovanna/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/datasets-br/"
    
    test_files = [
        dataset_dir + "output-of-benign-br-pcap-0.csv",
        dataset_dir + "output-of-malware-br-pcap.csv",
        dataset_dir + "output-of-phishing-br-pcap.csv",
        dataset_dir + "output-of-spam-br-pcap.csv"
    ]

    folder_model_1 = "melhor-br"
    folder_model_2 = "melhor-geral"

    compare_two_models(
        dir_model_a=folder_model_1,
        dir_model_b=folder_model_2,
        test_csv_files=test_files,
        name_a="Modelo BR (Dataset .br)",
        name_b="Modelo Geral (Dataset Internacional)",
        output_dir="comparacao_final_multiclasse"
    )