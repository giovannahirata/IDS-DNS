"""
Modelo DNS Híbrido Melhorado - Versão 2.0
Incorpora:
  - Pré-processamento especializado por ramo
  - Separação train/test antes de processing (sem data leakage)
  - Ramo lexical com texto puro + Embedding
  - Benchmarking: NN Híbrida vs XGBoost vs LightGBM vs Autoencoder
  - Ensemble final
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DNS Intrusion Detection - Hybrid Model v2.0")
print("="*80)

# ============================================================================
# PARTE 1: CARREGAMENTO DOS DADOS
# ============================================================================

dir_path = os.path.expanduser("~/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/")

benigns = ["output-of-benign-pcap-0.csv",
           "output-of-benign-pcap-1.csv",
           "output-of-benign-pcap-2.csv",
           "output-of-benign-pcap-3.csv"]

print("\n[1/5] Carregando datasets...")
df_benigns = [pd.read_csv(dir_path + f) for f in benigns]
df_benign = pd.concat(df_benigns)
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

df = pd.concat([df_benign, df_malware, df_phishing, df_spam]).reset_index(drop=True)

print(f"\nEstatísticas do dataset completo:")
print(f"  Total de amostras: {len(df)}")
print(f"  Classes: {df['maligno'].value_counts().to_dict()}")
print(f"  Proporção: {df['maligno'].value_counts(normalize=True).to_dict()}")

cols_remove = ['Unnamed: 0', 'flow_id', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
df = df.drop(columns=cols_remove)

# ============================================================================
# PARTE 2: DEFINIÇÃO DOS GRUPOS DE FEATURES
# ============================================================================

lexical_text_features = ['dns_domain_name', 'dns_top_level_domain', 'dns_second_level_domain']

lexical_numeric_features = [
    'dns_domain_name_length', 'dns_subdomain_name_length',
    'uni_gram_domain_name', 'bi_gram_domain_name', 'tri_gram_domain_name',
    'numerical_percentage', 'character_distribution', 'character_entropy',
    'max_continuous_numeric_len', 'max_continuous_alphabet_len',
    'max_continuous_consonants_len', 'max_continuous_same_alphabet_len',
    'vowels_consonant_ratio', 'conv_freq_vowels_consonants'
]

size_based_features = [
    'total_bytes', 'receiving_bytes', 'sending_bytes',
    'min_packets_len', 'max_packets_len', 'mean_packets_len',
    'median_packets_len', 'mode_packets_len',
    'standard_deviation_packets_len', 'variance_packets_len',
    'coefficient_of_variation_packets_len', 'skewness_packets_len',
    'min_receiving_packets_len', 'max_receiving_packets_len',
    'mean_receiving_packets_len', 'median_receiving_packets_len',
    'standard_deviation_receiving_packets_len', 'variance_receiving_packets_len',
    'coefficient_of_variation_receiving_packets_len', 'skewness_receiving_packets_len',
    'min_sending_packets_len', 'max_sending_packets_len',
    'mean_sending_packets_len', 'median_sending_packets_len',
    'standard_deviation_sending_packets_len', 'variance_sending_packets_len',
    'coefficient_of_variation_sending_packets_len', 'skewness_sending_packets_len'
]

delta_length_based_features = [
    'min_receiving_packets_delta_len', 'max_receiving_packets_delta_len',
    'mean_receiving_packets_delta_len', 'median_receiving_packets_delta_len',
    'standard_deviation_receiving_packets_delta_len', 'variance_receiving_packets_delta_len',
    'mode_receiving_packets_delta_len', 'coefficient_of_variation_receiving_packets_delta_len',
    'skewness_receiving_packets_delta_len',
    'min_sending_packets_delta_len', 'max_sending_packets_delta_len',
    'mean_sending_packets_delta_len', 'median_sending_packets_delta_len',
    'standard_deviation_sending_packets_delta_len', 'variance_sending_packets_delta_len',
    'mode_sending_packets_delta_len', 'coefficient_of_variation_sending_packets_delta_len',
    'skewness_sending_packets_delta_len'
]

delta_time_based_features = [
    'max_receiving_packets_delta_time', 'mean_receiving_packets_delta_time',
    'median_receiving_packets_delta_time', 'standard_deviation_receiving_packets_delta_time',
    'variance_receiving_packets_delta_time', 'mode_receiving_packets_delta_time',
    'coefficient_of_variation_receiving_packets_delta_time', 'skewness_sreceiving_packets_delta_time',
    'min_sending_packets_delta_time', 'max_sending_packets_delta_time',
    'mean_sending_packets_delta_time', 'median_sending_packets_delta_time',
    'standard_deviation_sending_packets_delta_time', 'variance_sending_packets_delta_time',
    'mode_sending_packets_delta_time', 'coefficient_of_variation_sending_packets_delta_time',
    'skewness_sending_packets_delta_time',
    'duration', 'handshake_duration', 'delta_start'
]

side_based_features = [
    'receiving_packets_numbers', 'sending_packets_numbers',
    'receiving_packets_rate', 'sending_packets_rate',
    'receiving_packets_len_rate', 'sending_packets_len_rate',
    'packets_numbers'
]

resource_record_based_features = [
    'distinct_ttl_values', 'ttl_values_min', 'ttl_values_max',
    'ttl_values_mean', 'ttl_values_mode', 'ttl_values_variance',
    'ttl_values_standard_deviation', 'ttl_values_median',
    'ttl_values_skewness', 'ttl_values_coefficient_of_variation',
    'distinct_A_records', 'distinct_NS_records',
    'average_authority_resource_records', 'average_additional_resource_records',
    'average_answer_resource_records',
    'query_resource_record_type', 'ans_resource_record_type',
    'query_resource_record_class', 'ans_resource_record_class'
]

statistical_based_features = ['packets_rate', 'packets_len_rate']

comportamental_features = (size_based_features + delta_length_based_features + 
                          delta_time_based_features + side_based_features)
protocolo_features = resource_record_based_features + statistical_based_features

print(f"\nDistribuição de features:")
print(f"  Lexical (texto): {len(lexical_text_features)}")
print(f"  Lexical (numérico): {len(lexical_numeric_features)}")
print(f"  Comportamental: {len(comportamental_features)}")
print(f"  Protocolo: {len(protocolo_features)}")

# ============================================================================
# PARTE 3: SEPARAR TRAIN/TEST ANTES DE PROCESSING (SEM DATA LEAKAGE)
# ============================================================================

print("\n[2/5] Separando train/test antes de pré-processamento...")

# Split estratificado
from sklearn.model_selection import train_test_split
df_train, df_valid = train_test_split(df, test_size=0.3, random_state=42, 
                                      stratify=df['maligno'])

y_train = df_train['maligno'].values.astype(np.float32)
y_valid = df_valid['maligno'].values.astype(np.float32)

print(f"  Treino: {len(df_train)} amostras")
print(f"  Validação: {len(df_valid)} amostras")

# ============================================================================
# PARTE 4: PRÉ-PROCESSAMENTO ESPECIALIZADO POR RAMO
# ============================================================================

print("\n[3/5] Pré-processamento especializado por ramo...")

# --- RAMO LEXICAL (TEXTO) ---
print("\n  [Ramo Lexical - TEXTO PURO]")
x_train_text_lex = df_train[lexical_text_features].fillna('unknown').astype(str)
x_valid_text_lex = df_valid[lexical_text_features].fillna('unknown').astype(str)
# Concatenar colunas de texto para formação de "sequência"
x_train_text_lex_seq = (x_train_text_lex['dns_domain_name'] + '_' + 
                        x_train_text_lex['dns_top_level_domain']).values
x_valid_text_lex_seq = (x_valid_text_lex['dns_domain_name'] + '_' + 
                        x_valid_text_lex['dns_top_level_domain']).values
print(f"    Exemplo: {x_train_text_lex_seq[0]}")

# --- RAMO LEXICAL (NUMÉRICO) ---
print("  [Ramo Lexical - NUMÉRICO]")
x_train_num_lex = df_train[lexical_numeric_features].copy()
x_valid_num_lex = df_valid[lexical_numeric_features].copy()

# Converter para numérico com coerce (força strings não-numéricas -> NaN)
for col in x_train_num_lex.columns:
    x_train_num_lex[col] = pd.to_numeric(x_train_num_lex[col], errors='coerce')
    x_valid_num_lex[col] = pd.to_numeric(x_valid_num_lex[col], errors='coerce')

# Fillna com mediana do treino
x_train_num_lex = x_train_num_lex.fillna(x_train_num_lex.median(numeric_only=True))
x_valid_num_lex = x_valid_num_lex.fillna(x_train_num_lex.median(numeric_only=True))

# Converter para float32 
x_train_num_lex = x_train_num_lex.astype(np.float32)
x_valid_num_lex = x_valid_num_lex.astype(np.float32)

# LogTransform para features de entropia
x_train_num_lex['character_entropy'] = np.log1p(x_train_num_lex['character_entropy'])
x_valid_num_lex['character_entropy'] = np.log1p(x_valid_num_lex['character_entropy'])

# StandardScaler fit só no treino
ss_lex = StandardScaler()
x_train_num_lex = ss_lex.fit_transform(x_train_num_lex)
x_valid_num_lex = ss_lex.transform(x_valid_num_lex)

# VarianceThreshold
vt_lex = VarianceThreshold(threshold=0.001)
x_train_num_lex = vt_lex.fit_transform(x_train_num_lex)
x_valid_num_lex = vt_lex.transform(x_valid_num_lex)
print(f"    Features após VarianceThreshold: {x_train_num_lex.shape[1]}")

# --- RAMO COMPORTAMENTAL ---
print("  [Ramo Comportamental]")
x_train_comp = df_train[comportamental_features].copy()
x_valid_comp = df_valid[comportamental_features].copy()

# Converter para numérico com coerce (força strings não-numéricas -> NaN)
for col in x_train_comp.columns:
    x_train_comp[col] = pd.to_numeric(x_train_comp[col], errors='coerce')
    x_valid_comp[col] = pd.to_numeric(x_valid_comp[col], errors='coerce')

# Fillna com 0 (comportamental)
x_train_comp = x_train_comp.fillna(0)
x_valid_comp = x_valid_comp.fillna(0)

# Converter para float32 
x_train_comp = x_train_comp.astype(np.float32)
x_valid_comp = x_valid_comp.astype(np.float32)

# RobustScaler (menos sensível a outliers)
robust_scaler_comp = RobustScaler()
x_train_comp = robust_scaler_comp.fit_transform(x_train_comp)
x_valid_comp = robust_scaler_comp.transform(x_valid_comp)

# VarianceThreshold
vt_comp = VarianceThreshold(threshold=0.01)
x_train_comp = vt_comp.fit_transform(x_train_comp)
x_valid_comp = vt_comp.transform(x_valid_comp)
print(f"    Features após VarianceThreshold: {x_train_comp.shape[1]}")

# --- RAMO PROTOCOLO ---
print("  [Ramo Protocolo]")
x_train_prot = df_train[protocolo_features].copy()
x_valid_prot = df_valid[protocolo_features].copy()

# Converter para numérico com coerce (força strings não-numéricas -> NaN)
for col in x_train_prot.columns:
    x_train_prot[col] = pd.to_numeric(x_train_prot[col], errors='coerce')
    x_valid_prot[col] = pd.to_numeric(x_valid_prot[col], errors='coerce')

# Fillna com 0 (protocolo)
x_train_prot = x_train_prot.fillna(0)
x_valid_prot = x_valid_prot.fillna(0)

# Converter para float32
x_train_prot = x_train_prot.astype(np.float32)
x_valid_prot = x_valid_prot.astype(np.float32)

# StandardScaler (dados limpos)
ss_prot = StandardScaler()
x_train_prot = ss_prot.fit_transform(x_train_prot)
x_valid_prot = ss_prot.transform(x_valid_prot)

# VarianceThreshold rigoroso
vt_prot = VarianceThreshold(threshold=0.05)
x_train_prot = vt_prot.fit_transform(x_train_prot)
x_valid_prot = vt_prot.transform(x_valid_prot)
print(f"    Features após VarianceThreshold: {x_train_prot.shape[1]}")

# --- RAMO TABULARES (para XGBoost, LightGBM, etc) ---
print("  [Dataset tabulares (para algoritmos clássicos)]")
x_train_all = np.hstack([x_train_num_lex, x_train_comp, x_train_prot])
x_valid_all = np.hstack([x_valid_num_lex, x_valid_comp, x_valid_prot])
print(f"    Total de features tabulares: {x_train_all.shape[1]}")

# ============================================================================
# PARTE 5: FUNÇÃO PARA TOKENIZAR TEXTO (CHARACTER-LEVEL)
# ============================================================================

def text_to_charseq(text_seq, max_len=100, vocab_size=256):
    """Converte sequência de texto em array numérico (ASCII codes)"""
    charseq = []
    for text in text_seq:
        if isinstance(text, str):
            codes = [ord(c) % vocab_size for c in text[:max_len]]
            codes += [0] * (max_len - len(codes))
            charseq.append(codes[:max_len])
        else:
            charseq.append([0] * max_len)
    return np.array(charseq, dtype=np.float32)

x_train_text_encoded = text_to_charseq(x_train_text_lex_seq, max_len=100)
x_valid_text_encoded = text_to_charseq(x_valid_text_lex_seq, max_len=100)
print(f"\n  Texto codificado: {x_train_text_encoded.shape}")

# ============================================================================
# PARTE 6: MODELO NEURAL HÍBRIDO MELHORADO
# ============================================================================

print("\n[4/5] Construindo e treinando modelos...")

def hybrid_model_v2(lexical_text_dim, lexical_num_dim, comportamental_dim, protocolo_dim):
    """
    Versão 2 do modelo híbrido com:
    - Ramo lexical de TEXTO COM EMBEDDING + Conv1D
    - Ramo lexical numérico com MLP
    - Ramo comportamental com MLP profundo (sem LSTM)
    - Ramo protocolo com MLP
    """
    
    # --- RAMO LEXICAL (TEXTO) ---
    input_text = layers.Input(shape=(lexical_text_dim,), name='input_text', dtype=tf.int32)
    x_text = layers.Embedding(256, 32, input_length=lexical_text_dim)(input_text)
    x_text = layers.Conv1D(64, 3, activation='relu', padding='same')(x_text)
    x_text = layers.MaxPooling1D(2, padding='same')(x_text)
    x_text = layers.Conv1D(32, 3, activation='relu', padding='same')(x_text)
    x_text = layers.GlobalAveragePooling1D()(x_text)
    x_text = layers.Dense(64, activation='relu')(x_text)
    x_text = layers.Dropout(0.3)(x_text)
    x_text = layers.Dense(32, activation='relu')(x_text)
    branch_text_output = x_text
    
    # --- RAMO LEXICAL (NUMÉRICO) ---
    input_lex_num = layers.Input(shape=(lexical_num_dim,), name='input_lex_num')
    x_lex_num = layers.Dense(64, activation='relu')(input_lex_num)
    x_lex_num = layers.Dropout(0.2)(x_lex_num)
    x_lex_num = layers.Dense(32, activation='relu')(x_lex_num)
    branch_lex_num_output = x_lex_num
    
    # --- RAMO COMPORTAMENTAL (MLP profundo) ---
    input_comp = layers.Input(shape=(comportamental_dim,), name='input_comp')
    x_comp = layers.Dense(256, activation='relu')(input_comp)
    x_comp = layers.BatchNormalization()(x_comp)
    x_comp = layers.Dropout(0.3)(x_comp)
    x_comp = layers.Dense(128, activation='relu')(x_comp)
    x_comp = layers.Dropout(0.3)(x_comp)
    x_comp = layers.Dense(64, activation='relu')(x_comp)
    x_comp = layers.Dropout(0.2)(x_comp)
    x_comp = layers.Dense(32, activation='relu')(x_comp)
    branch_comp_output = x_comp
    
    # --- RAMO PROTOCOLO ---
    input_prot = layers.Input(shape=(protocolo_dim,), name='input_prot')
    x_prot = layers.Dense(96, activation='relu')(input_prot)
    x_prot = layers.BatchNormalization()(x_prot)
    x_prot = layers.Dropout(0.2)(x_prot)
    x_prot = layers.Dense(64, activation='relu')(x_prot)
    x_prot = layers.Dropout(0.2)(x_prot)
    x_prot = layers.Dense(32, activation='relu')(x_prot)
    branch_prot_output = x_prot
    
    # --- FUSÃO ---
    merged = layers.Concatenate()([branch_text_output, branch_lex_num_output, 
                                   branch_comp_output, branch_prot_output])
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=[input_text, input_lex_num, input_comp, input_prot], 
                       outputs=output, name='HybridModelV2')
    return model

# Construir modelo
hybrid_nn_model = hybrid_model_v2(
    lexical_text_dim=x_train_text_encoded.shape[1],
    lexical_num_dim=x_train_num_lex.shape[1],
    comportamental_dim=x_train_comp.shape[1],
    protocolo_dim=x_train_prot.shape[1]
)

hybrid_nn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC()]
)

print("\nArquitetura do modelo híbrido v2:")
hybrid_nn_model.summary()

# Salva visualização da arquitetura
print("\n  Salvando visualização da arquitetura...")
try:
    plot_model(
        hybrid_nn_model,
        to_file='hybrid_model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',  # Top to Bottom
        dpi=150,
        show_layer_activations=True
    )
    print("Arquitetura salva em: hybrid_model_architecture.png")
except Exception as e:
    print(f"Erro ao salvar arquitetura: {e}")

# Treinar com early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
)

print("\nTreinando Modelo Híbrido Neural (v2)...")
history_hybrid = hybrid_nn_model.fit(
    [x_train_text_encoded, x_train_num_lex.astype(np.float32), 
     x_train_comp.astype(np.float32), x_train_prot.astype(np.float32)],
    y_train,
    validation_data=(
        [x_valid_text_encoded, x_valid_num_lex.astype(np.float32),
         x_valid_comp.astype(np.float32), x_valid_prot.astype(np.float32)],
        y_valid
    ),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

y_pred_hybrid_proba = hybrid_nn_model.predict(
    [x_valid_text_encoded, x_valid_num_lex.astype(np.float32),
     x_valid_comp.astype(np.float32), x_valid_prot.astype(np.float32)],
    verbose=0
).ravel()

# ============================================================================
# PARTE 7: COMPARAR COM OUTROS ALGORITMOS (BENCHMARKING)
# ============================================================================

print("\n" + "="*80)
print("BENCHMARKING: Comparando múltiplos algoritmos")
print("="*80)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

results_benchmark = {}

# --- XGBoost ---
print("\nTreinando XGBoost...")
xgb_model = XGBClassifier(
    max_depth=7,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=sum(y_train==0) / sum(y_train==1),
    random_state=42,
    verbosity=0
)
xgb_model.fit(x_train_all, y_train)
y_pred_xgb = xgb_model.predict_proba(x_valid_all)[:, 1]
y_pred_xgb_binary = (y_pred_xgb > 0.5).astype(int)
results_benchmark['XGBoost'] = {
    'accuracy': accuracy_score(y_valid, y_pred_xgb_binary),
    'precision': precision_score(y_valid, y_pred_xgb_binary),
    'recall': recall_score(y_valid, y_pred_xgb_binary),
    'f1': f1_score(y_valid, y_pred_xgb_binary),
    'auc': roc_auc_score(y_valid, y_pred_xgb),
    'model': xgb_model,
    'y_pred_proba': y_pred_xgb
}

# --- LightGBM ---
print("Treinando LightGBM...")
lgb_model = LGBMClassifier(
    max_depth=7,
    learning_rate=0.1,
    n_estimators=200,
    num_leaves=31,
    scale_pos_weight=sum(y_train==0) / sum(y_train==1),
    random_state=42,
    verbose=-1
)
lgb_model.fit(x_train_all, y_train)
y_pred_lgb = lgb_model.predict_proba(x_valid_all)[:, 1]
y_pred_lgb_binary = (y_pred_lgb > 0.5).astype(int)
results_benchmark['LightGBM'] = {
    'accuracy': accuracy_score(y_valid, y_pred_lgb_binary),
    'precision': precision_score(y_valid, y_pred_lgb_binary),
    'recall': recall_score(y_valid, y_pred_lgb_binary),
    'f1': f1_score(y_valid, y_pred_lgb_binary),
    'auc': roc_auc_score(y_valid, y_pred_lgb),
    'model': lgb_model,
    'y_pred_proba': y_pred_lgb
}

# --- RandomForest ---
print("Treinando RandomForest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(x_train_all, y_train)
y_pred_rf = rf_model.predict_proba(x_valid_all)[:, 1]
y_pred_rf_binary = (y_pred_rf > 0.5).astype(int)
results_benchmark['RandomForest'] = {
    'accuracy': accuracy_score(y_valid, y_pred_rf_binary),
    'precision': precision_score(y_valid, y_pred_rf_binary),
    'recall': recall_score(y_valid, y_pred_rf_binary),
    'f1': f1_score(y_valid, y_pred_rf_binary),
    'auc': roc_auc_score(y_valid, y_pred_rf),
    'model': rf_model,
    'y_pred_proba': y_pred_rf
}

# --- Neural Network Hybrid ---
y_pred_hybrid_binary = (y_pred_hybrid_proba > 0.5).astype(int)
results_benchmark['NeuralNet_Hybrid'] = {
    'accuracy': accuracy_score(y_valid, y_pred_hybrid_binary),
    'precision': precision_score(y_valid, y_pred_hybrid_binary),
    'recall': recall_score(y_valid, y_pred_hybrid_binary),
    'f1': f1_score(y_valid, y_pred_hybrid_binary),
    'auc': roc_auc_score(y_valid, y_pred_hybrid_proba),
    'model': hybrid_nn_model,
    'y_pred_proba': y_pred_hybrid_proba
}

# ============================================================================
# PARTE 8: COMPARAR E EXIBIR RESULTADOS
# ============================================================================

print("\n" + "="*80)
print("RESULTADOS")
print("="*80)

comparison_df = pd.DataFrame({
    model_name: {
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1'],
        'AUC-ROC': results['auc']
    }
    for model_name, results in results_benchmark.items()
}).T

print("\n" + comparison_df.to_string())

best_model_name = comparison_df['AUC-ROC'].idxmax()
best_auc = comparison_df['AUC-ROC'].max()

print(f"\nMelhor modelo: {best_model_name} (AUC = {best_auc:.4f})")

# ============================================================================
# PARTE 9: ENSEMBLE E VISUALIZAÇÕES
# ============================================================================

print("\n[5/5] Criando ensemble e visualizações...")

# --- GRÁFICO 1: Comparação de Modelos ---
print("\n  Gerando gráficos comparativos...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].bar(comparison_df.index, comparison_df['Accuracy'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
axes[0].set_title('Accuracy - Comparação de Modelos', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0, 1])
for i, v in enumerate(comparison_df['Accuracy']):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# AUC-ROC
axes[1].bar(comparison_df.index, comparison_df['AUC-ROC'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
axes[1].set_title('AUC-ROC - Comparação de Modelos', fontweight='bold', fontsize=12)
axes[1].set_ylabel('AUC-ROC')
axes[1].set_ylim([0, 1])
for i, v in enumerate(comparison_df['AUC-ROC']):
    axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_models.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gráfico salvo: comparison_models.png")

# --- GRÁFICO 2: Curvas ROC ---
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for idx, (model_name, results) in enumerate(results_benchmark.items()):
    fpr, tpr, _ = roc_curve(y_valid, results['y_pred_proba'])
    auc = results['auc']
    ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.4f})', linewidth=2.5, color=colors[idx])

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('Curvas ROC - Comparação de Modelos', fontweight='bold', fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gráfico salvo: roc_curves_comparison.png")

# --- GRÁFICO 3: Histórico de Treinamento da NN Hybrid ---
if history_hybrid:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history_hybrid.history['loss'], label='Loss (Treino)', linewidth=2)
    axes[0].plot(history_hybrid.history['val_loss'], label='Loss (Validação)', linewidth=2)
    axes[0].set_title('Loss - Treinamento do Modelo Híbrido', fontweight='bold')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history_hybrid.history['accuracy'], label='Accuracy (Treino)', linewidth=2)
    axes[1].plot(history_hybrid.history['val_accuracy'], label='Accuracy (Validação)', linewidth=2)
    axes[1].set_title('Accuracy - Treinamento do Modelo Híbrido', fontweight='bold')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_hybrid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: training_history_hybrid.png")

# Ensemble por média simples
y_pred_ensemble = np.mean([
    results_benchmark['XGBoost']['y_pred_proba'],
    results_benchmark['LightGBM']['y_pred_proba'],
    results_benchmark['NeuralNet_Hybrid']['y_pred_proba']
], axis=0)

y_pred_ensemble_binary = (y_pred_ensemble > 0.5).astype(int)
ensemble_auc = roc_auc_score(y_valid, y_pred_ensemble)
ensemble_acc = accuracy_score(y_valid, y_pred_ensemble_binary)
ensemble_precision = precision_score(y_valid, y_pred_ensemble_binary)
ensemble_recall = recall_score(y_valid, y_pred_ensemble_binary)
ensemble_f1 = f1_score(y_valid, y_pred_ensemble_binary)

print(f"\nEnsemble (Média):")
print(f"  Accuracy: {ensemble_acc:.4f}")
print(f"  Precision: {ensemble_precision:.4f}")
print(f"  Recall: {ensemble_recall:.4f}")
print(f"  F1-Score: {ensemble_f1:.4f}")
print(f"  AUC-ROC: {ensemble_auc:.4f}")

# Salva resultados
results_final = {
    'benchmark': comparison_df.to_dict(),
    'ensemble_accuracy': ensemble_acc,
    'ensemble_precision': ensemble_precision,
    'ensemble_recall': ensemble_recall,
    'ensemble_f1': ensemble_f1,
    'ensemble_auc': ensemble_auc,
    'best_model': best_model_name,
    'models': {name: res['model'] for name, res in results_benchmark.items()}
}

with open('results_benchmark.pkl', 'wb') as f:
    pickle.dump(results_final, f)

print("\n" + "="*80)
print("EXEUÇÃO CONCLUÍDA COM SUCESSO!")
print("="*80)
print("\nArquivos gerados:")
print("  1. hybrid_model_architecture.png - Visualização da arquitetura NN Híbrida")
print("  2. comparison_models.png - Comparação de Accuracy e AUC-ROC")
print("  3. roc_curves_comparison.png - Curvas ROC de todos os modelos")
print("  4. training_history_hybrid.png - Histórico de treinamento (Loss e Accuracy)")
print("  5. results_benchmark.pkl - Resultados em formato serializado")
print("\nResultados salvos em:", os.getcwd())
print("="*80)
