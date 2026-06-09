# aqui vou testar uma arquitetura hibrida de rede neural considerando
# as arquiteturas mais apropriadas para cada grupo de features baseando-se
# em suas propriedades (tipos de informação), são elas: features estatisticas,
# lexicais e de fluxo

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, roc_curve
import pickle
import shap
import signal
import sys

def feature_indices(features, category_features):
    """
    funcao que retorna os indices das features de uma categoria na 
    lista de features
    """
    indices=[]
    for feat in category_features:
        try:
            id = features.index(feat)
            indices.append(id)
        except ValueError:
            pass
    return indices

def hybrid_model(lexical_dim, comportamental_dim, protocolo_dim):
    """
    constrói o modelo híbrido com 3 ramos de especialização baseados em features:
    - ramo lexical (CNN 1D): para análise de padrões textuais
    - ramo comportamental (LSTM + MLP): para análise de séries temporais e comportamento
    - ramo protocolo (MLP): para metadados do protocolo DNS
    """

    # ramo lexical:
    if lexical_dim>0:
        input_lexical = layers.Input(shape=(lexical_dim,), name='input_lexical')

        # CNN1D (samples, steps, channels)
        x_lex = layers.Reshape((lexical_dim, 1))(input_lexical)

        # com filtros especializados em detectar padrões
        x_lex = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same',
                              name='cnn1d_1')(x_lex)
        x_lex = layers.BatchNormalization()(x_lex)
        x_lex = layers.MaxPooling1D(pool_size=2, padding='same')(x_lex)

        x_lex = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same',
                              name='cnn1d_2')(x_lex)
        x_lex = layers.BatchNormalization()(x_lex)
        x_lex = layers.GlobalAveragePooling1D()(x_lex)

        # abstração com mlp
        x_lex = layers.Dense(64, activation='relu', name='dense_lex_1')(x_lex)
        x_lex = layers.Dropout(0.3)(x_lex)
        x_lex = layers.Dense(32, activation='relu', name='dense_lex_2')(x_lex)
        x_lex = layers.Dropout(0.2)(x_lex)

        branch_lexical_output = x_lex

    else:
        input_lexical = None
        branch_lexical_output = None

    # ramo comportamental
    if comportamental_dim>0:
        input_comportamental = layers.Input(shape=(comportamental_dim,), name='input_comportamental')

        # reshape para LSTM - adiciona dimensao temporal
        x_comp = layers.Reshape((comportamental_dim, 1))(input_comportamental)

        # captura de dependências temporais com LSTM
        x_comp = layers.LSTM(64, return_sequences=True, name='lstm_1')(x_comp)
        x_comp = layers.Dropout(0.3)(x_comp)
        x_comp = layers.LSTM(32, return_sequences=False, name='lstm_2')(x_comp)
        x_comp = layers.Dropout(0.3)(x_comp)

        # refinamento com mlp
        x_comp = layers.Dense(128, activation='relu', name='dense_comp_1')(x_comp)
        x_comp = layers.BatchNormalization()(x_comp)
        x_comp = layers.Dropout(0.3)(x_comp)
        x_comp = layers.Dense(64, activation='relu', name='dense_comp_2')(x_comp)
        x_comp = layers.Dropout(0.2)(x_comp)
        x_comp = layers.Dense(32, activation='relu', name='dense_comp_3')(x_comp)

        branch_comportamental_output = x_comp

    else:
        input_comportamental = None
        branch_comportamental_output = None

    # ramo protocolo
    if protocolo_dim>0:
        input_protocolo = layers.Input(shape=(protocolo_dim,), name='input_protocolo')

        x_prot = layers.Dense(96, activation='relu', name='dense_prot_1')(input_protocolo)
        x_prot = layers.BatchNormalization()(x_prot)
        x_prot = layers.Dropout(0.3)(x_prot)
        x_prot = layers.Dense(64, activation='relu', name='dense_prot_2')(x_prot)
        x_prot = layers.Dropout(0.2)(x_prot)
        x_prot = layers.Dense(32, activation='relu', name='dense_prot_3')(x_prot)

        branch_protocolo_output = x_prot

    else:
        input_protocolo = None
        branch_protocolo_output = None

    # fusão dos outputs de todos os ramos no modelo
    branches_outputs=[]
    inputs=[]

    if input_lexical is not None:
        branches_outputs.append(branch_lexical_output)
        inputs.append(input_lexical)

    if input_comportamental is not None:
        branches_outputs.append(branch_comportamental_output)
        inputs.append(input_comportamental)

    if input_protocolo is not None:
        branches_outputs.append(branch_protocolo_output)
        inputs.append(input_protocolo)

    merged = layers.Concatenate(name='concatenate_branches')(branches_outputs)

    # camadas finais pra fusão
    x = layers.Dense(128, activation='relu', name='fusion_dense_1')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu', name='fusion_dense_2')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu', name='fusion_dense_3')(x)
    x = layers.Dropout(0.1)(x)

    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=output, name='HybridBinaryClassifier')

    return model


# carregamento e rotulagem dos dados

dir_path = "~/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/"

benigns = ["output-of-benign-pcap-0.csv",
           "output-of-benign-pcap-1.csv",
           "output-of-benign-pcap-2.csv",
           "output-of-benign-pcap-3.csv"]

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

df = pd.concat([df_benign, df_malware, df_phishing, df_spam])

print(f"\nAlgumas estatísticas do dataset:")
print(f"Total de amostras: {len(df)}")
print(f"Distribuição de classes:\n{df['maligno'].value_counts()}")
print(f"Proporção: {df['maligno'].value_counts(normalize=True)}")

cols_remove = ['Unnamed: 0', 'flow_id', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']

df = df.drop(columns=cols_remove)

lexical_based_features = [
    'dns_domain_name', 'dns_top_level_domain', 'dns_second_level_domain',
    'dns_domain_name_length', 'dns_subdomain_name_length',
    'uni_gram_domain_name', 'bi_gram_domain_name', 'tri_gram_domain_name',
    'numerical_percentage', 'character_distribution', 'character_entropy',
    'max_continuous_numeric_len', 'max_continuous_alphabet_len',
    'max_continuous_consonants_len', 'max_continuous_same_alphabet_len',
    'vowels_consonant_ratio', 'conv_freq_vowels_consonants'
]

size_based_features = size_features = [
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

comportamental_features = size_based_features + delta_length_based_features + delta_time_based_features + side_based_features
protocolo_features = resource_record_based_features + statistical_based_features

df = df.fillna(0)

# identifica colunas categoricas:
categorical_cols = df.select_dtypes(include=['object']).columns

# encoding de variaveis categoricas
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

x_train = df_train.drop(columns=["maligno", "tipo_maligno", "label"])
x_valid = df_valid.drop(columns=["maligno", "tipo_maligno", "label"])
y_train_bin = df_train["maligno"]
y_valid_bin = df_valid["maligno"]

print(f"\nFeatures originais: {x_train.shape[1]}")

# salva nomes das features
original_feature_names = list(x_train.columns)

# identifica coluna numericas:
numerical_cols = x_train.select_dtypes(include=['int64', 'float64']).columns
   
# normalização
scaler = StandardScaler()
x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
x_valid[numerical_cols] = scaler.transform(x_valid[numerical_cols])

# remove features com variancia zero ou muito baixa
selector = VarianceThreshold(threshold=0.01)
x_train_transformed = selector.fit_transform(x_train)
x_valid_transformed = selector.transform(x_valid)

# obtém nomes das features que foram mantidas após VarianceThreshold
selected_indices = selector.get_support(indices=True)
selected_feature_names = [original_feature_names[i] for i in selected_indices]

# verifica e corrige nomes duplicados (adiciona índice se houver duplicatas)
name_counts = {}
unique_feature_names = []
for name in selected_feature_names:
    if name in name_counts:
        name_counts[name] += 1
        unique_feature_names.append(f"{name}_{name_counts[name]}")
    else:
        name_counts[name] = 0
        unique_feature_names.append(name)

# verifica se há duplicatas
duplicates = [name for name, count in name_counts.items() if count > 0]
if duplicates:
    print(f"\nFeatures duplicadas encontradas e renomeadas: {duplicates}")
    selected_feature_names = unique_feature_names

print(f"Features após remoção: {x_train_transformed.shape[1]}")
print(f"Features selecionadas: {len(selected_feature_names)}")
print(f"Primeiras 10 features: {selected_feature_names[:10]}")


lexical_indices = feature_indices(selected_feature_names, lexical_based_features)
comportamental_indices = feature_indices(selected_feature_names, comportamental_features)
protocolo_indices = feature_indices(selected_feature_names, protocolo_features)

print("\nDistribuição de features por categoria")
print(f"Ramo lexical: {len(lexical_indices)} features")
print(f"Ramo comportamental: {len(comportamental_indices)} features")
print(f"Ramo protocolo: {len(protocolo_indices)} features")
print(f"Total: {len(lexical_indices)+len(comportamental_indices)+len(protocolo_indices)} features")

x_train = np.array(x_train_transformed, dtype=np.float32)
x_valid = np.array(x_valid_transformed, dtype=np.float32)
y_train_bin = np.array(y_train_bin, dtype=np.float32)
y_valid_bin = np.array(y_valid_bin, dtype=np.float32)

x_train_lexical = x_train[:, lexical_indices] if lexical_indices else np.array([]).reshape(x_train.shape[0],0)
x_valid_lexical = x_valid[:, lexical_indices] if lexical_indices else np.array([]).reshape(x_valid.shape[0],0)

x_train_comportamental = x_train[:, comportamental_indices] if comportamental_indices else np.array([]).reshape(x_train.shape[0],0)
x_valid_comportamental = x_valid[:, comportamental_indices] if comportamental_indices else np.array([]).reshape(x_valid.shape[0],0)

x_train_protocolo = x_train[:, protocolo_indices] if protocolo_indices else np.array([]).reshape(x_train.shape[0],0)
x_valid_protocolo = x_valid[:, protocolo_indices] if protocolo_indices else np.array([]).reshape(x_valid.shape[0],0)

print("\nShapes dos dados de treino:")
print(f"    Lexical: \t{x_train_lexical.shape}")
print(f"    Comportamental: \t{x_train_comportamental.shape}")
print(f"    Protocolo: {x_train_protocolo.shape}")

model = hybrid_model(
    lexical_dim=x_train_lexical.shape[1],
    comportamental_dim=x_train_comportamental.shape[1],
    protocolo_dim=x_train_protocolo.shape[1]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\nArquitetura do modelo híbrido")
model.summary()

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1
)

print("\nIniciando treinamento...")

history = model.fit(
    [x_train_lexical, x_train_comportamental, x_train_protocolo],
    y_train_bin,
    validation_data=(
        [x_valid_lexical, x_valid_comportamental, x_valid_protocolo],
        y_valid_bin
    ),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\nAvaliação do modelo:")
loss, accuracy, precision, recall = model.evaluate(
    [x_valid_lexical, x_valid_comportamental, x_valid_protocolo],
    y_valid_bin,
    verbose=0
)

print(f"\nAcurácia: {accuracy:.4f}")
print(f"\nPrecisão: {precision:.4f}")
print(f"\nRecall: {recall:.4f}")

y_pred_proba = model.predict([x_valid_lexical, x_valid_comportamental, x_valid_protocolo]).ravel()
y_pred = model.predict([x_valid_lexical, x_valid_comportamental, x_valid_protocolo])
y_pred_class = (y_pred>0.5).astype(int)

auc_score = roc_auc_score(y_valid_bin, y_pred_proba)
print(f'\nAUC-ROC Score: {auc_score:.4f}')

print('\nMatriz de confusão:')
cm = confusion_matrix(y_valid_bin, y_pred_class)
print(cm)

print('\nRelatório de classificação:')
print(classification_report(y_valid_bin, y_pred_class,
                            target_names=['Benigno', 'Maligno']))


print('\nValidação cruzada (5-Fold)')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores=[]
cv_accuracies=[]

for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train_bin)):
    print(f'\nFold {fold+1}/5', end=' ... ')

    X_fold_lex_train, X_fold_lex_val = x_train_lexical[train_idx], x_train_lexical[val_idx]
    X_fold_comp_train, X_fold_comp_val = x_train_comportamental[train_idx], x_train_comportamental[val_idx]
    X_fold_prot_train, X_fold_prot_val = x_train_protocolo[train_idx], x_train_protocolo[val_idx]
    y_fold_train, y_fold_val = y_train_bin[train_idx], y_train_bin[val_idx]

    model_fold = hybrid_model(
        lexical_dim=x_train_lexical.shape[1],
        comportamental_dim=x_train_comportamental.shape[1],
        protocolo_dim=x_train_protocolo.shape[1]
    )

    model_fold.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    history_fold = model_fold.fit(
        [X_fold_lex_train, X_fold_comp_train, X_fold_prot_train],
        y_fold_train,
        validation_data=(
            [X_fold_lex_val, X_fold_comp_val, X_fold_prot_val],
            y_fold_val,
        ),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    score = model_fold.evaluate(
        [X_fold_lex_val, X_fold_comp_val, X_fold_prot_val],
        y_fold_val,
        verbose=0
    )

    cv_scores.append(score[0])
    cv_accuracies.append(score[1])
    print(f'Acurácia: {score[1]:.4f}')

print(f'\nAcurácia média: {np.mean(cv_accuracies):.4f} (+/- {np.std(cv_accuracies):.4f})')
print(f'Melhor fold: {np.argmax(cv_accuracies)+1} com {np.max(cv_accuracies):.4f}')
print(f'Pior fold: {np.argmin(cv_accuracies)+1} com {np.min(cv_accuracies):.4f}')
print(f'Variação: {np.max(cv_accuracies)-np.min(cv_accuracies):.4f}')

print('\nVisualizações')

# loss e acurácia

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Treino', linewidth=2)
plt.plot(history.history['val_loss'], label='Validação', linewidth=2)
plt.title('Loss durante o Treinamento (Modelo Híbrido)', fontsize=12, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Treino', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validação', linewidth=2)
plt.title('Acurácia durante o Treinamento (Modelo Híbrido)', fontsize=12, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_hybrid.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gráfico de treinamento salvo: training_history_hybrid.png")

# curva ROC

fpr, tpr, thresholds = roc_curve(y_valid_bin, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Modelo Híbrido')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve_hybrid.png', dpi=150, bbox_inches='tight')
plt.close()
print("Curva ROC salva: roc_curve_hybrid.png")

# matriz de confusão

from sklearn.metrics import ConfusionMatrixDisplay
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benigno', 'Maligno'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusão - Modelo Híbrido')
plt.savefig('confusion_matrix_hybrid.png', dpi=150, bbox_inches='tight')
plt.close()
print("Matriz de confusão salva: confusion_matrix_hybrid.png")

# análise SHAP
print("\nAnálise SHAP (com amostragem reduzida para evitar timeout)")

BACKGROUND_SAMPLES = 50
TEST_SAMPLES = 10
TIMEOUT_SHAP = 300

def timeout_handler(signum, frame):
    print("\n[TIMEOUT] Análise SHAP excedeu o tempo limite. Abortando...")
    sys.exit(1)

try:
    x_train_comp_sample = x_train_comportamental[:BACKGROUND_SAMPLES]
    x_valid_comp_sample = x_valid_comportamental[:TEST_SAMPLES]
    
    print(f"Background: {len(x_train_comp_sample)} amostras")
    print(f"Teste: {len(x_valid_comp_sample)} amostras")
    print(f"Timeout SHAP: {TIMEOUT_SHAP}s")
    
    # define timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SHAP)
    
    print("\nGerando explicabilidade para Ramo Comportamental...")
    
    model_behavior_only = keras.Model(
        inputs=model.input[1],
        outputs=model.layers[-6].output
    )
    
    explainer_behavior = shap.DeepExplainer(model_behavior_only, x_train_comp_sample)
    shap_values_behavior = explainer_behavior.shap_values(x_valid_comp_sample)
    
    signal.alarm(0)  # cancela o timeout
    
    if isinstance(shap_values_behavior, list):
        shap_array_behavior = shap_values_behavior[0]
    else:
        shap_array_behavior = shap_values_behavior
    
    if len(shap_array_behavior.shape) == 3 and shap_array_behavior.shape[2] == 1:
        shap_array_behavior = shap_array_behavior[:, :, 0]
    
    comportamental_feature_names = [selected_feature_names[i] for i in comportamental_indices]
    
    # summary plot
    print("Salvando SHAP summary plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_array_behavior, x_valid_comp_sample,
                      feature_names=comportamental_feature_names,
                      max_display=15, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_behavior_branch.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP summary plot salvo: shap_summary_behavior_branch.png")

    print("Salvando SHAP bar plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_array_behavior, x_valid_comp_sample,
                      feature_names=comportamental_feature_names,
                      max_display=15, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_bar_behavior_branch.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP bar plot salvo: shap_bar_behavior_branch.png")
    print("\nAnálise SHAP concluída com sucesso")

except Exception as e:
    signal.alarm(0)  # cancela o timeout em caso de erro
    print(f"\nErro na análise SHAP: {e}")
    print("  Continuando com o resto do pipeline...")



# salva modelo
print('\nSalvando o modelo')
model.save('dns_intrusion_model_hybrid.keras')
print("Modelo salvo: dns_intrusion_model_hybrid.keras")

# salva informações sobre os ramos
ramo_info = {
    'lexical_indices': lexical_indices,
    'comportamental_indices': comportamental_indices,
    'protocolo_indices': protocolo_indices,
    'lexical_features': [selected_feature_names[i] for i in lexical_indices],
    'comportamental_features': [selected_feature_names[i] for i in comportamental_indices],
    'protocolo_features': [selected_feature_names[i] for i in protocolo_indices]
}

with open('ramo_indices.pkl', 'wb') as f:
    pickle.dump(ramo_info, f)

print("Informações sobre os ramos salvos (ramo_info.pkl)")

# salva o scaler e selector para pre-processamento
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('selector.pkl', 'wb') as f:
    pickle.dump(selector, f)
    
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
    
print("Pré-processadores salvos via Pickle (scaler.pkl, selector.pkl, label_encoders.pkl)")