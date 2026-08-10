import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

dir_path = os.path.expanduser("~/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/")

benigns = ["output-of-benign-pcap-0.csv", "output-of-benign-pcap-1.csv", "output-of-benign-pcap-2.csv", "output-of-benign-pcap-3.csv"]

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

cols_remove = ['Unnamed: 0', 'flow_id', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
df = df.drop(columns=cols_remove)

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

comportamental_features = size_based_features + delta_length_based_features + delta_time_based_features + side_based_features
protocolo_features = resource_record_based_features + statistical_based_features

df_train, df_valid = train_test_split(df, test_size=0.3, random_state=42, stratify=df['maligno'])

y_train = df_train['maligno'].values.astype(np.float32)
y_valid = df_valid['maligno'].values.astype(np.float32)

x_train_text_lex = df_train[lexical_text_features].fillna('unknown').astype(str)
x_valid_text_lex = df_valid[lexical_text_features].fillna('unknown').astype(str)

x_train_text_lex_seq = (x_train_text_lex['dns_domain_name'] + '_' + x_train_text_lex['dns_top_level_domain']).values
x_valid_text_lex_seq = (x_valid_text_lex['dns_domain_name'] + '_' + x_valid_text_lex['dns_top_level_domain']).values

x_train_num_lex = df_train[lexical_numeric_features].copy()
x_valid_num_lex = df_valid[lexical_numeric_features].copy()

for col in x_train_num_lex.columns:
    x_train_num_lex[col] = pd.to_numeric(x_train_num_lex[col], errors='coerce')
    x_valid_num_lex[col] = pd.to_numeric(x_valid_num_lex[col], errors='coerce')

x_train_num_lex = x_train_num_lex.fillna(x_train_num_lex.median(numeric_only=True))
x_valid_num_lex = x_valid_num_lex.fillna(x_train_num_lex.median(numeric_only=True))

x_train_num_lex = x_train_num_lex.astype(np.float32)
x_valid_num_lex = x_valid_num_lex.astype(np.float32)

x_train_num_lex['character_entropy'] = np.log1p(x_train_num_lex['character_entropy'])
x_valid_num_lex['character_entropy'] = np.log1p(x_valid_num_lex['character_entropy'])

ss_lex = StandardScaler()
x_train_num_lex = ss_lex.fit_transform(x_train_num_lex)
x_valid_num_lex = ss_lex.transform(x_valid_num_lex)

vt_lex = VarianceThreshold(threshold=0.001)
x_train_num_lex = vt_lex.fit_transform(x_train_num_lex)
x_valid_num_lex = vt_lex.transform(x_valid_num_lex)

x_train_comp = df_train[comportamental_features].copy()
x_valid_comp = df_valid[comportamental_features].copy()

x_train_comp['handshake_duration'] = x_train_comp['handshake_duration'].replace('not a tcp connection', -1)
x_valid_comp['handshake_duration'] = x_valid_comp['handshake_duration'].replace('not a tcp connection', -1)
x_train_comp['delta_start'] = x_train_comp['delta_start'].replace('not a tcp connection', -1)
x_valid_comp['delta_start'] = x_valid_comp['delta_start'].replace('not a tcp connection', -1)

for col in x_train_comp.columns:
    x_train_comp[col] = pd.to_numeric(x_train_comp[col], errors='coerce')
    x_valid_comp[col] = pd.to_numeric(x_valid_comp[col], errors='coerce')

x_train_comp = x_train_comp.fillna(0).astype(np.float32)
x_valid_comp = x_valid_comp.fillna(0).astype(np.float32)

robust_scaler_comp = RobustScaler()
x_train_comp = robust_scaler_comp.fit_transform(x_train_comp)
x_valid_comp = robust_scaler_comp.transform(x_valid_comp)

vt_comp = VarianceThreshold(threshold=0.01)
x_train_comp = vt_comp.fit_transform(x_train_comp)
x_valid_comp = vt_comp.transform(x_valid_comp)

x_train_prot = df_train[protocolo_features].copy()
x_valid_prot = df_valid[protocolo_features].copy()

for col in x_train_prot.columns:
    x_train_prot[col] = pd.to_numeric(x_train_prot[col], errors='coerce')
    x_valid_prot[col] = pd.to_numeric(x_valid_prot[col], errors='coerce')

x_train_prot = x_train_prot.fillna(0).astype(np.float32)
x_valid_prot = x_valid_prot.fillna(0).astype(np.float32)

ss_prot = StandardScaler()
x_train_prot = ss_prot.fit_transform(x_train_prot)
x_valid_prot = ss_prot.transform(x_valid_prot)

vt_prot = VarianceThreshold(threshold=0.05)
x_train_prot = vt_prot.fit_transform(x_train_prot)
x_valid_prot = vt_prot.transform(x_valid_prot)

x_train_all = np.hstack([x_train_num_lex, x_train_comp, x_train_prot])
x_valid_all = np.hstack([x_valid_num_lex, x_valid_comp, x_valid_prot])

def text_to_charseq(text_seq, max_len=100, vocab_size=256):
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

def hybrid_model_wide_and_deep(lexical_text_dim, lexical_num_dim, comportamental_dim, protocolo_dim):
    input_text = layers.Input(shape=(lexical_text_dim,), name='input_text', dtype=tf.int32)
    
    x_text = layers.Embedding(256, 16, input_length=lexical_text_dim)(input_text)
    x_text = layers.Conv1D(32, 3, activation='relu', padding='same')(x_text)
    x_text = layers.GlobalAveragePooling1D()(x_text)
    
    input_lex_num = layers.Input(shape=(lexical_num_dim,), name='input_lex_num')
    input_comp = layers.Input(shape=(comportamental_dim,), name='input_comp')
    input_prot = layers.Input(shape=(protocolo_dim,), name='input_prot')
    
    merged = layers.Concatenate(name='early_fusion')([
        x_text,
        input_lex_num,
        input_comp,
        input_prot
    ])
    
    x = layers.Dense(128, activation='relu', name='dense_128')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu', name='dense_64')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(32, activation='relu', name='dense_32')(x)
    x = layers.Dropout(0.1)(x)
    
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(
        inputs=[input_text, input_lex_num, input_comp, input_prot], 
        outputs=output, 
        name='Hybrid_Wide_and_Deep'
    )
    
    return model

hybrid_nn_model = hybrid_model_wide_and_deep(
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

class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
)

history_hybrid = hybrid_nn_model.fit(
    [x_train_text_encoded, x_train_num_lex, x_train_comp, x_train_prot],
    y_train,
    validation_data=(
        [x_valid_text_encoded, x_valid_num_lex, x_valid_comp, x_valid_prot],
        y_valid
    ),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

y_pred_hybrid_proba = hybrid_nn_model.predict(
    [x_valid_text_encoded, x_valid_num_lex, x_valid_comp, x_valid_prot],
    verbose=0
).ravel()

results_benchmark = {}

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

best_model_name = comparison_df['AUC-ROC'].idxmax()

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