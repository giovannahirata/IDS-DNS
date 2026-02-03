import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# carrega modelo e pré-processadores
model = tf.keras.models.load_model('dns_intrusion_model_nn1.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('selector.pkl', 'rb') as f:
    selector = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

def preprocess_new_data(df_new):
    """Pré-processa dados para inferencia/predição"""
    
    # remove colunas desnecessárias
    cols_remove = ['Unnamed: 0', 'flow_id', 'timestamp', 'src_ip', 'dst_ip']
    df_new = df_new.drop(columns=cols_remove, errors='ignore')
    df_new = df_new.fillna(0)
    
    # encoding categórico
    categorical_cols = df_new.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            # trata valores desconhecidos
            df_new[col] = df_new[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # normalização
    numerical_cols = df_new.select_dtypes(include=['int64', 'float64']).columns
    df_new[numerical_cols] = scaler.transform(df_new[numerical_cols])
    
    # seleção de features
    x_new = selector.transform(df_new)
    
    return np.array(x_new, dtype=np.float32)

def predict(df_new):
    """Faz inferencias/predições nos dados"""
    x_processed = preprocess_new_data(df_new)
    predictions = model.predict(x_processed)
    return (predictions > 0.5).astype(int).ravel()

if __name__ == "__main__":
    # carregar dados
    df_test = pd.read_csv("../output-of-benign-pcap-0.csv", nrows=10)
    
    # faz inferencias/predições
    predictions = predict(df_test)
    
    print("Predições:")
    print(predictions)
    print(f"\nBenigno: {np.sum(predictions == 0)}")
    print(f"Maligno: {np.sum(predictions == 1)}")