import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import os

# Reduz logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# carregamento e rotulamento dos dados
data_dir = "../"

benigns = ["output-of-benign-pcap-0.csv",
           "output-of-benign-pcap-1.csv",
           "output-of-benign-pcap-2.csv",
           "output-of-benign-pcap-3.csv"]

df_benigns = [pd.read_csv(data_dir + f, nrows=100) for f in benigns]
df_benign = pd.concat(df_benigns)
df_benign["maligno"] = 0
df_benign["tipo_maligno"] = "Benigno"

df_malware = pd.read_csv(data_dir + "output-of-malware-pcap.csv", nrows=400)
df_malware['maligno'] = 1
df_malware['tipo_maligno'] = 'Malware'

df_phishing = pd.read_csv(data_dir + "output-of-phishing-pcap.csv", nrows=400)
df_phishing['maligno'] = 1
df_phishing['tipo_maligno'] = 'Phishing'

df_spam = pd.read_csv(data_dir + "output-of-spam-pcap.csv", nrows=400)
df_spam['maligno'] = 1
df_spam['tipo_maligno'] = 'Spam'

df = pd.concat([df_benign, df_malware, df_phishing, df_spam])

# separar features e rótulos
X = df.drop(columns=['maligno', 'tipo_maligno'])
y_bin = df['maligno'].to_numpy()

# Pré-processamento
cat_cols = [col for col in X.columns if X[col].dtype == 'object' and X[col].nunique() < 50]
X = pd.get_dummies(X, columns=cat_cols)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = X.astype(np.float32).to_numpy()

# dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.3, random_state=42
)

print(f"Classes únicas: {np.unique(y_train)}")
print(f"Shape dos dados de treino: {X_train.shape}")
print(f"Shape dos labels de treino: {y_train.shape}")

# Definir função de construção do modelo para Keras Tuner
def build_model(hp):
    model = keras.Sequential()
    
    # Camada de entrada
    model.add(layers.Input(shape=(X_train.shape[1],)))
    
    # Primeira camada oculta
    model.add(layers.Dense(
        units=hp.Int('units_1', min_value=32, max_value=512, step=32),
        activation=hp.Choice('activation_1', values=['relu', 'tanh'])
    ))
    model.add(layers.Dropout(hp.Float('dropout_1', 0, 0.5, step=0.1)))
    
    # Segunda camada oculta (opcional)
    if hp.Boolean('second_layer'):
        model.add(layers.Dense(
            units=hp.Int('units_2', min_value=32, max_value=256, step=32),
            activation=hp.Choice('activation_2', values=['relu', 'tanh'])
        ))
        model.add(layers.Dropout(hp.Float('dropout_2', 0, 0.5, step=0.1)))
    
    # Camada de saída para classificação binária
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model

# Configurar o tuner (RandomSearch)
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=1,
    directory='keras_tuner_results',
    project_name='dns_intrusion_detection',
    overwrite=True
)

print("\n" + "="*50)
print("Iniciando busca de hiperparâmetros com Keras Tuner")
print("="*50 + "\n")

# Executar a busca
tuner.search(
    X_train, y_train,
    epochs=10,
    validation_split=0.2,
    verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# Obter o melhor modelo
best_model = tuner.get_best_models(num_models=1)[0]

# Exibir os melhores hiperparâmetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n" + "="*50)
print("Melhores hiperparâmetros encontrados:")
print("="*50)
print(f"units_1: {best_hps.get('units_1')}")
print(f"activation_1: {best_hps.get('activation_1')}")
print(f"dropout_1: {best_hps.get('dropout_1')}")
print(f"second_layer: {best_hps.get('second_layer')}")
if best_hps.get('second_layer'):
    print(f"units_2: {best_hps.get('units_2')}")
    print(f"activation_2: {best_hps.get('activation_2')}")
    print(f"dropout_2: {best_hps.get('dropout_2')}")
print(f"learning_rate: {best_hps.get('learning_rate')}")

# Fazer previsões
print("\n" + "="*50)
print("Avaliando o melhor modelo")
print("="*50 + "\n")

y_pred_prob = best_model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Avaliar
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno']))

# Avaliar no conjunto de teste
test_loss, test_accuracy, test_auc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Salvar o melhor modelo
best_model.save('best_model_keras_tuner.keras')
print("\nModelo salvo como 'best_model_keras_tuner.keras'")
