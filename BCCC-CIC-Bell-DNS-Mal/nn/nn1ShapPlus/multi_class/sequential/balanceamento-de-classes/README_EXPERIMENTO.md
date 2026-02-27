# Experimento MLP Multiclasse com Balanceamento de Classes

## Descrição

Este script realiza experimentos com uma rede neural MLP (Multi-Layer Perceptron) para classificação multiclasse de tráfego DNS, com foco em **balanceamento entre classes benignas e malignas**.

## Metodologia

### Balanceamento de Classes

O experimento implementa a seguinte estratégia de balanceamento:

1. **Dataset completo**: carrega todos os dados benignos e malignos
2. **Contagem de malignos**: soma total de todas as amostras malignas (Malware + Phishing + Spam)
3. **Amostragem aleatória**: seleciona **aleatoriamente** um número de amostras benignas igual ao total de malignos
4. **Preservação**: mantém **todas** as amostras malignas (nenhuma é removida)

### Repetições

O experimento é repetido **10 vezes**, onde:
- Cada repetição seleciona uma amostra aleatória diferente de dados benignos
- Isso garante 10 subdatasets balanceados diferentes
- A seed aleatória é variada para cada repetição: `42 + repetition`

### Split de Dados

Para cada repetição:
- **70% treino / 30% teste**
- Split **estratificado** para manter proporções das classes

### Arquitetura do Modelo

Rede neural sequencial:
```
Input → Dense(128, relu) → Dropout(0.3) → 
Dense(64, relu) → Dropout(0.3) → 
Dense(32, relu) → 
Dense(4, softmax)
```

### Treinamento

- **Otimizador**: Adam
- **Loss**: sparse_categorical_crossentropy
- **Épocas**: até 50 (com early stopping)
- **Batch size**: 32
- **Callbacks**:
  - Early Stopping (patience=10, monitora val_loss)
  - ReduceLROnPlateau (patience=5, factor=0.5)

## Métricas Avaliadas

### Métricas Gerais (weighted average)
- Acurácia
- Precisão
- Recall
- F1-Score
- AUC-ROC

### Métricas por Classe
- Precisão, Recall e F1-Score para cada uma das 4 classes:
  - Benigno
  - Malware
  - Phishing
  - Spam

### Tempos de Execução
- Tempo de treinamento
- Tempo de teste/inferência

## Arquivos Gerados

### Resultados
- `mlp_multiclass_10_repetitions_results.csv` - resultados detalhados de todas as repetições

### Modelos e Históricos
- `model_repetition_[1-10].keras` - 10 modelos treinados (um por repetição)
- `history_repetition_[1-10].pkl` - históricos de treinamento

### Visualizações
- `mlp_multiclass_10reps_overview.png` - visão geral das métricas em todas as repetições
- `mlp_multiclass_10reps_f1_per_class.png` - F1-Score por classe em todas as repetições
- `mlp_multiclass_10reps_boxplots.png` - boxplots comparativos de métricas

## Pré-requisitos

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib
```

## Uso

```bash
python3 nn1-classes-balanceadas.py
```

## Análise dos Resultados

O script automaticamente:
1. Calcula médias e desvios padrão de todas as métricas
2. Identifica a melhor e pior repetição (baseado em F1-Score)
3. Gera visualizações comparativas
4. Salva todos os resultados em CSV para análise posterior

## Estrutura do CSV de Resultados

Cada linha representa uma repetição e contém:
- Metadados: repetition, random_state, train_samples, test_samples, n_features
- Tempos: train_time, test_time
- Épocas: epochs
- Métricas gerais: accuracy, precision, recall, f1_score, auc_roc
- Métricas por classe: precision_*, recall_*, f1_* (para cada classe)

## Observações

- O balanceamento é feito **antes** do split treino/teste
- Cada repetição usa uma amostra diferente de benignos para garantir robustez
- A estratificação no split garante que as proporções das classes sejam mantidas em treino e teste
- O pré-processamento inclui normalização (StandardScaler) e remoção de features com baixa variância


Script desenvolvido para experimentos de detecção de intrusões em tráfego DNS
