# Detecção de Intrusões DNS com Redes Neurais (Pipeline do modelo em produção em tempo real)

Este repositório contém o pipeline automatizado para a inferência em tempo real e detecção de tráfego DNS anômalo/malicioso. O sistema utiliza uma arquitetura que une a análise comportamental de tráfego de rede à análise lexical de domínios através de modelos de Machine Learning (Redes Neurais).

## Como o Pipeline funciona

O fluxo de processamento é sequencial e composto pelas seguintes etapas:

1. **Estímulo e captura (Bash):** O script `analisador.sh` identifica o gerenciador de cache local do sistema operativo e o limpa. Em seguida, dispara requisições de resolução ao domínio alvo e grava o tráfego gerado (PCAP) através do `tcpdump`.
2. **Extração base de rede:** A ferramenta `ALFlowLyzer` lê a captura bruta e agrupa as transações, extraindo features e variáveis contínuas do tráfego (latência, tamanho de bytes, quantidade de pacotes).
3. **Análise lexical (Python):** O script `inferencia.py` processa e calcula as métricas textuais do domínio alvo, incluindo Entropia, N-grams (uni, bi, tri) e distribuições numéricas e de caracteres.
4. **Transformação de dados (Python):** Aplicação da persistência de objetos. O sistema carrega o modelo treinado previamente (modelo neural, `LabelEncoder`, `MinMaxScaler` e Selector) para garantir que os novos dados obedeçam estritamente à topologia matemática da matriz de treinamento original, preenchendo os dados de forma padronizada.
5. **Inferência (TensorFlow/Keras):** O vetor numérico processado é passado para a Rede Neural, que classifica cada tráfego como benigno (0) ou malicioso (1).

## Estrutura de Arquivos

Para o funcionamento do pipeline, a raiz do projeto deve conter:

*   `analisador.sh`: O orquestrador das ferramentas e capturas.
*   `inferencia.py`: Script de manipulação, extração lexical e inferência em ML.
*   `ALFlowLyzer`: Ferramenta ou diretório contendo o binário extrator de pacotes de rede.
*   Objetos Treinados do Modelo:
    *   `dns_intrusion_model_nn1.keras`
    *   `scaler.pkl`
    *   `selector.pkl`
    *   `label_encoders.pkl`

## Como Executar

O orquestrador requer privilégios de superusuário (`sudo`) devido às necessidades de captura do tráfego direto da interface de rede (via `tcpdump`) e manipulação dos serviços de sistema para limpeza do cache DNS.

**Sintaxe de execução:**
```bash
sudo ./analisador.sh <dominio_alvo>
```

Exemplo de uso:
```bash
sudo ./analisador.sh iuqerfsodp9ifjaposdfjhgosurijfaewrwergwea.com
```

## Artefatos Gerados

A cada execução bem-sucedida, o pipeline gera os seguintes arquivos de auditoria para garantir a reprodutibilidade e a integridade da análise:

    <dominio>.pcap: O arquivo de tráfego bruto (packet capture).

    output...csv: A saída bruta gerada unicamente pelo ALFlowLyzer (restrita à análise comportamental).

    output..._NATURAL.csv: O arquivo de auditoria em linguagem legível, contendo as features de rede unificadas às features lexicais textuais geradas pelo Python, antes da conversão para tensores matemáticos.

    output..._LIMPO.csv: A matriz final codificada (pós-LabelEncoder e MinMaxScaler) que foi inserida no modelo neural.