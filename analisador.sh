#!/bin/bash

DOMINIO=$1

if [ -z "$DOMINIO" ]; then
    echo "Erro: Forneça um domínio alvo."
    echo "Uso: sudo ./analisador.sh <dominio>"
    exit 1
fi

# configs de ambiente

PROJ_DIR="/home/giovanna/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais"
# INTERFACE="wlp0s20f3" # da minha máquina (para teste)
INTERFACE="eno1" # da lindor

DOMINIO_LIMPO="${DOMINIO//./_}"

# muda o PCAP bruto temporário para a pasta /tmp/ 
PCAP_BRUTO="/tmp/captura_bruta_temp_${DOMINIO_LIMPO}.pcap"
PCAP_LEGACY="$PROJ_DIR/tmp-pcap/${DOMINIO_LIMPO}_filtrado_legacy.pcap"
CSV_SAIDA="$PROJ_DIR/Extrator_ALFlowLyzer/ALFlowLyzer/output-of-${DOMINIO_LIMPO}-pcap-file.csv"
CONFIG_TEMP="$PROJ_DIR/Extrator_ALFlowLyzer/ALFlowLyzer/config.${DOMINIO_LIMPO}.json"

echo "========================================================"
echo " Iniciando Pipeline de ML para: $DOMINIO"
echo "========================================================"

# garante que o ambiente virtual do Python está ativo
source "$PROJ_DIR/venv/bin/activate"

# limpa o cache do sistema (tolerante a falhas)
echo "[1/6] Limpando cache DNS..."
if command -v resolvectl &> /dev/null; then
    resolvectl flush-caches
elif command -v systemd-resolve &> /dev/null; then
    systemd-resolve --flush-caches
else
    echo " -> (Aviso: Comando de cache não encontrado, pulando com segurança...)"
fi

# remove captura antiga por segurança
rm -f "$PCAP_BRUTO"

# captura em background
echo "[2/6] Iniciando captura na interface $INTERFACE..."
tcpdump -i "$INTERFACE" -w "$PCAP_BRUTO" udp port 53 > /dev/null 2>&1 &
TCPDUMP_PID=$!

sleep 1.5

# verifica o tcpdump 
if ! kill -0 $TCPDUMP_PID 2>/dev/null; then
    echo "ERRO: O tcpdump foi bloqueado ou falhou ao iniciar."
    exit 1
fi

# requisições do domínio
echo "[3/6] Disparando requisições para $DOMINIO..."
# dig +noedns +nocookie @8.8.8.8 "$DOMINIO" A > /dev/null
# sleep 0.5
# dig +noedns +nocookie @8.8.8.8 "$DOMINIO" AAAA > /dev/null
# sleep 0.5
# dig +noedns +nocookie @8.8.8.8 "$DOMINIO" MX > /dev/null
# sleep 0.5
# --
echo " -> Simulando botnet..."
for i in {1..15}; do
    dig +noedns +nocookie @8.8.8.8 "${i}xyz.$DOMINIO" A > /dev/null &
    dig +noedns +nocookie @8.8.8.8 "$DOMINIO" TXT > /dev/null &
    sleep 0.1
done
dig +noedns +nocookie @8.8.8.8 "$DOMINIO" ANY > /dev/null
sleep 1.5

# encerra o tcpdump e espera ele salvar o arquivo
kill $TCPDUMP_PID 2>/dev/null
wait $TCPDUMP_PID 2>/dev/null

if [ ! -f "$PCAP_BRUTO" ]; then
    echo "ERRO: O arquivo bruto não foi gerado na pasta /tmp/."
    exit 1
fi

# emparelhamento e correção de cabeçalho
echo "[4/6] Isolando pares perfeitos de Pergunta/Resposta..."
python3 "$PROJ_DIR/limpa_pcap.py" "$PCAP_BRUTO" "$PCAP_LEGACY" "$DOMINIO"

# gera um config temporário do ALFlowLyzer para este domínio
python3 - "$PROJ_DIR/Extrator_ALFlowLyzer/ALFlowLyzer/config.json" "$CONFIG_TEMP" "$PCAP_LEGACY" "$CSV_SAIDA" <<'PY'
import json
import sys

config_template, config_output, pcap_path, csv_path = sys.argv[1:5]

with open(config_template, 'r', encoding='utf-8') as f:
    config = json.load(f)

config['pcap_file_address'] = pcap_path
config['output_file_address'] = csv_path

with open(config_output, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
    f.write('\n')
PY

# extração de features (usando o ALFlowLyzer)
echo "[5/6] Extraindo features estatísticas e comportamentais..."
cd "$PROJ_DIR/Extrator_ALFlowLyzer/ALFlowLyzer"
alflowlyzer -c "$CONFIG_TEMP" > /dev/null 2>&1
cd "$PROJ_DIR"

# inferência da Rede Neural
echo "[6/6] Passando dados pelo modelo Neural..."
echo "--------------------------------------------------------"
python3 "$PROJ_DIR/inferencia.py" "$CSV_SAIDA" "$DOMINIO" "$PCAP_LEGACY"
echo "--------------------------------------------------------"

# limpeza
rm -f "$PCAP_BRUTO"
rm -f "$CONFIG_TEMP"

echo "Processo concluído com sucesso para $DOMINIO!"