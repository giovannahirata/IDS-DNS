#!/bin/bash

# Script robusto para rodar PyCaret sem interrupções
# Uso: bash run_pycaret_safe.sh

# Registrar tempo de início
START_TIME=$(date +%s)
START_DATE=$(date)

echo "========================================="
echo "Iniciando execução do PyCaret"
echo "Início: $START_DATE"
echo "========================================="

# 1. Ativar ambiente correto
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yourenvname

# 2. Verificar memória disponível
echo ""
echo "=== Memória disponível ==="
free -h

# 3. Verificar se joblib está correto
echo ""
echo "=== Verificando dependências ==="
python3 -c "import joblib; print(f'joblib version: {joblib.__version__}')" 2>&1
python3 -c "import pycaret; print(f'pycaret version: {pycaret.__version__}')" 2>&1

if [ $? -ne 0 ]; then
    echo ""
    echo "ERRO: Dependências não estão instaladas corretamente!"
    echo "Execute: pip install joblib==1.3.2"
    exit 1
fi

# 4. Rodar script com unbuffered output, capturar tudo E medir tempo
echo ""
echo "=== Iniciando treinamento ==="
echo ""

# Usar time para medir execução
{ time python3 -u autoMLPyCaret.py ; } 2>&1 | tee -a pycaret_complete_log.txt

# Capturar código de saída
EXIT_CODE=${PIPESTATUS[0]}

# Calcular tempo total
END_TIME=$(date +%s)
END_DATE=$(date)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "========================================="
echo "Execução finalizada"
echo "Fim: $END_DATE"
echo "Tempo total: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Exit Code: $EXIT_CODE"
echo "========================================="

# Mostrar uso de memória final
echo ""
echo "=== Memória ao final ==="
free -h

# Mensagens de status
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Execução completada com SUCESSO!"
    echo "✓ Logs salvos em: pycaret_complete_log.txt"
else
    echo "✗ Execução FALHOU com código $EXIT_CODE"
    echo "✗ Verifique os erros em: pycaret_complete_log.txt"
fi

exit $EXIT_CODE
