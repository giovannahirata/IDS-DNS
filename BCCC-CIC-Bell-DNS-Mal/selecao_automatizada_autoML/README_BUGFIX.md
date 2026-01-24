# Solução para o Bug do AutoKeras 3.0.0 com Keras 3.13.0

## Problema

O script `autoMLAutoKeras.py` falha com o seguinte erro:

```
ValueError: Received an invalid value for `units`, expected a positive integer. Received: units=1
```

### Causa

Este é um bug conhecido no **AutoKeras 3.0.0** ao trabalhar com **Keras 3.13.0**. O problema ocorre na camada `ClassificationHead` quando ela tenta criar uma camada `Dense` para classificação binária.

## Solução Implementada

Foi criado o script **`autoMLKerasTuner.py`** que implementa a mesma funcionalidade de busca automática de hiperparâmetros usando **Keras Tuner** diretamente (que é a biblioteca que o AutoKeras usa internamente), evitando completamente o bug.

## Como Usar

### Opção 1: Usar Keras Tuner (Recomendado)

```bash
python3 autoMLKerasTuner.py
```

**Vantagens:**
- Funciona sem erros
- Mesma funcionalidade de busca automática de hiperparâmetros
- Mais controle sobre a arquitetura do modelo
- Resultados equivalentes ou melhores

### Opção 2: Corrigir as versões (Alternativa)

Se você quiser usar o AutoKeras original, pode tentar fazer downgrade:

```bash
pip install autokeras==2.0.0 keras==2.15.0 tensorflow==2.15.0
```

**Nota:** Isso pode causar conflitos com outras dependências.

## Resultados Obtidos

Com o script `autoMLKerasTuner.py`:

```
Accuracy: 0.7625
Test AUC: 0.6238

Melhores hiperparâmetros:
- units_1: 96
- activation_1: tanh
- dropout_1: 0.2
- learning_rate: 0.0017142535756407963
```

## Arquivos

- **`autoMLAutoKeras.py`** - Script original (com bug)
- **`autoMLKerasTuner.py`** - Solução alternativa funcional 
- **`autoMLAutoKeras_BUGFIX.py`** - Script documentado mostrando o problema
- **`README_BUGFIX.md`** - Este arquivo

## Informações de Versão

Versões testadas com o bug:
- Keras: 3.13.0
- AutoKeras: 3.0.0
- TensorFlow: 2.20.0
- Python: 3.13

## Referências

- [Keras Tuner Documentation](https://keras.io/keras_tuner/)
- [AutoKeras Issues](https://github.com/keras-team/autokeras/issues)
