# Classificação Iris com MLP

## Objetivo

Classificar as três espécies da base Iris usando uma MLP com
exatamente uma camada escondida. Comparar algoritmo genético
e busca aleatória na seleção dos hiperparâmetros.


## Arquivos

- base.py: carrega a Iris e separa treino, validação e teste.
- mlp.py: cria a rede e fornece as funções de avaliação.
- requirements.txt: registra as dependências do ambiente.

## Dados

A base contém 150 flores, quatro características e três classes:

- 0: setosa
- 1: versicolor
- 2: virginica

Divisão estratificada, com random_state=42:

- Treino: 90 exemplos.
- Validação: 30 exemplos.
- Teste: 30 exemplos.

As entradas são padronizadas dentro de uma Pipeline.
Durante as buscas, a padronização é ajustada somente no treino.

## Rede neural

- Uma camada escondida.
- Ativação: ReLU.
- Otimizador: Adam.
- Máximo de 1.000 épocas, podendo encerrar antes por convergência.
- random_state=42.
- Demais opções mantidas nos padrões da versão registrada.

## Espaço de busca proposto

Ambos os métodos devem utilizar:

- neuronios: inteiro entre 4 e 30, inclusive.
- taxa_aprendizado: 0.001, 0.005, 0.01 ou 0.05. 

(Escolha aleatória)


## Integração das buscas

```python
from mlp import avaliar_configuracao

pontuacao = avaliar_configuracao({
    "neuronios": 12,
    "taxa_aprendizado": 0.01,
})
```

Cada chamada cria e treina um modelo novo.

O retorno é a acurácia de validação entre 0 e 1.
O objetivo das buscas é maximizar esse valor.

Os métodos devem usar o mesmo espaço de busca e o mesmo
orçamento de avaliações. 

## Avaliação final

Somente depois de cada busca selecionar sua melhor configuração:

```python
from mlp import avaliar_no_teste

resultado = avaliar_no_teste(melhores_parametros)
print(resultado)
```

A função treina um modelo novo com treino + validação e retorna
a acurácia de teste e a matriz de confusão.

Na matriz, as linhas são as classes reais e as colunas são as
classes previstas, na ordem: setosa, versicolor e virginica.

O teste não deve ser usado para ajustar hiperparâmetros nem
orientar as buscas.

## Verificação básica

```bash
python base.py
python mlp.py
```

Na execução inicial, a configuração com 12 neurônios e taxa
0.01 obteve 93,33% de acurácia de validação: 28 acertos em 30.

Esse resultado é uma verificação inicial.