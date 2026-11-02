from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix 

from base import carregar_base

def criar_modelo(parametros):
    # Cria a padronização e uma MLP com apenas uma camada escondida. 
    return Pipeline([
        ("padronizacao", StandardScaler()),
        ("mlp", MLPClassifier (
            hidden_layer_sizes=(parametros["neuronios"],),
            learning_rate_init=parametros["taxa_aprendizado"],
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
        )),
    ])

def avaliar_configuracao(parametros):
    # Treina uma nova rede e retorna a acurácia da validação.
    (
        X_treino,
        X_validacao,
        X_teste,
        y_treino,
        y_validacao,
        y_teste,
    ) = carregar_base()

    modelo = criar_modelo(parametros)

    modelo.fit(X_treino, y_treino) # Ajusta a padronização e os pesos usando somente o treino

    acuracia = modelo.score(X_validacao, y_validacao) # A validação é transformada usando a padronização já aprendida

    return float(acuracia)

def avaliar_no_teste(parametros): 
    # Avalia no teste uma configruação já escolhida pela busca.
    (
        X_treino,
        X_validacao,
        X_teste,
        y_treino,
        y_validacao,
        y_teste,
    ) = carregar_base()

    # Após a escolha dos hiperparâmetros, aprveitamos os dois conjuntos.
    X_final = np.concatenate((X_treino, X_validacao), axis=0)
    y_final = np.concatenate((y_treino, y_validacao), axis=0)

    # Cria um modelo novo, com padronização.
    modelo = criar_modelo(parametros)
    modelo.fit(X_final, y_final)

    previsoes = modelo.predict(X_teste)

    return {
        "acuracia_teste": float(accuracy_score(y_teste, previsoes)),
        "matriz_confusao": confusion_matrix(
            y_teste, previsoes, labels=[0,1,2]
        ).tolist(),
    }

if __name__ == "__main__":
    parametros = {
        "neuronios": 12,
        "taxa_aprendizado": 0.01,
    }

    pontuacao = avaliar_configuracao(parametros)

    print("Configuração:", parametros)
    print(f"Acurácia de validação: {pontuacao:.2%}")