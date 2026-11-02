from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def carregar_base():
    #Carrega a Iris e separa os dados em treino, validação e teste
    iris = load_iris()

    X = iris.data  # 4 características de cada flor"
    y = iris.target  # Espécie de cada flor, representada por 0, 1 ou 2

    # Reserva 20% da base para o teste final: 30 flores 
    X_restante, X_teste, y_restante, y_teste = train_test_split (X, y, test_size=0.20, random_state=42, stratify=y,)

    # Dos 80% restantes, reserva 25% para validação, que corresponderá
    # a 20% da base original: 30 flores

    X_treino, X_validacao, y_treino, y_validacao = train_test_split(X_restante, y_restante, test_size=0.25, random_state=42, stratify=y_restante,)

    return(
        X_treino,
        X_validacao,
        X_teste,
        y_treino,
        y_validacao,
        y_teste,
    )

if __name__== "__main__": 
    (
        X_treino,
        X_validacao,
        X_teste,
        y_treino,
        y_validacao,
        y_teste,
    ) = carregar_base()

    print("Treino:", X_treino.shape)
    print("Validação:", X_validacao.shape)
    print("Teste:", X_teste.shape)

    print("\nCaracterísticas da primeira flor de treino:", X_treino[0])
    print("Classe dessa flor: ", y_treino[0])