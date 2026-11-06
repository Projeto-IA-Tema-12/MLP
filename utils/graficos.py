import os
import warnings
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

from model.mlp import avaliar_configuracao, avaliar_no_teste
from algoritmos.algoritmo_genetico import algoritmo_genetico
from algoritmos.busca import busca_aleatoria

# Silencia avisos de não-convergência da MLP no terminal
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def plotar_busca_aleatoria(curva_best, acuracias_individuais):
    """Gráfico 1: Busca Aleatória (Best-so-Far + Avaliações Individuais)"""
    evals = range(1, len(curva_best) + 1)

    plt.figure(figsize=(9, 5))

    # Pontos de cada avaliação individual (amostragem pontual do sorteio)
    plt.scatter(
        evals,
        [acc * 100 for acc in acuracias_individuais],
        color="#d95f02",
        alpha=0.45,
        s=40,
        label="Avaliação Individual (Sorteio)",
    )

    # Linha do melhor acumulado (best-so-far)
    plt.plot(
        evals,
        [acc * 100 for acc in curva_best],
        label="Melhor Acumulado (Best-so-Far)",
        color="#d95f02",
        linestyle="--",
        marker="o",
        linewidth=2,
    )

    plt.title(
        "Busca Aleatória: Avaliações Individuais vs. Melhor Acumulado",
        fontsize=12,
        pad=12,
    )
    plt.xlabel("Número de Avaliações", fontsize=11)
    plt.ylabel("Acurácia de Validação (%)", fontsize=11)
    plt.ylim(70, 102)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig("graficos/grafico_busca_aleatoria.png", dpi=300)
    print("-> Gráfico 'grafico_busca_aleatoria.png' salvo com sucesso!")


def plotar_algoritmo_genetico(curva_best, historico_geracoes):
    """Gráfico 2: Algoritmo Genético (Best-so-Far + Média da População por Geração)"""
    evals = range(1, len(curva_best) + 1)

    plt.figure(figsize=(9, 5))

    # Linha da média da população por geração
    if historico_geracoes:
        eval_gen = [h["avaliacoes"] for h in historico_geracoes]
        media_gen = [h["media"] * 100 for h in historico_geracoes]
        plt.plot(
            eval_gen,
            media_gen,
            label="Média da População (Geração)",
            color="#7570b3",
            linestyle=":",
            marker="x",
            linewidth=2,
            alpha=0.8,
        )

    # Linha do melhor acumulado (best-so-far)
    plt.plot(
        evals,
        [acc * 100 for acc in curva_best],
        label="Melhor Acumulado (Best-so-Far)",
        color="#7570b3",
        marker="s",
        linewidth=2.5,
    )

    plt.title(
        "Algoritmo Genético: Média da População vs. Melhor Acumulado",
        fontsize=12,
        pad=12,
    )
    plt.xlabel("Número de Avaliações", fontsize=11)
    plt.ylabel("Acurácia de Validação (%)", fontsize=11)
    plt.ylim(70, 102)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig("graficos/grafico_algoritmo_genetico.png", dpi=300)
    print("-> Gráfico 'grafico_algoritmo_genetico.png' salvo com sucesso!")


def plotar_comparacao(curva_random, curva_ag):
    """Gráfico 3: Comparação Direta (Best-so-Far da Busca Aleatória vs. AG)"""
    evals = range(1, len(curva_random) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        evals,
        [acc * 100 for acc in curva_random],
        label="Busca Aleatória (Best-so-Far)",
        color="#d95f02",
        linestyle="--",
        marker="o",
        alpha=0.8,
        linewidth=2,
    )
    plt.plot(
        evals,
        [acc * 100 for acc in curva_ag],
        label="Algoritmo Genético (Best-so-Far)",
        color="#7570b3",
        marker="s",
        linewidth=2.5,
    )

    plt.title(
        "Comparação Direta: Busca Aleatória vs. Algoritmo Genético",
        fontsize=13,
        pad=15,
    )
    plt.xlabel("Número de Avaliações (Treinos da MLP)", fontsize=12)
    plt.ylabel("Melhor Acurácia Acumulada (%)", fontsize=12)
    plt.ylim(70, 102)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    plt.savefig("graficos/grafico_comparativo.png", dpi=300)
    print("-> Gráfico 'grafico_comparativo.png' salvo com sucesso!")


def executar_experimentos(seed=42):
    print("=" * 65)
    print(" 1. EXECUTANDO BUSCA ALEATÓRIA")
    print("=" * 65)
    res_random = busca_aleatoria(avaliar_configuracao, seed=seed, verbose=False)
    print(f"Melhor Configuração: {res_random['melhor_params']}")
    print(f"Maior Acurácia (Validação): {res_random['melhor_fitness']:.2%}")

    print("\n" + "=" * 65)
    print(" 2. EXECUTANDO ALGORITMO GENÉTICO")
    print("=" * 65)
    res_ag = algoritmo_genetico(avaliar_configuracao, seed=seed, verbose=False)
    print(f"Melhor Configuração: {res_ag['melhor_params']}")
    print(f"Maior Acurácia (Validação): {res_ag['melhor_fitness']:.2%}")

    print("\n" + "=" * 65)
    print(" 3. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
    print("=" * 65)
    teste_random = avaliar_no_teste(res_random["melhor_params"])
    print(f"[Busca Aleatória] Teste: {teste_random['acuracia_teste']:.2%}")

    teste_ag = avaliar_no_teste(res_ag["melhor_params"])
    print(f"[Algoritmo Genético] Teste: {teste_ag['acuracia_teste']:.2%}")

    print("\n" + "=" * 65)
    print(" 4. GERANDO E EXIBINDO OS 3 GRÁFICOS")
    print("=" * 65)
    plotar_busca_aleatoria(
        res_random["curva_avaliacoes"], res_random["fitness_avaliacoes"]
    )
    plotar_algoritmo_genetico(res_ag["curva_avaliacoes"], res_ag.get("historico"))
    plotar_comparacao(res_random["curva_avaliacoes"], res_ag["curva_avaliacoes"])

    # Exibe todas as janelas dos gráficos no final de uma só vez
    plt.show()


if __name__ == "__main__":
    executar_experimentos(seed=42)