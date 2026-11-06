import math
import random
import time

from config.espaco import ESPACO, ORCAMENTO 

def _para_interno(spec, v):
    return math.log(v) if spec[0] == "log" else float(v)


def _de_interno(spec, x):
    lo, hi = spec[1], spec[2]
    if spec[0] == "log":
        return min(max(math.exp(x), lo), hi)
    return int(min(max(round(x), lo), hi))


def _limites(spec):
    return _para_interno(spec, spec[1]), _para_interno(spec, spec[2])


def sortear_individuo(espaco, rng):
    ind = {}
    for nome, spec in espaco.items():
        if spec[0] == "cat":
            ind[nome] = rng.choice(spec[1])
        else:
            lo, hi = _limites(spec)
            ind[nome] = _de_interno(spec, rng.uniform(lo, hi))
    return ind


def selecao_torneio(populacao, fitness, k, rng):
    candidatos = rng.sample(range(len(populacao)), k)
    melhor = max(candidatos, key=lambda i: fitness[i])
    return populacao[melhor]


def cruzamento(p1, p2, espaco, rng, taxa=0.9):
    if rng.random() > taxa:
        return dict(p1), dict(p2)
    f1, f2 = {}, {}
    for nome, spec in espaco.items():
        a, b = p1[nome], p2[nome]
        if spec[0] == "cat":
            f1[nome], f2[nome] = (a, b) if rng.random() < 0.5 else (b, a)
        else:
            t = rng.random()
            xa, xb = _para_interno(spec, a), _para_interno(spec, b)
            f1[nome] = _de_interno(spec, t * xa + (1 - t) * xb)
            f2[nome] = _de_interno(spec, (1 - t) * xa + t * xb)
    return f1, f2


def mutacao(ind, espaco, rng, taxa=0.2, sigma=0.15):
    novo = dict(ind)
    for nome, spec in espaco.items():
        if rng.random() >= taxa:
            continue
        if spec[0] == "cat":
            novo[nome] = rng.choice(spec[1])
        else:
            lo, hi = _limites(spec)
            x = _para_interno(spec, novo[nome]) + rng.gauss(0, sigma * (hi - lo))
            novo[nome] = _de_interno(spec, min(max(x, lo), hi))
    return novo


def algoritmo_genetico(
    avaliar,
    espaco=ESPACO,
    tam_pop=6,
    max_avaliacoes=ORCAMENTO,
    elitismo=1,
    k_torneio=3,
    taxa_cruzamento=0.9,
    taxa_mutacao=0.2,
    sigma_mutacao=0.15,
    seed=None,
    verbose=False,
):
    rng = random.Random(seed)
    t0 = time.time()
    curva = []                      # best-so-far por avaliação
    melhor_global = {"fitness": -math.inf, "params": None}

    cache = {}

    def avaliar_e_registrar(ind):
        chave = tuple(sorted(ind.items()))
        if chave not in cache:
            cache[chave] = float(avaliar(ind))
        f = cache[chave]
        if f > melhor_global["fitness"]:
            melhor_global["fitness"], melhor_global["params"] = f, dict(ind)
        curva.append(melhor_global["fitness"])
        return f

    n_ini = min(tam_pop, max_avaliacoes)
    populacao = [sortear_individuo(espaco, rng) for _ in range(n_ini)]
    fitness = [avaliar_e_registrar(ind) for ind in populacao]

    historico = []

    def registrar(g):
        historico.append({
            "geracao": g,
            "melhor": max(fitness),
            "media": sum(fitness) / len(fitness),
            "avaliacoes": len(curva),
        })
        if verbose:
            h = historico[-1]
            print(f"Geração {g:2d} | melhor={h['melhor']:.4f} "
                  f"média={h['media']:.4f} | avaliações={h['avaliacoes']}")

    registrar(0)

    geracao = 1
    while len(curva) < max_avaliacoes:
        ordem = sorted(range(len(populacao)), key=lambda i: fitness[i], reverse=True)
        elite_idx = ordem[:elitismo]
        nova_pop = [populacao[i] for i in elite_idx]
        novo_fit = [fitness[i] for i in elite_idx]

        n_filhos = min(tam_pop - elitismo, max_avaliacoes - len(curva))
        filhos = []
        while len(filhos) < n_filhos:
            p1 = selecao_torneio(populacao, fitness, min(k_torneio, len(populacao)), rng)
            p2 = selecao_torneio(populacao, fitness, min(k_torneio, len(populacao)), rng)
            c1, c2 = cruzamento(p1, p2, espaco, rng, taxa_cruzamento)
            filhos += [mutacao(c1, espaco, rng, taxa_mutacao, sigma_mutacao),
                       mutacao(c2, espaco, rng, taxa_mutacao, sigma_mutacao)]
        filhos = filhos[:n_filhos]

        for f in filhos:
            nova_pop.append(f)
            novo_fit.append(avaliar_e_registrar(f))

        populacao, fitness = nova_pop, novo_fit
        registrar(geracao)
        geracao += 1

    return {
        "melhor_params": melhor_global["params"],
        "melhor_fitness": melhor_global["fitness"],
        "n_avaliacoes": len(curva),
        "n_configs_unicas": len(cache),
        "tempo_s": time.time() - t0,
        "curva_avaliacoes": curva,
        "historico": historico,
    }


def _avaliar_falsa(p):
    return 1 - abs(p["neuronios"] - 12) / 60 - abs(math.log10(p["taxa_aprendizado"]) + 2) / 5


if __name__ == "__main__":
    try:
        from model.mlp import avaliar_configuracao as avaliar
        print("Usando avaliar_configuracao do mlp.py")
    except ModuleNotFoundError as erro:
        if erro.name == "mlp":
            avaliar = _avaliar_falsa
            print("mlp.py não está na mesma pasta: usando função falsa")
        else:
            # Ex.: falta instalar scikit-learn/numpy. Mostra o erro real.
            raise
    res = algoritmo_genetico(avaliar, seed=0, verbose=True)
    print("\nMelhor fitness:", round(res["melhor_fitness"], 4))
    print("Melhores params:", res["melhor_params"])
    print("Avaliações:", res["n_avaliacoes"], "| configs únicas:", res["n_configs_unicas"],
          "| tempo:", round(res["tempo_s"], 3), "s")