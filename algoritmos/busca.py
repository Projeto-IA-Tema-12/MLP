import random
import time

from config.espaco import ESPACO, ORCAMENTO


def sortear_configuracao(espaco, rng):
  
    config = {}
    for nome, spec in espaco.items():
        if spec[0] == "cat":
            config[nome] = rng.choice(spec[1])
        elif spec[0] == "int":
            config[nome] = rng.randint(spec[1], spec[2]) 
        else:
            raise ValueError(f"Tipo de parâmetro não suportado: {spec[0]}")
    return config


def busca_aleatoria(
    avaliar,
    espaco=ESPACO,
    max_avaliacoes=ORCAMENTO,
    seed=None,
    verbose=False,
):
    
    rng = random.Random(seed)
    t0 = time.time()

    curva = []        
    fitness_todas = [] 
    melhor = {"fitness": float("-inf"), "params": None}
    cache = {}         

    for i in range(max_avaliacoes):
        config = sortear_configuracao(espaco, rng)
        chave = tuple(sorted(config.items()))
        if chave not in cache:
            cache[chave] = float(avaliar(config))
        f = cache[chave]

        if f > melhor["fitness"]:
            melhor["fitness"], melhor["params"] = f, dict(config)

        fitness_todas.append(f)
        curva.append(melhor["fitness"])

        if verbose:
            print(f"Avaliação {i + 1:2d} | {config} | fitness={f:.4f} "
                  f"| melhor={melhor['fitness']:.4f}")

    return {
        "melhor_params": melhor["params"],
        "melhor_fitness": melhor["fitness"],
        "n_avaliacoes": len(curva),
        "n_configs_unicas": len(cache),
        "tempo_s": time.time() - t0,
        "curva_avaliacoes": curva,
        "fitness_avaliacoes": fitness_todas,
    }


if __name__ == "__main__":
    from model.mlp import avaliar_configuracao

    res = busca_aleatoria(avaliar_configuracao, seed=0, verbose=True)
    print("\nMelhor fitness:", round(res["melhor_fitness"], 4))
    print("Melhores params:", res["melhor_params"])
    print("Avaliações:", res["n_avaliacoes"], "| configs únicas:", res["n_configs_unicas"],
          "| tempo:", round(res["tempo_s"], 3), "s")