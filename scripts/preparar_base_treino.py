"""
Gera a base de TREINO com rótulos de alta confiança e uma base de TESTE separada.

Passos:
  1. Junta o resultado da classificação (rotulo, confidence, probs) com os metadados da amostra.
  2. Separa um TESTE estratificado (rótulo x subreddit) de TODOS os dados, SEM filtro de confiança.
     Assim o teste representa a distribuição real, e nenhum texto do teste aparece no treino.
  3. TREINO = o restante, só com confidence > --conf-min (opcionalmente balanceado por classe).

Saídas (em dados_treino/):
  - treino_conf_<conf-min>.csv   (ou ..._balanceado.csv)
  - teste.csv
  - relatorio_base_treino.json    (parâmetros e contagens)

Uso:
    python scripts/preparar_base_treino.py
    python scripts/preparar_base_treino.py --conf-min 0.7 --balancear
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
ROTULOS = ("negativo", "neutro", "positivo")
COLUNAS = ["id", "texto", "rotulo", "confidence", "prob_negativo", "prob_neutro", "prob_positivo",
           "subreddit", "tipo", "data", "n_tokens"]


def ler(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def escrever(path, linhas):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS, extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)


def contagem(linhas):
    c = Counter(r["rotulo"] for r in linhas)
    return {"total": len(linhas), **{r: c[r] for r in ROTULOS},
            "por_subreddit": {s: {r: n for r, n in sorted(Counter(x["rotulo"] for x in linhas if x["subreddit"] == s).items())}
                              for s in sorted({x["subreddit"] for x in linhas})}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classificado", default=str(RAIZ / "dados_limpos" / "amostra_sentimento_classificado.csv"))
    p.add_argument("--amostra", default=str(RAIZ / "dados_limpos" / "amostra_sentimento.csv"))
    p.add_argument("--saida", default=str(RAIZ / "dados_treino"))
    p.add_argument("--conf-min", type=float, default=0.7, help="treino: confidence estritamente maior que este valor")
    p.add_argument("--frac-teste", type=float, default=0.15)
    p.add_argument("--balancear", action="store_true", help="undersampling para igualar as classes no treino")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    meta = {r["id"]: r for r in ler(args.amostra)}
    dados = []
    for r in ler(args.classificado):
        m = meta.get(r["id"], {})
        dados.append({**r, "subreddit": m.get("subreddit", ""), "tipo": m.get("tipo", ""),
                      "data": m.get("data", ""), "n_tokens": m.get("n_tokens", "")})

    # 1) teste estratificado por (rótulo, subreddit), sem filtro de confiança
    rng = random.Random(args.seed)
    estratos = defaultdict(list)
    for r in sorted(dados, key=lambda x: x["id"]):
        estratos[(r["rotulo"], r["subreddit"])].append(r)
    teste, resto = [], []
    for chave in sorted(estratos):
        grupo = estratos[chave]
        rng.shuffle(grupo)
        k = round(len(grupo) * args.frac_teste)
        teste += grupo[:k]
        resto += grupo[k:]

    # 2) treino: só alta confiança
    treino = [r for r in resto if float(r["confidence"]) > args.conf_min]
    if args.balancear:
        por_classe = defaultdict(list)
        for r in treino:
            por_classe[r["rotulo"]].append(r)
        n_min = min(len(v) for v in por_classe.values())
        treino = [r for c in ROTULOS for r in rng.sample(por_classe[c], n_min)]

    assert not ({r["id"] for r in treino} & {r["id"] for r in teste}), "vazamento treino/teste"
    rng.shuffle(treino)
    teste.sort(key=lambda r: r["id"])

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    nome_treino = f"treino_conf_{args.conf_min}" + ("_balanceado" if args.balancear else "") + ".csv"
    escrever(saida / nome_treino, treino)
    escrever(saida / "teste.csv", teste)

    rel = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/preparar_base_treino.py",
        "parametros": vars(args),
        "fonte_total": contagem(dados),
        "teste_sem_filtro": contagem(teste),
        "descartados_do_treino_por_confianca": len(resto) - len([r for r in resto if float(r["confidence"]) > args.conf_min]),
        "treino": contagem(treino),
    }
    with open(saida / "relatorio_base_treino.json", "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2)

    for nome, linhas in (("treino", treino), ("teste", teste)):
        c = contagem(linhas)
        print(f"{nome:<7} n={c['total']:<6} " + "  ".join(f"{r}={c[r]} ({c[r]/c['total']:.0%})" for r in ROTULOS))
    print(f"-> {saida / nome_treino}\n-> {saida / 'teste.csv'}")


if __name__ == "__main__":
    main()
