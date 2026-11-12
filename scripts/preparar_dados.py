"""
Prepara os dados do Reddit (dumps JSONL do Arctic Shift) para o classificador de sentimento.

Pipeline (por comunidade):
  1. Lê posts e comentários de data/<subreddit>/r_<subreddit>_{posts,comments}.jsonl (streaming).
  2. Filtros:
       - created_utc >= --desde (padrão 2022-01-01 UTC)
       - comentários: só depth 0 (parent_id começa com "t3_" = resposta direta ao post)
       - descarta removidos/deletados, AutoModerator e conteúdo fixado (stickied) de moderação
  3. Normalização do texto (ver normalizar()).
  4. Filtro de tamanho: --min-tokens <= n_tokens <= --max-tokens (tokenizer o200k_base, tiktoken).
  5. Remove textos duplicados (mesmo texto normalizado dentro da comunidade).
  6. Amostragem aleatória simples (seed fixa) de até --n-por-comunidade itens por comunidade,
     espalhada naturalmente por todo o período (não pega um bloco contíguo de datas).

Saídas (em --saida, padrão dados_limpos/):
  - amostra_sentimento.csv  -> entrada do classificar_sentimento.py (colunas id,texto + metadados)
  - relatorio_preparacao.json -> parâmetros, contagens de cada filtro e distribuição da amostra

Uso:
    python scripts/preparar_dados.py
    python scripts/preparar_dados.py --subreddits brdev datasciencebr --n-por-comunidade 10000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import tiktoken

RAIZ = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------------------
# Normalização
# --------------------------------------------------------------------------------------
RE_CODE_BLOCK = re.compile(r"```.*?```", re.S)
RE_INLINE_CODE = re.compile(r"`([^`\n]+)`")
RE_MD_LINK = re.compile(r"!?\[([^\]]*)\]\((?:[^()\s]|\([^()\s]*\))+\)")
RE_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
RE_USER = re.compile(r"(?<![\w/])/?u/[A-Za-z0-9_-]{2,}")
RE_SUB = re.compile(r"(?<![\w/])/?r/([A-Za-z0-9_]{2,})")
RE_HEADER = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
RE_QUOTE_LINE = re.compile(r"^\s*>.*$", re.M)
RE_EMPH = re.compile(r"(\*\*\*|\*\*|__|~~|\^)")
RE_SPOILER = re.compile(r">!(.*?)!<", re.S)
RE_HR = re.compile(r"^\s*([-*_]\s*){3,}$", re.M)
RE_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}.*$", re.M)
RE_INVISIVEIS = re.compile(r"[\u200b-\u200f\u2060\ufeff\u00ad]")  # zero-width, marcas de direção, BOM, hífen suave
RE_ESPACOS = re.compile(r"[ \t ]+")
RE_MUITAS_QUEBRAS = re.compile(r"\n{3,}")
RE_REPETICAO = re.compile(r"(.)\1{5,}")  # "kkkkkkkkkkkk" -> "kkkkk"


def normalizar(texto: str) -> str:
    """
    Normalização conservadora: remove ruído de formatação, mas preserva sinais de sentimento
    (caixa, pontuação, emojis, risadas, gírias).
      - HTML entities (&amp;, &gt; ...) -> caracteres; Unicode NFC; remove caracteres invisíveis
      - remove linhas citadas ("> ...") -> são falas de outra pessoa, não do autor
      - blocos de código -> "[código]"; código inline -> conteúdo sem crases
      - [texto](url) -> texto; URLs soltas -> "[link]"
      - u/usuario -> "@usuario" (anonimização); /r/sub -> r/sub
      - remove marcação markdown (**, __, ~~, #, separadores de tabela/linha)
      - repetições de 6+ caracteres iguais reduzidas a 5
      - colapsa espaços e quebras de linha
    """
    t = html.unescape(html.unescape(texto or ""))
    t = unicodedata.normalize("NFC", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = RE_INVISIVEIS.sub("", t)
    t = RE_CODE_BLOCK.sub(" [código] ", t)
    t = RE_SPOILER.sub(r"\1", t)
    t = RE_QUOTE_LINE.sub("", t)
    t = RE_MD_LINK.sub(lambda m: m.group(1) or "[link]", t)
    t = RE_URL.sub("[link]", t)
    t = RE_USER.sub("@usuario", t)
    t = RE_SUB.sub(r"r/\1", t)
    t = RE_INLINE_CODE.sub(r"\1", t)
    t = RE_TABLE_SEP.sub("", t)
    t = RE_HR.sub("", t)
    t = RE_HEADER.sub("", t)
    t = RE_EMPH.sub("", t)
    t = "".join(c for c in t if c == "\n" or unicodedata.category(c)[0] != "C")
    t = RE_REPETICAO.sub(lambda m: m.group(1) * 5, t)
    t = RE_ESPACOS.sub(" ", t)
    t = "\n".join(linha.strip() for linha in t.split("\n"))
    t = RE_MUITAS_QUEBRAS.sub("\n\n", t)
    return t.strip()


def so_ruido(t: str) -> bool:
    """True se não sobrou conteúdo textual (só [link], [código], pontuação...)."""
    resto = re.sub(r"\[(link|código)\]|@usuario", "", t)
    return not re.search(r"\w", resto)


# --------------------------------------------------------------------------------------
# Leitura e filtros
# --------------------------------------------------------------------------------------
INVALIDOS = {"", "[deleted]", "[removed]", "[deleted by user]"}
AUTORES_EXCLUIDOS = {"AutoModerator"}


def ler_jsonl(path: Path, cont: Counter, chave: str):
    with open(path, "rb") as f:
        for linha in f:
            try:
                yield json.loads(linha)
            except json.JSONDecodeError:
                cont[f"{chave}:linha_json_invalida"] += 1


def extrair(item: dict, tipo: str, desde: int, cont: Counter) -> tuple[str, dict] | None:
    """Aplica os filtros estruturais e devolve (texto_bruto, metadados) ou None."""
    k = tipo
    cont[f"{k}:lidos"] += 1
    ts = int(float(item.get("created_utc") or 0))
    if ts < desde:
        cont[f"{k}:antes_do_periodo"] += 1
        return None
    if item.get("author") in AUTORES_EXCLUIDOS:
        cont[f"{k}:automoderator"] += 1
        return None
    if item.get("stickied") or item.get("distinguished") == "moderator":
        cont[f"{k}:fixado_ou_moderacao"] += 1
        return None

    if tipo == "comentario":
        if not str(item.get("parent_id", "")).startswith("t3_"):
            cont[f"{k}:depth_maior_que_0"] += 1
            return None
        corpo = (item.get("body") or "").strip()
        if corpo in INVALIDOS:
            cont[f"{k}:removido_ou_deletado"] += 1
            return None
        bruto = corpo
        post_id = item.get("link_id") or item.get("parent_id")
    else:
        if item.get("removed_by_category") or (item.get("selftext") or "").strip() in ("[deleted]", "[removed]"):
            cont[f"{k}:removido_ou_deletado"] += 1
            return None
        titulo = (item.get("title") or "").strip()
        corpo = (item.get("selftext") or "").strip()
        bruto = f"{titulo}\n\n{corpo}" if corpo else titulo
        post_id = item.get("name") or f"t3_{item['id']}"

    fullname = item.get("name") or (("t1_" if tipo == "comentario" else "t3_") + item["id"])
    meta = {
        "id": fullname,
        "subreddit": item.get("subreddit"),
        "tipo": tipo,
        "created_utc": ts,
        "data": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d"),
        "post_id": post_id,
        "score": item.get("score"),
    }
    return bruto, meta


def processar_comunidade(sub: str, args, enc, cont: Counter) -> list[dict]:
    pasta = RAIZ / "data" / sub
    arquivos = {
        "post": pasta / f"r_{sub}_posts.jsonl",
        "comentario": pasta / f"r_{sub}_comments.jsonl",
    }
    pool: list[dict] = []
    vistos_texto: set[str] = set()
    vistos_id: set[str] = set()
    ultimo_ts: dict[str, int] = {}

    for tipo, path in arquivos.items():
        if not path.exists():
            print(f"  [aviso] arquivo não encontrado: {path}", file=sys.stderr)
            continue
        for item in ler_jsonl(path, cont, tipo):
            ultimo_ts[tipo] = max(ultimo_ts.get(tipo, 0), int(float(item.get("created_utc") or 0)))
            r = extrair(item, tipo, args.desde_ts, cont)
            if r is None:
                continue
            bruto, meta = r
            if meta["id"] in vistos_id:
                cont[f"{tipo}:id_duplicado"] += 1
                continue
            vistos_id.add(meta["id"])

            texto = normalizar(bruto)
            if not texto or so_ruido(texto):
                cont[f"{tipo}:vazio_apos_normalizacao"] += 1
                continue
            # atalho barato antes de tokenizar: textos muito longos certamente passam do limite
            if len(texto) > args.max_tokens * 12:
                cont[f"{tipo}:acima_max_tokens"] += 1
                continue
            n_tok = len(enc.encode(texto, disallowed_special=()))
            if n_tok < args.min_tokens:
                cont[f"{tipo}:abaixo_min_tokens"] += 1
                continue
            if n_tok > args.max_tokens:
                cont[f"{tipo}:acima_max_tokens"] += 1
                continue
            chave = hashlib.md5(texto.casefold().encode()).hexdigest()
            if chave in vistos_texto:
                cont[f"{tipo}:texto_duplicado"] += 1
                continue
            vistos_texto.add(chave)

            meta.update(texto=texto, n_tokens=n_tok)
            pool.append(meta)
            cont[f"{tipo}:elegiveis"] += 1

    for tipo, ts in ultimo_ts.items():
        cont[f"{tipo}:data_mais_recente_no_arquivo"] = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
    return pool


def resumo(amostra: list[dict]) -> dict:
    por_tipo = Counter(a["tipo"] for a in amostra)
    por_ano = Counter(a["data"][:4] for a in amostra)
    toks = sorted(a["n_tokens"] for a in amostra)
    q = lambda p: toks[min(len(toks) - 1, int(p * len(toks)))] if toks else None
    return {
        "total": len(amostra),
        "por_tipo": dict(por_tipo),
        "por_ano": dict(sorted(por_ano.items())),
        "tokens": {"min": q(0), "p25": q(0.25), "mediana": q(0.5), "p75": q(0.75), "p95": q(0.95), "max": toks[-1] if toks else None},
    }


def main():
    p = argparse.ArgumentParser(description="Seleciona e normaliza dados do Reddit para classificação de sentimento")
    p.add_argument("--subreddits", nargs="+", default=["brdev", "datasciencebr"])
    p.add_argument("--desde", default="2022-01-01", help="data mínima (UTC), AAAA-MM-DD")
    p.add_argument("--min-tokens", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--n-por-comunidade", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--saida", default=str(RAIZ / "dados_limpos"))
    args = p.parse_args()
    args.desde_ts = int(datetime.strptime(args.desde, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    enc = tiktoken.get_encoding("o200k_base")
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    relatorio = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/preparar_dados.py",
        "parametros": {k: v for k, v in vars(args).items() if k != "desde_ts"},
        "tokenizer": "tiktoken o200k_base (aproximação; o tokenizer do gpt-6-luna não é público)",
        "comunidades": {},
    }
    final: list[dict] = []

    for sub in args.subreddits:
        print(f"[{sub}] lendo e filtrando...")
        cont: Counter = Counter()
        pool = processar_comunidade(sub, args, enc, cont)
        rng = random.Random(f"{args.seed}-{sub}")
        amostra = pool if len(pool) <= args.n_por_comunidade else rng.sample(pool, args.n_por_comunidade)
        amostra.sort(key=lambda a: a["created_utc"])
        final.extend(amostra)
        relatorio["comunidades"][sub] = {
            "contagens_filtros": dict(sorted(cont.items())),
            "elegiveis": resumo(pool),
            "amostra": resumo(amostra),
        }
        print(f"[{sub}] elegíveis={len(pool)} amostrados={len(amostra)} {dict(Counter(a['tipo'] for a in amostra))}")

    colunas = ["id", "texto", "subreddit", "tipo", "data", "created_utc", "post_id", "score", "n_tokens"]
    arq_csv = saida / "amostra_sentimento.csv"
    with open(arq_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colunas, extrasaction="ignore")
        w.writeheader()
        w.writerows(final)
    with open(saida / "relatorio_preparacao.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(final)} linhas -> {arq_csv}")
    print(f"Relatório -> {saida / 'relatorio_preparacao.json'}")


if __name__ == "__main__":
    main()
