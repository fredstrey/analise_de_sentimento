"""
Classificação de sentimento de comentários do Reddit usando a OpenAI Decisions API.

Entrada : CSV com (no mínimo) uma coluna de texto e, opcionalmente, uma coluna de id.
Saída   : CSV com id,texto,rotulo,confidence,prob_negativo,prob_neutro,prob_positivo
          (rotulo ∈ {negativo, neutro, positivo}; confidence e probs vêm da Decisions API)
Erros   : CSV com id,texto,erro,tentativas,timestamp  (amostras que falharam após todos os retries)

Características
- Requests em paralelo com asyncio + Semaphore (limita concorrência -> evita rate limit).
- Até N tentativas por amostra (padrão 3) com backoff exponencial + jitter; respeita Retry-After.
- Persistência incremental: cada resultado é gravado (flush + fsync) assim que chega.
- Retomada: ao reiniciar, ids já presentes na saída são pulados.
- Falhas definitivas vão para o arquivo de erros; rode com --retry-erros para reprocessá-las.

Uso
    Coloque a chave no arquivo .env na raiz do projeto:  OPENAI_API_KEY=sk-...
    (ou defina a variável de ambiente; PowerShell: $env:OPENAI_API_KEY="sk-...")
    python classificar_sentimento.py entrada.csv
    python classificar_sentimento.py entrada.csv --retry-erros
    python classificar_sentimento.py entrada.csv --mock          (testa sem chave de API)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Carrega a chave do arquivo .env na raiz do projeto.
# Variáveis já definidas no ambiente têm prioridade sobre o .env.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# --------------------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------------------
API_URL = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
ROTULOS = ("negativo", "neutro", "positivo")

QUESTION = {
    "type": "choice",
    "name": "sentimento",
    "instructions": (
        "Classifique o sentimento geral expresso pelo autor deste comentário do Reddit. "
        "Considere sarcasmo, ironia e gírias da internet. "
        "Se o comentário for apenas informativo, uma pergunta sem carga emocional, "
        "ou misturar positivo e negativo sem predominância, escolha 'neutro'."
    ),
    "choices": [
        {"value": "negativo", "description": "Raiva, frustração, tristeza, crítica, reclamação, desprezo ou sarcasmo negativo."},
        {"value": "neutro", "description": "Factual, informativo, pergunta neutra, ou sentimento misto/ausente."},
        {"value": "positivo", "description": "Elogio, gratidão, alegria, entusiasmo, concordância ou apoio."},
    ],
}

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # comentários longos
log = logging.getLogger("sentimento")


class ErroRetentavel(Exception):
    """Falha transitória (429, 5xx, timeout, rede) - vale tentar de novo."""

    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class ErroDefinitivo(Exception):
    """Falha que não adianta repetir (400, 401, 403, resposta inválida...)."""


@dataclass
class Amostra:
    id: str
    texto: str


@dataclass
class Resultado:
    rotulo: str
    confidence: float | None
    probs: dict  # {"negativo": p, "neutro": p, "positivo": p}


HEADER_SAIDA = ["id", "texto", "rotulo", "confidence"] + [f"prob_{r}" for r in ROTULOS]


def fmt(x) -> str:
    return "" if x is None else f"{float(x):.6f}"


# --------------------------------------------------------------------------------------
# Escrita segura em CSV (append + flush + fsync, um writer por vez)
# --------------------------------------------------------------------------------------
class CsvAppender:
    def __init__(self, path: Path, header: list[str]):
        self.path = path
        novo = not path.exists() or path.stat().st_size == 0
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.lock = asyncio.Lock()
        if novo:
            self.w.writerow(header)
            self._sync()

    def _sync(self):
        self.f.flush()
        os.fsync(self.f.fileno())

    async def write(self, row: list):
        async with self.lock:
            self.w.writerow(row)
            self._sync()

    def close(self):
        self.f.close()


def ler_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"] for r in csv.DictReader(f) if r.get("id")}


def ler_erros_pendentes(path_erros: Path, ids_ok: set[str]) -> dict[str, Amostra]:
    """Último erro registrado de cada id que ainda não está na saída."""
    pend: dict[str, Amostra] = {}
    if not path_erros.exists():
        return pend
    with open(path_erros, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["id"] not in ids_ok:
                pend[r["id"]] = Amostra(r["id"], r["texto"])
    return pend


def compactar_erros(path_erros: Path, ids_ok: set[str]):
    """Reescreve o arquivo de erros mantendo só o último erro de ids ainda não resolvidos."""
    if not path_erros.exists():
        return
    with open(path_erros, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        ultimos = {r["id"]: r for r in reader if r["id"] not in ids_ok}
    tmp = path_erros.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(ultimos.values())
    os.replace(tmp, path_erros)


def iterar_entrada(path: Path, col_id: str | None, col_texto: str):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if col_texto not in cols:
            raise SystemExit(f"Coluna de texto '{col_texto}' não encontrada. Colunas: {cols}")
        usar_id = col_id and col_id in cols
        if not usar_id:
            log.warning("Coluna de id '%s' não encontrada; usando o número da linha como id.", col_id)
        for i, r in enumerate(reader):
            yield Amostra(str(r[col_id]).strip() if usar_id else str(i), r[col_texto] or "")


# --------------------------------------------------------------------------------------
# Chamada à API
# --------------------------------------------------------------------------------------
async def chamar_decisions(client: httpx.AsyncClient, texto: str) -> str:
    payload = {"model": MODEL, "input": texto, "questions": [QUESTION]}
    try:
        resp = await client.post(API_URL, json=payload)
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise ErroRetentavel(f"{type(e).__name__}: {e}") from e

    if resp.status_code == 429 or resp.status_code >= 500:
        ra = resp.headers.get("retry-after")
        raise ErroRetentavel(
            f"HTTP {resp.status_code}: {resp.text[:500]}",
            retry_after=float(ra) if ra and ra.replace(".", "", 1).isdigit() else None,
        )
    if resp.status_code >= 400:
        raise ErroDefinitivo(f"HTTP {resp.status_code}: {resp.text[:500]}")

    try:
        answer = resp.json()["answers"][0]
    except Exception as e:
        raise ErroRetentavel(f"Resposta inesperada: {resp.text[:500]}") from e

    if answer.get("type") == "refusal":
        raise ErroDefinitivo(f"Modelo recusou: {answer}")
    rotulo = answer.get("choice")
    if rotulo not in ROTULOS:
        raise ErroRetentavel(f"Rótulo inválido retornado: {answer}")
    probs = {p.get("value"): p.get("probability") for p in answer.get("probabilities") or []}
    return Resultado(rotulo, answer.get("confidence"), probs)


async def chamar_mock(_client, texto: str) -> Resultado:
    """Simula a API (latência + falhas aleatórias) para testar sem chave."""
    await asyncio.sleep(random.uniform(0.05, 0.3))
    r = random.random()
    if r < 0.15:
        raise ErroRetentavel("HTTP 429: mock rate limit", retry_after=0.2)
    if r < 0.20:
        raise ErroRetentavel("HTTP 503: mock server error")
    t = texto.lower()
    if any(p in t for p in ("ódio", "horrível", "péssimo", "lixo", "hate", "terrible", "awful")):
        rotulo = "negativo"
    elif any(p in t for p in ("amei", "ótimo", "incrível", "love", "great", "awesome")):
        rotulo = "positivo"
    else:
        rotulo = "neutro"
    p_top = random.uniform(0.4, 0.98)
    resto = (1 - p_top) / 2
    probs = {r_: (p_top if r_ == rotulo else resto) for r_ in ROTULOS}
    return Resultado(rotulo, round(p_top, 4), probs)


# --------------------------------------------------------------------------------------
# Processamento
# --------------------------------------------------------------------------------------
class Processador:
    def __init__(self, args, client, chamar):
        self.args = args
        self.client = client
        self.chamar = chamar
        self.sem = asyncio.Semaphore(args.concorrencia)
        self.ok = 0
        self.falhas = 0
        self.total = 0

    async def classificar(self, a: Amostra, out: CsvAppender, err: CsvAppender):
        ultimo_erro = ""
        tentativas = 0
        try:
            if not a.texto.strip():
                raise ErroDefinitivo("Texto vazio")
            for tentativa in range(1, self.args.tentativas + 1):
                tentativas = tentativa
                try:
                    res = await self.chamar(self.client, a.texto)
                    await out.write([a.id, a.texto, res.rotulo, fmt(res.confidence)]
                                    + [fmt(res.probs.get(r)) for r in ROTULOS])
                    self.ok += 1
                    return
                except ErroDefinitivo:
                    raise
                except ErroRetentavel as e:
                    ultimo_erro = str(e)
                    if tentativa < self.args.tentativas:
                        espera = e.retry_after or min(60, self.args.backoff * 2 ** (tentativa - 1))
                        espera += random.uniform(0, espera * 0.5)  # jitter
                        log.warning("id=%s tentativa %d/%d falhou (%s). Nova tentativa em %.1fs",
                                    a.id, tentativa, self.args.tentativas, ultimo_erro[:150], espera)
                        await asyncio.sleep(espera)
                except Exception as e:  # erro inesperado: trata como retentável
                    ultimo_erro = f"{type(e).__name__}: {e}"
                    log.exception("id=%s erro inesperado", a.id)
        except ErroDefinitivo as e:
            ultimo_erro = str(e)

        self.falhas += 1
        log.error("id=%s FALHOU após %d tentativa(s): %s", a.id, tentativas, ultimo_erro[:300])
        await err.write([a.id, a.texto, ultimo_erro, tentativas,
                         datetime.now(timezone.utc).isoformat(timespec="seconds")])

    async def _executar(self, a: Amostra, out, err):
        try:
            await self.classificar(a, out, err)
        finally:
            self.sem.release()
            feitos = self.ok + self.falhas
            if feitos % self.args.log_cada == 0:
                log.info("Progresso: %d/%d (ok=%d, falhas=%d)", feitos, self.total, self.ok, self.falhas)

    async def rodar(self, amostras: list[Amostra], out: CsvAppender, err: CsvAppender):
        self.total = len(amostras)
        tarefas = set()
        for a in amostras:
            await self.sem.acquire()  # controla quantas requests ficam em voo
            t = asyncio.create_task(self._executar(a, out, err))
            tarefas.add(t)
            t.add_done_callback(tarefas.discard)
        if tarefas:
            await asyncio.gather(*tarefas)


async def main_async(args):
    entrada = Path(args.entrada)
    saida = Path(args.saida or entrada.with_name(entrada.stem + "_classificado.csv"))
    erros = Path(args.erros or entrada.with_name(entrada.stem + "_erros.csv"))

    if saida.exists() and saida.stat().st_size > 0:
        with open(saida, newline="", encoding="utf-8") as f:
            header_existente = next(csv.reader(f), [])
        if header_existente != HEADER_SAIDA:
            raise SystemExit(
                f"O arquivo de saída {saida} tem colunas {header_existente}, mas o formato atual é "
                f"{HEADER_SAIDA}. Renomeie/remova o arquivo antigo ou use --saida com outro nome."
            )

    ids_ok = ler_ids(saida)
    pendentes_erro = ler_erros_pendentes(erros, ids_ok)

    if args.retry_erros:
        amostras = list(pendentes_erro.values())
        log.info("Modo retry: %d amostra(s) com erro para reprocessar.", len(amostras))
    else:
        vistos: set[str] = set()
        amostras = []
        dup = 0
        for a in iterar_entrada(entrada, args.col_id, args.col_texto):
            if a.id in vistos:
                dup += 1
                continue
            vistos.add(a.id)
            if a.id in ids_ok or a.id in pendentes_erro:
                continue
            amostras.append(a)
        log.info("Já classificadas: %d | com erro pendente: %d (use --retry-erros) | "
                  "ids duplicados ignorados: %d | a processar: %d",
                  len(ids_ok), len(pendentes_erro), dup, len(amostras))
        if args.limite:
            amostras = amostras[: args.limite]

    if not amostras:
        log.info("Nada a processar.")
        return

    if args.mock:
        chamar, client = chamar_mock, None
    else:
        key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key or key.startswith("cole-sua-chave"):
            raise SystemExit("OPENAI_API_KEY não definida. Preencha o arquivo .env na raiz do projeto "
                             "(OPENAI_API_KEY=sk-...) ou use --mock para testar.")
        chamar = chamar_decisions
        client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(args.timeout),
            limits=httpx.Limits(max_connections=args.concorrencia, max_keepalive_connections=args.concorrencia),
        )

    out = CsvAppender(saida, HEADER_SAIDA)
    err = CsvAppender(erros, ["id", "texto", "erro", "tentativas", "timestamp"])
    proc = Processador(args, client, chamar)
    try:
        await proc.rodar(amostras, out, err)
    finally:
        out.close()
        err.close()
        if client:
            await client.aclose()
        compactar_erros(erros, ler_ids(saida))
        log.info("Fim. ok=%d falhas=%d | saída: %s | erros: %s", proc.ok, proc.falhas, saida, erros)


def main():
    p = argparse.ArgumentParser(description="Classificação de sentimento (Reddit) via OpenAI Decisions API")
    p.add_argument("entrada", help="CSV de entrada")
    p.add_argument("--saida", help="CSV de saída (padrão: <entrada>_classificado.csv)")
    p.add_argument("--erros", help="CSV de erros (padrão: <entrada>_erros.csv)")
    p.add_argument("--col-id", default="id", help="coluna de id (padrão: id; se não existir usa nº da linha)")
    p.add_argument("--col-texto", default="texto", help="coluna de texto (padrão: texto)")
    p.add_argument("--concorrencia", type=int, default=10, help="requests simultâneas (semaphore)")
    p.add_argument("--tentativas", type=int, default=3, help="tentativas por amostra")
    p.add_argument("--backoff", type=float, default=2.0, help="backoff base em segundos")
    p.add_argument("--timeout", type=float, default=60.0, help="timeout por request (s)")
    p.add_argument("--retry-erros", action="store_true", help="reprocessa apenas as amostras do CSV de erros")
    p.add_argument("--limite", type=int, help="processa no máximo N amostras novas (teste)")
    p.add_argument("--log-cada", type=int, default=50, help="loga progresso a cada N amostras")
    p.add_argument("--log-arquivo", default="classificacao.log", help="arquivo de log")
    p.add_argument("--mock", action="store_true", help="simula a API (sem chave)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_arquivo, encoding="utf-8")],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log.warning("Interrompido. Rode novamente o mesmo comando para continuar de onde parou.")


if __name__ == "__main__":
    main()
