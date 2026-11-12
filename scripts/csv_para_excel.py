"""
Converte o resultado da classificação para Excel (.xlsx), com números como números.

Uso:
    python scripts/csv_para_excel.py
    python scripts/csv_para_excel.py --classificado dados_limpos/x_classificado.csv --amostra dados_limpos/x.csv --saida x.xlsx
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill

RAIZ = Path(__file__).resolve().parent.parent
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

COLUNAS = [
    # (nome, origem, tipo, largura, formato)
    ("id", "id", "str", 13, None),
    ("subreddit", "subreddit", "str", 14, None),
    ("tipo", "tipo", "str", 11, None),
    ("data", "data", "str", 11, None),
    ("rotulo", "rotulo", "str", 10, None),
    ("confidence", "confidence", "num", 11, "0.00"),
    ("prob_negativo", "prob_negativo", "num", 13, "0.00"),
    ("prob_neutro", "prob_neutro", "num", 12, "0.00"),
    ("prob_positivo", "prob_positivo", "num", 13, "0.00"),
    ("n_tokens", "n_tokens", "int", 9, "0"),
    ("texto", "texto", "str", 100, None),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classificado", default=str(RAIZ / "dados_limpos" / "amostra_sentimento_classificado.csv"))
    p.add_argument("--amostra", default=str(RAIZ / "dados_limpos" / "amostra_sentimento.csv"))
    p.add_argument("--saida", default=str(RAIZ / "dados_limpos" / "amostra_sentimento_classificado.xlsx"))
    args = p.parse_args()

    meta = {}
    if Path(args.amostra).exists():
        with open(args.amostra, newline="", encoding="utf-8") as f:
            meta = {r["id"]: r for r in csv.DictReader(f)}

    wb = Workbook()
    ws = wb.active
    ws.title = "classificacao"
    fonte = Font(name="Arial", size=10)
    fonte_cab = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    fundo_cab = PatternFill("solid", start_color="305496")
    sem_quebra = Alignment(wrap_text=False, vertical="top")

    ws.append([c[0] for c in COLUNAS])
    for cell in ws[1]:
        cell.font, cell.fill = fonte_cab, fundo_cab

    n = 0
    with open(args.classificado, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = meta.get(r["id"], {})
            linha = []
            for _, origem, tipo, _, _ in COLUNAS:
                v = r.get(origem) if origem in r else m.get(origem)
                if tipo == "num":
                    v = float(v) if v not in (None, "") else None
                elif tipo == "int":
                    v = int(v) if v not in (None, "") else None
                else:
                    v = ILLEGAL_CHARACTERS_RE.sub("", v or "")
                linha.append(v)
            ws.append(linha)
            n += 1
            # texto começando com "=" não pode virar fórmula
            for i, (_, _, tipo, _, _) in enumerate(COLUNAS, start=1):
                c = ws.cell(row=n + 1, column=i)
                c.font = fonte
                c.alignment = sem_quebra
                if tipo == "str":
                    c.data_type = "s"

    for i, (_, _, _, largura, formato) in enumerate(COLUNAS, start=1):
        letra = ws.cell(row=1, column=i).column_letter
        ws.column_dimensions[letra].width = largura
        if formato:
            for (c,) in ws.iter_rows(min_row=2, min_col=i, max_col=i):
                c.number_format = formato

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(args.saida)
    print(f"OK: {n} linhas -> {args.saida}")


if __name__ == "__main__":
    main()
