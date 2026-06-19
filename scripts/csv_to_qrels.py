#!/usr/bin/env python
"""
csv_to_qrels.py  --  converte a planilha anotada em eval/qrels.tsv (formato TREC)
---------------------------------------------------------------------------------
Lê eval/pool_to_annotate.csv (com a coluna REL preenchida) e gera eval/qrels.tsv.
Linhas com REL vazio são tratadas como 0 (não-relevante), convenção de pooling.

Uso:
    python scripts/csv_to_qrels.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "eval" / "pool_to_annotate.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "qrels.tsv")
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"Não achei {args.csv}. Rode build_pool.py e preencha a coluna REL.")

    n_total = n_rel1 = n_rel2 = n_blank = n_bad = 0
    lines = []
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid, docid = row["qid"].strip(), row["doc_id"].strip()
            raw = (row.get("REL") or "").strip()
            if raw == "":
                rel, n_blank = 0, n_blank + 1
            else:
                try:
                    rel = int(float(raw))
                except ValueError:
                    n_bad += 1
                    print(f"[aviso] REL inválido em {qid}/{docid}: {raw!r} -> tratado como 0")
                    rel = 0
                if rel not in (0, 1, 2):
                    print(f"[aviso] REL fora de 0/1/2 em {qid}/{docid}: {rel} -> 0")
                    rel = 0
            n_total += 1
            n_rel1 += rel == 1
            n_rel2 += rel == 2
            lines.append(f"{qid}\t0\t{docid}\t{rel}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"qrels gerado: {args.out}")
    print(f"  {n_total} julgamentos | relevantes: {n_rel1} (rel=1) + {n_rel2} (rel=2)"
          f" | em branco->0: {n_blank}")
    rel_total = n_rel1 + n_rel2
    if rel_total == 0:
        print("[ATENÇÃO] Nenhum documento marcado como relevante. As métricas "
              "darão 0. Preencha a coluna REL com 1/2 nos relevantes.")


if __name__ == "__main__":
    main()
