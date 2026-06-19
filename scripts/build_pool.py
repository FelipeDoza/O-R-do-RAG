#!/usr/bin/env python
"""
build_pool.py  --  monta o POOL de anotação (pooling) numa planilha CSV
-----------------------------------------------------------------------
Depois de gerar as runs (scripts/build_runs.py), este script junta o top-N de
TODOS os sistemas para cada query e cria uma planilha CSV pronta para anotar:

    eval/pool_to_annotate.csv   (colunas: qid, query, n_sistemas, melhor_rank,
                                 doc_id, title, abstract, REL)

Você abre essa planilha no Excel/Google Sheets, lê o título/abstract e digita
na última coluna (REL): 2 = altamente relevante, 1 = relevante, 0 = não.
Os candidatos vêm ordenados por quantos sistemas os recuperaram e pelo melhor
rank, então os mais prováveis ficam no topo (anotação mais rápida).

Depois, rode scripts/csv_to_qrels.py para gerar eval/qrels.tsv.

Uso:
    python scripts/build_pool.py --top 10
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_jsonl, load_queries


def read_run(path: Path):
    run = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            qid, docid, rank = p[0], p[2], int(p[3])
            run[qid].append((rank, docid))
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus.jsonl")
    ap.add_argument("--queries", type=Path, default=ROOT / "eval" / "queries.tsv")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "notebooks" / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "pool_to_annotate.csv")
    ap.add_argument("--top", type=int, default=10, help="top-N de cada run no pool")
    args = ap.parse_args()

    docs = {d["arxiv_id"]: d for d in load_jsonl(args.corpus)}
    queries = dict(load_queries(args.queries))

    run_files = sorted(args.runs_dir.glob("*.trec"))
    if not run_files:
        sys.exit(f"Nenhuma run encontrada em {args.runs_dir}. Rode build_runs.py antes.")
    runs = {f.stem: read_run(f) for f in run_files}
    print("Runs no pool:", ", ".join(runs.keys()))

    rows = []
    for qid, qtext in queries.items():
        # agrega: para cada doc, em quantos sistemas apareceu e o melhor rank
        agg = defaultdict(lambda: {"n": 0, "best": 10**9})
        for run in runs.values():
            for rank, docid in run.get(qid, [])[: args.top]:
                agg[docid]["n"] += 1
                agg[docid]["best"] = min(agg[docid]["best"], rank)
        # ordena: mais sistemas primeiro, depois melhor rank
        ordered = sorted(agg.items(), key=lambda x: (-x[1]["n"], x[1]["best"]))
        for docid, info in ordered:
            d = docs.get(docid, {})
            rows.append({
                "qid": qid,
                "query": qtext,
                "n_sistemas": info["n"],
                "melhor_rank": info["best"],
                "doc_id": docid,
                "title": d.get("title", ""),
                "abstract": (d.get("abstract", "") or "")[:500],
                "REL": "",   # <- você preenche: 0, 1 ou 2
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig para o Excel abrir acentos corretamente
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "query", "n_sistemas",
                                          "melhor_rank", "doc_id", "title",
                                          "abstract", "REL"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nPool gerado: {len(rows)} pares (query, doc) -> {args.out}")
    print(f"Queries: {len(queries)} | docs únicos por query (médio): "
          f"{len(rows)/max(len(queries),1):.1f}")
    print("\nAgora: abra o CSV, preencha a coluna REL (0/1/2) e rode:")
    print("  python scripts/csv_to_qrels.py")


if __name__ == "__main__":
    main()
