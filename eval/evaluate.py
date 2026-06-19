"""
evaluate.py
-----------
Script de avaliacao para o trabalho pratico de IA (FACOM/UFMS, 2026/1).

Le um arquivo de qrels (relevance judgments) e um ou mais arquivos de run
no formato TREC, e reporta P@k, R@k, MAP e nDCG@k para cada sistema.

Uso:
    python evaluate.py --qrels qrels.tsv --runs ../notebooks/runs/bm25.trec \\
                       ../notebooks/runs/knn.trec --k 10

Dependencias:
    pip install pandas numpy

Observacao: Este script tem implementacoes simples das metricas para fins
didaticos. Em producao/pesquisa, prefira `pytrec_eval` ou `ir_measures`.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Leitura de arquivos
# ---------------------------------------------------------------------------

def read_qrels(path: Path):
    """Le qrels no formato TREC: qid 0 doc_id rel. Comentarios com '#'."""
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            qrels[qid][docid] = rel
    return qrels


def read_run(path: Path):
    """Le run no formato TREC: qid Q0 doc_id rank score system."""
    run = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            qid, _, docid, rank, score = parts[0], parts[1], parts[2], int(parts[3]), float(parts[4])
            run[qid].append((rank, score, docid))
    # ordena por rank (caso o arquivo nao venha ordenado)
    for qid in run:
        run[qid].sort(key=lambda x: x[0])
    return run


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def precision_at_k(ranked_docs, qrels_q, k):
    if not ranked_docs:
        return 0.0
    top = ranked_docs[:k]
    rel = sum(1 for d in top if qrels_q.get(d, 0) > 0)
    return rel / k


def recall_at_k(ranked_docs, qrels_q, k):
    total_rel = sum(1 for v in qrels_q.values() if v > 0)
    if total_rel == 0:
        return 0.0
    top = ranked_docs[:k]
    rel = sum(1 for d in top if qrels_q.get(d, 0) > 0)
    return rel / total_rel


def average_precision(ranked_docs, qrels_q):
    total_rel = sum(1 for v in qrels_q.values() if v > 0)
    if total_rel == 0:
        return 0.0
    score = 0.0
    hits = 0
    for i, d in enumerate(ranked_docs, 1):
        if qrels_q.get(d, 0) > 0:
            hits += 1
            score += hits / i
    return score / total_rel


def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_docs, qrels_q, k):
    top = ranked_docs[:k]
    gains = [qrels_q.get(d, 0) for d in top]
    ideal = sorted(qrels_q.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(gains) / idcg


# ---------------------------------------------------------------------------
# Avaliacao
# ---------------------------------------------------------------------------

def evaluate(qrels, run, k):
    rows = []
    qids = sorted(set(qrels.keys()) & set(run.keys()))
    for qid in qids:
        ranked = [docid for _, _, docid in run[qid]]
        rows.append({
            "qid": qid,
            f"P@{k}": precision_at_k(ranked, qrels[qid], k),
            f"R@{k}": recall_at_k(ranked, qrels[qid], k),
            "AP": average_precision(ranked, qrels[qid]),
            f"nDCG@{k}": ndcg_at_k(ranked, qrels[qid], k),
        })
    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser(description="Avaliacao IR no formato TREC.")
    ap.add_argument("--qrels", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path, nargs="+")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    qrels = read_qrels(args.qrels)
    if not qrels:
        print(f"[erro] qrels vazio em {args.qrels}", file=sys.stderr)
        sys.exit(1)

    summary = []
    for run_path in args.runs:
        run = read_run(run_path)
        df = evaluate(qrels, run, args.k)
        if df.empty:
            print(f"[aviso] sem queries em comum entre qrels e {run_path.name}")
            continue
        print(f"\n=== {run_path.name} ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        means = df.mean(numeric_only=True)
        print("\nMedias:")
        print(means.to_string(float_format=lambda x: f"{x:.4f}"))
        summary.append({
            "system": run_path.stem,
            **{k: float(v) for k, v in means.items()},
        })

    if summary:
        print("\n=== Resumo (medias) ===")
        sm = pd.DataFrame(summary)
        print(sm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
