#!/usr/bin/env python
"""
build_runs.py
-------------
Constrói os recuperadores sobre data/corpus.jsonl, roda todas as queries de
eval/queries.tsv e grava as runs em notebooks/runs/ no formato TREC.

Runs geradas:
  - bm25.trec          (esparso, obrigatório)
  - knn_tfidf.trec     (denso clássico, obrigatório)
  - knn_specter2.trec  (denso neural, se --dense for ativado)
  - hybrid_rrf.trec    (módulo M5: fusão BM25 + denso via RRF)

Exemplos:
  # Só os baselines sem rede neural (rápido, não baixa modelos):
  python scripts/build_runs.py

  # Com SPECTER2 + híbrido (baixa o modelo na 1a vez):
  python scripts/build_runs.py --dense specter2

  # Com SPECTER v1 via sentence-transformers (mais leve que SPECTER2):
  python scripts/build_runs.py --dense st --st-model sentence-transformers/allenai-specter
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite rodar o script de qualquer lugar (adiciona a raiz do projeto ao path).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_jsonl, load_queries, write_trec_run, set_seed
from src.retrievers import (
    BM25Retriever, TfidfKnnRetriever, DenseKnnRetriever,
    make_specter2_encoder, make_st_encoder,
)
from src.fusion import reciprocal_rank_fusion


def main():
    ap = argparse.ArgumentParser(description="Gera as runs TREC dos recuperadores.")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus.jsonl")
    ap.add_argument("--queries", type=Path, default=ROOT / "eval" / "queries.tsv")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "notebooks" / "runs")
    ap.add_argument("--k", type=int, default=100, help="tamanho do ranking (run)")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--dense", choices=["none", "specter2", "st"], default="none",
                    help="recuperador denso neural a usar (e habilita o híbrido)")
    ap.add_argument("--st-model", default="sentence-transformers/allenai-specter")
    ap.add_argument("--rrf-c", type=int, default=60)
    args = ap.parse_args()

    set_seed(42)
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Carregando corpus de {args.corpus} ...")
    docs = load_jsonl(args.corpus)
    queries = load_queries(args.queries)
    print(f"  {len(docs)} documentos | {len(queries)} queries de teste")

    # ---- BM25 (obrigatório) -------------------------------------------------
    print("Construindo BM25 ...")
    bm25 = BM25Retriever(docs, k1=args.k1, b=args.b)
    bm25_runs = bm25.run_for_queries(queries, k=args.k)
    write_trec_run(args.runs_dir / "bm25.trec", bm25_runs, system="bm25")
    print("  -> bm25.trec")

    # ---- KNN TF-IDF (obrigatório) ------------------------------------------
    print("Construindo KNN TF-IDF ...")
    tfidf = TfidfKnnRetriever(docs)
    tfidf_runs = tfidf.run_for_queries(queries, k=args.k)
    write_trec_run(args.runs_dir / "knn_tfidf.trec", tfidf_runs, system="knn_tfidf")
    print("  -> knn_tfidf.trec")

    # ---- KNN denso neural + Híbrido M5 (opcional) --------------------------
    if args.dense != "none":
        if args.dense == "specter2":
            print("Carregando SPECTER2 (pode baixar pesos na 1a vez) ...")
            encoder = make_specter2_encoder()
            dense_name = "knn_specter2"
        else:
            print(f"Carregando sentence-transformers: {args.st_model} ...")
            encoder = make_st_encoder(args.st_model)
            dense_name = "knn_st"

        dense = DenseKnnRetriever(docs, encoder=encoder)
        dense_runs = dense.run_for_queries(queries, k=args.k)
        write_trec_run(args.runs_dir / f"{dense_name}.trec", dense_runs, system=dense_name)
        print(f"  -> {dense_name}.trec")

        # M5: fusão RRF do BM25 com o denso neural
        print("Fundindo (M5 / RRF) ...")
        hybrid_runs = {
            qid: reciprocal_rank_fusion(
                [bm25_runs[qid], dense_runs[qid]], c=args.rrf_c, k=args.k
            )
            for qid, _ in queries
        }
        write_trec_run(args.runs_dir / "hybrid_rrf.trec", hybrid_runs, system="hybrid_rrf")
        print("  -> hybrid_rrf.trec")
    else:
        print("[info] --dense none: pulei SPECTER2 e o híbrido. "
              "Rode com --dense specter2 para gerar knn_specter2.trec e hybrid_rrf.trec.")

    print("\nPronto. Agora avalie com:")
    print(f"  python eval/evaluate.py --qrels eval/qrels.tsv "
          f"--runs {args.runs_dir}/*.trec --k {args.k}")


if __name__ == "__main__":
    main()
