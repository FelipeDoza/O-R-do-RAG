#!/usr/bin/env python
"""
demo.py  --  Demonstração mínima de uso (entregável obrigatório #3)
-------------------------------------------------------------------
Recebe uma CONSULTA em texto livre e devolve a lista ranqueada de artigos
da coleção, conforme o recuperador escolhido. É o script que o professor
roda para ver o sistema funcionando com pelo menos uma consulta-exemplo.

Exemplos:
  python scripts/demo.py "machine learning for student dropout prediction"
  python scripts/demo.py --method tfidf  "knowledge tracing with deep learning"
  python scripts/demo.py --method specter2 "gamification to reduce task abandonment"
  python scripts/demo.py --method hybrid --dense specter2 "adaptive learning for dyslexia"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_jsonl, set_seed
from src.retrievers import (
    BM25Retriever, TfidfKnnRetriever, DenseKnnRetriever,
    make_specter2_encoder, make_st_encoder,
)
from src.fusion import reciprocal_rank_fusion


def main():
    ap = argparse.ArgumentParser(description="Demo: consulta -> ranking de artigos.")
    ap.add_argument("query", help="consulta em texto livre (entre aspas)")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus.jsonl")
    ap.add_argument("--method", choices=["bm25", "tfidf", "specter2", "st", "hybrid"],
                    default="bm25")
    ap.add_argument("--dense", choices=["specter2", "st"], default="specter2",
                    help="qual denso usar quando --method hybrid")
    ap.add_argument("--st-model", default="sentence-transformers/allenai-specter")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    set_seed(42)
    docs = load_jsonl(args.corpus)
    by_id = {d["arxiv_id"]: d for d in docs}

    def dense_encoder():
        if args.dense == "specter2" or args.method == "specter2":
            return make_specter2_encoder()
        return make_st_encoder(args.st_model)

    if args.method == "bm25":
        ranking = BM25Retriever(docs).search(args.query, k=args.k)
    elif args.method == "tfidf":
        ranking = TfidfKnnRetriever(docs).search(args.query, k=args.k)
    elif args.method in ("specter2", "st"):
        ranking = DenseKnnRetriever(docs, encoder=dense_encoder()).search(args.query, k=args.k)
    else:  # hybrid (M5)
        bm25 = BM25Retriever(docs).search(args.query, k=100)
        dense = DenseKnnRetriever(docs, encoder=dense_encoder()).search(args.query, k=100)
        ranking = reciprocal_rank_fusion([bm25, dense], k=args.k)

    print(f'\nConsulta: "{args.query}"   (método={args.method}, top-{args.k})\n')
    for rank, (doc_id, score) in enumerate(ranking, 1):
        d = by_id.get(doc_id, {})
        title = d.get("title", "?")
        year = (d.get("published") or "")[:4]
        cat = d.get("primary_category", "")
        print(f"{rank:>2}. [{score:7.4f}] {title}")
        print(f"     {doc_id} | {cat} | {year}")
        abs = (d.get('abstract') or '')[:180]
        print(f"     {abs}...\n")


if __name__ == "__main__":
    main()
