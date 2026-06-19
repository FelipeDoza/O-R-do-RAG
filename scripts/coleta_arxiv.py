#!/usr/bin/env python
"""
coleta_arxiv.py
---------------
Coleta a coleção do ArXiv para o tema do trabalho e gera data/corpus.jsonl.
Baseado no notebook do professor, com coleta robusta (backoff contra HTTP
429/503) e salvamento incremental: se cair, basta rodar de novo que retoma.

================================  EDITE AQUI  ================================
O ESCOPO abaixo já está pré-configurado para o NÚCLEO de ML da sua dissertação
(gamificação adaptativa + ML + engajamento/evasão + aprendizagem personalizada).
Revise as palavras-chave com seu olhar de especialista no domínio: tire o que
trouxer ruído, acrescente termos que você sabe que são centrais. Lembre-se:
esta é a QUERY DE COLETA (ampla, um grande OR) -- NÃO confunda com as queries
de teste (específicas) que ficam em eval/queries.tsv.
============================================================================
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import arxiv
import pandas as pd
from tqdm import tqdm

# ----------------------------- ESCOPO DA COLEÇÃO ----------------------------
KEYWORDS = [
    "gamification",
    "adaptive learning",
    "personalized learning",
    "learning analytics",
    "knowledge tracing",
    "student engagement",
    "student dropout prediction",
    "intelligent tutoring system",
    "educational data mining",
    "student performance prediction",
    "reinforcement learning for education",
    "adaptive educational system",
]
# Categorias ArXiv onde tecnologia educacional/IHC/ML aparecem.
CATEGORIES = ["cs.CY", "cs.HC", "cs.LG", "cs.AI"]
YEAR_FROM = 2015           # era do deep learning aplicado à educação
YEAR_TO = 2026
TARGET_SIZE = 2500         # entre 1.000 e 5.000 (enunciado)
PAGE_SIZE = 50             # 50 é mais robusto que 100 contra 429/503
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_PATH = OUTPUT_DIR / "arxiv_raw.jsonl"
CORPUS_PATH = OUTPUT_DIR / "corpus.jsonl"


def build_query(keywords, categories):
    kw = " OR ".join(f'all:"{k}"' for k in keywords) if keywords else ""
    cat = " OR ".join(f"cat:{c}" for c in categories) if categories else ""
    parts = [p for p in [f"({kw})" if kw else "", f"({cat})" if cat else ""] if p]
    return " AND ".join(parts) if parts else "all:*"


def already_collected_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["arxiv_id"])
            except Exception:
                continue
    return ids


def collect(query, target, page_size, y0, y1, out_path,
            max_attempts=6, backoff=60):
    client = arxiv.Client(page_size=page_size, delay_seconds=5, num_retries=8)
    seen = already_collected_ids(out_path)
    print(f"Já coletados: {len(seen)}")
    offset, attempt = 0, 0
    while len(seen) < target and attempt < max_attempts:
        try:
            search = arxiv.Search(
                query=query, max_results=target * 3,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            print(f"Retomando do offset={offset} (salvos={len(seen)}/{target})")
            with open(out_path, "a", encoding="utf-8") as f:
                pbar = tqdm(initial=len(seen), total=target, desc="coletando")
                for r in client.results(search, offset=offset):
                    offset += 1
                    year = r.published.year if r.published else None
                    if y0 and (year is None or year < y0):
                        continue
                    if y1 and (year is None or year > y1):
                        continue
                    aid = r.get_short_id().split("v")[0]
                    if aid in seen:
                        continue
                    rec = {
                        "arxiv_id": aid,
                        "title": (r.title or "").strip(),
                        "abstract": (r.summary or "").strip().replace("\n", " "),
                        "authors": [a.name for a in r.authors],
                        "categories": list(r.categories or []),
                        "primary_category": r.primary_category,
                        "published": r.published.isoformat() if r.published else None,
                        "updated": r.updated.isoformat() if r.updated else None,
                        "doi": r.doi,
                        "pdf_url": r.pdf_url,
                        "entry_id": r.entry_id,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    seen.add(aid)
                    pbar.update(1); pbar.set_postfix(offset=offset)
                    if len(seen) >= target:
                        break
                pbar.close()
            break
        except Exception as e:
            attempt += 1
            wait = min(backoff * (2 ** (attempt - 1)), 600)
            print(f"[aviso] falha {attempt}/{max_attempts}: {type(e).__name__}: {e}")
            print(f"[aviso] aguardando {wait}s e retomando do offset={offset}...")
            time.sleep(wait)
    return len(seen)


def main():
    query = build_query(KEYWORDS, CATEGORIES)
    print("Query de coleta:\n", query, "\n")
    collect(query, TARGET_SIZE, PAGE_SIZE, YEAR_FROM, YEAR_TO, RAW_PATH)

    # Deduplicação e limpeza -> corpus.jsonl
    raw = [json.loads(l) for l in open(RAW_PATH, encoding="utf-8")]
    df = pd.DataFrame(raw)
    df["updated_dt"] = pd.to_datetime(df["updated"], errors="coerce")
    df = df.sort_values("updated_dt").drop_duplicates("arxiv_id", keep="last")
    df = df[df["title"].str.len() > 0]
    df = df[df["abstract"].str.len() > 50]   # abstracts curtos = ruído
    cols = ["arxiv_id", "title", "abstract", "authors", "categories",
            "primary_category", "published", "doi", "pdf_url"]
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for _, row in df[cols].iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"\nCorpus final: {len(df)} documentos -> {CORPUS_PATH}")

    df["year"] = pd.to_datetime(df["published"], errors="coerce").dt.year
    print("\nDistribuição por ano:\n", df["year"].value_counts().sort_index())
    print("\nTop categorias primárias:\n",
          df["primary_category"].value_counts().head(10))


if __name__ == "__main__":
    main()
