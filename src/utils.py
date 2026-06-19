"""
utils.py
--------
Funções utilitárias: leitura de corpus, fixação de sementes e escrita de
arquivos de run no formato TREC (compatível com o eval/evaluate.py do professor).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fixa sementes para reprodutibilidade (rubrica C9)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_jsonl(path: str | Path) -> List[dict]:
    """Carrega um arquivo .jsonl (uma linha JSON por documento)."""
    path = Path(path)
    docs: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def load_queries(path: str | Path) -> List[Tuple[str, str]]:
    """Lê queries.tsv no formato 'qid<TAB>texto'. Retorna [(qid, texto), ...]."""
    path = Path(path)
    queries: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            qid, _, text = line.partition("\t")
            queries.append((qid.strip(), text.strip()))
    return queries


def write_trec_run(
    path: str | Path,
    runs: Dict[str, List[Tuple[str, float]]],
    system: str,
) -> None:
    """Escreve um arquivo de run no formato TREC.

    Formato (idêntico ao baseline do professor, lido pelo evaluate.py):
        qid Q0 doc_id rank score system

    Parâmetros
    ----------
    runs : {qid: [(doc_id, score), ...ordenado por score desc]}
    system : rótulo do sistema (e.g. 'bm25', 'knn_tfidf', 'hybrid_rrf')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked in runs.items():
            for rank, (doc_id, score) in enumerate(ranked, 1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {system}\n")
