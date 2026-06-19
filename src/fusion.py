"""
fusion.py  --  Módulo de aprofundamento M5 (Ranking Híbrido sparse+dense)
-------------------------------------------------------------------------
Combina os rankings de BM25 (esparso) e do recuperador denso (SPECTER2/TF-IDF)
em um único ranking, buscando o melhor dos dois mundos: o lexical é forte quando
a query usa exatamente os termos do documento; o denso é forte em paráfrases e
sinônimos. A fusão tende a recuperar relevantes que um dos dois sozinho perderia.

Oferecemos duas estratégias clássicas:

1) Reciprocal Rank Fusion (RRF) -- robusta e SEM necessidade de normalizar
   escores (que vivem em escalas diferentes: BM25 não é limitado, cosseno e [-1,1]).
   score_RRF(d) = soma_sistemas 1 / (c + rank_sistema(d)),  c=60 (valor canônico).

2) Soma ponderada de escores normalizados (min-max) -- permite controlar o peso
   alpha do componente denso vs. esparso, e conecta com o módulo M4 (otimizar alpha).
   score(d) = alpha * dense_norm(d) + (1-alpha) * sparse_norm(d).
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    c: int = 60,
    k: int = 100,
) -> List[Tuple[str, float]]:
    """Funde várias listas ranqueadas via RRF.

    rankings : lista de rankings; cada ranking é [(doc_id, score), ...] ordenado.
    Os SCORES originais são ignorados pelo RRF (ele usa apenas as POSIÇÕES).
    """
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (c + rank)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ordered[:k]


def _minmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1e-12
    return {d: (s - lo) / rng for d, s in scores.items()}


def weighted_sum_fusion(
    sparse_ranking: List[Tuple[str, float]],
    dense_ranking: List[Tuple[str, float]],
    alpha: float = 0.5,
    k: int = 100,
) -> List[Tuple[str, float]]:
    """Funde dois rankings por soma ponderada de escores normalizados (min-max).

    alpha em [0,1]: peso do componente DENSO. alpha=0 -> só esparso;
    alpha=1 -> só denso. Documentos ausentes em um dos rankings recebem 0
    naquele componente.
    """
    sparse = _minmax(dict(sparse_ranking))
    dense = _minmax(dict(dense_ranking))
    all_docs = set(sparse) | set(dense)
    fused = {
        d: alpha * dense.get(d, 0.0) + (1 - alpha) * sparse.get(d, 0.0)
        for d in all_docs
    }
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ordered[:k]
