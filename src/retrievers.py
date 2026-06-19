"""
retrievers.py
-------------
Três recuperadores sobre a MESMA coleção (título + abstract):

  1) BM25Retriever        -> recuperação esparsa/lexical (etapa c, obrigatória)
  2) TfidfKnnRetriever    -> recuperação densa clássica via TF-IDF (etapa d)
  3) DenseKnnRetriever    -> recuperação densa neural via embeddings (etapa d)

Todos expõem a mesma interface:
    .search(query: str, k: int) -> List[(doc_id, score)]
    .run_for_queries(queries, k) -> {qid: [(doc_id, score), ...]}

Conexões com a disciplina (para o relatório):
  - BM25  <-> Naïve Bayes / modelos probabilísticos (relevância como probabilidade,
            independência entre termos, ponderação por TF e raridade/IDF).
  - TF-IDF + cosseno + top-K <-> aula de KNN, agora aplicada a RECUPERAÇÃO em vez
            de classificação: em vez de votar uma classe entre os vizinhos,
            devolvemos os próprios vizinhos como ranking.
  - Embeddings densos <-> Redes Neurais (a rede pré-treinada produz o vetor;
            NÓS implementamos a busca por similaridade, conforme exige o enunciado).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .preprocessing import doc_text, tokenize


# ===========================================================================
# Funções de similaridade implementadas "à mão" (numpy), como pede o enunciado
# para o recuperador denso ("a busca deve ser implementada por você").
# ===========================================================================
def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normaliza cada linha para norma 1 (transforma produto interno em cosseno)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return matrix / norms


def cosine_topk(
    query_vec: np.ndarray,
    doc_matrix_normalized: np.ndarray,
    k: int,
) -> List[Tuple[int, float]]:
    """Top-K por similaridade do cosseno.

    Assume que doc_matrix_normalized já está L2-normalizada. Normaliza a
    consulta e calcula o produto interno (= cosseno). Retorna [(indice, score)].
    """
    q = query_vec.astype(np.float32)
    qn = np.linalg.norm(q)
    if qn == 0:
        return []
    q = q / qn
    sims = doc_matrix_normalized @ q  # vetor de similaridades
    k = min(k, sims.shape[0])
    # argpartition é O(n); depois ordenamos só os k candidatos.
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


# ===========================================================================
# 1) BM25 (esparso)
# ===========================================================================
class BM25Retriever:
    """Recuperador BM25 (rank_bm25.BM25Okapi).

    Hiperparâmetros (justificar no relatório, C3):
      - k1 (saturação do TF): controla quão rápido repetições de um termo param
        de aumentar o score. Faixa típica 1.2-2.0; usamos 1.5 (meio-termo
        consagrado na literatura, e.g. Robertson & Zaragoza 2009).
      - b  (normalização por tamanho): penaliza documentos longos. Padrão da
        literatura b=0.75 (abstracts têm tamanho parecido, então a penalização
        moderada é adequada).
    """

    def __init__(self, docs: Sequence[dict], k1: float = 1.5, b: float = 0.75,
                 **tok_kwargs):
        self.docs = list(docs)
        self.doc_ids = [d["arxiv_id"] for d in self.docs]
        self.tok_kwargs = tok_kwargs
        self._tokenized = [tokenize(doc_text(d), **tok_kwargs) for d in self.docs]
        self.k1, self.b = k1, b
        self.bm25 = BM25Okapi(self._tokenized, k1=k1, b=b)

    def search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        q_tokens = tokenize(query, **self.tok_kwargs)
        scores = self.bm25.get_scores(q_tokens)
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.doc_ids[i], float(scores[i])) for i in idx]

    def run_for_queries(self, queries, k: int = 100):
        return {qid: self.search(text, k) for qid, text in queries}


# ===========================================================================
# 2) KNN sobre TF-IDF (denso clássico)
# ===========================================================================
class TfidfKnnRetriever:
    """KNN por similaridade do cosseno sobre vetores TF-IDF.

    Usa o TfidfVectorizer do scikit-learn para vetorizar; a busca top-K por
    cosseno é feita à mão (cosine_topk). Conexão direta com a disciplina:
    ponderação de termos (IDF) + busca por vizinhos (KNN).
    """

    def __init__(self, docs: Sequence[dict], **tok_kwargs):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.docs = list(docs)
        self.doc_ids = [d["arxiv_id"] for d in self.docs]
        self.tok_kwargs = tok_kwargs

        # Passamos nosso tokenizer para garantir o MESMO pré-processamento.
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda t: tokenize(t, **tok_kwargs),
            preprocessor=lambda x: x,
            token_pattern=None,
            min_df=2,         # ignora termos que aparecem em <2 docs (ruído/erros)
            sublinear_tf=True,  # 1+log(tf): amortece termos muito frequentes
        )
        texts = [doc_text(d) for d in self.docs]
        self.doc_matrix = self.vectorizer.fit_transform(texts)  # esparsa
        # Pré-normaliza as linhas (TF-IDF do sklearn já é L2 por padrão, mas
        # garantimos explicitamente para o cosseno via produto interno).
        from sklearn.preprocessing import normalize
        self.doc_matrix = normalize(self.doc_matrix, norm="l2", axis=1)

    def search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        q = self.vectorizer.transform([query])
        from sklearn.preprocessing import normalize
        q = normalize(q, norm="l2", axis=1)
        sims = (self.doc_matrix @ q.T).toarray().ravel()  # cosseno (já normalizado)
        k = min(k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.doc_ids[i], float(sims[i])) for i in idx]

    def run_for_queries(self, queries, k: int = 100):
        return {qid: self.search(text, k) for qid, text in queries}


# ===========================================================================
# 3) KNN sobre embeddings densos (SPECTER2 / Sentence-Transformers)
# ===========================================================================
class DenseKnnRetriever:
    """KNN por cosseno sobre embeddings densos pré-treinados.

    'encoder' é qualquer função texts:List[str] -> np.ndarray (n, dim).
    Use os encoders abaixo:
      - make_specter2_encoder()  -> SPECTER2 (recomendado p/ artigo científico)
      - make_st_encoder(name)    -> qualquer modelo sentence-transformers

    Os embeddings dos documentos são calculados uma vez no __init__ e
    L2-normalizados; a busca top-K por cosseno é implementada à mão.
    """

    def __init__(self, docs: Sequence[dict], encoder: Callable[[List[str]], np.ndarray]):
        self.docs = list(docs)
        self.doc_ids = [d["arxiv_id"] for d in self.docs]
        self.encoder = encoder
        texts = [doc_text(d) for d in self.docs]
        emb = np.asarray(self.encoder(texts), dtype=np.float32)
        self.doc_matrix = _l2_normalize(emb)

    def search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        q_emb = np.asarray(self.encoder([query]), dtype=np.float32)[0]
        hits = cosine_topk(q_emb, self.doc_matrix, k)
        return [(self.doc_ids[i], score) for i, score in hits]

    def run_for_queries(self, queries, k: int = 100):
        return {qid: self.search(text, k) for qid, text in queries}


# ---------------------------------------------------------------------------
# Encoders (lazy imports: só exigem torch/transformers se forem usados)
# ---------------------------------------------------------------------------
def make_st_encoder(model_name: str = "sentence-transformers/allenai-specter",
                    batch_size: int = 32):
    """Encoder via sentence-transformers. Default = SPECTER (v1), 1 linha e robusto.

    Alternativas: 'sentence-transformers/all-MiniLM-L6-v2' (leve/rápido).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(texts: List[str]) -> np.ndarray:
        return model.encode(texts, batch_size=batch_size,
                            show_progress_bar=len(texts) > 64,
                            convert_to_numpy=True)
    return encode


def make_specter2_encoder(batch_size: int = 16, max_length: int = 512):
    """Encoder SPECTER2 oficial (allenai/specter2_base + adapter de proximidade).

    Requer: pip install transformers adapters torch
    SPECTER2 foi treinado especificamente sobre artigos científicos e espera
    a entrada no formato 'titulo [SEP] abstract'. Como concatenamos
    'titulo. abstract' em doc_text, aqui apenas repassamos o texto; para máxima
    fidelidade você pode adaptar para inserir o token [SEP] entre os campos.
    """
    import torch
    from transformers import AutoTokenizer
    try:
        from adapters import AutoAdapterModel
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "SPECTER2 requer o pacote 'adapters'. Instale com: pip install adapters\n"
            "Ou troque para make_st_encoder() (SPECTER v1 / MiniLM)."
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf",
                       load_as="proximity", set_active=True)
    model.to(device).eval()

    @torch.no_grad()
    def encode(texts: List[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tok(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
            emb = model(**inputs).last_hidden_state[:, 0, :]  # token [CLS]
            out.append(emb.cpu().numpy())
        return np.vstack(out)
    return encode
