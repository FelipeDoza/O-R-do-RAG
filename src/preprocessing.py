"""
preprocessing.py
----------------
Pré-processamento textual compartilhado entre documentos e consultas.

Decisão de projeto importante (documentar no relatório, Seção 6/C2):
o MESMO pipeline de pré-processamento é aplicado tanto aos documentos da
coleção quanto às consultas. Se a consulta e o documento fossem tokenizados
de formas diferentes, os termos não casariam e tanto o BM25 quanto o
recuperador TF-IDF perderiam recall.

Conexão com a disciplina: a tokenização + remoção de stopwords + ponderação
posterior (IDF) é exatamente a etapa de "representação de texto como
características" que precede tanto o paradigma probabilístico (Naïve Bayes /
BM25) quanto a busca por similaridade (KNN).
"""
from __future__ import annotations

import re
from typing import List

import nltk

# Garante as stopwords do NLTK (baixa só na primeira vez).
try:
    from nltk.corpus import stopwords
    _STOP_EN = set(stopwords.words("english"))
except LookupError:  # pragma: no cover
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    _STOP_EN = set(stopwords.words("english"))

# Captura tokens alfabéticos (permitindo hífen interno, e.g. "self-supervised").
# Optamos por NÃO manter números porque, em abstracts científicos, números
# soltos (anos, métricas) raramente discriminam tópico e só inflam o vocabulário.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")

# Stemmer de Porter (opcional). Discussão para o relatório:
# - PRÓ: aproxima formas flexionadas ("learning"/"learn", "network"/"networks"),
#        aumentando recall em queries curtas.
# - CONTRA: pode colapsar termos técnicos distintos e prejudicar precisão
#        ("transformer" vs "transform"). Por isso é uma FLAG, não o default.
_PORTER = nltk.PorterStemmer()


def tokenize(
    text: str,
    remove_stopwords: bool = True,
    min_len: int = 3,
    stemming: bool = False,
) -> List[str]:
    """Converte um texto livre em uma lista de tokens normalizados.

    Etapas:
      1. lower-casing (normalização de caixa);
      2. extração de tokens alfabéticos via regex (remove pontuação/dígitos);
      3. remoção de stopwords em inglês (opcional);
      4. descarte de tokens muito curtos (ruído);
      5. stemming de Porter (opcional).

    Os mesmos parâmetros devem ser usados para documentos e consultas.
    """
    text = (text or "").lower()
    tokens = _TOKEN_RE.findall(text)

    out: List[str] = []
    for tok in tokens:
        if len(tok) < min_len:
            continue
        if remove_stopwords and tok in _STOP_EN:
            continue
        if stemming:
            tok = _PORTER.stem(tok)
        out.append(tok)
    return out


def doc_text(doc: dict) -> str:
    """Texto indexável de um documento = título + abstract.

    O enunciado pede explicitamente recuperação sobre 'título + abstract'.
    O título costuma carregar os termos mais discriminativos do artigo, por
    isso é concatenado ao abstract (que dá contexto e cobertura de vocabulário).
    """
    title = (doc.get("title") or "").strip()
    abstract = (doc.get("abstract") or "").strip()
    return f"{title}. {abstract}"
