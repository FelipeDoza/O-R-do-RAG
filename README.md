# Trabalho Prático — Inteligência Artificial (FACOM/UFMS, 2026/1)
## Construindo o "R" do RAG: recuperação de artigos científicos

**Aluno:** Felipe Dôza de Andrade
**Nível:** Mestrado
**Tema da coleção:** Núcleo de ML da dissertação — *aprendizagem adaptativa/personalizada,
predição de engajamento e evasão, gamificação e modelagem do aluno* (educação +
ML, coletado do ArXiv).

> Este sistema implementa o componente de **recuperação** (o "R" do RAG):
> dada uma consulta em texto, devolve uma lista ranqueada de artigos da coleção.
> Implementa **três** recuperadores sobre a mesma coleção — BM25 (esparso),
> KNN-TF-IDF (denso clássico) e KNN-SPECTER2 (denso neural) — e um módulo de
> aprofundamento **M5 (ranking híbrido)** que funde esparso + denso via RRF.

---

## Estrutura

```
.
├── README.md
├── requirements.txt
├── LINKS.txt                <- link do vídeo
├── CHECKLIST.md
├── data/                    <- corpus.jsonl 
├── src/
│   ├── preprocessing.py     <- tokenização/stopwords/stemming (docs e queries)
│   ├── retrievers.py        <- BM25, KNN-TFIDF, KNN-denso (SPECTER2)
│   ├── fusion.py            <- M5: RRF e soma ponderada
│   └── utils.py             <- IO jsonl, sementes, escrita de run TREC
├── scripts/
│   ├── coleta_arxiv.py      <- coleta a coleção 
│   ├── build_runs.py        <- gera todas as runs .trec
│   └── demo.py              <- DEMONSTRAÇÃO MÍNIMA: consulta -> ranking
├── eval/
│   ├── queries.tsv          <- 15 queries de teste 
│   ├── qrels.tsv            <- gabarito (Anotado à mão via pooling)
│   └── evaluate.py          <- métricas P@k, R@k, MAP, nDCG (do professor)
├── notebooks/runs/          <- saídas .trec dos modelos
└── relatorio/               <- relatorio.tex / relatorio.pdf (formato SBC)
```

## Como reproduzir

```bash
# 1. Ambiente
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"

# 2. Coletar a coleção (ajuste as palavras-chave no topo do script, se quiser)
python scripts/coleta_arxiv.py            # gera data/corpus.jsonl

# 3. Demonstração mínima (uma consulta -> ranking)
python scripts/demo.py "machine learning for student dropout prediction"
python scripts/demo.py --method specter2 "gamification to reduce task abandonment"

# 4. Gerar todas as runs (com SPECTER2 + híbrido M5)
python scripts/build_runs.py --dense specter2

# 5. Construir o qrels (pooling): anote eval/qrels.tsv à mão a partir das runs
#    (veja as instruções dentro do próprio eval/qrels.tsv)

# 6. Avaliar e comparar os sistemas
python eval/evaluate.py --qrels eval/qrels.tsv \
    --runs notebooks/runs/bm25.trec notebooks/runs/knn_tfidf.trec \
           notebooks/runs/knn_specter2.trec notebooks/runs/hybrid_rrf.trec \
    --k 10
```

### Notas
- O recuperador denso usa **SPECTER2** (`allenai/specter2_base` + adapter). Se o
  ambiente não tiver GPU ou der problema com o pacote `adapters`, use a opção
  mais leve: `python scripts/build_runs.py --dense st --st-model sentence-transformers/all-MiniLM-L6-v2`.
- A **busca** densa (cosseno + top-K) é implementada à mão em `src/retrievers.py`
  (`cosine_topk`), conforme exige o enunciado; só os *embeddings* vêm de modelo pronto.
- Sementes fixadas em 42 (`src/utils.set_seed`).

## Uso de assistentes de IA generativa

Conforme exigido pelo enunciado, o uso de GenAI será declarado aqui e no relatório:
apoio à estruturação do código dos recuperadores, revisão de
texto do relatório, sugestões de hiperparâmetros - sempre com revisão e decisão final do autor.

## Vídeo de apresentação

URL: _((https://youtu.be/SPoxzb4csHg))_
