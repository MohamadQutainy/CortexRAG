# CortexRAG
### CortexRAG - Production-grade RAG framework built from scratch.
### Hybrid retrieval, modular vector backends, LLM reranking and a full 3-layer evaluation suite. Zero LangChain. Pure Python.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/LLM-OpenAI%20%7C%20Local-green?style=for-the-badge&logo=openai" />
  <img src="https://img.shields.io/badge/Vector%20DB-Chroma%20%7C%20FAISS%20%7C%20Pinecone%20%7C%20Milvus-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/RAG-Advanced%20%7C%20Agentic%20%7C%20Graph-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Evaluation-Hit%40k%20%7C%20MRR%20%7C%20nDCG-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/UI-Gradio-yellow?style=for-the-badge" />
</p>

---

> **CortexRAG** is not a wrapper around existing libraries it is a ground-up, modular, production-ready RAG framework engineered with clean abstractions, a centralized configuration system, and a full evaluation suite. It ships with a domain-agnostic synthetic data generator (demonstrated on a real-estate domain) that produces 110+ richly structured documents and a scored Q&A evaluation set giving you a realistic,end-to-end test bed out of the box.

---

## Table of Contents

1. [Why CortexRAG?](#1-why-cortexrag)
2. [System Architecture](#2-system-architecture)
3. [Full Directory Structure](#3-full-directory-structure)
4. [Feature Matrix](#4-feature-matrix)
5. [Synthetic Data Generator](#5-synthetic-data-generator)
6. [Pipeline Deep Dive](#6-pipeline-deep-dive)
   - [Ingestion Flow](#a-ingestion-flow)
   - [Query Flow](#b-query-flow)
7. [Configuration System](#7-configuration-system)
8. [Vector Store Backends](#8-vector-store-backends)
9. [Retrieval Strategies](#9-retrieval-strategies)
10. [Advanced RAG Modules](#10-advanced-rag-modules)
11. [Evaluation Framework](#11-evaluation-framework)
12. [Results](#12-results)
13. [Quick Start](#13-quick-start)
14. [Gradual Feature Activation Guide](#14-gradual-feature-activation-guide)
15. [Key Design Decisions](#15-key-design-decisions)
16. [Extending the Framework](#16-extending-the-framework)
17. [Limitations](#17-limitations)
18. [Requirements](#18-requirements)


---

## 1. Why CortexRAG?

Many RAG repositories are useful as demos but become difficult to extend, evaluate, or debug once the pipeline grows beyond a basic prototype. CortexRAG is designed as a readable experimentation framework: components are separated behind simple interfaces, behavior is driven through configuration, and evaluation is part of the development loop rather than an afterthought.

| Problem in Typical RAG Repos | How CortexRAG Addresses It |
|---|---|
| Hardcoded model names, chunk sizes, and strategies | Single `config.yaml` controls the entire system zero code changes needed to swap components |
| No reusable test data | Includes a synthetic dataset generator for building a linked corpus and paired evaluation set for repeatable experiments |
| Single retrieval strategy | Supports Vector, BM25, and Hybrid (RRF) retrieval switchable at config level |
| No re-ranking | LLM-based and Cross-Encoder re-ranking built in as interchangeable modules |
| No evaluation beyond spot checks | Includes retrieval metrics, answer scoring, and semantic similarity evaluation |
| Tight coupling to one framework | Core components are implemented as project-native modules with clear extension points |
| Limited operational visibility | Includes caching, logging, and timing hooks to support debugging and profiling |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CortexRAG Framework                          │
│                                                                     │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐  │
│  │  config.yaml │────▶│  config.py  (Factory + Loader)           │  │
│  └──────────────┘     └──────────────────┬───────────────────────┘  │
│                                          │                           │
│              ┌───────────────────────────▼──────────────────────┐   │
│              │               rag/pipeline.py                     │   │
│              │           build_generator()                       │   │
│              └───┬─────────────┬──────────────┬─────────────────┘   │
│                  │             │              │                      │
│         ┌────────▼──┐  ┌───────▼──────┐  ┌───▼──────────┐          │
│         │ Ingestion │  │   Retrieval  │  │  Generation  │          │
│         │  Flow     │  │   + Rerank   │  │  + Prompts   │          │
│         └────┬──────┘  └──────┬───────┘  └──────┬───────┘          │
│              │                │                  │                  │
│   ┌──────────▼──┐    ┌────────▼────────┐  ┌──────▼──────────────┐  │
│   │  Chunking   │    │  Vector / BM25  │  │  Query Rewriter /   │  │
│   │  Embeddings │    │  Hybrid (RRF)   │  │  Query Expander     │  │
│   │  VectorStore│    │  Cross-Encoder  │  │  LLM Generator      │  │
│   └─────────────┘    └─────────────────┘  └─────────────────────┘  │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Advanced Modules: Graph RAG │ Hierarchical RAG │ Agentic   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Evaluation: Retrieval Metrics │ LLM Judge │ Semantic Sim.  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   ┌──────────────────────┐        ┌──────────────────────────────┐  │
│   │  main.py  (CLI)      │        │  app.py  (Gradio UI)         │  │
│   │  ingest / query /    │        │  Interactive chat + context  │  │
│   │  evaluate            │        │  source viewer               │  │
│   └──────────────────────┘        └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

The two runtime paths **Ingestion** and **Query** are fully decoupled. Ingestion runs once (or on data updates) and writes to the vector store. The Query path reads from it at inference time. Both paths share the same config and pipeline orchestrator, ensuring consistency.

---

## 3. Full Directory Structure

```
CortexRAG/
│
├── config.yaml                     # Central control models, strategies, sizes, weights
├── config.py                       # YAML loader + Factory (create_* functions)
├── main.py                         # CLI entry point: ingest | query | evaluate
├── app.py                          # Gradio web UI with source context viewer
├── generate_dataset.py             # Synthetic data generator (fast | llm | hybrid modes)
├── requirements.txt                # Layered dependencies (core + optional)
├── .env                            # API keys (gitignored)
├── eval_app.py                     # Gradio interface for displaying evaluation quickly
│
├── rag/                            # Core framework
│   ├── __init__.py
│   ├── pipeline.py                 # build_generator() assembles the full pipeline
│   │
│   ├── ingestion/
│   │   ├── loader.py               # File loader with LangChain upgrade notes
│   │   └── preprocessor.py         # Text cleaning and normalization
│   │
│   ├── chunking/
│   │   ├── base.py                 # Abstract chunker interface
│   │   ├── semantic_chunker.py     # Embedding-based semantic splitting
│   │   └── recursive_chunker.py    # Recursive character splitting
│   │
│   ├── embeddings/
│   │   ├── base.py
│   │   ├── openai_embedder.py      # text-embedding-3-large (default)
│   │   └── sentence_transformer.py # Local HuggingFace embeddings (free)
│   │
│   ├── vector_store/
│   │   ├── base.py
│   │   ├── chroma_store.py         # ChromaDB - local default
│   │   ├── faiss_store.py          # FAISS - high-speed in-memory
│   │   ├── pinecone_store.py       # Pinecone - managed cloud
│   │   └── milvus_store.py         # Milvus - large-scale distributed
│   │
│   ├── retrieval/
│   │   ├── base.py
│   │   ├── vector_retriever.py     # Dense semantic search
│   │   ├── bm25_retriever.py       # Sparse keyword search
│   │   └── hybrid_retriever.py     # RRF fusion of vector + BM25
│   │
│   ├── reranking/
│   │   ├── base.py
│   │   ├── llm_reranker.py         # LLM-based contextual reranking
│   │   └── cross_encoder.py        # Local Cross-Encoder reranking
│   │
│   ├── generation/
│   │   ├── generator.py            # Core RAG generator - answer() method
│   │   └── prompts.py              # System and query prompt templates
│   │
│   ├── query/
│   │   ├── rewriter.py             # Ambiguous query rewriting
│   │   └── expander.py             # Multi-query expansion for recall boost
│   │
│   ├── advanced/
│   │   ├── hierarchical.py         # Summary → detail hierarchical retrieval
│   │   ├── graph_rag.py            # Entity/relationship graph retrieval
│   │   └── agentic_rag.py          # Self-directed multi-step search agent
│   │
│   ├── cache/
│   │   └── cache.py                # File-based LLM + embedding cache with TTL
│   │
│   └── observability/
│       └── logger.py               # Structured logger + @timed decorator
│
├── evaluation/
│   ├── test.py                     # Q&A data model definition
│   ├── tests.jsonl                 # 150 evaluation questions (ground truth)
│   ├── retrieval_eval.py           # Hit@k, MRR@k, nDCG@k, Keyword Recall@k
│   ├── answer_eval.py              # LLM-as-Judge: Accuracy, Completeness, Relevance
│   └── semantic_eval.py            # Cosine similarity between answer embeddings
│
└── knowledge-base/                 # Raw documents (drop your data here)
```

---

## 4. Feature Matrix

### Core Pipeline

| Feature | Implementation | Config Key |
|---|---|---|
| Recursive chunking | `chunking/recursive_chunker.py` | `chunking.strategy: recursive` |
| Semantic chunking | `chunking/semantic_chunker.py` | `chunking.strategy: semantic` |
| OpenAI embeddings | `embeddings/openai_embedder.py` | `embeddings.provider: openai` |
| Local embeddings | `embeddings/sentence_transformer.py` | `embeddings.provider: sentence_transformer` |
| ChromaDB | `vector_store/chroma_store.py` | `vector_store.backend: chroma` |
| FAISS | `vector_store/faiss_store.py` | `vector_store.backend: faiss` |
| Pinecone | `vector_store/pinecone_store.py` | `vector_store.backend: pinecone` |
| Milvus | `vector_store/milvus_store.py` | `vector_store.backend: milvus` |

### Retrieval & Ranking

| Feature | Implementation | Config Key |
|---|---|---|
| Dense vector search | `retrieval/vector_retriever.py` | `retrieval.strategy: vector` |
| BM25 keyword search | `retrieval/bm25_retriever.py` | `retrieval.strategy: bm25` |
| Hybrid RRF fusion | `retrieval/hybrid_retriever.py` | `retrieval.strategy: hybrid` |
| LLM re-ranking | `reranking/llm_reranker.py` | `reranking.strategy: llm` |
| Cross-Encoder re-ranking | `reranking/cross_encoder.py` | `reranking.strategy: cross_encoder` |
| Query rewriting | `query/rewriter.py` | `enable_query_rewriting: true` |
| Query expansion | `query/expander.py` | `enable_query_expansion: true` |

### Advanced Modules

| Module | File | Use Case |
|---|---|---|
| Agentic RAG | `advanced/agentic_rag.py` | Multi-step self-directed search for complex queries |
| Graph RAG | `advanced/graph_rag.py` | Entity/relationship traversal across documents |
| Hierarchical RAG | `advanced/hierarchical.py` | Summary-first retrieval for large document corpora |

### Evaluation

| Metric | Layer | File |
|---|---|---|
| Hit@k | Retrieval | `retrieval_eval.py` |
| MRR@k | Retrieval | `retrieval_eval.py` |
| nDCG@k | Retrieval | `retrieval_eval.py` |
| Keyword Recall@k | Retrieval | `retrieval_eval.py` |
| Accuracy (50%) | Answer Quality | `answer_eval.py` |
| Completeness (30%) | Answer Quality | `answer_eval.py` |
| Relevance (20%) | Answer Quality | `answer_eval.py` |
| Cosine Similarity | Semantic | `semantic_eval.py` |

---

## 5. Synthetic Data Generator

CortexRAG includes `generate_dataset.py`, a standalone data synthesis utility for building a linked corpus and paired evaluation artifacts. The included example targets real estate under *Sham Real Estate Group*, but the generation flow can be adapted to other domains by changing the templates and schema assumptions.

### What it generates

- **100+ structured Markdown documents** across categories such as properties, clients, contracts, transactions, reports, and emails
- **RAG-oriented formatting** with explicit metadata blocks and predictable section headers
- **A paired evaluation set** (`qa_eval.jsonl`) with reference answers and keyword anchors for automated scoring
- **Controlled noise injection** to introduce mild inconsistencies and cross-document dependencies

### Generation Modes

| Mode | Command | Speed | Cost | Best For |
|---|---|---|---|---|
| Fast | `python generate_dataset.py --mode fast` | Fast | Free | Structural testing and local iteration |
| LLM | `python generate_dataset.py --mode llm` | Slow | API credits | Richer document variation |
| Hybrid *(recommended)* | `python generate_dataset.py --mode hybrid --seed 42` | Moderate | Low to moderate | Balanced realism and cost |

> **Reproducibility:** The `--seed` flag helps keep generated outputs stable across runs, which is useful when comparing retrieval configurations on the same synthetic corpus.

### Integrating generated data

```bash
# 1. Generate synthetic dataset (documents + evaluation set)
python generate_dataset.py --mode hybrid --seed 42

```

> **Note:** The system reads all subdirectory names under `knowledge-base/` and uses them as document category labels automatically. No code changes required.

> **Note:** Ensure that both the generated documents and the evaluation set are located within the `knowledge-base/` directory. The system assumes a unified data layout where all content, including evaluation artifacts, resides under the same root for consistent ingestion and experimentation.
---

## 6. Pipeline Deep Dive

### A. Ingestion Flow

Runs once, or on any knowledge base update. Writes a persistent vector index to disk.

```
Raw Files (MD / TXT / PDF)
        │
        ▼
  loader.py  ──────────── Reads files, extracts raw text, tags source metadata
        │
        ▼
preprocessor.py ────────── Cleans whitespace, normalizes encoding, strips noise
        │
        ▼
  chunker  ────────────── Splits text into overlapping, semantically coherent chunks
  (recursive or semantic)        Chunk size and overlap controlled via config.yaml
        │
        ▼
  embedder  ───────────── Encodes each chunk into a dense vector
  (OpenAI or local)              Vectors carry semantic meaning, not literal text
        │
        ▼
vector_store  ───────────── Persists (vector, text, metadata) tuples
(Chroma / FAISS / Pinecone / Milvus)    Ready for sub-second retrieval
```

### B. Query Flow

Runs on every user query. Reads from the pre-built vector index.

```
User Question
      │
      ▼
Query Rewriter (optional) ──── Clarifies ambiguous questions before search
      │
      ▼
Query Expander (optional) ──── Generates N paraphrase variants to increase recall
      │
      ▼
Retriever ────────────────────── Searches vector store for top-K relevant chunks
(vector | BM25 | hybrid RRF)       Hybrid mode merges both signals via RRF fusion
      │
      ▼
Re-ranker (optional) ─────────── Reorders retrieved chunks by true relevance
(LLM or Cross-Encoder)             Ensures the most useful context is at position 1
      │
      ▼
Generator ────────────────────── Injects ranked context into the system prompt
(generator.py + prompts.py)        LLM produces a grounded, citation-aware answer
      │
      ▼
Response + Source Context
```

---

## 7. Configuration System

The framework is primarily controlled through `config.yaml`. `config.py` loads the file, merges it with defaults, and exposes factory functions (`create_embedder()`, `create_retriever()`, `create_vector_store()`, `create_chunker()`, and others) that `pipeline.py` uses to assemble the runtime pipeline.

**Representative `config.yaml` structure:**

```yaml
# --- Project ---
project:
  knowledge_base_path: "knowledge-base"
  vector_db_path: "vector_db"
  collection_name: "docs"

# --- Chunking ---
chunking:
  strategy: "recursive"         # "recursive" | "semantic"
  average_chunk_size: 200       # semantic mode
  recursive_chunk_size: 4000    # recursive mode
  recursive_chunk_overlap: 0

# --- Embeddings ---
embeddings:
  provider: "openai"            # "openai" | "sentence_transformer"
  model_name: "text-embedding-3-large"

# --- Vector Store ---
vector_store:
  provider: "chroma"            # "chroma" | "faiss" | "pinecone" | "milvus"

# --- Retrieval ---
retrieval:
  strategy: "hybrid"            # "vector" | "bm25" | "hybrid"
  top_k: 30
  final_k: 10
  hybrid_alpha: 0.4
  enable_query_rewriting: false
  enable_query_expansion: false
  expansion_count: 2

# --- Re-ranking ---
reranking:
  strategy: "cross_encoder"     # "none" | "llm" | "cross_encoder"

# --- Generation ---
llm:
  model_name: "gpt-4o-mini"
  temperature: 0.0

# --- Cache ---
cache:
  enabled: true
  directory: ".cache"
  cache_embeddings: true
  cache_llm_responses: true
  ttl_hours: 24
```

> In most cases, swapping a component is a small config change rather than a pipeline rewrite.

---

## 8. Vector Store Backends

| Backend | Type | Best For | Persistence |
|---|---|---|---|
| **ChromaDB** | Local embedded | Development, prototyping | Disk (default) |
| **FAISS** | Local in-memory | High-throughput batch retrieval | Disk (serialized) |
| **Pinecone** | Managed cloud | Production, team deployments | Cloud |
| **Milvus** | Distributed | Enterprise scale (billions of vectors) | Distributed storage |

All backends implement the same `VectorStoreBase` interface. Switching backends requires only a config change no pipeline modifications.

> **Critical:** If you change the embedding model or provider, you must delete `vector_db/` and re-run `python main.py ingest`. Vectors produced by different models occupy incompatible mathematical spaces.

---

## 9. Retrieval Strategies

### Vector Retrieval
Dense semantic search using cosine similarity between query and chunk embeddings. Excels at conceptual and paraphrase matching. Can miss exact names, codes, or technical identifiers.

### BM25 Retrieval
Sparse keyword-based retrieval using term frequency statistics. Excels at exact matches for product codes, names, and rare technical terms. No semantic understanding.

### Hybrid Retrieval (Recommended)
Combines both signals using **Reciprocal Rank Fusion (RRF)**. Each result is scored by its rank position in both lists, then fused. This eliminates the weaknesses of either approach alone.

```
hybrid_score(doc) = 1/(k + rank_vector) + 1/(k + rank_bm25)
```

The `hybrid_alpha` parameter in `config.yaml` controls the relative weight of each signal, allowing fine-tuning per domain without code changes.

---

## 10. Advanced RAG Modules

These modules are architecturally independent and not enabled in the default `config.yaml`. Each represents a distinct retrieval paradigm suited to specific problem types.

### Agentic RAG (`agentic_rag.py`)

The standard pipeline searches once and answers. The Agentic module enters a **reasoning loop**: it searches, reads the result, decides if it has enough information, and if not, generates a new targeted sub-query before searching again. This repeats until the agent judges its context sufficient.

**When to use:** Multi-part questions, comparative analysis, questions that require information from documents that don't share obvious keywords.

```python
# Usage pattern
from rag.advanced.agentic_rag import AgenticRAG

agent = AgenticRAG(retriever=your_retriever, generator=your_generator)
answer = agent.answer("Compare Q1 and Q2 sales policy bonus structures")
```

### Graph RAG (`graph_rag.py`)

Extracts entities and relationships from documents during ingestion and builds a structured knowledge graph. At query time, it traverses this graph to surface connected information that plain vector search would miss.

**When to use:** Organizational hierarchies, supply chains, legal entity networks, any domain where *who is connected to what* matters as much as raw text content.

### Hierarchical RAG (`hierarchical.py`)

Solves the scale problem: when a corpus is too large for any single retrieval pass to be reliable. The module builds a two-level index a top layer of LLM-generated document summaries, and a bottom layer of fine-grained chunks. Query routing hits the summary layer first to identify the right document, then dives into its detailed chunks.

**When to use:** Corpora of thousands of long documents, legal archives, multi-year report collections.

---

## 11. Evaluation Framework

CortexRAG implements a three-layer evaluation stack that separates retrieval quality from answer quality. This makes it easier to diagnose whether a regression comes from indexing, retrieval, reranking, or generation.

```bash
python main.py evaluate --sample 50
```

### Layer 1 Retrieval Quality (`retrieval_eval.py`)

Evaluates whether the correct chunks were retrieved *before* the LLM sees them. Measures the retrieval mechanism in isolation.

| Metric | Definition |
|---|---|
| **Hit@k** | Did any of the top-k chunks contain the answer? (Binary) |
| **MRR@k** | Mean Reciprocal Rank how early did the correct chunk appear? |
| **nDCG@k** | Normalized Discounted Cumulative Gain ranks the full ordering quality |
| **Keyword Recall@k** | Did the retrieved chunks contain the ground-truth answer keywords? |

### Layer 2 Answer Quality (`answer_eval.py`)

Uses a configurable LLM judge. The judge receives the generated answer alongside the reference answer from the evaluation set and scores three axes:

| Dimension | Weight | Purpose |
|---|---|---|
| **Accuracy** | 50% | Penalizes hallucination factual correctness is paramount |
| **Completeness** | 30% | Rewards answers that cover all aspects of the question |
| **Relevance** | 20% | Penalizes padding and off-topic content |

This weighting biases the score toward factual accuracy while still rewarding coverage and answer focus.

### Layer 3 Semantic Similarity (`semantic_eval.py`)

Encodes both the generated answer and the ground-truth answer into embedding space and computes **cosine similarity**. This catches cases where the model answered correctly but in different words protecting against false negatives from exact-match scoring.

### Reading Evaluation Results

```
┌─────────────────────────────────────────────┐
│          Evaluation Summary (n=50)          │
├─────────────────────────────────────────────┤
│  Retrieval Hit@5          :  0.84           │
│  Retrieval MRR@5          :  0.71           │
│  Retrieval nDCG@5         :  0.76           │
│  Keyword Recall@5         :  0.79           │
├─────────────────────────────────────────────┤
│  Answer Accuracy (50%)    :  4.2 / 5.0      │
│  Answer Completeness (30%):  3.9 / 5.0      │
│  Answer Relevance (20%)   :  4.4 / 5.0      │
│  Weighted Score           :  4.13 / 5.0     │
├─────────────────────────────────────────────┤
│  Semantic Similarity      :  0.87           │
└─────────────────────────────────────────────┘
```

---

## 12. Results

The numbers below are representative results from a synthetic real-estate evaluation set and are intended to show the expected direction of change as features are enabled. They should be treated as a benchmark for this repository, not as a claim of production performance on external data.

| Configuration | Hit@5 | MRR@5 | nDCG@5 | Answer Score |
|--------------|------|------|-------|--------------|
| Baseline: vector + recursive + no rerank | 0.72 | 0.58 | 0.63 | 3.74 / 5 |
| + Hybrid retrieval | 0.81 | 0.66 | 0.71 | 4.01 / 5 |
| + Cross-encoder rerank | 0.84 | 0.71 | 0.76 | 4.13 / 5 |
| + Query rewriting | 0.86 | 0.73 | 0.78 | 4.19 / 5 |

In practice, the largest gains usually come from improving retrieval recall first, then using reranking to improve the quality of the context passed to generation.

---

## 13. Quick Start

### Prerequisites

```bash
python >= 3.10
pip install -r requirements.txt
```

### Environment

```bash
# .env
OPENAI_API_KEY=sk-...
```

### Step-by-Step

```bash
# 1. Generate a synthetic knowledge base with test data
python generate_dataset.py --mode hybrid --seed 42

Note: Ensure that both the generated documents and the evaluation set are located within the `knowledge-base/` directory. The system assumes a unified data layout where all content, including evaluation artifacts, resides under the same root for consistent ingestion and experimentation.

# 4. Build the vector index
python main.py ingest

# 5. Query via CLI
python main.py query "What is the price of PROP-001 and what type of property is it?"

# 6. Query via web UI
python app.py
# Opens at http://127.0.0.1:7860

# 7. Run evaluation
python eval_app.py 
```

---

## 14. Gradual Feature Activation Guide

Do not enable everything at once. The right engineering practice is to establish a baseline, measure it, then activate features one at a time and quantify their impact.

### Stage 1 Baseline RAG

```yaml
retrieval:
  strategy: "vector"
  enable_query_rewriting: false
  enable_query_expansion: false
reranking:
  strategy: "none"
chunking:
  strategy: "recursive"
```

**Goal:** Understand the floor. Run `evaluate` and record your Hit@5 and weighted answer score.

---

### Stage 2 Add Query Rewriting

```yaml
retrieval:
  enable_query_rewriting: true
```

**Goal:** Catch ambiguous follow-up queries ("why?" → "why is product X expensive?"). Check if MRR improves on short or implicit questions.

---

### Stage 3 Upgrade to Hybrid Retrieval

```yaml
retrieval:
  strategy: "hybrid"
  hybrid_alpha: 0.7
```

**Goal:** Stop missing exact identifiers and codes that pure vector search scores poorly. Expect Hit@5 to increase noticeably on questions involving specific names, IDs, or numbers.

---

### Stage 4 Enable Re-ranking

```yaml
reranking:
  strategy: "cross_encoder"
```

**Goal:** Improve MRR and nDCG by reordering retrieved chunks so the most relevant appears first. This directly impacts the quality of the generation context.

---

### Stage 5 Semantic Chunking

```yaml
chunking:
  strategy: "semantic"
```

> Requires re-running `python main.py ingest` from scratch after changing chunking strategy.

**Goal:** Better chunk boundaries for narrative or loosely structured documents. Most impactful when documents have long paragraphs with topic shifts.

---

## 15. Key Design Decisions

### Why not LangChain by default?

LangChain is a powerful abstraction, but abstractions hide logic. CortexRAG was built to make every retrieval and generation decision explicit and debuggable. Every component is a thin, readable Python class. Where LangChain drop-in replacements exist, they are documented directly in the relevant source file as inline comments so teams that *do* need LangChain for integrations can upgrade specific components without rebuilding the framework.

### Why YAML + Factory pattern?

YAML is human-readable and version-controllable. The Factory pattern (`create_embedder()`, `create_retriever()`, etc. in `config.py`) decouples component selection from component usage. `pipeline.py` does not know or care whether it is using Chroma or Pinecone it just calls `create_vector_store()` and receives a compatible object.

### Why build a synthetic data generator?

Evaluation without realistic data is meaningless. Collecting real domain data has legal, privacy, and availability constraints. A synthetic generator with controllable noise, deliberate cross-document references, and a paired evaluation set gives every developer an identical, reproducible test bed regardless of their data access.

### Why a three-layer evaluation stack?

Each layer catches different failure modes:
- **Retrieval eval** catches bad indexing or chunking decisions.
- **LLM Judge eval** catches generation failures (hallucination, incompleteness) independent of retrieval.
- **Semantic eval** catches false negatives from lexical mismatch in the ground truth.

Using only one layer of evaluation gives an incomplete and often misleading signal.

---

## 16. Extending the Framework

### Adding a new vector store backend

1. Create `rag/vector_store/your_store.py`
2. Inherit from `VectorStoreBase` in `base.py` and implement `add()`, `search()`, and `delete()`
3. Add a case for your backend in `config.py` → `create_vector_store()`
4. Set `vector_store.provider: your_store` in `config.yaml`

### Adding a new chunking strategy

1. Create `rag/chunking/your_chunker.py`
2. Inherit from `ChunkerBase` and implement `split()`
3. Register it in the factory and config

### Adding a new evaluation metric

1. Add your metric function to the appropriate eval file (or create a new one)
2. Call it inside `main.py`'s `evaluate` command handler
3. Include it in the summary output

The framework's base class pattern ensures that any new component is automatically compatible with the rest of the system.

---

## 17. Limitations

- The included evaluation workflow is strongest on the synthetic dataset generated for this repository; results should be revalidated on a domain-specific corpus before drawing broader conclusions.
- Advanced modules such as Graph RAG, Hierarchical RAG, and Agentic RAG are useful extension points, but they should be treated as experimental building blocks rather than drop-in defaults.
- The project demonstrates local experimentation, evaluation, and configurability, but it does not include an API layer, deployment packaging, or CI/CD automation out of the box.
- Synthetic documents are valuable for controlled testing, but they do not fully capture the noise, policy constraints, and operational edge cases of production document systems.

---

## 18. Requirements

### Core

```
openai>=1.0.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
rank-bm25>=0.2.2
gradio>=4.0.0
pyyaml>=6.0
python-dotenv>=1.0.0
numpy>=1.24.0
```

### Optional (by backend)

```
faiss-cpu>=1.7.4          # FAISS backend
pinecone-client>=3.0.0    # Pinecone backend
pymilvus>=2.3.0           # Milvus backend
```

### Optional (advanced modules)

```
networkx>=3.0             # Graph RAG
torch>=2.0.0              # Cross-Encoder re-ranking
transformers>=4.30.0      # Cross-Encoder re-ranking
```

---

