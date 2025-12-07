---
tags:
- sentence-transformers
- sentence-similarity
- sparse-encoder
- sparse
- splade
- feature-extraction
- telepix
pipeline_tag: feature-extraction
library_name: sentence-transformers
license: apache-2.0
---
<p align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/61d6f4a4d49065ee28a1ee7e/V8n2En7BlMNHoi1YXVv8Q.png" width="400"/>
<p>
  
# PIXIE-Splade-Preview
**PIXIE-Splade-Preview** is a Korean-only [SPLADE](https://arxiv.org/abs/2403.06789) (Sparse Lexical and Expansion) retriever, developed by [TelePIX Co., Ltd](https://telepix.net/). 
**PIXIE** stands for Tele**PIX** **I**ntelligent **E**mbedding, representing TelePIX’s high-performance embedding technology. 
This model is trained exclusively on Korean data and outputs sparse lexical vectors that are directly 
compatible with inverted indexing (e.g., Lucene/Elasticsearch). 
Because each non-zero weight corresponds to a Korean subword/token, 
interpretability is built-in: you can inspect which tokens drive retrieval.

## Why SPLADE for Search?
- **Inverted Index Ready**: Directly index weighted tokens in standard IR stacks (Lucene/Elasticsearch).
- **Interpretable by Design**: Top-k contributing tokens per query/document explain *why* a hit matched.
- **Production-Friendly**: Fast candidate generation at web scale; memory/latency tunable via sparsity thresholds.
- **Hybrid-Retrieval Friendly**: Combine with dense retrievers via score fusion.

## Model Description
- **Model Type:** SPLADE Sparse Encoder
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 8192 tokens
- **Output Dimensionality:** 50000 dimensions
- **Similarity Function:** Dot Product
- **Language:** Korean
- **License:** apache-2.0 

### Full Model Architecture

```
SparseEncoder(
  (0): MLMTransformer({'max_seq_length': 8192, 'do_lower_case': False, 'architecture': 'ModernBertForMaskedLM'})
  (1): SpladePooling({'pooling_strategy': 'max', 'activation_function': 'relu', 'word_embedding_dimension': 50000})
)
```

## Quality Benchmarks
**PIXIE-Splade-Preview** delivers consistently strong performance across a diverse set of domain-specific and open-domain benchmarks in Korean, demonstrating its effectiveness in real-world search applications. 
The table below presents the retrieval performance of several embedding models evaluated on a variety of Korean MTEB benchmarks. 
We report **Normalized Discounted Cumulative Gain (NDCG)** scores, which measure how well a ranked list of documents aligns with ground truth relevance. Higher values indicate better retrieval quality.
- **Avg. NDCG**: Average of NDCG@1, @3, @5, and @10 across all benchmark datasets.  
- **NDCG@k**: Relevance quality of the top-*k* retrieved results.

All evaluations were conducted using the open-source **[Korean-MTEB-Retrieval-Evaluators](https://github.com/BM-K/Korean-MTEB-Retrieval-Evaluators)** codebase to ensure consistent dataset handling, indexing, retrieval, and NDCG@k computation across models.

### 6 Datasets of MTEB (Korean)
Our model, **telepix/PIXIE-Splade-Preview**, achieves strong performance across most metrics and benchmarks, 
demonstrating strong generalization across domains such as multi-hop QA, long-document retrieval, public health, and e-commerce.

Descriptions of the benchmark datasets used for evaluation are as follows:
- **Ko-StrategyQA**  
  A Korean multi-hop open-domain question answering dataset designed for complex reasoning over multiple documents.
- **AutoRAGRetrieval**  
  A domain-diverse retrieval dataset covering finance, government, healthcare, legal, and e-commerce sectors.
- **MIRACLRetrieval**  
  A document retrieval benchmark built on Korean Wikipedia articles.
- **PublicHealthQA**  
  A retrieval dataset focused on medical and public health topics.
- **BelebeleRetrieval**  
  A dataset for retrieving relevant content from web and news articles in Korean.
- **MultiLongDocRetrieval**  
  A long-document retrieval benchmark based on Korean Wikipedia and mC4 corpus.

> **Tip:**
> While many benchmark datasets are available for evaluation, in this project we chose to use only those that contain clean positive documents for each query. Keep in mind that a benchmark dataset is just that a benchmark. For real-world applications, it is best to construct an evaluation dataset tailored to your specific domain and evaluate embedding models, such as PIXIE, in that environment to determine the most suitable one.

#### Sparse Embedding
| Model Name | # params | Avg. NDCG | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| telepix/PIXIE-Splade-Preview | 0.1B | 0.7253 | 0.6799 | 0.7217 | 0.7416 | 0.7579 |
|  |  |  |  |  |  |  |
| [BM25](https://github.com/xhluca/bm25s) | N/A | 0.4714 | 0.4194 | 0.4708 | 0.4886 | 0.5071 |
| naver/splade-v3 | 0.1B | 0.0582 | 0.0462 | 0.0566 | 0.0612 | 0.0685 |

#### Dense Embedding
| Model Name | # params | Avg. NDCG | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| telepix/PIXIE-Spell-Preview-1.7B | 1.7B | 0.7567 | 0.7149 | 0.7541 | 0.7696 | 0.7882 |
| telepix/PIXIE-Spell-Preview-0.6B | 0.6B | 0.7280 | 0.6804 | 0.7258 | 0.7448 | 0.7612 |
| telepix/PIXIE-Rune-Preview | 0.5B | 0.7383 | 0.6936 | 0.7356 | 0.7545 | 0.7698 |
|  |  |  |  |  |  |  |
| nlpai-lab/KURE-v1 | 0.5B | 0.7312 | 0.6826 | 0.7303 | 0.7478 | 0.7642 |
| BAAI/bge-m3 | 0.5B | 0.7126 | 0.6613 | 0.7107 | 0.7301 | 0.7483 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.5B | 0.7050 | 0.6570 | 0.7015 | 0.7226 | 0.7390 |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 0.6872 | 0.6423 | 0.6833 | 0.7017 | 0.7215 |
| jinaai/jina-embeddings-v3 | 0.5B | 0.6731 | 0.6224 | 0.6715 | 0.6899 | 0.7088 |
| SamilPwC-AXNode-GenAI/PwC-Embedding_expr | 0.5B | 0.6709 | 0.6221 | 0.6694 | 0.6852 | 0.7069 | 
| Alibaba-NLP/gte-multilingual-base | 0.3B | 0.6679 | 0.6068 | 0.6673 | 0.6892 | 0.7084 |
| openai/text-embedding-3-large | N/A | 0.6465 | 0.5895 | 0.6467 | 0.6646 | 0.6853 |

## Direct Use (Inverted-Index Retrieval)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
from sentence_transformers import SparseEncoder

MODEL_NAME = "telepix/PIXIE-Splade-Preview"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _to_dense_numpy(x) -> np.ndarray:
    """
    Safely converts a tensor returned by SparseEncoder to a dense numpy array.
    """
    if hasattr(x, "to_dense"):
        return x.to_dense().float().cpu().numpy()
    # If it's already a numpy array or a dense tensor
    if isinstance(x, torch.Tensor):
        return x.float().cpu().numpy()
    return np.asarray(x)

def _filter_special_ids(ids: List[int], tokenizer) -> List[int]:
    """
    Filters out special token IDs from a list of token IDs.
    """
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    return [i for i in ids if i not in special]

def build_inverted_index(
    model: SparseEncoder,
    tokenizer,
    documents: List[str],
    batch_size: int = 8,
    min_weight: float = 0.0,
) -> Tuple[Dict[int, List[Tuple[int, float]]], List[str]]:
    """
    Generates document embeddings and constructs an inverted index.
    The index maps token_id to a list of (doc_idx, weight) tuples.
    index[token_id] = [(doc_idx, weight), ...]
    """
    with torch.no_grad():
        doc_emb = model.encode_document(documents, batch_size=batch_size)
    doc_dense = _to_dense_numpy(doc_emb)

    index: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    for doc_idx, vec in enumerate(doc_dense):
        # Extract only active tokens (those with weight above the threshold)
        nz = np.flatnonzero(vec > min_weight)
        # Optionally, remove special tokens
        nz = _filter_special_ids(nz.tolist(), tokenizer)

        for token_id in nz:
            index[token_id].append((doc_idx, float(vec[token_id])))

    return index

# -------------------------
# Search + Token Overlap Explanation
# -------------------------
def splade_token_overlap_inverted(
    model: SparseEncoder,
    tokenizer,
    inverted_index: Dict[int, List[Tuple[int, float]]],
    documents: List[str],
    queries: List[str],
    top_k_docs: int = 3,
    top_k_tokens: int = 10,
    min_weight: float = 0.0,
):
    """
    Calculates SPLADE similarity using an inverted index and shows the
    contribution (qw*dw) of the top_k_tokens 'overlapping tokens' for each top-ranked document.
    """
    for qi, qtext in enumerate(queries):
        with torch.no_grad():
            q_vec = model.encode_query(qtext)
        q_vec = _to_dense_numpy(q_vec).ravel()

        # Active query tokens
        q_nz = np.flatnonzero(q_vec > min_weight).tolist()
        q_nz = _filter_special_ids(q_nz, tokenizer)

        scores: Dict[int, float] = defaultdict(float)
        # Token contribution per document: token_id -> (qw, dw, qw*dw)
        per_doc_contrib: Dict[int, Dict[int, Tuple[float, float, float]]] = defaultdict(dict)

        for tid in q_nz:
            qw = float(q_vec[tid])
            postings = inverted_index.get(tid, [])
            for doc_idx, dw in postings:
                prod = qw * dw
                scores[doc_idx] += prod
                # Store per-token contribution (can be summed if needed)
                per_doc_contrib[doc_idx][tid] = (qw, dw, prod)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k_docs]

        print("\n============================")
        print(f"[Query {qi}] {qtext}")
        print("============================")

        if not ranked:
            print("→ 일치 토큰이 없어 문서 스코어가 생성되지 않았습니다.")
            continue

        for rank, (doc_idx, score) in enumerate(ranked, start=1):
            doc = documents[doc_idx]
            print(f"\n→ Rank {rank} | Document {doc_idx}: {doc}")
            print(f"  [Similarity Score ({score:.6f})]")

            contrib = per_doc_contrib[doc_idx]
            if not contrib:
                print("(겹치는 토큰이 없습니다.)")
                continue

            # Extract top K contributing tokens
            top = sorted(contrib.items(), key=lambda kv: kv[1][2], reverse=True)[:top_k_tokens]
            token_ids = [tid for tid, _ in top]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            print("  [Top Contributing Tokens]")
            for (tid, (qw, dw, prod)), tok in zip(top, tokens):
                print(f"    {tok:20} {prod:.6f}")

if __name__ == "__main__":
    # 1) Load model and tokenizer
    model = SparseEncoder(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2) Example data
    queries = [
        "텔레픽스는 어떤 산업 분야에서 위성 데이터를 활용하나요?",
        "국방 분야에 어떤 위성 서비스가 제공되나요?",
        "텔레픽스의 기술 수준은 어느 정도인가요?",
    ]
    documents = [
        "텔레픽스는 해양, 자원, 농업 등 다양한 분야에서 위성 데이터를 분석하여 서비스를 제공합니다.",
        "정찰 및 감시 목적의 위성 영상을 통해 국방 관련 정밀 분석 서비스를 제공합니다.",
        "TelePIX의 광학 탑재체 및 AI 분석 기술은 Global standard를 상회하는 수준으로 평가받고 있습니다.",
        "텔레픽스는 우주에서 수집한 정보를 분석하여 '우주 경제(Space Economy)'라는 새로운 가치를 창출하고 있습니다.",
        "텔레픽스는 위성 영상 획득부터 분석, 서비스 제공까지 전 주기를 아우르는 솔루션을 제공합니다.",
    ]

    # 3) Build document index (inverted index)
    inverted_index = build_inverted_index(
        model=model,
        tokenizer=tokenizer,
        documents=documents,
        batch_size=8,
        min_weight=0.0,  # Adjust to 1e-6 ~ 1e-4 to filter out very small noise
    )

    # 4) Search and explain token overlap
    splade_token_overlap_inverted(
        model=model,
        tokenizer=tokenizer,
        inverted_index=inverted_index,
        documents=documents,
        queries=queries,
        top_k_docs=2,     # Print only the top 3 documents
        top_k_tokens=5,  # Top 10 contributing tokens for each document
        min_weight=0.0,
    )
```

## License
The PIXIE-Splade-Preview model is licensed under Apache License 2.0.

## Citation
```
@software{TelePIX-PIXIE-Splade-Preview,
  title={PIXIE-Splade-Preview},
  author={TelePIX AI Research Team and Bongmin Kim},
  year={2025},
  url={https://huggingface.co/telepix/PIXIE-Splade-Preview}
}
```

## Contact

If you have any suggestions or questions about the PIXIE, please reach out to the authors at bmkim@telepix.net.