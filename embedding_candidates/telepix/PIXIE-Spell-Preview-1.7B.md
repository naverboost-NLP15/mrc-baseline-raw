---
tags:
- sentence-transformers
- sentence-similarity
- dense-encoder
- dense
- feature-extraction
- telepix
pipeline_tag: feature-extraction
library_name: sentence-transformers
license: apache-2.0
---
<p align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/61d6f4a4d49065ee28a1ee7e/V8n2En7BlMNHoi1YXVv8Q.png" width="400"/>
<p>

# PIXIE-Spell-Preview-1.7B
**PIXIE-Spell-Preview-1.7B** is a decoder-based embedding model trained on Korean and English dataset, 
developed by [TelePIX Co., Ltd](https://telepix.net/).
**PIXIE** stands for Tele**PIX** **I**ntelligent **E**mbedding, representing TelePIX’s high-performance embedding technology.
This model is specifically optimized for semantic retrieval tasks in Korean and English, and demonstrates strong performance in aerospace domain applications. Through extensive fine-tuning and domain-specific evaluation, PIXIE shows robust retrieval quality for real-world use cases such as document understanding, technical QA, and semantic search in aerospace and related high-precision fields.
It also performs competitively across a wide range of open-domain Korean and English retrieval benchmarks, making it a versatile foundation for multilingual semantic search systems.


## Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 8192 tokens
- **Output Dimensionality:** 2048 dimensions
- **Similarity Function:** Cosine Similarity
- **Language:** Multilingual — optimized for high performance in Korean and English
- **Domain Specialization:** Aerospace semantic search
- **License:** apache-2.0 

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 8192, 'do_lower_case': False, 'architecture': 'Qwen3Model'})
  (1): Pooling({'word_embedding_dimension': 2048, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': True, 'include_prompt': True})
  (2): Normalize()
)
```

## Quality Benchmarks
**PIXIE-Spell-Preview-1.7B** is a multilingual embedding model specialized for Korean and English retrieval tasks. 
It delivers consistently strong performance across a diverse set of domain-specific and open-domain benchmarks in both languages, demonstrating its effectiveness in real-world semantic search applications.
The table below presents the retrieval performance of several embedding models evaluated on a variety of Korean and English benchmarks.
We report **Normalized Discounted Cumulative Gain (NDCG)** scores, which measure how well a ranked list of documents aligns with ground truth relevance. Higher values indicate better retrieval quality.  
- **Avg. NDCG**: Average of NDCG@1, @3, @5, and @10 across all benchmark datasets.  
- **NDCG@k**: Relevance quality of the top-*k* retrieved results.
  
All evaluations were conducted using the open-source **[Korean-MTEB-Retrieval-Evaluators](https://github.com/BM-K/Korean-MTEB-Retrieval-Evaluators)** codebase to ensure consistent dataset handling, indexing, retrieval, and NDCG@k computation across models.

#### 6 Datasets of MTEB (Korean)
Our model, **telepix/PIXIE-Spell-Preview-1.7B**, achieves strong performance across most metrics and benchmarks, demonstrating strong generalization across domains such as multi-hop QA, long-document retrieval, public health, and e-commerce.

| Model Name | # params | Avg. NDCG | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| telepix/PIXIE-Spell-Preview-1.7B | 1.7B | 0.7567 | 0.7149 | 0.7541 | 0.7696 | 0.7882 |
| telepix/PIXIE-Spell-Preview-0.6B | 0.6B | 0.7280 | 0.6804 | 0.7258 | 0.7448 | 0.7612 |
| telepix/PIXIE-Rune-Preview | 0.5B | 0.7383 | 0.6936 | 0.7356 | 0.7545 | 0.7698 |
| telepix/PIXIE-Splade-Preview | 0.1B | 0.7253 | 0.6799 | 0.7217 | 0.7416 | 0.7579 |
|  |  |  |  |  |  |  |
| nlpai-lab/KURE-v1 | 0.5B | 0.7312 | 0.6826 | 0.7303 | 0.7478 | 0.7642 |
| BAAI/bge-m3 | 0.5B | 0.7126 | 0.6613 | 0.7107 | 0.7301 | 0.7483 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.5B | 0.7050 | 0.6570 | 0.7015 | 0.7226 | 0.7390 |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 0.6872 | 0.6423 | 0.6833 | 0.7017 | 0.7215 |
| jinaai/jina-embeddings-v3 | 0.5B | 0.6731 | 0.6224 | 0.6715 | 0.6899 | 0.7088 |
| SamilPwC-AXNode-GenAI/PwC-Embedding_expr | 0.5B | 0.6709 | 0.6221 | 0.6694 | 0.6852 | 0.7069 | 
| Alibaba-NLP/gte-multilingual-base | 0.3B | 0.6679 | 0.6068 | 0.6673 | 0.6892 | 0.7084 |
| openai/text-embedding-3-large | N/A | 0.6465 | 0.5895 | 0.6467 | 0.6646 | 0.6853 |

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

#### 7 Datasets of BEIR (English)
Our model, **telepix/PIXIE-Spell-Preview-1.7B**, achieves strong performance on a wide range of tasks, including fact verification, multi-hop question answering, financial QA, and scientific document retrieval, demonstrating competitive generalization across diverse domains.
 
| Model Name | # params | Avg. NDCG | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| telepix/PIXIE-Spell-Preview-1.7B | 1.7B | 0.5630 | 0.5446 | 0.5529 | 0.5660 | 0.5885 |
| telepix/PIXIE-Spell-Preview-0.6B | 0.6B | 0.5354 | 0.5208 | 0.5241 | 0.5376 | 0.5589 |
| telepix/PIXIE-Rune-Preview | 0.5B | 0.5781 | 0.5691 | 0.5663 | 0.5791 | 0.5979 |
|  |  |  |  |  |  |  |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.5B | 0.5812 | 0.5725 | 0.5705 | 0.5811 | 0.6006 |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 0.5558 | 0.5321 | 0.5451 | 0.5620 | 0.5839 |
| Alibaba-NLP/gte-multilingual-base | 0.3B | 0.5541 | 0.5446 | 0.5426 | 0.5574 | 0.5746 |
| BAAI/bge-m3 | 0.5B | 0.5318 | 0.5078 | 0.5231 | 0.5389 | 0.5573 |
| nlpai-lab/KURE-v1 | 0.5B | 0.5272 | 0.5017 | 0.5171 | 0.5353 | 0.5548 |
| SamilPwC-AXNode-GenAI/PwC-Embedding_expr | 0.5B | 0.5111 | 0.4766 | 0.5006 | 0.5212 | 0.5460 |
| jinaai/jina-embeddings-v3 | 0.6B | 0.4482 | 0.4116 | 0.4379 | 0.4573 | 0.4861 |

Descriptions of the benchmark datasets used for evaluation are as follows:
- **ArguAna**  
  A dataset for argument retrieval based on claim-counterclaim pairs from online debate forums.
- **FEVER**  
  A fact verification dataset using Wikipedia for evidence-based claim validation.
- **FiQA-2018**  
  A retrieval benchmark tailored to the finance domain with real-world questions and answers.
- **HotpotQA**  
  A multi-hop open-domain QA dataset requiring reasoning across multiple documents.
- **MSMARCO**  
  A large-scale benchmark using real Bing search queries and corresponding web documents.
- **NQ**  
  A Google QA dataset where user questions are answered using Wikipedia articles.
- **SCIDOCS**  
  A citation-based document retrieval dataset focused on scientific papers.
  
## Direct Use (Semantic Search)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer
  
# Load the model
model_name = 'telepix/PIXIE-Spell-Preview-1.7B'
model = SentenceTransformer(model_name)

# Define the queries and documents
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

# Compute embeddings: use `prompt_name="query"` to encode queries!
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

# Compute cosine similarity scores
scores = model.similarity(query_embeddings, document_embeddings)

# Output the results
for query, query_scores in zip(queries, scores):
    doc_score_pairs = list(zip(documents, query_scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    print("Query:", query)
    for document, score in doc_score_pairs:
        print(score, document)

```

## License
The PIXIE-Spell-Preview-1.7B model is licensed under Apache License 2.0.

## Citation
```
@software{TelePIX-PIXIE-Spell-Preview-1.7B,
  title={PIXIE-Spell-Preview-1.7B},
  author={TelePIX AI Research Team and Bongmin Kim},
  year={2025},
  url={https://huggingface.co/telepix/PIXIE-Spell-Preview-1.7B}
}
```

## Contact

If you have any suggestions or questions about the PIXIE, please reach out to the authors at bmkim@telepix.net.