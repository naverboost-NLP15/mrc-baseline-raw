---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- loss:CachedGISTEmbedLoss
base_model: Snowflake/snowflake-arctic-embed-l-v2.0
license: apache-2.0
language:
- ko
- en
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

<img src="https://cdn-uploads.huggingface.co/production/uploads/642b0c2fecec03b4464a1d9b/9uN5ypGY-GRGgakLs_s1o.png" width="600">

# SentenceTransformer based on Snowflake/snowflake-arctic-embed-l-v2.0

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) on the clustered datasets. It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search.  

The **Snowflake/snowflake-arctic-embed-l-v2.0** model has been further trained with Korean data to enhance its performance in **Korean retrieval tasks**. It is a powerful model that achieves **state-of-the-art (SOTA) performance across multiple retrieval benchmarks**.



## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) <!-- at revision 7f311bb640ad3babc0a4e3a8873240dcba44c9d2 -->
- **Maximum Sequence Length:** 8192 tokens
- **Output Dimensionality:** 1024 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Datasets:**
    - AI Hub Dataset
      - 행정 문서 대상 기계 독해
      - 기계 독해
      - 뉴스 기사 기계독해
      - 도서 자료 기계독해
      - 숫자 연산 기계독해
      - 금융 법률 문서 기계독해
- **Language:** Korean, English


### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 8192, 'do_lower_case': False}) with Transformer model: XLMRobertaModel 
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```


## Usage

First install the Sentence Transformers library and xformers library

```bash
pip install -U sentence-transformers
pip install xformers

```

Then you can load this model and run inference.

### Using Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# Load the model
# Please use bf16 when inferring with half precision
model_name = 'dragonkue/snowflake-arctic-embed-l-v2.0-ko'
model = SentenceTransformer(model_name)

# Define the queries and documents
queries = ['대한민국의 수도는 어디인가?', '한글을 만든 사람은 누구인가?']
documents = ['대한민국의 수도는 서울이다.', '한글은 세종대왕이 창제하였다.']

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

### Using Huggingface Transformers


You can use the transformers package to use Snowflake's arctic-embed model, as shown below. For optimal retrieval quality, use the CLS token to embed each text portion and use the query prefix below (just on the query).

```python
import torch
from transformers import AutoModel, AutoTokenizer

# Load the model
# Please use bf16 when inferring with half precision
model_name = 'dragonkue/snowflake-arctic-embed-l-v2.0-ko'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
model.eval()

# Define the queries and documents
query_prefix = 'query: '
queries  = ['대한민국의 수도는 어디인가?', '한글을 만든 사람은 누구인가?']
queries_with_prefix = ["{}{}".format(query_prefix, i) for i in queries]
query_tokens = tokenizer(queries_with_prefix, padding=True, truncation=True, return_tensors='pt', max_length=8192)

documents = ['대한민국의 수도는 서울이다.', '한글은 세종대왕이 창제하였다.']
document_tokens = tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=8192)

# Compute token embeddings
with torch.no_grad():
    query_embeddings = model(**query_tokens)[0][:, 0]
    document_embeddings = model(**document_tokens)[0][:, 0]

# Normalize embeddings
query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
document_embeddings = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)

scores = torch.mm(query_embeddings, document_embeddings.transpose(0, 1))

for query, query_scores in zip(queries, scores):
    doc_score_pairs = list(zip(documents, query_scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    # Output passages & scores
    print("Query:", query)
    for document, score in doc_score_pairs:
        print(score, document)

```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

- This evaluation references the KURE GitHub repository. (https://github.com/nlpai-lab/KURE)
- We conducted an evaluation on all **Korean Retrieval Benchmarks** registered in [MTEB](https://github.com/embeddings-benchmark/mteb).

### Korean Retrieval Benchmark
- [Ko-StrategyQA](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA): A Korean **ODQA multi-hop retrieval dataset**, translated from StrategyQA.
- [AutoRAGRetrieval](https://huggingface.co/datasets/yjoonjang/markers_bm): A **Korean document retrieval dataset** constructed by parsing PDFs from five domains: **finance, public, medical, legal, and commerce**.
- [MIRACLRetrieval](https://huggingface.co/datasets/miracl/miracl): A **Korean document retrieval dataset** based on Wikipedia.
- [PublicHealthQA](https://huggingface.co/datasets/xhluca/publichealth-qa): A **retrieval dataset** focused on **medical and public health domains** in Korean.
- [BelebeleRetrieval](https://huggingface.co/datasets/facebook/belebele): A **Korean document retrieval dataset** based on FLORES-200.
- [MrTidyRetrieval](https://huggingface.co/datasets/mteb/mrtidy): A **Wikipedia-based Korean document retrieval dataset**.
- [MultiLongDocRetrieval](https://huggingface.co/datasets/Shitao/MLDR): A **long-document retrieval dataset** covering various domains in Korean.
- [XPQARetrieval](https://huggingface.co/datasets/jinaai/xpqa): A **cross-domain Korean document retrieval dataset**.

### Metrics

* Standard metric : NDCG@10

### Information Retrieval

- Achieves state-of-the-art (SOTA) performance across various benchmarks.
- For each benchmark, the **highest score** is highlighted in bold, and the _second-highest score_ is italicized.  

| Model                                                                                            | Average      | MrTidyRetrieval   | MIRACLRetrieval   | XPQARetrieval   | BelebeleRetrieval   | PublicHealthQA   | AutoRAGRetrieval   | Ko-StrategyQA   |
|:-------------------------------------------------------------------------------------------------|:-------------|:------------------|:------------------|:----------------|:--------------------|:-----------------|:-------------------|:----------------|
| dragonkue/snowflake-arctic-embed-l-v2.0-ko                                                       | **0.740433** | 0.57121           | 0.66846           | **0.4436**      | **0.95177**         | 0.83374          | **0.90927**        | _0.80498_       |
| dragonkue/BGE-m3-ko                                                                              | _0.729993_   | 0.60992           | 0.68331           | 0.38131         | _0.95027_           | 0.81545          | _0.87379_          | 0.7959          |
| nlpai-lab/KURE-v1                                                                                | 0.727739     | 0.59092           | 0.68157           | 0.38158         | 0.95019             | 0.81925          | 0.87076            | 0.7999          |
| BAAI/bge-m3                                                                                      | 0.724169     | **0.64708**       | _0.70146_         | 0.36075         | 0.93164             | 0.80412          | 0.83008            | 0.79405         |
| Snowflake/snowflake-arctic-embed-l-v2.0                                                          | 0.724104     | 0.59071           | 0.66077           | _0.43018_       | 0.9271              | 0.81679          | 0.83863            | 0.80455         |
| intfloat/multilingual-e5-large                                                                   | 0.721607     | _0.64211_         | 0.66486           | 0.3571          | 0.94499             | 0.82534          | 0.81337            | 0.80348         |
| nlpai-lab/KoE5                                                                                   | 0.711356     | 0.58411           | 0.62347           | 0.35086         | 0.94251             | 0.83507          | 0.84339            | 0.80008         |
| BAAI/bge-multilingual-gemma2                                                                     | 0.704274     | 0.47521           | **0.70315**       | 0.37446         | 0.95001             | _0.87102_        | 0.76535            | 0.79072         |
| jinaai/jina-embeddings-v3                                                                        | 0.701314     | 0.55759           | 0.63716           | 0.41272         | 0.91203             | 0.83059          | 0.76104            | 0.79807         |
| SamilPwC-AXNode-GenAI/PwC-Embedding_expr                                                         | 0.699483     | 0.56656	          | 0.63214	          | 0.36388         | 0.91669             | 0.83462	         | 0.78493	          | 0.79756	        |            
| intfloat/multilingual-e5-large-instruct                                                          | 0.69837      | 0.52877           | 0.59914           | 0.39712         | 0.936               | 0.84967          | 0.77996            | 0.79793         |
| nomic-ai/nomic-embed-text-v2-moe                                                                 | 0.693773     | 0.53766           | 0.65913           | 0.36871         | 0.93636             | 0.78448          | 0.80682            | 0.76325         |
| intfloat/multilingual-e5-base                                                                    | 0.689429     | 0.58082           | 0.6227            | 0.3607          | 0.92868             | 0.77203          | 0.79752            | 0.76355         |
| intfloat/e5-mistral-7b-instruct                                                                  | 0.683734     | 0.52444           | 0.58709           | 0.39159         | 0.92403             | **0.88733**      | 0.67849            | 0.79317         |
| Alibaba-NLP/gte-Qwen2-7B-instruct                                                                | 0.680323     | 0.46571           | 0.53375           | 0.37866         | 0.94808             | 0.85844          | 0.76682            | **0.8108**      |
| Qwen/Qwen3-Embedding-0.6B	                                                                       | 0.676200     | 0.48987           | 0.60021           | 0.33440         | 0.91601             | 0.80290          | 0.82405            | 0.76596         |   
| Alibaba-NLP/gte-multilingual-base                                                                | 0.663766     | 0.56464           | 0.62697           | 0.30702         | 0.8796              | 0.74584          | 0.77108            | 0.75121         |
| openai/text-embedding-3-large                                                                    | 0.662239     | 0.44728           | 0.56248           | 0.37423         | 0.89451             | 0.85617          | 0.76466            | 0.73634         |
| upskyy/bge-m3-korean                                                                             | 0.6567       | 0.55011           | 0.59892           | 0.31695         | 0.8731              | 0.77559          | 0.72946            | 0.75277         |
| Salesforce/SFR-Embedding-2_R                                                                     | 0.65591      | 0.40347           | 0.55798           | 0.37371         | 0.91747             | 0.8605           | 0.70782            | 0.77042         |
| ibm-granite/granite-embedding-278m-multilingual                                                  | 0.641935     | nan               | 0.59216           | 0.23058         | 0.83231             | 0.77668          | 0.70226            | 0.71762         |
| jhgan/ko-sroberta-multitask                                                                      | 0.526301     | 0.29475           | 0.36698           | 0.27961         | 0.81636             | 0.69212          | 0.58332            | 0.65097         |


#### Capabilities Beyond Benchmarks

This model is designed to handle various retrieval scenarios that are not directly measured in benchmarks:

1. Supports phrase-based queries in addition to full-sentence queries.

    Example: "What products does Samsung sell?" or "Samsung's products"

2. Trained to handle diverse query formats, regardless of phrasing variations.

    Example: "Tell me about Samsung.", "I'm curious about Samsung.", "What is Samsung?"

3. Optimized for Markdown table search, allowing retrieval of answers embedded within tables when present in documents.

4. Efficient clustering without hard negatives:

   - Samples within the same batch are clustered together.
   - Uses efficient embedding formation for clustering by truncating embeddings from the Snowflake/snowflake-arctic-embed-l-v2.0 model to 256 dimensions.
   - The clustering approach is inspired by the findings in the following papers:
     - *Embedding And Clustering Your Data Can Improve Contrastive Pretraining*
     - *CONTEXTUAL DOCUMENT EMBEDDINGS*

5. Strong performance across different domains:

    - The *Arctic-Embed 2.0: Multilingual Retrieval Without Compromise* paper states:  
     *"While models like mE5, mGTE, and BGE-M3 excel on MIRACL, their performance on CLEF is notably weaker compared to ours and closed-source offerings, suggesting the potential of overfitting to MIRACL or its Wikipedia-based domain."*  
    - Based on my own experience, **Snowflake/snowflake-arctic-embed-l-v2.0** has consistently outperformed **BGE-M3** in different domains, further validating this observation.


## Bias, Risks and Limitations

To prevent excessive GPU usage costs, the model was trained with a maximum sequence length of **1300** tokens. As a result, its performance may degrade on benchmarks like MultiLongDocRetrieval (MLDR).  

The previous model, **BGE-m3-ko**, was trained with a token length of **1024**, which imposed limitations on its MLDR benchmark performance.  

In the case of **snowflake-arctic-embed-l-v2.0-ko**, if the document length exceeds **1300** tokens or approximately **2500** characters, it is recommended to consider the following models instead.



| Model                                                                                            |   MultiLongDocRetrieval |
|:-------------------------------------------------------------------------------------------------|------------------------:|
| Alibaba-NLP/gte-multilingual-base/Alibaba-NLP/gte-multilingual-base                              |             **0.48402** |
| nlpai-lab/KURE-v1/nlpai-lab_KURE-v1                                                              |               _0.47528_ |
| dragonkue/snowflake-arctic-embed-l-v2.0-ko                                                       |                 0.4459  |
| BAAI/bge-m3/BAAI_bge-m3                                                                          |                 0.43011 |
| Snowflake/snowflake-arctic-embed-l-v2.0                                                          |                 0.40401 |
| dragonkue/BGE-m3-ko/dragonkue_BGE-m3-ko                                                          |                 0.40135 |
| openai/text-embedding-3-large                                                                    |                 0.31108 |
| BAAI/bge-multilingual-gemma2                                                                     |                 0.31021 |
| nlpai-lab/KoE5                                                                                   |                 0.30869 |
| jinaai/jina-embeddings-v3/jinaai__jina-embeddings-v3                                             |                 0.30512 |
| Alibaba-NLP/gte-Qwen2-7B-instruct/Alibaba-NLP__gte-Qwen2-7B-instruct                             |                 0.30313 |
| intfloat/multilingual-e5-large-instruct/intfloat__multilingual-e5-large-instruct                 |                 0.27973 |
| nomic-ai/nomic-embed-text-v2-moe                                                                 |                 0.27135 |
| intfloat/e5-mistral-7b-instruct/intfloat__e5-mistral-7b-instruct                                 |                 0.2583  |
| intfloat/multilingual-e5-large/intfloat__multilingual-e5-large                                   |                 0.24596 |
| Salesforce/SFR-Embedding-2_R/Salesforce__SFR-Embedding-2_R                                       |                 0.24346 |
| intfloat/multilingual-e5-base/intfloat__multilingual-e5-base                                     |                 0.23766 |
| upskyy/bge-m3-korean/upskyy__bge-m3-korean                                                       |                 0.21968 |
| ibm-granite/granite-embedding-278m-multilingual/ibm-granite__granite-embedding-278m-multilingual |                 0.20781 |
| jhgan/ko-sroberta-multitask/jhgan__ko-sroberta-multitask                                         |                 0.20416 |


<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

* Loss: [<code>CachedGISTEmbedLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedgistembedloss) with these parameters:



### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 20000
- `per_device_eval_batch_size`: 4096
- `learning_rate`: 2e-05
- `num_train_epochs`: 2
- `lr_scheduler_type`: warmup_stable_decay
- `lr_scheduler_kwargs`: {'num_decay_steps': 160}
- `warmup_ratio`: 0.05
- `bf16`: True
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 10000
- `per_device_eval_batch_size`: 4096
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: warmup_stable_decay
- `lr_scheduler_kwargs`: {'num_decay_steps': 160}
- `warmup_ratio`: 0.05
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: True
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>


### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.4.1
- Transformers: 4.49.0
- PyTorch: 2.6.0+cu124
- Accelerate: 1.4.0
- Datasets: 3.3.2
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```
#### KURE
```bibtex
@misc{KURE,
  publisher = {Youngjoon Jang, Junyoung Son, Taemin Lee},
  year = {2024},
  url = {https://github.com/nlpai-lab/KURE}
}
```
#### Arctic-Embed 2.0
```bibtex
@article{yu2024arcticembed,
  title = "Arctic-Embed 2.0: Multilingual Retrieval Without Compromise",
  author = "Puxuan Yu, Luke Merrick, Gaurav Nuti, Daniel Campos",
  journal = "arXiv preprint arXiv:2412.04506",
  year = "2024",
  url = "https://arxiv.org/abs/2412.04506"
}
```
#### Embedding And Clustering Your Data Can Improve Contrastive Pretraining
```bibtex
@article{merrick2024embedding,
  title = "Embedding And Clustering Your Data Can Improve Contrastive Pretraining",
  author = "Luke Merrick",
  journal = "arXiv preprint arXiv:2407.18887",
  year = "2024",
  url = "https://arxiv.org/abs/2407.18887"
}
```
#### Contextual Document Embeddings
```bibtex
@article{morris2024contextual,
  title = "Contextual Document Embeddings",
  author = "John X. Morris, Alexander M. Rush",
  journal = "arXiv preprint arXiv:2410.02525",
  year = "2024",
  url = "https://arxiv.org/abs/2410.02525"
}
```

## License

Arctic is licensed under the **Apache-2**. The released models can be used for commercial purposes free of charge.



<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->