# 개선된 베이스라인 가이드 (Qdrant & Hybrid Retrieval)

## 소개

이 코드는 기존 MRC 베이스라인을 기반으로 **Qdrant 벡터 DB를 활용한 Hybrid Retrieval(Dense + SPLADE)** 및 **KorQuad 데이터셋 추가 학습** 기능을 포함하여 성능을 개선한 버전입니다.

## 주요 변경 사항

1.  **Hybrid Retrieval**: Qdrant를 사용하여 Dense Vector(예: BGE-M3, PIXIE)와 Sparse Vector(SPLADE/BM25)를 결합한 하이브리드 검색을 수행합니다.
2.  **Vector DB**: FAISS, 로컬 Pickle 대신 Qdrant를 사용하여 대규모 벡터 검색 및 필터링을 효율적으로 처리합니다.
3.  **KorQuad 학습**: 학습 시 `--add_korquad` 옵션을 통해 KorQuad v1.0 데이터를 추가하여 MRC 모델의 일반화 성능을 높일 수 있습니다.
4.  **Reranking**: 검색된 문서에 대해 Reranker 모델(BGE-Reranker 등)을 사용하여 검색 정밀도를 높입니다.

## 설치 및 환경 설정

### 요구 사항

필요한 패키지를 설치합니다.
```bash
pip install -r requirements.txt
```

### Qdrant 서버 설정

이 베이스라인은 외부 또는 로컬 Qdrant 서버가 필요합니다. 
`src/qdrant_indexing/build_qdrant_hybrid_index.py` 및 `src/retrieval_qdrant_final.py` 내의 `QDRANT_HOST`, `QDRANT_PORT`, `api_key` 설정을 본인의 환경에 맞게 수정해야 할 수 있습니다. 

> **참고**: 현재 기본 설정은 `lori2mai11ya.asuscomm.com:6333`으로 되어 있습니다.

## 데이터 구축 (Indexing)

ODQA 수행 전, 위키피디아 문서를 Qdrant에 인덱싱해야 합니다. 인덱싱 스크립트는 Dense와 Sparse 임베딩을 모두 생성하여 저장합니다.

```bash
# Qdrant에 하이브리드 인덱스 구축 (Dense + SPLADE + BM25)
python src/qdrant_indexing/build_qdrant_hybrid_index.py \
    --data_path ../data \
    --context_path wikipedia_documents.json \
    --dense_model_name "telepix/PIXIE-Spell-Preview-1.7B" \
    --sparse_model_name "telepix/PIXIE-Splade-Preview" \
    --collection_name "wiki_hybrid_PIXIE_splade_bm25"
```
*   위 스크립트는 청크 단위로 분할된 문서를 임베딩하여 업로드합니다.
*   GPU 환경에서 실행하는 것을 권장합니다.

## 실행 방법

### 1. Train (MRC 모델 학습)

`train.py`는 MRC(Reader) 모델을 학습합니다. `--add_korquad` 플래그를 사용하여 KorQuad 1.0 데이터를 학습 데이터에 추가할 수 있습니다.

```bash
python src/train.py \
    --output_dir ./models/train_dataset \
    --do_train \
    --dataset_name ../data/train_dataset \
    --add_korquad True \
    --num_train_epochs 3
```

### 2. Eval (평가)

학습된 모델을 검증 데이터셋(Validation set)으로 평가합니다.

```bash
python src/train.py \
    --output_dir ./outputs/train_dataset \
    --model_name_or_path ./models/train_dataset \
    --do_eval \
    --dataset_name ../data/train_dataset
```

### 3. Inference (ODQA 추론)

Qdrant를 이용한 검색과 학습된 MRC 모델을 결합하여 최종 답변을 생성합니다. `inference.py`는 내부적으로 `QdrantHybridRetrieval`을 사용합니다.

```bash
python src/inference.py \
    --output_dir ./outputs/test_dataset/ \
    --dataset_name ../data/test_dataset/ \
    --model_name_or_path ./models/train_dataset/ \
    --do_predict \
    --top_k_retrieval 10 \
    --eval_retrieval True \
    --dense_weight 0.5 \
    --fp16
```
*   `--dense_weight`: Hybrid 검색 시 Dense Vector의 가중치입니다. (0.0 ~ 1.0). Sparse 가중치는 `1.0 - dense_weight`가 됩니다.
*   `--fp16`: 인퍼런스 시 모델을 fp16으로 로드하여 속도를 높입니다.

## 파일 구성

```bash
code/
├── assets/                # 이미지 리소스
├── src/
│   ├── retrieval_qdrant_final.py # Qdrant 기반 Hybrid Retrieval 클래스
│   ├── inference.py              # ODQA 추론 (QdrantHybridRetrieval 사용)
│   ├── train.py                  # MRC 학습 (KorQuad 추가 가능)
│   ├── trainer_qa.py             # QA Trainer
│   ├── arguments.py              # 실행 인자 정의
│   ├── utils_qa.py               # QA 유틸리티
│   └── qdrant_indexing/
│       └── build_qdrant_hybrid_index.py # Qdrant 인덱싱 스크립트
├── requirements.txt       # 의존성 패키지 목록
└── README.md              # 가이드 문서 (본 파일)
```

## 주의 사항

1.  **Qdrant 연결**: `retrieval_qdrant_final.py`와 `build_qdrant_hybrid_index.py` 상단의 호스트 설정을 확인하세요. 서버가 실행 중이어야 검색이 가능합니다.
2.  **Collection 매칭**: 인덱싱할 때 생성한 `collection_name`이 `retrieval_qdrant_final.py`에서 참조하는 이름과 일치하는지 확인하세요. (기본값: `hybird_collection_v1` 또는 모델명 기반 자동 생성)
3.  **Overwrite Cache**: 모델 학습 시 `--overwrite_cache`를 사용하지 않으면 이전 캐시된 데이터가 로드될 수 있습니다. 데이터나 전처리가 변경되었다면 캐시를 덮어써주세요.


## 현재 리더보드에 올라온 모델 진행방식

- `klue/roberta-large` 모델 사용
- `train_data` + `korquad` 로 1차 파인튜닝
- `train_data` 로 2차 파인튜닝(이게 더 좋은 방법인지는 모르겠습니다.)
  - 2차 파인튜닝 결과
    - EM: 70.83 -> 71.67 상승
    - F1: 80.88 -> 80.69 소폭 하락?
- topk = 1
  - 현재 topk 개의 문서를 join해서 하나의 text로 만드는 과정에서 문제가 있어 보임
  - 문제를 해결하면 top k를 늘렸을 때 더 좋은 성능이 나올 것으로 예상됨.


### 1차 학습 스크립트 및 하이퍼파라미터
```bash
uv run code/src/train.py \
--output_dir code/models/train_dataset/{roberta_large_korquad} \
--do_train \
--do_eval \
--model_name_or_path klue/roberta-large \
--dataset_name raw/data/train_dataset \
--add_korquad True \
\
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \
--per_device_eval_batch_size 32 \
\
--learning_rate 1.5e-5 \
--weight_decay 0.01 \
--num_train_epochs 3 \
--warmup_ratio 0.1 \
\
--logging_steps 500 \
--eval_strategy steps \
--eval_steps 500 \
--save_steps 500 \
--save_total_limit 2 \
\
--load_best_model_at_end \
--metric_for_best_model exact_match \
--overwrite_output_dir \
--fp16 \
--seed 42
```

### 2차 학습 스크립트 및 하이퍼파라미터

```bash
uv run code/src/train.py \
--output_dir code/models/train_dataset/baseline_korquad_opt_finetuned_final \
--do_train \
--do_eval \
--model_name_or_path ./code/models/train_dataset/baseline_korquad_opt \
--dataset_name raw/data/train_dataset \
--add_korquad False \
\
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--per_device_eval_batch_size 32 \
\
--learning_rate 1e-5 \
--weight_decay 0.01 \
--num_train_epochs 5 \
--warmup_ratio 0.1 \
\
--logging_steps 100 \
--eval_strategy steps \
--eval_steps 100 \
--save_steps 100 \
--save_total_limit 2 \
\
--load_best_model_at_end \
--metric_for_best_model exact_match \
--overwrite_output_dir \
--fp16 \
--seed 42
```