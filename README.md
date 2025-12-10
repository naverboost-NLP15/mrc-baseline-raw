# 개선된 베이스라인 가이드 (Qdrant & Hybrid Retrieval)

## 설치 및 환경 설정

### 요구 사항

필요한 패키지를 설치합니다.
```bash
uv sync
```

### Qdrant 서버 설정

이 베이스라인은 외부 또는 로컬 Qdrant 서버(석진님께서 제공)가 필요합니다. 
`src/retrieval_qdrant_final.py` 내의 `QDRANT_HOST`, `QDRANT_PORT`, `api_key` 설정을 본인의 환경에 맞게 수정해야 할 수 있습니다. 

> **참고**: 현재 기본 설정은 `lori2mai11ya.asuscomm.com:6333`으로 되어 있습니다.


## 실행 방법

### 1. Train (MRC 모델 학습) 및 Eval

`train.py`는 extractive MRC(Reader) 모델을 학습합니다. 
- 하단은 `klue/roberta-large`로 진행했던 1차 학습 스크립트 사용 예시이며, 2차 학습 스크립트는 README 하단에 기재해 두었습니다.

*   `--add_korquad True` 플래그를 사용하여 KorQuad 1.0 데이터를 학습 데이터에 추가할 수 있습니다.


**example**

```bash
uv run code/src/train.py \
--output_dir code/models/train_dataset/모델저장경로 \
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

### 2. Inference (ODQA 추론)

Qdrant를 이용한 검색과 학습된 MRC 모델을 결합하여 최종 답변을 생성합니다. `inference.py`는 내부적으로 `QdrantHybridRetrieval`을 사용합니다.


*   `--do_eval`: eval 데이터로 추론
*   `--do_predict`: test 데이터로 추론(제출 직전에 사용)
*   `--alpha`: Hybrid 검색 시 Dense Vector의 가중치입니다. (기본값 0.5). Sparse 가중치는 `1.0 - alpha`가 됩니다.
*   `--fp16`: 인퍼런스 시 모델(Splade, Spell2, Reranking)을 fp16으로 로드하여 속도를 높입니다.

**example**

```bash
uv run code/src/inference.py \
--output_dir code/predictions/baseline_real \
--dataset_name raw/data/train_dataset/ \
--model_name_or_path code/models/train_dataset/roberta_large_korquad \
--eval_retrieval \
--top_k_retrieval 1 \
--do_eval \
--alpha 0.5
```

## 파일 구성

```bash
code/
├── src/
│   ├── retrieval_qdrant_final.py # Qdrant 기반 Hybrid Retrieval 클래스
│   ├── inference.py              # ODQA 추론 (QdrantHybridRetrieval 사용)
│   ├── train.py                  # MRC 학습 (KorQuad 추가 가능)
│   ├── trainer_qa.py             # QA Trainer
│   ├── arguments.py              # 실행 인자 정의
│   └── utils_qa.py               # QA 유틸리티
└── README.md              # 가이드 문서 (본 파일)
```

## 주의 사항

1.  **Collection 매칭**: 인덱싱할 때 생성한 `collection_name`이 `retrieval_qdrant_final.py`에서 참조하는 이름과 일치하는지 확인하세요. (기본값: `hybird_collection_v1` 또는 모델명 기반 자동 생성)
2.  **Overwrite Cache**: 모델 학습 시 `--overwrite_cache`를 사용하지 않으면 이전 캐시된 데이터가 로드될 수 있습니다. 데이터나 전처리가 변경되었다면 캐시를 덮어써주세요.


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
--output_dir code/models/train_dataset/모델저장경로 \
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



### Eval data 추론(with retriever)
```bash
uv run code/src/inference.py \
--output_dir code/predictions/baseline_real \
--dataset_name raw/data/train_dataset/ \
--model_name_or_path code/models/train_dataset/roberta_large_korquad \
--eval_retrieval \
--top_k_retrieval 1 \
--do_eval \
--alpha 0.5
```

### Test data 추론(with retriever)
```bash
uv run code/src/inference.py \
--output_dir code/predictions/예측결과저장경로 \
--dataset_name raw/data/test_dataset/ \
--model_name_or_path code/models/train_dataset/추론에사용될모델경로 \
--eval_retrieval \
--top_k_retrieval 1 \
--do_predict \
--alpha 0.5
```