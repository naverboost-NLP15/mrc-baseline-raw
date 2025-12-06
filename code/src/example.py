import os
import json
import time
import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Union, Optional, Tuple
from datasets import Dataset

import torch
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

from numpy.typing import NDArray


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class HybridRetrieval:
    def __init__(
        self,
        tokenize_fn=None,
        data_path: str = "./raw/data",
        context_path: str = "wikipedia_documents.json",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        use_reranker: bool = True,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
    ) -> None:
        """
        BM25(Sparse) + FAISS(Dense) + Reranker를 결합한 Hybrid Retriever 개선판

        Args:
            tokenize_fn (None): Defaults to Kiwi tokenizer
            data_path (str | None, optional): Defaults to "./raw/data".
            context_path (str | None, optional): Defaults to "wikipedia_documents.json".
            chunk_size (int, optional): Chunk size. Defaults to 1000.
            chunk_overlap (int, optional): Chunk overlap. Defaults to 100.
            use_reranker (bool): Reranker 사용 여부.
            reranker_model_name (str): Reranker 모델 이름.
        """
        self.data_path = data_path
        self.context_path = context_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_reranker = use_reranker

        # Kiwi 형태소 분석기 초기화 (Chunking에 사용하기 위해 미리 선언)
        self.kiwi = Kiwi()

        # 문서 로드
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki: dict = json.load(f)

        self.texts = []
        self.titles = []
        self.ids = []
        self.doc_ids = []  # 원본 문서 ID

        # 중복 제거 및 데이터 리스트 생성 (Chunking 적용)
        print("문서 로드 및 Chunking 중...")
        seen_texts = set()

        for v in tqdm(wiki.values(), desc="Processing Wiki"):
            text, title, doc_id = v["text"], v["title"], v["document_id"]

            if text in seen_texts:
                continue

            seen_texts.add(text)

            # Chunking (Improved: 문장 단위 분리)
            chunks = self.split_text(text, self.chunk_size, self.chunk_overlap)

            for chunk in chunks:
                self.texts.append(chunk)
                self.titles.append(title)
                self.doc_ids.append(doc_id)
                self.ids.append(len(self.ids))  # Chunk ID

        print(f"Total Chunks: {len(self.texts)}")

        # Dense Model Load
        self.embedding_model_name = "intfloat/multilingual-e5-large-instruct"
        self.encoder = SentenceTransformer(self.embedding_model_name)

        # Reranker Load
        self.reranker = None
        if self.use_reranker:
            print(f"Loading Reranker ({reranker_model_name})...")
            self.reranker = CrossEncoder(reranker_model_name)

        self.bm25 = None
        self.faiss_index = None

    def split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Kiwi를 사용하여 문장 단위로 분리한 뒤, chunk_size(글자 수) 제한에 맞춰 병합합니다.
        Overlap을 적용하여 문맥 단절을 방지합니다.
        """
        if not text:
            return []

        # 문장 분리
        try:
            sents = [s.text for s in self.kiwi.split_into_sents(text)]
        except Exception:
            sents = text.split(". ")

        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sents:
            # 글자 수 기준 (기존 word count -> character count로 변경)
            sent_len = len(sent)

            if current_len + sent_len > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                # Overlap 적용: 이전 문장들을 역순으로 탐색하여 overlap 크기만큼 가져옴
                new_chunk = []
                new_len = 0

                for prev_sent in reversed(current_chunk):
                    if new_len + len(prev_sent) > chunk_overlap:
                        break
                    new_chunk.insert(0, prev_sent)
                    new_len += len(prev_sent)

                # 새 청크 시작 (Overlap 된 부분 + 현재 문장)
                current_chunk = new_chunk + [sent]
                current_len = new_len + sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def kiwi_tokenizer(self, text: str) -> list[str]:
        """
        형태소의 품사가 명사, 동사, 형용사, 숫자, 외국어, 한자 등인 형태소만 추출
        """
        tokens = self.kiwi.tokenize(text)
        return [
            t.form
            for t in tokens
            if t.tag.startswith("N")
            or t.tag.startswith("V")
            or t.tag.startswith("SN")
            or t.tag.startswith("SL")
            or t.tag.startswith("SH")
        ]

    def get_embedding(self) -> None:
        """
        BM25 Retriever와 Dense Retriever를 생성하거나 로드하여 Ensemble Retriever를 구축
        """
        # sparse, dense embedding 저장 경로 설정
        # _v2 suffix를 붙여 기존 인덱스와 충돌 방지 및 새 로직 적용
        model_name_str = self.embedding_model_name.replace("/", "_")
        chunk_suffix = f"_chunk{self.chunk_size}_overlap{self.chunk_overlap}_v2"

        bm25_path = os.path.join(self.data_path, f"bm25_wiki{chunk_suffix}.pkl")
        faiss_path = os.path.join(
            self.data_path, f"faiss_{model_name_str}{chunk_suffix}.index"
        )

        # Sparse(BM25) 설정
        if os.path.isfile(bm25_path):
            print("BM25 retriever 로딩 중...")
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            print("BM25 retriever 구축 중...")
            tokenized_corpus = [
                self.kiwi_tokenizer(tit + " " + txt)
                for tit, txt in zip(self.titles, self.texts)
            ]
            self.bm25 = BM25Okapi(corpus=tokenized_corpus)

            with open(bm25_path, "wb") as f:
                pickle.dump(self.bm25, f)
            print("BM25 retriever saved.")

        # Dense 설정
        if os.path.exists(faiss_path):
            print("FAISS index 로딩 중...")
            self.faiss_index = faiss.read_index(faiss_path)
        else:
            print("FAISS index 구축 중...")
            docs = [f"{tit}\n{txt}" for tit, txt in zip(self.titles, self.texts)]
            embeddings = self.encoder.encode(
                sentences=docs,
                batch_size=32,  # Batch size 증가
                show_progress_bar=True,
                normalize_embeddings=True,
            )

            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(embeddings)
            faiss.write_index(self.faiss_index, faiss_path)
            print("FAISS index 저장이 완료되었습니다.")

    def retrieve(
        self,
        query_or_dataset: str | Dataset,
        topk: int = 20,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> pd.DataFrame:
        """
        Hybrid Search 수행 함수 (Candidate Expansion -> Fusion -> Reranking)
        """
        assert (
            self.bm25 is not None and self.faiss_index is not None
        ), "get_embedding() 이 먼저 호출되어야 합니다."

        if isinstance(query_or_dataset, str):
            queries = [query_or_dataset]
        elif isinstance(query_or_dataset, Dataset):
            queries = query_or_dataset["question"]
        else:
            queries = query_or_dataset

        total_results = []

        # 1. Candidate Expansion: 최종 topk보다 더 많은 후보를 검색
        candidate_topk = min(len(self.ids), topk * 5)
        if candidate_topk < 50:
            candidate_topk = 50

        # 1. Sparse Search (BM25)
        print(f"Sparse 검색 중 (Top-{candidate_topk})...")
        bm25_scores_list = []
        bm25_indices_list = []

        for query in tqdm(queries, desc="BM25"):
            tokenized_query = self.kiwi_tokenizer(query)
            scores = self.bm25.get_scores(tokenized_query)
            topk_indices = np.argsort(scores)[::-1][:candidate_topk]
            bm25_scores_list.append(scores[topk_indices])
            bm25_indices_list.append(topk_indices)

        # 2. Dense Search (FAISS)
        print(f"Dense 검색 중 (Top-{candidate_topk})...")
        dense_queries = [f"query: {q}" for q in queries]
        query_embeds = self.encoder.encode(
            dense_queries,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        dense_scores_list, dense_indices_list = self.faiss_index.search(
            query_embeds, candidate_topk
        )

        # 3. Fusion (Weighted RRF) & Reranking
        print("Fusing & Reranking...")
        k = 60

        for i, query in enumerate(tqdm(queries, desc="Fusion & Rerank")):
            doc_score_map = {}

            # Sparse Scores (Rank based)
            for rank, idx in enumerate(bm25_indices_list[i]):
                if idx not in doc_score_map:
                    doc_score_map[idx] = 0
                doc_score_map[idx] += bm25_weight * (1 / (k + rank + 1))

            # Dense Scores (Rank based)
            for rank, idx in enumerate(dense_indices_list[i]):
                if idx not in doc_score_map:
                    doc_score_map[idx] = 0
                doc_score_map[idx] += dense_weight * (1 / (k + rank + 1))

            # Sort by fused score
            sorted_docs = sorted(
                doc_score_map.items(), key=lambda x: x[1], reverse=True
            )

            # Reranking 후보군 선정 (예: 상위 50개)
            rerank_candidates_count = min(len(sorted_docs), 50)
            rerank_candidates = sorted_docs[:rerank_candidates_count]

            final_indices = []

            if self.use_reranker and self.reranker:
                # CrossEncoder 입력 생성: (Query, Document Title + Text)
                pairs = []
                candidate_indices = [x[0] for x in rerank_candidates]

                for idx in candidate_indices:
                    doc_text = f"{self.titles[idx]} {self.texts[idx]}"
                    pairs.append([query, doc_text])

                # Reranking 점수 계산
                rerank_scores = self.reranker.predict(pairs)

                # 점수 기준 재정렬
                reranked_results = sorted(
                    zip(candidate_indices, rerank_scores),
                    key=lambda x: x[1],
                    reverse=True,
                )

                # 최종 Top-K 추출
                final_indices = [x[0] for x in reranked_results[:topk]]
            else:
                # Reranker 미사용 시 RRF 결과 그대로 사용
                final_indices = [x[0] for x in rerank_candidates[:topk]]

            # 최종 Context 생성
            context = "\n\n".join([self.texts[idx] for idx in final_indices])

            tmp = {
                "question": query,
                "id": (
                    query_or_dataset[i]["id"]
                    if isinstance(query_or_dataset, Dataset)
                    else i
                ),
                "context": context,
            }

            if isinstance(query_or_dataset, Dataset):
                if "answers" in query_or_dataset.features:
                    tmp["answers"] = query_or_dataset[i]["answers"]
                if "context" in query_or_dataset.features:
                    tmp["original_context"] = query_or_dataset[i]["context"]

            total_results.append(tmp)

        return pd.DataFrame(total_results)


if __name__ == "__main__":
    import argparse
    from datasets import load_from_disk, concatenate_datasets

    parser = argparse.ArgumentParser(description="Hybrid Retrieval Test (Example)")
    parser.add_argument("--dataset_name", type=str, default="raw/data/train_dataset")
    parser.add_argument("--data_path", type=str, default="raw/data")
    parser.add_argument("--context_path", type=str, default="wikipedia_documents.json")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--bm25_weight", type=float, default=0.5)
    parser.add_argument("--dense_weight", type=float, default=0.5)
    parser.add_argument("--use_reranker", action="store_true", help="Use Reranker")
    parser.add_argument(
        "--no_reranker",
        action="store_false",
        dest="use_reranker",
        help="Disable Reranker",
    )
    parser.set_defaults(use_reranker=True)

    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_name}...")
    original_dataset = load_from_disk(args.dataset_name)
    try:
        full_ds = concatenate_datasets(
            [
                original_dataset["train"].flatten_indices(),
                original_dataset["validation"].flatten_indices(),
            ]
        )
    except KeyError:
        full_ds = original_dataset["validation"]

    print("*" * 40, "Query Dataset Info", "*" * 40)
    print(full_ds)

    retriever = HybridRetrieval(
        tokenize_fn=None,
        data_path=args.data_path,
        context_path=args.context_path,
        use_reranker=args.use_reranker,
    )

    retriever.get_embedding()

    with timer("Bulk query by Hybrid search"):
        df = retriever.retrieve(
            query_or_dataset=full_ds,
            topk=args.topk,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
        )

        if "original_context" in df.columns:
            correct_count = 0
            for idx, row in df.iterrows():
                if row["original_context"] in row["context"]:
                    correct_count += 1
            acc = correct_count / len(df)
            print(f"Top-{args.topk} Retrieval Accuracy: {acc:.4f}")
        else:
            print("ground truth context가 없습니다. 성능 체크를 스킵합니다.")

    test_query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    print(f"\n[Test Single Query]: {test_query}")
    res_df = retriever.retrieve(test_query, topk=5)
    print("Result Context Sample:")
    print(res_df.iloc[0]["context"])
