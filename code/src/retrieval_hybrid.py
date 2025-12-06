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
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        chunk_overlap: int = 200,
        use_reranker: bool = True,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
    ) -> None:
        """
        Sparse + Dense + Reranker를 결합한 Hybrid Retriever

        Args:
            tokenize_fn (None): Defaults to Kiwi tokenizer
            data_path (str | None, optional): _description_. Defaults to "./raw/data".
            context_path (str | None, optional): _description_. Defaults to "wikipedia_documents.json".
            chunk_size (int, optional): Chunk size. Defaults to 1000.
            chunk_overlap (int, optional): Chunk overlap. Defaults to 100.
            use_reranker (bool): use Reranker or not
            reranker_model_name (str): Reranker model name
        """
        self.data_path = data_path
        self.context_path = context_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_reranker = use_reranker

        # Kiwi 형태소 분석기
        self.kiwi = Kiwi()

        # Wikipedia 문서 로드
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

            # Chunking
            chunks = self.split_text(text, self.chunk_size, self.chunk_overlap)

            for chunk in chunks:
                self.texts.append(chunk)
                self.titles.append(title)
                self.doc_ids.append(doc_id)
                self.ids.append(len(self.ids))  # Chunk ID (0부터 시작하는 고유 ID)

        print(f"Total Chunks: {len(self.texts)}")

        # Dense Model Load
        self.embedding_model_name = (
            "telepix/PIXIE-Rune-Preview"  # TODO:더 정밀한 성능 평가 필요
        )
        self.encoder = SentenceTransformer(self.embedding_model_name)

        self.bm25 = None
        self.faiss_index = None

        # TODO: Reranker Load
        self.reranker = None
        if self.use_reranker:
            print(f"Reranker 로딩 중...({reranker_model_name})")
            self.reranker = CrossEncoder(reranker_model_name)

    def split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        text를 문장 단위로 분리한 후, RecursiveCharacterTextSplitter를 사용하여 chunking을 진행합니다.
        (문장 분리로 Kiwi Tokenizer를 사용합니다.)
        """
        if not text:
            return []

        # 1. Kiwi로 문장 단위 분리 (한국어 문맥 보존)
        try:
            # type: ignore
            sents = [sent.text for sent in self.kiwi.split_into_sents(text)]
        except Exception:
            sents = text.split(". ")

        # 2. 문장들을 다시 합치되, 확실한 구분자(\n\n) 사용
        preprocessed_text = "\n\n".join(sents)

        # 3. RecursiveCharacterTextSplitter 사용
        # separators를 설정하여 우리가 넣은 구분자(\n\n)를 최우선으로 자르도록 유도
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],  # 문장 단위(\n\n) 우선 분할 시도
            length_function=len,
        )

        return text_splitter.split_text(preprocessed_text)

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
        # sparse, dense embedding 저장 경로 설정 (Chunk 정보 포함)
        model_name_str = self.embedding_model_name.replace("/", "_")
        chunk_suffix = f"_chunk{self.chunk_size}_overlap{self.chunk_overlap}_v3"

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
            # ! text + title로 합쳐서 인덱싱합니다.
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
                batch_size=8,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(
                dim
            )  # Inner Product (Cosine Similarity because normalized)
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
        Hybrid Search + Reranking 수행 함수
        """

        assert (
            self.bm25 is not None and self.faiss_index is not None
        ), "get_embedding() 이 먼저 호출되어야 합니다."

        if isinstance(query_or_dataset, str):
            queries = [query_or_dataset]
        elif isinstance(query_or_dataset, Dataset):
            queries = query_or_dataset["question"]
        else:
            queries = query_or_dataset  # list

        total_results = []

        # 1. Sparse Search (BM25)
        print("Sparse 검색 중...")
        bm25_scores_list: list[NDArray[np.float64]] | list = []
        bm25_indices_list: list[NDArray[np.int64]] | list = []

        for query in tqdm(queries, desc="BM25"):
            tokenized_query = self.kiwi_tokenizer(query)
            scores = self.bm25.get_scores(tokenized_query)

            # Top-k 인덱스만 뽑아옵니다.
            topk_indices = np.argsort(scores)[::-1][:topk]
            bm25_scores_list.append(scores[topk_indices])
            bm25_indices_list.append(topk_indices)

        # 2. Dense Search (FAISS)
        print("Dense 검색 중...")

        query_embeds = self.encoder.encode(
            queries,
            batch_size=8,
            show_progress_bar=True,
            normalize_embeddings=True,
            prompt_name="query",
        )
        dense_scores_list, dense_indices_list = self.faiss_index.search(
            query_embeds, topk
        )

        # 3. Fusion
        print("Fusing...")
        k = 60  # 보통 60이 쓰임.

        for i, query in enumerate(tqdm(queries, desc="Fusion")):
            doc_score_map: dict[int, float] = {}

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

            # Sort by fused
            sorted_docs = sorted(
                doc_score_map.items(), key=lambda x: x[1], reverse=True
            )[:topk]
            final_indices = [x[0] for x in sorted_docs]

            # Final docs construct
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

            # Final docs construct
            context = "\n\n".join([self.texts[idx] for idx in final_indices])
            retrieved_doc_ids = [self.doc_ids[idx] for idx in final_indices]

            tmp = {
                "question": query,
                "id": (
                    query_or_dataset[i]["id"]
                    if isinstance(query_or_dataset, Dataset)
                    else i
                ),
                "context": context,
                "retrieved_doc_ids": retrieved_doc_ids,
            }

            if isinstance(query_or_dataset, Dataset):
                if "answers" in query_or_dataset.features:
                    tmp["answers"] = query_or_dataset[i]["answers"]
                if "context" in query_or_dataset.features:
                    tmp["original_context"] = query_or_dataset[i]["context"]
                if "document_id" in query_or_dataset.features:
                    tmp["original_document_id"] = query_or_dataset[i]["document_id"]

            total_results.append(tmp)
        return pd.DataFrame(total_results)


if __name__ == "__main__":

    import argparse
    from datasets import load_from_disk, concatenate_datasets

    # Arugment 설정 (기존 retrieval.py와 호환성 유지 + Hybrid 파라미터 추가)
    parser = argparse.ArgumentParser(description="Hybrid Retrieval Test")
    parser.add_argument(
        "--dataset_name",
        metavar="./data/train_dataset",
        type=str,
        default="raw/data/train_dataset",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--data_path",
        metavar="./data",
        type=str,
        default="raw/data",
        help="Path to the directory",
    )
    parser.add_argument(
        "--context_path",
        metavar="wikipedia_documents.json",
        type=str,
        default="wikipedia_documents.json",
        help="Context(Document) file name",
    )
    parser.add_argument(
        "--topk",
        metavar=20,
        type=int,
        default=20,
        help="Number of passages to retrieve",
    )
    parser.add_argument(
        "--bm25_weight",
        metavar=1.0,
        type=float,
        default=1.0,
        help="Weight for BM25 results",
    )
    parser.add_argument(
        "--dense_weight",
        metavar=1.0,
        type=float,
        default=1.0,
        help="Weight for Dense results",
    )

    args = parser.parse_args()

    # 데이터셋 로드
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
        # train_dataset이 아닌, tset_dataset인 경우 validation만 존재하므로, 예외처리
        full_ds = original_dataset["validation"]

    print("*" * 40, "Query Dataset Info", "*" * 40)
    print(full_ds)

    # Hybrid Retrieval 초기화
    retriever = HybridRetrieval(
        tokenize_fn=None,  # 내부적으로 kiwi tokenzier 사용
        data_path=args.data_path,
        context_path=args.context_path,
    )

    # 임베딩 생성 또는 로드
    retriever.get_embedding()

    # Retrieval 성능 테스트
    # Ground Truth(original_context)가 검색된 Top-K 문서들(context) 안에 포함되어 있는지 확인.
    with timer("Bulk query by Hybrid search"):
        df = retriever.retrieve(
            query_or_dataset=full_ds,
            topk=args.topk,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
        )

        if "answers" in df.columns:
            correct_count = 0
            for idx, row in df.iterrows():
                # 방식 B: 검색된 Context 안에 실제 정답 텍스트가 포함되어 있는지 확인
                # answers는 {'text': ['정답1', '정답2'], 'answer_start': [10]} 형태
                answer_texts = row["answers"]["text"]
                if any(ans in row["context"] for ans in answer_texts):
                    correct_count += 1

            acc = correct_count / len(df)
            print(f"Top-{args.topk} Retrieval Accuracy (Answer Match): {acc:.4f}")

        elif "original_document_id" in df.columns:
            # Test 데이터셋처럼 정답이 없고 문서 ID만 있는 경우 (혹은 호환성 유지)
            correct_count = 0
            for idx, row in df.iterrows():
                if row["original_document_id"] in row["retrieved_doc_ids"]:
                    correct_count += 1
            acc = correct_count / len(df)
            print(f"Top-{args.topk} Retrieval Accuracy (Doc ID Match): {acc:.4f}")

        else:
            print("ground truth info가 없습니다. 성능 체크를 스킵합니다.")

    # Single Query Test
    test_query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    print(f"\n[Test Single Query]: {test_query}")
    res_df = retriever.retrieve(test_query, topk=5)
    print("Result Context Sample:")
    print(res_df.iloc[0]["context"])
    print("실험 종료")
