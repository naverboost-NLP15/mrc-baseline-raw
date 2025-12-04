import os
import json
import time
import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Union, Optional
from datasets import Dataset

import torch
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer
import faiss


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class HybridRetrieval:
    def __init__(
        self,
        tokenize_fn: Callable,
        data_path: str | None = "./raw/data",
        context_path: str | None = "wikipedia_documents.json",
    ) -> None:
        """
        BM25(Sparse) + FAISS(Dense)를 결합한 Hybrid Retriever

        Args:
            tokenize_fn (Callable): _description_
            data_path (str | None, optional): _description_. Defaults to "./raw/data".
            context_path (str | None, optional): _description_. Defaults to "wikipedia_documents.json".
        """
        self.tokenize_fn = tokenize_fn
        self.data_path = data_path
        self.context_path = context_path

        # 문서 로드
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki: dict = json.load(f)

        # 중복 제거 및 Langchain Document 형식으로 변환
        # text + title 정보 추가, 중복 제거는 text 기준으로만 변경

        self.documents = []
        seen_texts = set()

        print("document 처리 중 (Title + Text 전략)...")
        for v in wiki.values():
            text = v["text"]
            title = v["title"]
            doc_id = v["document_id"]

            if text in seen_texts:
                continue

            seen_texts.add(text)

            combined_context = f"{title}\n{text}"

            self.documents.append(
                Document(
                    page_content=combined_context,
                    metadata={
                        "id": doc_id,
                        "title": title,
                        "original_text": text,
                    },
                )
            )

        # Retriever 초기화
        self.retriever: EnsembleRetriever | None = None

    def get_embedding(self) -> None:
        """
        BM25 Retriever와 Dense Retriever를 생성하거나 로드하여 Ensemble Retriever를 구축합니다.
        """

        # sparse, dense embedding 저장 경로 설정
        bm25_path = os.path.join(self.data_path, "bm25_retriever.pkl")
        faiss_path = os.path.join(self.data_path, "faiss_index")

        # Dense 설정
        # TODO: 한국어 성능이 좋은 모델 선정...
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        encode_kwargs = {"normalize_embeddings": True}

        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        if os.path.exists(faiss_path):
            print("FAISS index 로딩 중...")
            vectorstore = FAISS.load_local(
                folder_path=faiss_path,
                embeddings=hf,
                allow_dangerous_deserialization=True,
            )
        else:
            print("FAISS index 구축 중...")
            vectorstore = FAISS.from_documents(self.documents, hf)
            vectorstore.save_local(folder_path=faiss_path)
            print("FAISS index 저장이 완료되었습니다.")

        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        # Sparse 설정
        if os.path.isfile(bm25_path):
            print("BM25 retriever 로딩 중...")
            with open(bm25_path, "rb") as f:
                bm25_retriever = pickle.load(f)

        else:
            print("BM25 retriever 구축 중...")
            # tokenize_fn을 사용하여 토큰화 후 BM25 생성
            # LangChain BM25는 기본적으로 공백 split을 사용하므로,
            # 한글 처리를 위해 preprocess_func를 커스텀하거나 토큰화된 리스트를 넘겨야 함.
            # 여기서는 편의상 from_documents 사용 (내부적으로 기본 토크나이저 사용됨)
            # TODO: 성능을 높이려면 konlpy Mecab 등의 토크나이저로 전처리가 필요함.

            bm25_retriever = BM25Retriever.from_documents(
                self.documents,
                preprocess_func=self.tokenize_fn,
            )
            bm25_retriever.k = 10

            with open(bm25_path, "wb") as f:
                pickle.dump(bm25_retriever, f)
            print("BM25 retriever saved.")

        # 3. Ensemble 설정
        # weights: [BM25 가중치, Dense 가중치]. 보통 0.3:0.7 또는 0.5:0.5 사용
        print("Ensemble Retriever 초기화 중...")
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5],
        )

    def retrieve(
        self,
        query_or_dataset: str | Dataset,
        topk: int | None,
    ) -> pd.DataFrame | list[Document]:
        """
        LangChain Ensemble Retriever를 사용하여 문서 검색

        Args:
            query_or_dataset (str | Dataset):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 langchain의 `retriever.invoke`를 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우, 다수의 query를 처리하기 위해 for 루프를 통해 여러 번 invoke를 호출하며, 얻은 docs를 total 리스트에 저장합니다.
            topk (int | None): 상위 k개의 문서를 반환합니다.

        Returns:
            pd.DataFrame: _description_
        """

        assert self.retriever is not None, "get_embedding() 이 먼저 호출되어야 합니다."

        # topk 설정 업데이트 (Retrievers의 k값 조정)
        self.retriever.retrievers[0].k = topk  # BM25
        self.retriever.retrievers[1].search_kwargs["k"] = topk  # Dense

        total = []  # Multi-Query 대비 list

        # 단일 쿼리를 받는 경우
        if isinstance(query_or_dataset, str):
            with timer("Single Query Search"):
                docs = self.retriever.invoke(query_or_dataset)
                print(f"[Query]: {query_or_dataset}")

                for i, doc in enumerate(docs):
                    print(f"Top-{i+1}: {doc.page_content[:50]}...")

            return docs

        # 다수 쿼리를 받는 경우
        elif isinstance(query_or_dataset, Dataset):
            queries = query_or_dataset["question"]

            print(f"Retrieving for {len(queries)} queries...")

            # LangChain의 invoke는 단일 쿼리용이므로 loop를 돌아야 함.
            # 배치 처리를 위해 vectorstore의 search를 직접 배치로 돌리는 방법도 있으나
            # EnsembleRetriever 구조상 loop가 가장 안전하다고 함.

            for idx, query in enumerate(
                tqdm(queries, desc="Hybrid Retrieval on mulit queries")
            ):
                retrieved_docs = self.retriever.invoke(query)

                context = "\n".join([doc.page_content for doc in retrieved_docs])

                tmp = {
                    "question": query,
                    "id": query_or_dataset[idx]["id"],
                    "context": context,
                }

                if (
                    "context" in query_or_dataset.features
                    and "answers" in query_or_dataset.features
                ):
                    tmp["original_context"] = query_or_dataset[idx]["context"]
                    tmp["answers"] = query_or_dataset[idx]["answers"]

                total.append(tmp)

            return pd.DataFrame(total)


import torch


if __name__ == "__main__":
    pass
