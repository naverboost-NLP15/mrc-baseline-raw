import os
import json
import time
import pickle
from turtle import title
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
        tokenize_fn=None,
        data_path: str = "./raw/data",
        context_path: str = "wikipedia_documents.json",
    ) -> None:
        """
        BM25(Sparse) + FAISS(Dense)를 결합한 Hybrid Retriever

        Args:
            tokenize_fn (None): Defaults to Kiwi tokenizer
            data_path (str | None, optional): _description_. Defaults to "./raw/data".
            context_path (str | None, optional): _description_. Defaults to "wikipedia_documents.json".
        """
        self.data_path = data_path
        self.context_path = context_path

        # 문서 로드
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki: dict = json.load(f)

        self.texts = []
        self.titles = []
        self.ids = []

        # 중복 제거 및 데이터 리스트 생성
        print("중복 text 제거 중...")
        seen_texts = set()  # 모든 id값은 고유하지만, 중복되는 text가 존재하므로, 제거
        for v in wiki.values():
            text, title, id = v["text"], v["title"], v["document_id"]
            if text not in seen_texts:
                self.texts.append(text)
                self.titles.append(title)
                self.ids.append(id)
                seen_texts.add(text)

        # Kiwi 형태소 분석기
        self.kiwi = Kiwi()
        # Dense Model Load
        self.embedding_model_name = (
            "Qwen/Qwen3-Embedding-0.6B"  # TODO: Fix Embedding Model
        )
        self.encoder = SentenceTransformer(self.embedding_model_name)

        self.bm25 = None
        self.faiss_index = None

    # TODO: Ensemble Retriever 구현

    def kiwi_tokenizer(self, text: str) -> list[str]:
        """
        형태소의 품사가 명사, 동사, 형용사 등인 형태소만 추출

        Args:
            text (str): 원본 text

        Returns:
            list[str]: 추출된 형태소 Tokens
        """
        tokens = self.kiwi.tokenize(text)
        return [
            t.form for t in tokens if t.tag.startswith("N") or t.tag.startswith("V")
        ]

    def get_embedding(self) -> None:
        """
        BM25 Retriever와 Dense Retriever를 생성하거나 로드하여 Ensemble Retriever를 구축
        """

        # sparse, dense embedding 저장 경로 설정
        embedding_model_name_splitted = self.embedding_model_name.split("/")[1]

        bm25_path = os.path.join(self.data_path, "bm25_wiki.pkl")
        faiss_path = os.path.join(
            self.data_path, f"faiss_{embedding_model_name_splitted}.index"
        )

        # Sparse(BM25) 설정
        if os.path.isfile(bm25_path):
            print("BM25 retriever 로딩 중...")
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)

        else:
            print("BM25 retriever 구축 중...")
            # text + title로 합쳐서 인덱싱합니다.

            tokenized_corpus = [
                self.kiwi_tokenizer(tit + " " + txt)
                for tit, txt in zip(self.titles, self.texts)
            ]
            self.bm25 = BM25Okapi(corpus=tokenized_corpus)

            with open(bm25_path, "wb") as f:
                pickle.dump(self.bm25, f)
            print("BM25 retriever saved.")
            # TODO: bm25_retriever.k = 10

        # Dense 설정

        if os.path.exists(faiss_path):
            print("FAISS index 로딩 중...")
            self.faiss_index = faiss.read_index(faiss_path)
        else:
            print("FAISS index 구축 중...")
            # vectorstore = FAISS.from_documents(self.documents, hf)
            docs = [f"{tit}\n{txt}" for tit, txt in zip(self.titles, self.texts)]
            embeddings = self.encoder.encode(
                sentences=docs,
                batch_size=32,
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
