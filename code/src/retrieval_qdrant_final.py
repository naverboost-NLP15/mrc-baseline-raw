import os
import json
import time
import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import wandb
from contextlib import contextmanager
from typing import List, Union, Optional
from datasets import Dataset
from rank_bm25 import BM25Okapi

import torch
import binascii
from collections import Counter
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, CrossEncoder, SparseEncoder
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from scipy.sparse import csr_matrix

from numpy.typing import NDArray

# Qdrant Configuration (Default)
QDRANT_HOST = "lori2mai11ya.asuscomm.com"
QDRANT_PORT = 6333


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


def compute_bm25_weight(tf, doc_len, avgdl, idf, k1=1.2, b=0.75):
    # BM25의 (IDF * TF component) 계산
    # Qdrant 검색 시 Query Vector Value = 1.0 (또는 Query TF)으로 설정하면 됨.
    numerator = tf * (k1 + 1)
    denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
    return idf * (numerator / denominator)


def hash_token(token):
    # unsigned 32-bit int로 변환
    return binascii.crc32(token.encode("utf-8")) & 0xFFFFFFFF


class QdrantHybridRetrieval:
    def __init__(
        self,
        data_path: str = "./raw/data",
        context_path: str = "wikipedia_documents.json",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_reranker: bool = True,
        use_fp16: bool = False,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        dense_model_name: str = "telepix/PIXIE-Spell-Preview-1.7B",
        sparse_model_name: str = "telepix/PIXIE-Splade-Preview",
        collection_name: Optional[str] = None,
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT,
        qdrant_api_key: str = "boostcamp",
    ) -> None:
        """
        Qdrant 기반 Hybrid Retriever (Dense + Sparse/Splade + Reranker)
        """
        self.data_path = data_path
        self.context_path = context_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_reranker = use_reranker
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name

        # Collection Name
        if collection_name is None:
            # Default fallback (build_qdrant_hybrid_index.py 규칙)
            safe_dense = dense_model_name.replace("/", "_").replace("-", "_")
            self.collection_name = f"hybird_collection_v1"
        else:
            self.collection_name = collection_name

        # Kiwi 형태소 분석기 (BM25용)
        self.kiwi = Kiwi()

        # 1. Data Loading (Chunking Cache) - Reranking 및 텍스트 반환용
        chunk_cache_path = os.path.join(
            data_path, f"wiki_chunks_{chunk_size}_overlap{chunk_overlap}.pkl"
        )

        if os.path.isfile(chunk_cache_path):
            print(f"Chunking Cache 로드 중... ({chunk_cache_path})")
            with open(chunk_cache_path, "rb") as f:
                data = pickle.load(f)
            self.texts = data["texts"]
            self.titles = data["titles"]
            self.doc_ids = data["doc_ids"]
            self.ids = data["ids"]
            print(f"Cached Total Chunks: {len(self.texts)}")
        else:
            # Cache가 없으면 원본 문서 로드 및 청킹 수행
            with open(
                os.path.join(data_path, context_path), "r", encoding="utf-8"
            ) as f:
                wiki: dict = json.load(f)

            self.texts = []
            self.titles = []
            self.ids = []
            self.doc_ids = []

            print("문서 로드 및 Chunking 중...")
            seen_texts = set()

            for v in tqdm(wiki.values(), desc="Processing Wiki"):
                text, title, doc_id = v["text"], v["title"], v["document_id"]
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                chunks = self.split_text(text, self.chunk_size, self.chunk_overlap)
                for chunk in chunks:
                    self.texts.append(chunk)
                    self.titles.append(title)
                    self.doc_ids.append(doc_id)
                    self.ids.append(len(self.ids))

            with open(chunk_cache_path, "wb") as f:
                pickle.dump(
                    {
                        "texts": self.texts,
                        "titles": self.titles,
                        "doc_ids": self.doc_ids,
                        "ids": self.ids,
                    },
                    f,
                )
            print("Chunking 결과 저장 완료")

        # 2. Load Models
        model_kwargs = (
            {"torch_dtype": torch.float16}
            if use_fp16 and torch.cuda.is_available()
            else {}
        )

        # Dense Encoder
        print(f"Loading Dense Encoder: {self.dense_model_name} (FP16={use_fp16})")
        self.dense_encoder = SentenceTransformer(
            self.dense_model_name, model_kwargs=model_kwargs
        )

        # Sparse (SPLADE) Encoder
        print(
            f"Loading Sparse (SPLADE) Model & Tokenizer: {self.sparse_model_name} (FP16={use_fp16})"
        )
        self.sparse_encoder = SparseEncoder(
            self.sparse_model_name, model_kwargs=model_kwargs
        )
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(self.sparse_model_name)
        self.special_token_ids = set(self.sparse_tokenizer.all_special_ids)

        if torch.cuda.is_available():
            self.sparse_encoder.to("cuda")

        # Qdrant Client
        print(f"Connecting to Qdrant ({qdrant_host}:{qdrant_port})...")
        self.client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            api_key=qdrant_api_key,
            https=False,
            timeout=60,
        )

        if not self.client.collection_exists(self.collection_name):
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist in Qdrant!"
            )

        print(f"Target Collection: {self.collection_name}")

        # BM25 (Local Object for tokenizer usage, not for search index)
        # We need this to keep consistency in tokenization if needed, but Qdrant handles retrieval.
        # Actually we don't need the full BM25 index object here for Qdrant retrieval,
        # but if we wanted to use 'bm25' sparse vector in Qdrant, we just need to tokenize query.
        self.bm25 = None

        # Reranker
        self.reranker = None
        if self.use_reranker:
            print(f"Reranker 로딩 중...({reranker_model_name})")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.reranker = CrossEncoder(
                reranker_model_name,
                device=device,
                model_kwargs={"dtype": torch.float16 if use_fp16 else torch.float32},
            )

    def split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        if not text:
            return []
        try:
            sents = [sent.text for sent in self.kiwi.split_into_sents(text)]
        except Exception:
            sents = text.split(". ")
        preprocessed_text = "\n\n".join(sents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
        return text_splitter.split_text(preprocessed_text)

    def kiwi_tokenizer(self, text: str) -> list[str]:
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

    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: int = 20,
        topn: int = 50,
        alpha: float = 0.5,  # Weight between Dense and Sparse (0.0 = Sparse Only, 1.0 = Dense Only)
        sparse_type: str = "splade",  # "splade" or "bm25"
    ) -> pd.DataFrame:
        """
        Hybrid Search using Qdrant (Dense + Sparse) -> Reranking
        Args:
            alpha: Weight for Dense (1 - alpha for Sparse)
            sparse_type: Which sparse vector to use ("splade", "bm25", or "custom_sparse")
        """

        if isinstance(query_or_dataset, str):
            queries = [query_or_dataset]
        elif isinstance(query_or_dataset, Dataset):
            queries = query_or_dataset["question"]
        else:
            queries = query_or_dataset

        total_results = []

        # Candidate Expansion
        topn_candidate = min(len(self.ids), topk * 5)
        if topn_candidate < topn:
            topn_candidate = topn

        # 1. Query Encoding
        print(
            f"Encoding Queries... (Dense: {self.dense_model_name}, Sparse: {sparse_type})"
        )

        # Dense Encoding
        dense_query_embs = self.dense_encoder.encode(
            queries,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True,
            prompt_name="query",
        )

        # Sparse Encoding
        sparse_query_vecs = []

        if sparse_type == "splade":
            # Batch encode for SPLADE
            batch_size = 16
            for i in tqdm(range(0, len(queries), batch_size), desc="SPLADE Encoding"):
                batch_queries = queries[i : i + batch_size]
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        q_embs = self.sparse_encoder.encode_query(batch_queries)

                if isinstance(q_embs, torch.Tensor):
                    if q_embs.is_sparse:
                        q_embs = q_embs.to_dense()
                    q_embs = q_embs.cpu().numpy()

                if q_embs.dtype == np.float16:
                    q_embs = q_embs.astype(np.float32)

                # Filter special tokens
                if hasattr(self, "special_token_ids"):
                    for sp_id in self.special_token_ids:
                        if sp_id < q_embs.shape[1]:
                            q_embs[:, sp_id] = 0

                # Convert to Qdrant Sparse Vector format
                for row in q_embs:
                    indices = np.nonzero(row)[0].tolist()
                    values = row[indices].tolist()
                    sparse_query_vecs.append(
                        models.SparseVector(indices=indices, values=values)
                    )

        elif sparse_type == "bm25":
            # On-the-fly BM25 Query Vector Generation
            # Note: We need IDF?
            # In our build script (method 1), we multiplied IDF into Doc Vector.
            # So Query Vector should just have TF (usually 1.0 for binary query term presence).
            for query in tqdm(queries, desc="BM25 Encoding"):
                tokens = self.kiwi_tokenizer(query)
                token_counts = Counter(tokens)

                indices = []
                values = []
                for token, count in token_counts.items():
                    idx = hash_token(token)
                    # Query weight = 1.0 (or count)
                    # Simple approach: weight = 1.0 * count
                    indices.append(idx)
                    values.append(float(count))  # Query TF

                sparse_query_vecs.append(
                    models.SparseVector(indices=indices, values=values)
                )

        elif sparse_type == "custom_sparse":
            # Placeholder for custom sparse logic
            print(
                "Warning: custom_sparse query encoding not implemented. Using empty vectors."
            )
            for _ in queries:
                sparse_query_vecs.append(models.SparseVector(indices=[], values=[]))

        else:
            raise ValueError(f"Unknown sparse_type: {sparse_type}")

        # 2. Qdrant Batch Search (Hybrid)
        print(f"Searching Qdrant (Top-{topn_candidate})...")

        search_requests = []
        for dense_vec, sparse_vec in zip(dense_query_embs, sparse_query_vecs):
            # Hybrid Query using Prefetch (Two-stage or Fusion)
            # Qdrant supports Hybrid Query directly?
            # Currently Qdrant Python Client supports query fusion via `prefetch` since v1.10?
            # Or we can use the older RRF method by doing 2 searches.
            # But here we want to use Qdrant's internal hybrid capabilities if possible,
            # OR replicate the RRF logic from retrieval_hybrid.py.
            # The prompt asked to replicate the logic of retrieval_hybrid.py (which uses RRF).
            # So we will perform 2 searches per query (one Dense, one Sparse) and fuse them.

            # Wait, sending 2*N requests might be slow.
            # Qdrant's newer API allows Fusion. But let's stick to RRF for consistency with retrieval_hybrid.py.
            pass

        # To optimize, we will use client.search_batch twice: once for dense, once for sparse.

        # A. Dense Search Batch
        dense_requests = [
            models.SearchRequest(
                vector=models.NamedVector(name="dense", vector=dense_vec.tolist()),
                limit=topn_candidate,
                with_payload=False,
            )
            for dense_vec in dense_query_embs
        ]

        # B. Sparse Search Batch
        sparse_requests = [
            models.SearchRequest(
                vector=models.NamedSparseVector(name=sparse_type, vector=sparse_vec),
                limit=topn_candidate,
                with_payload=False,
            )
            for sparse_vec in sparse_query_vecs
        ]

        # Execute Search in Batches to avoid Timeout
        search_batch_size = 64

        print("Executing Dense Batch Search...")
        dense_results_batch = []
        for i in tqdm(
            range(0, len(dense_requests), search_batch_size),
            desc="Dense Search Batches",
        ):
            batch_reqs = dense_requests[i : i + search_batch_size]
            batch_res = self.client.search_batch(
                collection_name=self.collection_name, requests=batch_reqs
            )
            dense_results_batch.extend(batch_res)

        print("Executing Sparse Batch Search...")
        sparse_results_batch = []
        for i in tqdm(
            range(0, len(sparse_requests), search_batch_size),
            desc="Sparse Search Batches",
        ):
            batch_reqs = sparse_requests[i : i + search_batch_size]
            batch_res = self.client.search_batch(
                collection_name=self.collection_name, requests=batch_reqs
            )
            sparse_results_batch.extend(batch_res)

        # 3. Fusion (RRF)
        print("Fusing Results...")
        k = 60

        # Weights from alpha
        # alpha = 1.0 -> Dense Only
        # alpha = 0.0 -> Sparse Only
        dense_weight = alpha
        sparse_weight = 1.0 - alpha

        for i, query in enumerate(tqdm(queries, desc="Fusion")):
            doc_score_map = {}

            # Dense Results
            for rank, hit in enumerate(dense_results_batch[i]):
                if hit.id not in doc_score_map:
                    doc_score_map[hit.id] = 0
                doc_score_map[hit.id] += dense_weight * (1 / (k + rank + 1))

            # Sparse Results
            for rank, hit in enumerate(sparse_results_batch[i]):
                if hit.id not in doc_score_map:
                    doc_score_map[hit.id] = 0
                doc_score_map[hit.id] += sparse_weight * (1 / (k + rank + 1))

            # Sort
            sorted_docs = sorted(doc_score_map.items(), key=lambda x: -x[1])
            rerank_candidates_count = min(len(sorted_docs), topn)
            rerank_candidates = sorted_docs[:rerank_candidates_count]

            # 4. Reranking
            final_indices = []
            if self.use_reranker and self.reranker:
                pairs = []
                candidates_indices = [c[0] for c in rerank_candidates]
                for idx in candidates_indices:
                    # Retrieve Text from Local Cache
                    doc_text = f"{self.titles[idx]} {self.texts[idx]}"
                    pairs.append((query, doc_text))

                rerank_scores = self.reranker.predict(pairs)
                reranked_results = sorted(
                    zip(candidates_indices, rerank_scores), key=lambda x: -x[1]
                )
                final_indices = [x[0] for x in reranked_results[:topk]]
            else:
                final_indices = [x[0] for x in rerank_candidates[:topk]]

            # Result Construction
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

    parser = argparse.ArgumentParser(description="Qdrant Hybrid Retrieval Final")
    parser.add_argument("--dataset_name", type=str, default="raw/data/train_dataset")
    parser.add_argument("--data_path", type=str, default="raw/data")
    parser.add_argument("--context_path", type=str, default="wikipedia_documents.json")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--topn", type=int, default=50)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for Dense (0.0~1.0). 0.5 means Dense 0.5 + Sparse 0.5",
    )
    parser.add_argument(
        "--sparse_type",
        type=str,
        default="splade",
        choices=["splade", "bm25", "custom_sparse"],
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="hybird_collection_v1",
        help="Collection name in Qdrant",
    )
    parser.add_argument(
        "--dense_model", type=str, default="telepix/PIXIE-Spell-Preview-1.7B"
    )
    parser.add_argument(
        "--sparse_model", type=str, default="telepix/PIXIE-Splade-Preview"
    )
    parser.add_argument("--no_reranker", dest="use_reranker", action="store_false")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 for models to save memory and speed up inference",
    )
    parser.set_defaults(use_reranker=True)

    # WandB arguments
    parser.add_argument(
        "--wandb_project", type=str, default="QDQA_Retrieval", help="WandB project name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="WandB entity name"
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name")

    args = parser.parse_args()

    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=vars(args),
    )

    # Load Dataset
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

    # Initialize Retriever
    retriever = QdrantHybridRetrieval(
        data_path=args.data_path,
        context_path=args.context_path,
        collection_name=args.collection_name,
        dense_model_name=args.dense_model,
        sparse_model_name=args.sparse_model,
        use_reranker=args.use_reranker,
        use_fp16=args.fp16,
    )

    # Retrieve
    with timer("Hybrid Search (Qdrant)"):
        df = retriever.retrieve(
            query_or_dataset=full_ds,
            topk=args.topk,
            topn=args.topn,
            alpha=args.alpha,
            sparse_type=args.sparse_type,
        )

    # Evaluation
    if "answers" in df.columns:
        correct_count = 0
        for idx, row in df.iterrows():
            answer_texts = row["answers"]["text"]
            if any(ans in row["context"] for ans in answer_texts):
                correct_count += 1
        acc = correct_count / len(df)
        print(f"Top-{args.topk} Accuracy: {acc:.4f}")

        wandb.log(
            {
                f"top{args.topk}_accuracy": acc,
                "correct_count": correct_count,
                "total_count": len(df),
            }
        )
