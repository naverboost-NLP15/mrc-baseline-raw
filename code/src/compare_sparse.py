import os
import json
import time
import pickle
import argparse
import pandas as pd
import numpy as np
import torch
import faiss

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Union, Optional, Tuple, Dict
from datasets import Dataset, load_from_disk, concatenate_datasets
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, CrossEncoder, SparseEncoder
from transformers import AutoTokenizer  # Added
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scipy.sparse import csr_matrix, save_npz, load_npz
from numpy.typing import NDArray


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrievalComparison:
    def __init__(
        self,
        data_path: str = "./raw/data",
        context_path: str = "wikipedia_documents.json",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_type: str = "bm25",  # bm25, splade, bge-m3
    ) -> None:
        self.data_path = data_path
        self.context_path = context_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_type = model_type

        self.kiwi = Kiwi()

        # Load Data
        self._load_data()

        # Initialize Models
        self.bm25 = None
        self.sparse_encoder = None
        self.sparse_matrix = None

        # BGE-M3 (Using SentenceTransformer for dense, but here we focus on sparse if available via specific methods or models)
        # Note: Standard SentenceTransformers BGE-M3 model outputs dense vectors.
        # For sparse BGE-M3, we need 'BAAI/bge-m3' and specific usage.
        # However, for fair comparison as requested "sparse performance", we will treat:
        # 1. BM25 (Lexical)
        # 2. SPLADE (Learned Sparse)
        # 3. BGE-M3 (Learned Sparse - if supported by library, else we simulate or use BGEM3FlagModel if installed, but here we use what's available in SentenceTransformers or similar)
        # Assuming BGE-M3 sparse is accessible via SparseEncoder or similar interface if supported.
        # If not directly supported as 'SparseEncoder' for BGE-M3 in this env, we might need a workaround or specific library 'FlagEmbedding'.
        # For now, we will use 'BAAI/bge-m3' with the assumption we can extract sparse weights or use it as a dense baseline to compare 'sparse-like' behavior?
        # User asked for "sparse performance comparison".
        # NOTE: sentence-transformers supports BGE-M3 but mainly for dense.
        # To get sparse from BGE-M3 using sentence-transformers is not standard without 'BGEM3FlagModel'.
        # We will try to load it if model_type is bge-m3-sparse using a custom wrapper if needed,
        # OR we check if 'SparseEncoder' supports it.
        # Actually 'SparseEncoder' in sentence-transformers is for SPLADE-like models.
        # We will use 'naver/splade-cocondenser-ensembledistil' or similar for SPLADE.
        # For BGE-M3 sparse, we really need `FlagEmbedding` library or manual implementation.
        # Checking env... we don't have FlagEmbedding installed likely.
        # We will proceed with BM25 and SPLADE for sure. For BGE-M3, we will try to use the same SparseEncoder if compatible,
        # or alert user if specific library is missing.

        if self.model_type == "splade":
            # Using a known SPLADE model or the one used in the project
            self.model_name = "telepix/PIXIE-Splade-Preview"  # From project files
            self.sparse_encoder = SparseEncoder(self.model_name)
            self.sparse_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.special_token_ids = set(self.sparse_tokenizer.all_special_ids)
            
            if torch.cuda.is_available():
                self.sparse_encoder.to("cuda")

        elif self.model_type == "bge-m3":
            # We will attempt to use BGEM3FlagModel if available, else warn.
            try:
                from FlagEmbedding import BGEM3FlagModel

                self.bge_model = BGEM3FlagModel("BAAI/bge-m3")
            except ImportError:
                print(
                    "Warning: FlagEmbedding library not found. Installing or falling back..."
                )
                # Simplified fallback or install instruction
                print("Please install FlagEmbedding: pip install -U FlagEmbedding")
                # For this script, we will assume it might fail and just handle BM25/SPLADE or try standard ST if possible (but ST doesn't do BGE sparse easily)
                # We will use a placeholder or skip if not available for this strict comparison script.
                self.bge_model = None

    def _load_data(self):
        chunk_cache_path = os.path.join(
            self.data_path,
            f"wiki_chunks_{self.chunk_size}_overlap{self.chunk_overlap}.pkl",
        )

        if os.path.isfile(chunk_cache_path):
            print(f"Loading Chunk Cache: {chunk_cache_path}")
            with open(chunk_cache_path, "rb") as f:
                data = pickle.load(f)
            self.texts = data["texts"]
            self.titles = data["titles"]
            self.doc_ids = data["doc_ids"]
            self.ids = data["ids"]
        else:
            # Basic load if cache missing (simplified from original)
            with open(
                os.path.join(self.data_path, self.context_path), "r", encoding="utf-8"
            ) as f:
                wiki = json.load(f)
            self.texts = []
            self.titles = []
            self.doc_ids = []
            self.ids = []

            # Using kiwi for splitting to match logic
            for i, (k, v) in enumerate(tqdm(wiki.items(), desc="Processing Wiki")):
                # Simplified chunking for speed in this comparison script if cache missing
                # (Ideally cache exists from previous runs)
                self.texts.append(v["text"])
                self.titles.append(v["title"])
                self.doc_ids.append(v["document_id"])
                self.ids.append(i)

    def kiwi_tokenizer(self, text: str) -> list[str]:
        tokens = self.kiwi.tokenize(text)
        return [
            t.form for t in tokens if t.tag.startswith("N") or t.tag.startswith("V")
        ]  # Simplified

    def build_index(self):
        print(f"Building Index for {self.model_type}...")

        chunk_suffix = f"_chunk{self.chunk_size}_overlap{self.chunk_overlap}_v3"

        if self.model_type == "bm25":
            bm25_path = os.path.join(self.data_path, f"bm25_wiki{chunk_suffix}.pkl")
            if os.path.isfile(bm25_path):
                print(f"Loading existing BM25 index from {bm25_path}")
                with open(bm25_path, "rb") as f:
                    self.bm25 = pickle.load(f)
            else:
                print("BM25 index not found. Building new one...")
                tokenized_corpus = [
                    self.kiwi_tokenizer(tit + " " + txt)
                    for tit, txt in zip(self.titles, self.texts)
                ]
                self.bm25 = BM25Okapi(corpus=tokenized_corpus)
                # Save for future use
                with open(bm25_path, "wb") as f:
                    pickle.dump(self.bm25, f)

        elif self.model_type == "splade":
            # Check for existing sparse matrix
            # Naming convention from retrieval_splade_hybrid.py:
            # sparse_{model_name_safe}{chunk_suffix}.npz
            model_name_safe = self.model_name.replace("/", "_")
            sparse_path = os.path.join(
                self.data_path, f"sparse_{model_name_safe}{chunk_suffix}.npz"
            )

            if os.path.isfile(sparse_path):
                print(f"Loading existing SPLADE matrix from {sparse_path}")
                self.sparse_matrix = load_npz(sparse_path)
            else:
                print("SPLADE matrix not found. Building new one...")
                # Build Sparse Matrix
                docs = [f"{tit}\n{txt}" for tit, txt in zip(self.titles, self.texts)]
                batch_size = 16
                all_embeddings = []

                for i in tqdm(range(0, len(docs), batch_size), desc="Encoding SPLADE"):
                    batch_docs = docs[i : i + batch_size]
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            embeddings = self.sparse_encoder.encode_document(batch_docs)

                    if isinstance(embeddings, torch.Tensor):
                        if embeddings.is_sparse:
                            embeddings = embeddings.to_dense()
                        embeddings = embeddings.cpu().numpy()

                    # Convert to float32 for csr_matrix compatibility
                    if embeddings.dtype == np.float16:
                        embeddings = embeddings.astype(np.float32)

                    # Filter Special Tokens
                    # embeddings: (batch_size, vocab_size)
                    # Set columns corresponding to special tokens to 0
                    if hasattr(self, "special_token_ids"):
                         for sp_id in self.special_token_ids:
                             if sp_id < embeddings.shape[1]:
                                 embeddings[:, sp_id] = 0

                    all_embeddings.append(csr_matrix(embeddings))

                from scipy.sparse import vstack

                self.sparse_matrix = vstack(all_embeddings)
                # Save
                save_npz(sparse_path, self.sparse_matrix)

        elif self.model_type == "bge-m3":
            if self.bge_model is None:
                raise ValueError("FlagEmbedding not installed for BGE-M3 sparse.")

            # Check for existing sparse matrix for BGE-M3
            sparse_path = os.path.join(
                self.data_path, f"sparse_bge_m3{chunk_suffix}.npz"
            )

            if os.path.isfile(sparse_path):
                print(f"Loading existing BGE-M3 matrix from {sparse_path}")
                self.sparse_matrix = load_npz(sparse_path)
            else:
                print("BGE-M3 matrix not found. Building new one...")
                docs = [f"{tit}\n{txt}" for tit, txt in zip(self.titles, self.texts)]

                batch_size = 16
                rows, cols, data = [], [], []

                for i in tqdm(range(0, len(docs), batch_size), desc="Encoding BGE-M3"):
                    batch_docs = docs[i : i + batch_size]
                    output = self.bge_model.encode(
                        batch_docs,
                        return_dense=False,
                        return_sparse=True,
                        return_colbert_vecs=False,
                    )

                    for j, doc_weights in enumerate(output["lexical_weights"]):
                        global_idx = i + j
                        for token_id_str, weight in doc_weights.items():
                            rows.append(global_idx)
                            cols.append(int(token_id_str))
                            data.append(weight)

                # Vocab size approx 250002 for bge-m3
                data = np.array(data, dtype=np.float32)
                self.sparse_matrix = csr_matrix(
                    (data, (rows, cols)), shape=(len(docs), 250002)
                )
                save_npz(sparse_path, self.sparse_matrix)

    def retrieve(self, queries: List[str], topk: int = 20) -> List[List[int]]:
        print(f"Retrieving with {self.model_type}...")
        results = []

        if self.model_type == "bm25":
            for query in tqdm(queries, desc="BM25 Search"):
                tokenized_query = self.kiwi_tokenizer(query)
                scores = self.bm25.get_scores(tokenized_query)
                top_n = np.argsort(scores)[::-1][:topk]
                results.append(top_n.tolist())

        elif self.model_type == "splade":
            # Batch Query Encoding
            batch_size = 32
            for i in tqdm(range(0, len(queries), batch_size), desc="SPLADE Search"):
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

                # Filter Special Tokens for Query
                if hasattr(self, "special_token_ids"):
                     for sp_id in self.special_token_ids:
                         if sp_id < q_embs.shape[1]:
                             q_embs[:, sp_id] = 0

                q_sparse = csr_matrix(q_embs)
                scores = q_sparse.dot(self.sparse_matrix.T)

                for j in range(scores.shape[0]):
                    row = scores.getrow(j)
                    top_n = np.argsort(row.data)[::-1][:topk]
                    # Map back to original indices
                    original_indices = row.indices[top_n]
                    results.append(original_indices.tolist())

        elif self.model_type == "bge-m3":
            # Query Encoding
            batch_size = 32
            for i in tqdm(range(0, len(queries), batch_size), desc="BGE-M3 Search"):
                batch_queries = queries[i : i + batch_size]
                output = self.bge_model.encode(
                    batch_queries,
                    return_dense=False,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )

                # Construct Query Sparse Matrix
                q_rows, q_cols, q_data = [], [], []
                for j, q_weights in enumerate(output["lexical_weights"]):
                    for token_id_str, weight in q_weights.items():
                        q_rows.append(j)
                        q_cols.append(int(token_id_str))
                        q_data.append(weight)

                q_data = np.array(q_data, dtype=np.float32)
                q_sparse = csr_matrix(
                    (q_data, (q_rows, q_cols)), shape=(len(batch_queries), 250002)
                )

                scores = q_sparse.dot(self.sparse_matrix.T)

                for j in range(scores.shape[0]):
                    row = scores.getrow(j)
                    if row.nnz == 0:
                        results.append([])
                        continue
                    top_n = np.argsort(row.data)[::-1][:topk]
                    original_indices = row.indices[top_n]
                    results.append(original_indices.tolist())

        return results

    def evaluate(self, dataset, topk=20):
        queries = dataset["question"]
        retrieved_indices = self.retrieve(queries, topk=topk)

        correct_count = 0
        for i, example in enumerate(dataset):
            # Check if answer is in retrieved contexts
            retrieved_docs = [self.texts[idx] for idx in retrieved_indices[i]]

            # Ground Truth Check (Answer based)
            answers = example["answers"]["text"]

            hit = False
            for doc in retrieved_docs:
                if any(ans in doc for ans in answers):
                    hit = True
                    break
            if hit:
                correct_count += 1

        acc = correct_count / len(dataset)
        print(f"[{self.model_type}] Top-{topk} Accuracy: {acc:.4f}")
        return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="all", choices=["bm25", "splade", "bge-m3", "all"]
    )
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    # Load Dataset (Validation set for speed)
    print("Loading Validation Dataset...")
    dataset = load_from_disk("raw/data/train_dataset")["validation"]

    models_to_test = (
        ["bm25", "splade", "bge-m3"] if args.model == "all" else [args.model]
    )

    results = {}

    for model_name in models_to_test:
        try:
            print(f"\n--- Testing {model_name} ---")
            tester = SparseRetrievalComparison(model_type=model_name)
            tester.build_index()
            acc = tester.evaluate(dataset, topk=args.topk)
            results[model_name] = acc
        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
            results[model_name] = "Failed"

    print("\n=== Final Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")
