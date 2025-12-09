import os
import argparse
import time
import pickle  # Added
from tqdm import tqdm
import torch
import numpy as np
import binascii
from collections import Counter
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer, SparseEncoder
from transformers import AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    PointStruct,
)
from retrieval_hybrid import HybridRetrieval
from rank_bm25 import BM25Okapi  # Added

# Qdrant Configuration
QDRANT_HOST = "lori2mai11ya.asuscomm.com"
QDRANT_PORT = 6333


def compute_bm25_weight(tf, doc_len, avgdl, idf, k1=1.2, b=0.75):
    # BM25의 (IDF * TF component) 계산
    # 방법 1: Doc Vector에 IDF를 미리 곱해서 저장.
    # 이렇게 하면 쿼리 시에는 단순히 TF(보통 1)만 보내도 BM25 점수가 근사됨.
    numerator = tf * (k1 + 1)
    denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
    return idf * (numerator / denominator)


def hash_token(token):
    # unsigned 32-bit int로 변환
    return binascii.crc32(token.encode("utf-8")) & 0xFFFFFFFF


def build_qdrant_hybrid_index(
    data_path,
    context_path,
    chunk_size,
    chunk_overlap,
    dense_model_name,
    sparse_model_name,
    collection_name=None,
    api_key=None,
):
    # 1. Initialize HybridRetrieval for data loading & BM25
    print("Initializing HybridRetrieval for data loading & BM25...")
    retriever = HybridRetrieval(
        data_path=data_path,
        context_path=context_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_reranker=False,
    )

    # BM25 객체 로드 (FAISS 구축 방지)
    if retriever.bm25 is None:
        print("Loading BM25 index only...")
        chunk_suffix = f"_chunk{chunk_size}_overlap{chunk_overlap}_v3"
        bm25_path = os.path.join(data_path, f"bm25_wiki{chunk_suffix}.pkl")

        if os.path.isfile(bm25_path):
            print(f"Loading existing BM25 index from {bm25_path}")
            with open(bm25_path, "rb") as f:
                retriever.bm25 = pickle.load(f)
        else:
            print("BM25 index not found. Building new one...")
            tokenized_corpus = [
                retriever.kiwi_tokenizer(tit + " " + txt)
                for tit, txt in zip(retriever.titles, retriever.texts)
            ]
            retriever.bm25 = BM25Okapi(corpus=tokenized_corpus)
            # Save for future use
            with open(bm25_path, "wb") as f:
                pickle.dump(retriever.bm25, f)
            print("BM25 index built and saved.")

    if not hasattr(retriever, "texts") or not retriever.texts:
        print("Error: No texts found in retriever.")
        return

    # BM25 Stats
    avgdl = retriever.bm25.avgdl
    print(f"BM25 Average Document Length: {avgdl}")

    print(f"Total documents to index: {len(retriever.texts)}")

    # 2. Load Models
    print(f"Loading Dense Model: {dense_model_name}")
    dense_encoder = SentenceTransformer(dense_model_name)

    print(f"Loading Sparse (SPLADE) Model & Tokenizer: {sparse_model_name}")
    sparse_encoder = SparseEncoder(sparse_model_name)
    sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_model_name)

    # Get Special Token IDs
    special_token_ids = set(sparse_tokenizer.all_special_ids)
    print(f"Special tokens to filter: {special_token_ids}")

    if torch.cuda.is_available():
        sparse_encoder.to("cuda")

        # 3. Connect to Qdrant
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        client = QdrantClient(
            host=QDRANT_HOST, 
            port=QDRANT_PORT, 
            api_key=api_key, 
            https=False,
            timeout=300
        )
    
        # 4. Define Collection
        if collection_name is None:
            safe_dense = dense_model_name.replace("/", "_").replace("-", "_")
            safe_sparse = sparse_model_name.replace("/", "_").replace("-", "_")
            collection_name = f"wiki_hybrid_{safe_dense}_splade_bm25"
    
        print(f"Target Collection Name: {collection_name}")
    
        # Get dense dimension
        sample_dense = dense_encoder.encode(["test"], normalize_embeddings=True)
        dense_dim = sample_dense.shape[1]
        print(f"Dense Vector dimension: {dense_dim}")
    
        # Recreate Collection with Hybrid Config (Dense + SPLADE + BM25 + Custom Sparse)
        print(f"Creating/Recreating collection '{collection_name}' with Hybrid Config...")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "splade": SparseVectorParams(),
                "bm25": SparseVectorParams(),
                "custom_sparse": SparseVectorParams(),  # Placeholder for future custom sparse vectors
            },
        )
    
        # 5. Batch Encoding & Uploading
        batch_size = 8  # Adjust based on GPU VRAM
        total_docs = len(retriever.texts)
    
        print("Starting Hybrid Indexing (Dense + SPLADE + BM25)...")
        for i in tqdm(range(0, total_docs, batch_size), desc="Indexing to Qdrant"):
            batch_texts = retriever.texts[i : i + batch_size]
            batch_titles = retriever.titles[i : i + batch_size]
            batch_doc_ids = retriever.doc_ids[i : i + batch_size]
            batch_ids = retriever.ids[i : i + batch_size]
    
            # Prepare text for embedding (Title + Text)
            docs_for_embed = [f"{t}\n{txt}" for t, txt in zip(batch_titles, batch_texts)]
    
            # --- Dense Encoding ---
            dense_embeddings = dense_encoder.encode(
                docs_for_embed,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
    
            # --- Sparse (SPLADE) Encoding ---
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    sparse_output = sparse_encoder.encode_document(docs_for_embed)
    
            # Convert SPLADE to CSR
            if isinstance(sparse_output, torch.Tensor):
                if sparse_output.is_sparse:
                    sparse_output = sparse_output.to_dense()
                sparse_output = sparse_output.cpu().numpy()
    
            if sparse_output.dtype == np.float16:
                sparse_output = sparse_output.astype(np.float32)
    
            splade_csr = csr_matrix(sparse_output)
    
            # --- Prepare Points ---
            points = []
            for j in range(len(batch_texts)):
                idx = batch_ids[j]
                doc_text_combined = docs_for_embed[j]
    
                # 1. Dense Vector
                dense_vec = dense_embeddings[j].tolist()
    
                # 2. SPLADE Vector (with Special Token Filtering)
                splade_row = splade_csr.getrow(j)
                splade_indices = splade_row.indices.tolist()
                splade_values = splade_row.data.tolist()
                
                # Filter special tokens
                filtered_indices = []
                filtered_values = []
                for sp_idx, sp_val in zip(splade_indices, splade_values):
                    if sp_idx not in special_token_ids:
                        filtered_indices.append(sp_idx)
                        filtered_values.append(sp_val)
                
                splade_indices = filtered_indices
                splade_values = filtered_values
    
                # 3. BM25 Vector (On-the-fly calculation)
                # 토큰화 (Kiwi)
                tokens = retriever.kiwi_tokenizer(doc_text_combined)
                doc_len = len(tokens)
                token_counts = Counter(tokens)
    
                bm25_indices = []
                bm25_values = []
    
                for token, count in token_counts.items():
                    # Hash Token -> Index
                    token_idx = hash_token(token)
                    
                    # Get IDF
                    # rank_bm25 라이브러리의 idf 딕셔너리 사용
                    idf = retriever.bm25.idf.get(token, 0.0)
                    
                    # Calculate Weight (IDF * TF component)
                    weight = compute_bm25_weight(count, doc_len, avgdl, idf)
    
                    bm25_indices.append(token_idx)
                    bm25_values.append(weight)
    
                points.append(
                    PointStruct(
                        id=idx,
                        vector={
                            "dense": dense_vec,
                            "splade": SparseVector(
                                indices=splade_indices, values=splade_values
                            ),
                            "bm25": SparseVector(indices=bm25_indices, values=bm25_values),
                            "custom_sparse": SparseVector(indices=[], values=[]),
                        },
                        payload={
                            "title": batch_titles[j],
                            "text": batch_texts[j],
                            "doc_id": batch_doc_ids[j],
                            "chunk_id": idx,
                        },
                    )
                )
    
            # Upload
            client.upsert(collection_name=collection_name, points=points)
            
            # Explicit Garbage Collection
            del dense_embeddings
            del sparse_output
            del splade_csr
            torch.cuda.empty_cache()
    print("Indexing complete!")
    print(
        f"Total vectors in collection '{collection_name}': {client.count(collection_name=collection_name).count}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Qdrant Hybrid Index (Dense + SPLADE)"
    )
    parser.add_argument(
        "--data_path", type=str, default="raw/data", help="Path to data directory"
    )
    parser.add_argument(
        "--context_path",
        type=str,
        default="wikipedia_documents.json",
        help="Context file",
    )
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    parser.add_argument(
        "--dense_model_name",
        type=str,
        default="telepix/PIXIE-Spell-Preview-1.7B",
        help="Dense embedding model name",
    )
    parser.add_argument(
        "--sparse_model_name",
        type=str,
        default="telepix/PIXIE-Splade-Preview",
        help="Sparse (SPLADE) embedding model name",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="hybird_collection_v1",
        help="Qdrant collection name (optional)",
    )
    parser.add_argument(
        "--api_key", type=str, default="boostcamp", help="Qdrant API Key"
    )

    args = parser.parse_args()

    build_qdrant_hybrid_index(
        data_path=args.data_path,
        context_path=args.context_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        dense_model_name=args.dense_model_name,
        sparse_model_name=args.sparse_model_name,
        collection_name=args.collection_name,
        api_key=args.api_key,
    )
