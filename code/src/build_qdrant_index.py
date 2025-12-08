import os
import argparse
import time
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from retrieval_hybrid import HybridRetrieval

# Qdrant Configuration
QDRANT_HOST = "lori2mai11ya.asuscomm.com"
QDRANT_PORT = 6333

def build_qdrant_index(data_path, context_path, chunk_size, chunk_overlap, model_name, collection_name=None, api_key=None):
    # 1. Initialize HybridRetrieval to load and chunk data
    print("Initializing HybridRetrieval for data loading...")
    retriever = HybridRetrieval(
        data_path=data_path,
        context_path=context_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_reranker=False # Not needed for indexing
    )
    
    # Ensure data is loaded
    if not hasattr(retriever, 'texts') or not retriever.texts:
        pass

    print(f"Total documents to index: {len(retriever.texts)}")

    # 2. Load Embedding Model
    print(f"Loading embedding model: {model_name}")
    encoder = SentenceTransformer(model_name)
    
    # 3. Connect to Qdrant
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=api_key)

    # 4. Define Collection
    # Set collection name if not provided
    if collection_name is None:
        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        collection_name = f"wiki_{safe_model_name}_chunk{chunk_size}"
    
    print(f"Target Collection Name: {collection_name}")

    # Get embedding dimension from the model
    sample_embedding = encoder.encode(["test"], normalize_embeddings=True)
    vector_size = sample_embedding.shape[1]
    print(f"Vector dimension: {vector_size}")

    if not client.collection_exists(collection_name=collection_name):
        print(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{collection_name}' already exists.")

    # 5. Batch Encoding & Uploading
    batch_size = 64  # Adjust based on GPU memory
    total_docs = len(retriever.texts)
    
    print("Starting indexing...")
    for i in tqdm(range(0, total_docs, batch_size), desc="Indexing to Qdrant"):
        batch_texts = retriever.texts[i : i + batch_size]
        batch_titles = retriever.titles[i : i + batch_size]
        batch_doc_ids = retriever.doc_ids[i : i + batch_size]
        batch_ids = retriever.ids[i : i + batch_size]

        # Prepare text for embedding (Title + Text)
        docs_for_embed = [f"{t}\n{txt}" for t, txt in zip(batch_titles, batch_texts)]

        # Generate Embeddings using the specified encoder
        embeddings = encoder.encode(
            docs_for_embed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Prepare Points
        points = []
        for j, embedding in enumerate(embeddings):
            # Global index
            idx = batch_ids[j] 
            
            points.append(
                PointStruct(
                    id=idx, 
                    vector=embedding.tolist(),
                    payload={
                        "title": batch_titles[j],
                        "text": batch_texts[j],
                        "doc_id": batch_doc_ids[j],
                        "chunk_id": idx
                    }
                )
            )

        # Upload
        client.upsert(
            collection_name=collection_name,
            points=points
        )

    print("Indexing complete!")
    print(f"Total vectors in collection '{collection_name}': {client.count(collection_name=collection_name).count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Qdrant Index with specific model")
    parser.add_argument("--data_path", type=str, default="raw/data", help="Path to data directory")
    parser.add_argument("--context_path", type=str, default="wikipedia_documents.json", help="Context file")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    parser.add_argument("--model_name", type=str, default="telepix/PIXIE-Rune-Preview", help="HuggingFace model name")
    parser.add_argument("--collection_name", type=str, default=None, help="Qdrant collection name (optional)")
    parser.add_argument("--api_key", type=str, default=None, help="Qdrant API Key")
    
    args = parser.parse_args()

    build_qdrant_index(
        data_path=args.data_path,
        context_path=args.context_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model_name,
        collection_name=args.collection_name,
        api_key=args.api_key
    )
