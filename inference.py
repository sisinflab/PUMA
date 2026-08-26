import json
import torch
import numpy as np
from typing import Optional
from collections import defaultdict

from sae import SparseAutoencoder, SAEConfig
from train_sae import load_checkpoint


class SparseEncoder:
    """Encode dense embeddings as sparse feature-weight dictionaries."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 256,
        override_k: Optional[int] = None,
    ):
        self.device = device
        self.batch_size = batch_size
        self.sae = load_checkpoint(checkpoint_path, device)
        self.sae.eval()
        self.checkpoint_k = int(self.sae.current_k)
        self.set_active_k(override_k)

        print(
            f"Loaded sparse encoder ({type(self.sae).__name__}): "
            f"{self.sae.config.input_dim}d → "
            f"{self.sae.config.dict_size}d, checkpoint_k={self.checkpoint_k}, "
            f"active_k={self.active_k}"
        )

    @property
    def active_k(self) -> int:
        return int(self.sae.current_k)

    def set_active_k(self, override_k: Optional[int] = None) -> int:
        if override_k is None:
            self.sae.current_k = self.checkpoint_k
        else:
            self.sae.current_k = int(override_k)
            if self.sae.current_k != int(override_k):
                print(f"Requested sparse k={override_k} was clamped to k={self.sae.current_k}")
        return self.active_k
    
    @torch.no_grad()
    def encode(
        self, embeddings: np.ndarray, batch_size: Optional[int] = None
    ) -> list[dict[int, float]]:
        """
        Encode dense embeddings into sparse vectors.
        
        Args:
            embeddings: (N, d) numpy array of dense embeddings
        
        Returns:
            List of dicts: [{feature_id: weight}, ...]
            Each dict has exactly k entries.
        """
        batch_size = batch_size or self.batch_size
        sparse_docs = []
        for i in range(0, len(embeddings), batch_size):
            batch_np = np.array(embeddings[i : i + batch_size], dtype=np.float32, copy=True)
            batch = torch.from_numpy(batch_np).to(self.device)
            indices, values = self.sae.get_active_features(batch)
            
            for j in range(len(batch)):
                doc = {}
                for idx, val in zip(
                    indices[j].cpu().tolist(),
                    values[j].cpu().tolist()
                ):
                    if val > 0:
                        doc[int(idx)] = float(val)
                sparse_docs.append(doc)
        
        return sparse_docs
    
    @torch.no_grad()
    def encode_single(
        self, embedding: np.ndarray
    ) -> dict[int, float]:
        """Encode a single dense embedding to sparse."""
        return self.encode(embedding[np.newaxis, :])[0]


class SimpleInvertedIndex:

    def __init__(self):
        self.posting_lists: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self.doc_norms: dict[int, float] = {}
        self.num_docs = 0
        self.doc_lengths: dict[int, int] = {}
    
    def add_document(self, doc_id: int, sparse_vec: dict[int, float]):
        """Add a document to the index."""
        norm = 0.0
        for feature_id, weight in sparse_vec.items():
            self.posting_lists[feature_id].append((doc_id, weight))
            norm += weight ** 2
        
        self.doc_norms[doc_id] = np.sqrt(norm)
        self.doc_lengths[doc_id] = len(sparse_vec)
        self.num_docs += 1
    
    def add_batch(self, sparse_vecs: list[dict[int, float]], start_id: int = 0):
        """Add multiple documents."""
        for i, vec in enumerate(sparse_vecs):
            self.add_document(start_id + i, vec)
    
    def search(
        self,
        query_vec: dict[int, float],
        top_k: int = 10,
        scoring: str = "dot",
    ) -> list[tuple[int, float]]:
        """
        Retrieve top-k documents for a sparse query.
        
        Scoring options:
        - "dot":    sum of (query_weight * doc_weight) for matching features
        - "cosine": dot product normalized by query and doc norms
        """
        if scoring not in {"dot", "cosine"}:
            raise ValueError(f"Unsupported scoring method: {scoring}")

        scores: dict[int, float] = defaultdict(float)

        for feature_id, q_weight in query_vec.items():
            if feature_id not in self.posting_lists:
                continue

            postings = self.posting_lists[feature_id]

            for doc_id, d_weight in postings:
                if scoring == "dot":
                    scores[doc_id] += q_weight * d_weight
                elif scoring == "cosine":
                    scores[doc_id] += q_weight * d_weight

        if scoring == "cosine":
            q_norm = np.sqrt(sum(w**2 for w in query_vec.values()))
            for doc_id in scores:
                scores[doc_id] /= (q_norm * self.doc_norms[doc_id] + 1e-8)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    
    def stats(self) -> dict:
        """Index statistics."""
        num_features_used = len(self.posting_lists)
        avg_posting_length = np.mean([
            len(pl) for pl in self.posting_lists.values()
        ]) if self.posting_lists else 0
        
        return {
            "num_docs": self.num_docs,
            "num_features_used": num_features_used,
            "avg_posting_length": f"{avg_posting_length:.1f}",
            "avg_doc_length": f"{np.mean(list(self.doc_lengths.values())):.1f}",
        }
    
    def save(self, path: str):
        """Save index to disk."""
        data = {
            "posting_lists": {
                str(k): v for k, v in self.posting_lists.items()
            },
            "doc_norms": self.doc_norms,
            "doc_lengths": self.doc_lengths,
            "num_docs": self.num_docs,
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load index from disk."""
        with open(path) as f:
            data = json.load(f)
        
        self.posting_lists = defaultdict(list)
        for k, v in data["posting_lists"].items():
            self.posting_lists[int(k)] = [(d, w) for d, w in v]
        
        self.doc_norms = {int(k): v for k, v in data["doc_norms"].items()}
        self.doc_lengths = {int(k): v for k, v in data["doc_lengths"].items()}
        self.num_docs = data["num_docs"]

if __name__ == "__main__":
    print("=== Sparse Multimodal Retrieval Demo ===\n")

    config = SAEConfig(input_dim=1536, dict_size=24576, k_initial=32, k_final=32)
    sae = SparseAutoencoder(config)
    sae.eval()

    print("Creating synthetic corpus...")
    np.random.seed(42)
    corpus_embs = np.random.randn(1000, 1536).astype(np.float32)
    corpus_embs = corpus_embs / np.linalg.norm(corpus_embs, axis=-1, keepdims=True)

    print("Encoding corpus to sparse vectors...")
    with torch.no_grad():
        x = torch.from_numpy(corpus_embs)
        all_indices, all_values = sae.get_active_features(x)
    
    sparse_corpus = []
    for i in range(len(corpus_embs)):
        doc = {}
        for idx, val in zip(
            all_indices[i].cpu().tolist(),
            all_values[i].cpu().tolist()
        ):
            if val > 0:
                doc[idx] = val
        sparse_corpus.append(doc)
    
    print("Building inverted index...")
    index = SimpleInvertedIndex()
    index.add_batch(sparse_corpus)
    
    stats = index.stats()
    print(f"\nIndex stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\nSearching with doc_0 as query...")
    query = sparse_corpus[0]
    results = index.search(query, top_k=5, scoring="dot")
    
    print(f"\nTop-5 results:")
    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"  {rank}. doc_{doc_id}  score={score:.4f}")
    
    active_counts = [len(doc) for doc in sparse_corpus]
    print(f"\nSparsity analysis:")
    print(f"  Active dims per doc: {np.mean(active_counts):.1f} ± {np.std(active_counts):.1f}")
    print(f"  Sparsity: {1 - np.mean(active_counts) / config.dict_size:.4%}")
    print(f"  Dictionary utilization: {stats['num_features_used']} / {config.dict_size}")
