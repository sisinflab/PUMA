"""
Evaluation: Measure sparse retrieval quality and efficiency.

Compares:
1. Dense retrieval (original VLM embeddings, exact search)
2. Sparse retrieval (SAE sparse codes, inverted index)
3. Reconstruction quality (how well SAE preserves original embeddings)

Metrics:
- Recall@K (1, 5, 10, 100)
- nDCG@10
- MRR
- Sparsity statistics
- Cross-modal alignment quality
"""

import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

from sae import SparseAutoencoder, SAEConfig
from train_sae import load_checkpoint
from inference import SimpleInvertedIndex


def compute_sparse_index_stats(sparse_vecs: list[dict[int, float]]) -> dict:
    posting_lengths: dict[int, int] = defaultdict(int)
    doc_lengths = []
    for vec in sparse_vecs:
        doc_lengths.append(len(vec))
        for feature_id in vec:
            posting_lengths[int(feature_id)] += 1

    avg_posting_length = np.mean(list(posting_lengths.values())) if posting_lengths else 0.0
    avg_doc_length = np.mean(doc_lengths) if doc_lengths else 0.0
    return {
        "num_docs": len(sparse_vecs),
        "num_features_used": len(posting_lengths),
        "avg_posting_length": f"{avg_posting_length:.1f}",
        "avg_doc_length": f"{avg_doc_length:.1f}",
    }


def build_torch_sparse_doc_matrix(
    sparse_docs: list[dict[int, float]],
    *,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_ptr = [0]
    col_indices = []
    values = []
    norms = []

    for vec in sparse_docs:
        norm_sq = 0.0
        for feature_id, weight in vec.items():
            col_indices.append(int(feature_id))
            values.append(float(weight))
            norm_sq += float(weight) * float(weight)
        row_ptr.append(len(col_indices))
        norms.append(norm_sq**0.5)

    crow_indices = torch.tensor(row_ptr, dtype=torch.int64, device=device)
    col_tensor = torch.tensor(col_indices, dtype=torch.int64, device=device)
    value_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    doc_matrix = torch.sparse_csr_tensor(
        crow_indices,
        col_tensor,
        value_tensor,
        size=(len(sparse_docs), vocab_size),
        device=device,
    )
    doc_norms = torch.tensor(norms, dtype=torch.float32, device=device)
    return doc_matrix, doc_norms


def sparse_query_batch_to_dense(
    sparse_queries: list[dict[int, float]],
    *,
    start: int,
    end: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = torch.zeros((end - start, vocab_size), dtype=torch.float32, device=device)
    norms = torch.zeros((end - start,), dtype=torch.float32, device=device)
    for row_idx, vec in enumerate(sparse_queries[start:end]):
        if not vec:
            continue
        cols = torch.tensor(list(vec.keys()), dtype=torch.int64, device=device)
        vals = torch.tensor(list(vec.values()), dtype=torch.float32, device=device)
        batch[row_idx, cols] = vals
        norms[row_idx] = torch.linalg.vector_norm(vals)
    return batch, norms


def recall_at_k(
    rankings: list[list[int]],  # per-query ranked doc IDs
    relevant: list[set[int]],   # per-query relevant doc IDs
    k: int,
) -> float:
    """Recall@K: fraction of relevant docs found in top-K."""
    hits = 0
    total = 0
    for ranked, rel in zip(rankings, relevant):
        top_k = set(ranked[:k])
        hits += len(top_k & rel)
        total += len(rel)
    return hits / max(total, 1)


def mrr(
    rankings: list[list[int]],
    relevant: list[set[int]],
) -> float:
    """Mean Reciprocal Rank."""
    rr_sum = 0
    for ranked, rel in zip(rankings, relevant):
        for i, doc_id in enumerate(ranked):
            if doc_id in rel:
                rr_sum += 1.0 / (i + 1)
                break
    return rr_sum / len(rankings)


def ndcg_at_k(
    rankings: list[list[int]],
    relevant: list[set[int]],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    ndcg_sum = 0
    for ranked, rel in zip(rankings, relevant):
        dcg = 0
        for i, doc_id in enumerate(ranked[:k]):
            if doc_id in rel:
                dcg += 1.0 / np.log2(i + 2)
        
        # Ideal DCG: all relevant docs at top
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
        
        ndcg_sum += dcg / max(ideal_dcg, 1e-8)
    
    return ndcg_sum / len(rankings)


@torch.no_grad()
def evaluate_reconstruction(
    sae: SparseAutoencoder,
    embeddings: np.ndarray,
    batch_size: int = 1024,
) -> dict:
    """
    Measure how well the SAE reconstructs original embeddings.
    
    Good reconstruction is necessary but not sufficient for good retrieval.
    """
    device = next(sae.parameters()).device
    
    cos_sims = []
    l2_errors = []
    
    for i in range(0, len(embeddings), batch_size):
        batch = torch.from_numpy(embeddings[i:i+batch_size]).to(device)
        out = sae(batch)
        
        cos_sim = F.cosine_similarity(out["x"], out["x_hat"], dim=-1)
        l2_err = (out["x"] - out["x_hat"]).norm(dim=-1)
        
        cos_sims.append(cos_sim.cpu().numpy())
        l2_errors.append(l2_err.cpu().numpy())
    
    cos_sims = np.concatenate(cos_sims)
    l2_errors = np.concatenate(l2_errors)
    
    return {
        "cosine_similarity_mean": float(cos_sims.mean()),
        "cosine_similarity_std": float(cos_sims.std()),
        "l2_error_mean": float(l2_errors.mean()),
        "l2_error_std": float(l2_errors.std()),
    }


@torch.no_grad()
def evaluate_cross_modal_alignment(
    sae: SparseAutoencoder,
    img_embeddings: np.ndarray,
    txt_embeddings: np.ndarray,
    batch_size: int = 1024,
) -> dict:
    """
    Measure whether paired image-text activate similar sparse features.
    
    This is the critical test for cross-modal retrieval:
    if an image and its caption produce very different sparse codes,
    cross-modal retrieval via inverted index will fail.
    """
    device = next(sae.parameters()).device
    
    overlaps = []
    sparse_cos_sims = []
    dense_cos_sims = []
    
    n = min(len(img_embeddings), len(txt_embeddings))
    
    for i in range(0, n, batch_size):
        img_batch = torch.from_numpy(img_embeddings[i:i+batch_size]).to(device)
        txt_batch = torch.from_numpy(txt_embeddings[i:i+batch_size]).to(device)
        
        out_img = sae(img_batch)
        out_txt = sae(txt_batch)
        
        active_img = (out_img["z"] != 0).float()
        active_txt = (out_txt["z"] != 0).float()
        intersection = (active_img * active_txt).sum(dim=-1)
        union = ((active_img + active_txt) > 0).float().sum(dim=-1)
        jaccard = intersection / (union + 1e-8)
        overlaps.append(jaccard.cpu().numpy())
        
        sparse_cos = F.cosine_similarity(out_img["z"], out_txt["z"], dim=-1)
        sparse_cos_sims.append(sparse_cos.cpu().numpy())
        
        dense_cos = F.cosine_similarity(img_batch, txt_batch, dim=-1)
        dense_cos_sims.append(dense_cos.cpu().numpy())
    
    overlaps = np.concatenate(overlaps)
    sparse_cos_sims = np.concatenate(sparse_cos_sims)
    dense_cos_sims = np.concatenate(dense_cos_sims)
    
    return {
        "feature_overlap_jaccard_mean": float(overlaps.mean()),
        "sparse_cosine_mean": float(sparse_cos_sims.mean()),
        "dense_cosine_mean": float(dense_cos_sims.mean()),
        "alignment_retention": float(
            sparse_cos_sims.mean() / max(dense_cos_sims.mean(), 1e-8)
        ),
    }


@torch.no_grad()
def evaluate_retrieval(
    sae: SparseAutoencoder,
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    relevance: list[set[int]],  # per-query relevant doc IDs
    scoring: str = "dot",
    batch_size: int = 256,
    sparse_backend: str = "python",
) -> dict:
    """
    Full retrieval evaluation: sparse vs dense.
    
    Encodes queries and docs through SAE, builds inverted index,
    retrieves, and computes ranking metrics.
    """
    device = next(sae.parameters()).device
    top_k = min(100, len(doc_embeddings))
    if top_k <= 0:
        raise ValueError("Document collection is empty")
    
    print("  Encoding documents...")
    t0 = time.time()
    sparse_docs = []
    for i in range(0, len(doc_embeddings), batch_size):
        batch_np = np.array(doc_embeddings[i:i+batch_size], copy=True)
        batch = torch.from_numpy(batch_np).to(device)
        indices, values = sae.get_active_features(batch)
        for j in range(len(batch)):
            doc = {}
            for idx, val in zip(indices[j].cpu().tolist(), values[j].cpu().tolist()):
                if val > 0:
                    doc[idx] = val
            sparse_docs.append(doc)
    encode_time = time.time() - t0
    
    print("  Encoding queries...")
    sparse_queries = []
    for i in range(0, len(query_embeddings), batch_size):
        batch_np = np.array(query_embeddings[i:i+batch_size], copy=True)
        batch = torch.from_numpy(batch_np).to(device)
        indices, values = sae.get_active_features(batch)
        for j in range(len(batch)):
            q = {}
            for idx, val in zip(indices[j].cpu().tolist(), values[j].cpu().tolist()):
                if val > 0:
                    q[idx] = val
            sparse_queries.append(q)
    
    print("  Building index...")
    if sparse_backend == "python":
        t0 = time.time()
        index = SimpleInvertedIndex()
        index.add_batch(sparse_docs)
        index_time = time.time() - t0

        print("  Retrieving (sparse)...")
        t0 = time.time()
        sparse_rankings = []
        for q in sparse_queries:
            results = index.search(q, top_k=top_k, scoring=scoring)
            sparse_rankings.append([doc_id for doc_id, _ in results])
        sparse_search_time = time.time() - t0
        index_stats = index.stats()
    elif sparse_backend == "torch_sparse":
        if scoring not in {"dot", "cosine"}:
            raise ValueError("torch_sparse backend currently supports only dot and cosine scoring.")

        vocab_size = sae.config.dict_size
        t0 = time.time()
        doc_matrix, doc_norms = build_torch_sparse_doc_matrix(
            sparse_docs,
            vocab_size=vocab_size,
            device=device,
        )
        index_time = time.time() - t0

        print("  Retrieving (sparse)...")
        t0 = time.time()
        sparse_rankings = []
        k = min(top_k, len(sparse_docs))
        for start in range(0, len(sparse_queries), batch_size):
            end = min(start + batch_size, len(sparse_queries))
            query_batch, query_norms = sparse_query_batch_to_dense(
                sparse_queries,
                start=start,
                end=end,
                vocab_size=vocab_size,
                device=device,
            )
            scores = torch.sparse.mm(doc_matrix, query_batch.t())
            if scoring == "cosine":
                denom = (doc_norms[:, None] * query_norms[None, :]).clamp_min(1e-8)
                scores = scores / denom
            for col in range(scores.shape[1]):
                col_scores = scores[:, col]
                positive_mask = col_scores > 0
                if not torch.any(positive_mask):
                    sparse_rankings.append([])
                    continue
                positive_indices = torch.nonzero(positive_mask, as_tuple=False).squeeze(-1)
                positive_scores = col_scores[positive_indices]
                take_k = min(k, positive_scores.shape[0])
                _, order = torch.topk(positive_scores, k=take_k, dim=0)
                sparse_rankings.append(positive_indices[order].cpu().tolist())
        sparse_search_time = time.time() - t0
        index_stats = compute_sparse_index_stats(sparse_docs)
    else:
        raise ValueError(f"Unsupported sparse_backend: {sparse_backend}")
    
    print("  Retrieving (dense, exact)...")
    t0 = time.time()
    q_tensor = torch.from_numpy(np.array(query_embeddings, copy=True)).to(device)
    d_tensor = torch.from_numpy(np.array(doc_embeddings, copy=True)).to(device)
    
    dense_rankings = []
    for i in range(0, len(q_tensor), batch_size):
        q_batch = q_tensor[i:i+batch_size]
        scores = torch.mm(q_batch, d_tensor.t())
        _, top_indices = scores.topk(top_k, dim=-1)
        for j in range(len(q_batch)):
            dense_rankings.append(top_indices[j].cpu().tolist())
    dense_search_time = time.time() - t0
    
    results = {
        "sparse": {
            "recall@1": recall_at_k(sparse_rankings, relevance, 1),
            "recall@5": recall_at_k(sparse_rankings, relevance, 5),
            "recall@10": recall_at_k(sparse_rankings, relevance, 10),
            "recall@100": recall_at_k(sparse_rankings, relevance, 100),
            "mrr": mrr(sparse_rankings, relevance),
            "ndcg@10": ndcg_at_k(sparse_rankings, relevance, 10),
            "search_time_s": sparse_search_time,
        },
        "dense": {
            "recall@1": recall_at_k(dense_rankings, relevance, 1),
            "recall@5": recall_at_k(dense_rankings, relevance, 5),
            "recall@10": recall_at_k(dense_rankings, relevance, 10),
            "recall@100": recall_at_k(dense_rankings, relevance, 100),
            "mrr": mrr(dense_rankings, relevance),
            "ndcg@10": ndcg_at_k(dense_rankings, relevance, 10),
            "search_time_s": dense_search_time,
        },
        "efficiency": {
            "encode_time_s": encode_time,
            "index_build_time_s": index_time,
            "sparse_search_time_s": sparse_search_time,
            "dense_search_time_s": dense_search_time,
            "speedup": dense_search_time / max(sparse_search_time, 1e-8),
        },
        "index_stats": index_stats,
    }
    
    return results


def evaluate_sparsity_quality_tradeoff(
    sae: SparseAutoencoder,
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    relevance: list[set[int]],
    k_values: list[int] = [4, 8, 16, 32, 64, 128, 256],
) -> dict:
    """
    Measure retrieval quality at different sparsity levels.
    
    This is the key experiment: how much quality do we lose
    as we make the representation sparser?
    """
    results = {}
    original_k = sae.current_k
    
    for k in k_values:
        print(f"\n  Evaluating k={k}...")
        sae.current_k = k
        
        r = evaluate_retrieval(
            sae, query_embeddings, doc_embeddings, relevance
        )
        
        results[k] = {
            "recall@10": r["sparse"]["recall@10"],
            "ndcg@10": r["sparse"]["ndcg@10"],
            "mrr": r["sparse"]["mrr"],
            "vs_dense_recall@10": (
                r["sparse"]["recall@10"] / max(r["dense"]["recall@10"], 1e-8)
            ),
        }
    
    sae.current_k = original_k
    return results


if __name__ == "__main__":
    print("=== Evaluation Demo (synthetic data) ===\n")
    
    config = SAEConfig(input_dim=256, dict_size=4096, k_initial=32, k_final=32)
    sae = SparseAutoencoder(config)
    sae.eval()
    
    np.random.seed(42)
    n_queries = 100
    n_docs = 5000
    d = 256
    
    base = np.random.randn(n_queries, d).astype(np.float32)
    
    queries = base + 0.1 * np.random.randn(n_queries, d).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=-1, keepdims=True)
    
    docs = np.random.randn(n_docs, d).astype(np.float32)
    docs[:n_queries] = base + 0.1 * np.random.randn(n_queries, d).astype(np.float32)
    docs = docs / np.linalg.norm(docs, axis=-1, keepdims=True)
    
    relevance = [{i} for i in range(n_queries)]
    
    print("1. Reconstruction quality:")
    recon = evaluate_reconstruction(sae, docs[:1000])
    for k, v in recon.items():
        print(f"   {k}: {v:.4f}")
    
    print("\n2. Retrieval quality (sparse vs dense):")
    retrieval = evaluate_retrieval(sae, queries, docs, relevance)
    
    print("\n   Sparse retrieval:")
    for k, v in retrieval["sparse"].items():
        print(f"     {k}: {v:.4f}" if isinstance(v, float) else f"     {k}: {v}")
    
    print("\n   Dense retrieval (baseline):")
    for k, v in retrieval["dense"].items():
        print(f"     {k}: {v:.4f}" if isinstance(v, float) else f"     {k}: {v}")
    
    print(f"\n   Speedup: {retrieval['efficiency']['speedup']:.1f}×")
    
    print("\n3. Sparsity-quality tradeoff:")
    tradeoff = evaluate_sparsity_quality_tradeoff(
        sae, queries, docs, relevance,
        k_values=[4, 8, 16, 32, 64, 128],
    )
    
    print(f"\n   {'k':>6s}  {'R@10':>8s}  {'nDCG@10':>8s}  {'vs_dense':>8s}")
    print("   " + "-" * 38)
    for k, m in sorted(tradeoff.items()):
        print(
            f"   {k:>6d}  {m['recall@10']:>8.4f}  "
            f"{m['ndcg@10']:>8.4f}  {m['vs_dense_recall@10']:>8.2%}"
        )
