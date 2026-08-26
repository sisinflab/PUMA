"""
Dataset for loading cached paired embeddings from Stage 1.

Supports two modes:
- PairedMode: returns (image_emb, text_emb) pairs for SAE training with alignment
- TripletMode: returns (query, positive, negatives) for contrastive fine-tuning
- RetrievalMode: returns CIRR/M-BEIR query/doc pairs with optional hard negatives
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional


def infer_cache_layout(cache_dir: str | Path) -> str:
    """Infer which cache layout is stored in a directory."""
    cache_dir = Path(cache_dir)

    paired_files = [
        cache_dir / "metadata.json",
        cache_dir / "image_embeddings.npy",
        cache_dir / "text_embeddings.npy",
    ]
    retrieval_files = [
        cache_dir / "query_meta.json",
        cache_dir / "query_embeddings.npy",
        cache_dir / "query_ids.json",
        cache_dir / "doc_meta.json",
        cache_dir / "doc_embeddings.npy",
        cache_dir / "doc_ids.json",
    ]

    if all(path.exists() for path in paired_files):
        return "paired"
    if all(path.exists() for path in retrieval_files):
        return "retrieval"

    raise FileNotFoundError(
        f"Could not infer cache layout under {cache_dir}. "
        "Expected either paired cache files (metadata.json, image_embeddings.npy, "
        "text_embeddings.npy) or retrieval cache files "
        "(query/doc embeddings + ids + meta)."
    )


def load_qrels(qrels_path: str | Path) -> dict[str, list[str]]:
    """Load qrels into qid -> [doc_id, ...] for all relevance > 0."""
    positives: dict[str, list[str]] = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, doc_id, rel = parts[:4]
            try:
                rel_value = float(rel)
            except ValueError:
                continue
            if rel_value <= 0:
                continue
            positives.setdefault(qid, []).append(doc_id)
    return positives


def load_query_annotations(query_path: str | Path) -> dict[str, dict]:
    """Load M-BEIR query JSONL rows keyed by qid."""
    annotations: dict[str, dict] = {}
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("qid")
            if qid is None:
                continue
            annotations[str(qid)] = row
    return annotations


class PairedEmbeddingDataset(Dataset):
    """
    Loads paired (image, text) embeddings from memory-mapped files.
    
    Used in Stage 2 (SAE training) where we need pairs for the
    cross-modal alignment loss.
    """

    def __init__(
        self,
        cache_dir: str = "./cached_embeddings",
        split: str = "train",         # "train" or "val"
        val_fraction: float = 0.02,   # 2% held out for validation
    ):
        cache_dir = Path(cache_dir)
        
        with open(cache_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        n = self.metadata["num_samples"]
        d = self.metadata["embedding_dim"]
        
        # Load memory-mapped arrays (no RAM cost until accessed)
        self.img_embs = np.memmap(
            str(cache_dir / "image_embeddings.npy"),
            dtype=np.float32, mode="r", shape=(n, d)
        )
        self.txt_embs = np.memmap(
            str(cache_dir / "text_embeddings.npy"),
            dtype=np.float32, mode="r", shape=(n, d)
        )
        
        val_size = int(n * val_fraction)
        if split == "train":
            self.start = 0
            self.end = n - val_size
        else:
            self.start = n - val_size
            self.end = n
        
        self.embedding_dim = d
    
    def __len__(self) -> int:
        return self.end - self.start
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        actual_idx = self.start + idx
        image_emb = torch.from_numpy(self.img_embs[actual_idx].copy())
        text_emb = torch.from_numpy(self.txt_embs[actual_idx].copy())
        
        return {
            "image_emb": image_emb,
            "text_emb": text_emb,
            "query_emb": text_emb,
            "positive_emb": image_emb,
        }


class RetrievalEmbeddingDataset(Dataset):
    """
    Load benchmark-style retrieval caches backed by query/doc memmaps.

    Each item is a query embedding paired with one relevant document embedding.
    This matches composed retrieval datasets such as CIRR, where training should
    align query(image+text) embeddings with target image embeddings.
    """

    def __init__(
        self,
        cache_dir: str,
        qrels_path: str,
        query_jsonl: Optional[str] = None,
        split: str = "train",         # "train", "val", or "all"
        val_fraction: float = 0.02,   # held-out query fraction
        split_seed: int = 0,
        num_hard_negatives: int = 0,
        negative_sampling_strategy: str = "fast",
    ):
        cache_dir = Path(cache_dir)
        qrels_path = Path(qrels_path)
        query_jsonl_path = Path(query_jsonl) if query_jsonl else None

        with open(cache_dir / "query_meta.json", "r", encoding="utf-8") as f:
            self.query_meta = json.load(f)
        with open(cache_dir / "doc_meta.json", "r", encoding="utf-8") as f:
            self.doc_meta = json.load(f)

        q_n = self.query_meta["count"]
        d_n = self.doc_meta["count"]
        q_dim = self.query_meta["embedding_dim"]
        d_dim = self.doc_meta["embedding_dim"]
        if q_dim != d_dim:
            raise ValueError(
                f"Query/doc embedding dims differ: {q_dim} vs {d_dim}"
            )

        self.query_embs = np.memmap(
            str(cache_dir / "query_embeddings.npy"),
            dtype=np.float32, mode="r", shape=(q_n, q_dim)
        )
        self.doc_embs = np.memmap(
            str(cache_dir / "doc_embeddings.npy"),
            dtype=np.float32, mode="r", shape=(d_n, d_dim)
        )

        with open(cache_dir / "query_ids.json", "r", encoding="utf-8") as f:
            self.query_ids = json.load(f)
        with open(cache_dir / "doc_ids.json", "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)

        if len(self.query_ids) != q_n or len(self.doc_ids) != d_n:
            raise ValueError("Cache ids do not match embedding counts")

        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        positives_by_qid = load_qrels(qrels_path)
        annotations_by_qid = (
            load_query_annotations(query_jsonl_path)
            if query_jsonl_path is not None
            else {}
        )

        self.relevance_by_query_row: dict[int, tuple[int, ...]] = {}
        self.hard_negatives_by_query_row: dict[int, tuple[int, ...]] = {}
        valid_query_rows: list[int] = []
        for query_row, qid in enumerate(self.query_ids):
            annotation = annotations_by_qid.get(qid, {})
            annotated_positives = annotation.get("pos_cand_list") or []
            qrel_positives = positives_by_qid.get(qid, [])
            positive_rows = sorted(
                {
                    doc_id_to_idx[doc_id]
                    for doc_id in [*annotated_positives, *qrel_positives]
                    if doc_id in doc_id_to_idx
                }
            )
            if not positive_rows:
                continue
            self.relevance_by_query_row[query_row] = tuple(positive_rows)

            negative_rows = sorted(
                {
                    doc_id_to_idx[doc_id]
                    for doc_id in (annotation.get("neg_cand_list") or [])
                    if doc_id in doc_id_to_idx and doc_id_to_idx[doc_id] not in positive_rows
                }
            )
            self.hard_negatives_by_query_row[query_row] = tuple(negative_rows)
            valid_query_rows.append(query_row)

        if not valid_query_rows:
            raise ValueError(
                f"No qrels from {qrels_path} matched document ids in {cache_dir}"
            )

        ordered_rows = np.asarray(valid_query_rows, dtype=np.int64)
        if split not in {"train", "val", "all"}:
            raise ValueError(f"Unsupported split: {split}")

        if split == "all":
            self.query_rows = ordered_rows
        else:
            rng = np.random.default_rng(split_seed)
            shuffled_rows = ordered_rows.copy()
            rng.shuffle(shuffled_rows)

            if len(shuffled_rows) <= 1:
                val_size = 0
            else:
                val_size = int(round(len(shuffled_rows) * val_fraction))
                val_size = min(max(val_size, 1), len(shuffled_rows) - 1)

            if split == "train":
                self.query_rows = shuffled_rows[:-val_size] if val_size else shuffled_rows
            else:
                self.query_rows = shuffled_rows[-val_size:] if val_size else shuffled_rows[:0]

        self.embedding_dim = q_dim
        self.num_docs = d_n
        self.doc_row_ids = np.arange(self.num_docs, dtype=np.int64)
        self.num_hard_negatives = max(0, int(num_hard_negatives))
        if negative_sampling_strategy not in {"fast", "legacy"}:
            raise ValueError(
                "negative_sampling_strategy must be 'fast' or 'legacy', "
                f"got {negative_sampling_strategy!r}"
            )
        self.negative_sampling_strategy = str(negative_sampling_strategy)
        self.query_jsonl = str(query_jsonl_path) if query_jsonl_path is not None else None

    def __len__(self) -> int:
        return int(len(self.query_rows))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        query_row = int(self.query_rows[idx])
        positive_rows = self.relevance_by_query_row[query_row]
        if len(positive_rows) == 1:
            positive_row = int(positive_rows[0])
        else:
            positive_row = int(np.random.choice(np.asarray(positive_rows, dtype=np.int64)))

        batch = {
            "query_emb": torch.from_numpy(self.query_embs[query_row].copy()),
            "positive_emb": torch.from_numpy(self.doc_embs[positive_row].copy()),
            "query_index": torch.tensor(query_row, dtype=torch.long),
            "positive_index": torch.tensor(positive_row, dtype=torch.long),
        }

        if self.num_hard_negatives > 0:
            negative_rows = self._sample_negative_rows(query_row, positive_rows)
            batch["hard_negative_embs"] = torch.from_numpy(self.doc_embs[negative_rows].copy())
            batch["hard_negative_indices"] = torch.from_numpy(negative_rows.astype(np.int64, copy=False))

        return batch

    def _sample_negative_rows(
        self,
        query_row: int,
        positive_rows: tuple[int, ...],
    ) -> np.ndarray:
        if self.negative_sampling_strategy == "legacy":
            return self._sample_negative_rows_legacy(query_row, positive_rows)
        return self._sample_negative_rows_fast(query_row, positive_rows)

    def _sample_negative_rows_fast(
        self,
        query_row: int,
        positive_rows: tuple[int, ...],
    ) -> np.ndarray:
        selected: list[int] = []
        excluded = set(int(row) for row in positive_rows)
        target_negatives = min(
            self.num_hard_negatives,
            max(0, self.num_docs - len(excluded)),
        )
        explicit_negatives = self.hard_negatives_by_query_row.get(query_row, ())
        if explicit_negatives:
            take = min(target_negatives, len(explicit_negatives))
            chosen = np.random.choice(
                np.asarray(explicit_negatives, dtype=np.int64),
                size=take,
                replace=False,
            )
            for value in np.atleast_1d(chosen).tolist():
                value = int(value)
                selected.append(value)
                excluded.add(value)

        need = target_negatives - len(selected)
        if need > 0:
            sampled = set(selected)
            attempts = 0
            while need > 0 and attempts < 8:
                candidate_count = max(32, 4 * need)
                candidates = np.random.randint(0, self.num_docs, size=candidate_count, dtype=np.int64)
                for value in candidates.tolist():
                    value = int(value)
                    if value in excluded or value in sampled:
                        continue
                    selected.append(value)
                    sampled.add(value)
                    need -= 1
                    if need == 0:
                        break
                attempts += 1

            if need > 0:
                excluded_rows = np.fromiter(
                    excluded.union(sampled),
                    dtype=np.int64,
                    count=len(excluded) + len(sampled),
                )
                available = np.setdiff1d(self.doc_row_ids, excluded_rows, assume_unique=False)
                if len(available) > 0:
                    take = min(need, len(available))
                    chosen = np.random.choice(available, size=take, replace=False)
                    selected.extend(int(value) for value in np.atleast_1d(chosen).tolist())

        return np.asarray(selected, dtype=np.int64)

    def _sample_negative_rows_legacy(
        self,
        query_row: int,
        positive_rows: tuple[int, ...],
    ) -> np.ndarray:
        selected: list[int] = []
        excluded = set(int(row) for row in positive_rows)
        target_negatives = min(
            self.num_hard_negatives,
            max(0, self.num_docs - len(excluded)),
        )
        explicit_negatives = self.hard_negatives_by_query_row.get(query_row, ())
        if explicit_negatives:
            take = min(target_negatives, len(explicit_negatives))
            chosen = np.random.choice(
                np.asarray(explicit_negatives, dtype=np.int64),
                size=take,
                replace=False,
            )
            for value in np.atleast_1d(chosen).tolist():
                value = int(value)
                selected.append(value)
                excluded.add(value)

        need = target_negatives - len(selected)
        if need > 0:
            available = np.asarray(
                [row for row in range(self.num_docs) if row not in excluded],
                dtype=np.int64,
            )
            if len(available) > 0:
                take = min(need, len(available))
                chosen = np.random.choice(available, size=take, replace=False)
                selected.extend(int(value) for value in np.atleast_1d(chosen).tolist())

        return np.asarray(selected, dtype=np.int64)

    def get_query_embeddings(self, limit: Optional[int] = None) -> np.ndarray:
        query_rows = self.query_rows if limit is None else self.query_rows[:limit]
        return np.asarray(self.query_embs[query_rows], dtype=np.float32)

    def get_doc_embeddings(self) -> np.ndarray:
        return np.asarray(self.doc_embs, dtype=np.float32)

    def get_relevance_sets(self, limit: Optional[int] = None) -> list[set[int]]:
        query_rows = self.query_rows if limit is None else self.query_rows[:limit]
        return [
            set(self.relevance_by_query_row[int(query_row)])
            for query_row in query_rows
        ]

    def get_alignment_pairs(
        self,
        limit: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_rows = self.query_rows if limit is None else self.query_rows[:limit]
        positive_rows = [
            self.relevance_by_query_row[int(query_row)][0]
            for query_row in query_rows
        ]
        query_embs = np.asarray(self.query_embs[query_rows], dtype=np.float32)
        positive_embs = np.asarray(self.doc_embs[positive_rows], dtype=np.float32)
        return query_embs, positive_embs


class RetrievalTripletDataset(Dataset):
    """
    Generates (query, positive, hard_negatives) triplets for Stage 3.
    
    Hard negatives are sampled from the same batch or pre-mined.
    For simplicity, this uses random negatives — in practice you'd
    want BM25 or dense retrieval hard negatives.
    """

    def __init__(
        self,
        cache_dir: str = "./cached_embeddings",
        num_negatives: int = 7,       # negatives per query
        query_modality: str = "both", # "text", "image", or "both"
    ):
        cache_dir = Path(cache_dir)
        
        with open(cache_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        n = self.metadata["num_samples"]
        d = self.metadata["embedding_dim"]
        
        self.img_embs = np.memmap(
            str(cache_dir / "image_embeddings.npy"),
            dtype=np.float32, mode="r", shape=(n, d)
        )
        self.txt_embs = np.memmap(
            str(cache_dir / "text_embeddings.npy"),
            dtype=np.float32, mode="r", shape=(n, d)
        )
        
        self.n = n
        self.d = d
        self.num_negatives = num_negatives
        self.query_modality = query_modality
    
    def __len__(self) -> int:
        return self.n
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.query_modality == "both":
            use_text_query = (idx % 2 == 0)
        else:
            use_text_query = (self.query_modality == "text")
        
        if use_text_query:
            query = torch.from_numpy(self.txt_embs[idx].copy())
            positive = torch.from_numpy(self.img_embs[idx].copy())
        else:
            query = torch.from_numpy(self.img_embs[idx].copy())
            positive = torch.from_numpy(self.txt_embs[idx].copy())
        
        # Random negatives (in practice: use hard negatives!)
        neg_indices = np.random.choice(
            self.n, size=self.num_negatives, replace=False
        )
        # Avoid sampling the positive as a negative
        neg_indices = neg_indices[neg_indices != idx][:self.num_negatives]
        
        if use_text_query:
            negatives = torch.from_numpy(
                self.img_embs[neg_indices].copy()
            )
        else:
            negatives = torch.from_numpy(
                self.txt_embs[neg_indices].copy()
            )
        
        return {
            "query": query,
            "positive": positive,
            "negatives": negatives,
        }


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 4096,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """Create a DataLoader with appropriate settings for embedding data."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    return DataLoader(
        **loader_kwargs,
    )
