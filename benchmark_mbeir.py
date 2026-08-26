"""
Benchmark a trained SAE on a local M-BEIR split.

This script can:
1. Resolve query/candidate/qrels files from an M-BEIR root.
2. Encode queries and documents with Qwen3-VL-Embedding via vLLM.
3. Cache dense embeddings to disk for reuse.
4. Evaluate dense vs sparse retrieval with local metrics.

  CUDA_VISIBLE_DEVICES=3 python benchmark_mbeir.py \
    --dataset cirr_task7 \
    --split test \
    --mbeir_root ./M-BEIR \
    --model_path Qwen/Qwen3-VL-Embedding-2B \
    --checkpoint_path ./checkpoints/sae_final.pt \
    --embedding_dim 2048 \
    --batch_size 64 \
    --image_workers 16 \
    --sparse_batch_size 256 \
    --device cuda \
    --dense_cache_dir ./benchmark_cache/cirr_task7_test \
    --output_dir ./benchmark_runs/cirr_task7_test

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import mrr, ndcg_at_k, recall_at_k
from extract_qwen import (
    build_vllm_prompt,
    embedding_from_vllm_output,
    resolve_image_safe,
)
from inference import SimpleInvertedIndex, SparseEncoder


DEFAULT_EMBED_SYSTEM_INSTRUCTION = "Represent the user's input."
METRIC_KEYS = [
    "recall@1",
    "recall@5",
    "recall@10",
    "recall@100",
    "hit_rate@1",
    "hit_rate@5",
    "hit_rate@10",
    "hit_rate@100",
    "mrr",
    "ndcg@10",
]


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_jsonl(path: Path, max_items: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_items > 0 and len(rows) >= max_items:
                break
    return rows


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, did, rel = parts[:4]
            if int(rel) > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


def default_query_text_prefix(dataset: str) -> str:
    dataset = dataset.lower()
    dataset_key = dataset.split("_task", 1)[0]
    prompts = {
        "fashioniq": "Find a fashion image that aligns with the reference image and style note.",
        "fashion200k": "Find a fashion image that aligns with the reference image and style note.",
        "cirr": "Retrieve a day-to-day image that aligns with the modification instructions of the provided image.",
    }
    return prompts.get(
        dataset_key,
        "Retrieve the target image that best matches the reference image and the textual modification.",
    )


def item_modality(record: dict, *, is_query: bool) -> set[str]:
    key = "query_modality" if is_query else "modality"
    raw_modality = record.get(key)
    if isinstance(raw_modality, str) and raw_modality.strip():
        return {part.strip().lower() for part in raw_modality.split(",") if part.strip()}

    has_text = bool((record.get("query_txt") if is_query else record.get("txt")) or "")
    has_image = bool(record.get("query_img_path") if is_query else record.get("img_path"))
    modalities = set()
    if has_text:
        modalities.add("text")
    if has_image:
        modalities.add("image")
    return modalities


def record_id(record: dict, *, is_query: bool) -> str:
    return str(record["qid"] if is_query else record["did"])


def hash_ids(ids: list[str]) -> str:
    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_mbeir_paths(
    *,
    mbeir_root: Path,
    dataset: str,
    split: str,
    candidate_scope: str,
    query_path: Optional[str],
    cand_pool_path: Optional[str],
    qrels_path: Optional[str],
) -> tuple[Path, Path, Path]:
    dataset_base = dataset.split("_task", 1)[0]

    if query_path:
        query_file = Path(query_path)
    else:
        query_candidates = [mbeir_root / "query" / split / f"mbeir_{dataset}_{split}.jsonl"]
        if split == "train" and dataset_base != dataset:
            query_candidates.append(
                mbeir_root / "query" / split / f"mbeir_{dataset_base}_{split}.jsonl"
            )
        query_file = next((path for path in query_candidates if path.exists()), query_candidates[0])

    if qrels_path:
        qrels_file = Path(qrels_path)
    else:
        qrels_candidates = [mbeir_root / "qrels" / split / f"mbeir_{dataset}_{split}_qrels.txt"]
        if split == "train" and dataset_base != dataset:
            qrels_candidates.append(
                mbeir_root / "qrels" / split / f"mbeir_{dataset_base}_{split}_qrels.txt"
            )
        qrels_file = next((path for path in qrels_candidates if path.exists()), qrels_candidates[0])

    if cand_pool_path:
        cand_file = Path(cand_pool_path)
    else:
        candidates = []
        if candidate_scope == "local":
            candidates.extend(
                [
                    mbeir_root / "cand_pool" / "local" / f"mbeir_{dataset}_cand_pool.jsonl",
                    mbeir_root / "cand_pool" / "local" / f"mbeir_{dataset}_{split}_cand_pool.jsonl",
                ]
            )
            if dataset_base != dataset:
                candidates.extend(
                    [
                        mbeir_root / "cand_pool" / "local" / f"mbeir_{dataset_base}_cand_pool.jsonl",
                        mbeir_root / "cand_pool" / "local" / f"mbeir_{dataset_base}_{split}_cand_pool.jsonl",
                    ]
                )
        else:
            candidates.append(
                mbeir_root / "cand_pool" / "global" / f"mbeir_union_{split}_cand_pool.jsonl"
            )
        cand_file = next((path for path in candidates if path.exists()), None)
        if cand_file is None and candidate_scope == "local":
            task_matches = sorted(
                (mbeir_root / "cand_pool" / "local").glob(f"mbeir_{dataset_base}_task*_cand_pool.jsonl")
            )
            if len(task_matches) == 1:
                cand_file = task_matches[0]
            elif len(task_matches) > 1:
                joined = ", ".join(str(path) for path in task_matches)
                raise FileNotFoundError(
                    "Multiple task-specific candidate pools matched. "
                    f"Please pass --cand_pool_path explicitly. Matches: {joined}"
                )
        if cand_file is None:
            searched = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(f"Could not resolve candidate pool. Tried: {searched}")

    for path, label in [
        (query_file, "query file"),
        (cand_file, "candidate pool"),
        (qrels_file, "qrels file"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    return query_file, cand_file, qrels_file


def item_id_text_image(
    record: dict,
    *,
    is_query: bool,
    mbeir_root: Path,
    text_prefix: str,
) -> tuple[str, str, Optional[Path]]:
    modalities = item_modality(record, is_query=is_query)
    if is_query:
        item_id = str(record["qid"])
        raw_text = record.get("query_txt") or ""
        image_path = record.get("query_img_path")
    else:
        item_id = str(record["did"])
        raw_text = record.get("txt") or ""
        image_path = record.get("img_path")

    text = raw_text.strip() if "text" in modalities else ""
    if text and text_prefix:
        text = f"{text_prefix}\n{text}"

    resolved_path = None
    if "image" in modalities and image_path:
        image_candidate = Path(image_path)
        resolved_path = image_candidate if image_candidate.is_absolute() else mbeir_root / image_candidate

    return item_id, text, resolved_path


def build_cache_metadata(
    *,
    model_name: str,
    count: int,
    embedding_dim: int,
    batch_size: int,
    system_instruction: str,
    text_prefix: str,
    kind: str,
    ids: list[str],
) -> dict:
    return {
        "cache_version": 2,
        "model_name": model_name,
        "count": count,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "system_instruction": system_instruction,
        "text_prefix": text_prefix,
        "kind": kind,
        "id_sha256": hash_ids(ids),
        "first_id": ids[0] if ids else None,
        "last_id": ids[-1] if ids else None,
    }


def cache_mismatch_reasons(metadata: dict, expected: dict) -> list[str]:
    reasons = []
    for key, expected_value in expected.items():
        if expected_value is None:
            continue
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            reasons.append(f"{key}: expected {expected_value!r}, found {actual_value!r}")
    return reasons


def inspect_dense_cache(cache_dir: Path, prefix: str, expected_meta: dict) -> tuple[bool, list[str]]:
    emb_path = cache_dir / f"{prefix}_embeddings.npy"
    ids_path = cache_dir / f"{prefix}_ids.json"
    meta_path = cache_dir / f"{prefix}_meta.json"

    missing = [str(path.name) for path in (emb_path, ids_path, meta_path) if not path.exists()]
    if missing:
        return False, [f"missing files: {', '.join(missing)}"]

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    normalized_metadata = dict(metadata)
    need_ids = any(
        key not in normalized_metadata
        for key in ("id_sha256", "first_id", "last_id")
    )
    ids = None
    if need_ids:
        with ids_path.open("r", encoding="utf-8") as f:
            ids = json.load(f)

    if "system_instruction" not in normalized_metadata and "instruction" in normalized_metadata:
        normalized_metadata["system_instruction"] = normalized_metadata["instruction"]
    if "text_prefix" not in normalized_metadata:
        normalized_metadata["text_prefix"] = ""
    if "cache_version" not in normalized_metadata:
        normalized_metadata["cache_version"] = 2
    if ids is not None:
        normalized_metadata.setdefault("id_sha256", hash_ids(ids))
        normalized_metadata.setdefault("first_id", ids[0] if ids else None)
        normalized_metadata.setdefault("last_id", ids[-1] if ids else None)

    reasons = cache_mismatch_reasons(normalized_metadata, expected_meta)
    return not reasons, reasons


def load_cached_dense(cache_dir: Path, prefix: str) -> tuple[np.memmap, list[str], dict]:
    emb_path = cache_dir / f"{prefix}_embeddings.npy"
    ids_path = cache_dir / f"{prefix}_ids.json"
    meta_path = cache_dir / f"{prefix}_meta.json"

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    with ids_path.open("r", encoding="utf-8") as f:
        ids = json.load(f)
    embeddings = np.memmap(
        emb_path,
        dtype=np.float32,
        mode="r",
        shape=(metadata["count"], metadata["embedding_dim"]),
    )
    return embeddings, ids, metadata


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


def write_trec_run(path: Path, query_ids: list[str], rankings: list[list[int]], doc_ids: list[str], tag: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for query_id, ranking in zip(query_ids, rankings):
            for rank, doc_idx in enumerate(ranking, start=1):
                score = 1.0 / rank
                f.write(f"{query_id} Q0 {doc_ids[doc_idx]} {rank} {score:.6f} {tag}\n")


class VLLMEncoder:
    def __init__(self, model_name: str, batch_size: int, image_workers: int):
        # vLLM's library API defaults to fork on this install, which breaks once
        # CUDA has been touched in the parent process. Force spawn unless the
        # caller already chose a method explicitly.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM

        self.model_name = model_name
        self.batch_size = batch_size
        self.image_workers = max(1, image_workers)

        llm_kwargs = {
            "model": model_name,
            "runner": "pooling",
            "convert": "embed",
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.7,
            "enforce_eager": True,
            # Match the reference provider's multimodal vLLM path as closely as possible.
            "limit_mm_per_prompt": {"image": 1},
            "mm_encoder_attn_backend": "TORCH_SDPA",
        }

        self.llm = LLM(**llm_kwargs)
        self.apply_chat_template = self.llm.llm_engine.tokenizer.apply_chat_template

    def _build_requests(
        self,
        batch: list[dict],
        mbeir_root: Path,
        is_query: bool,
        system_instruction: str,
        text_prefix: str,
    ) -> list[tuple[str, dict]]:
        work_items = [
            item_id_text_image(
                record,
                is_query=is_query,
                mbeir_root=mbeir_root,
                text_prefix=text_prefix,
            )
            for record in batch
        ]

        def load_image(path: Optional[Path]):
            if path is None:
                return None
            return resolve_image_safe({"image": str(path)})

        with ThreadPoolExecutor(max_workers=self.image_workers) as pool:
            images = list(pool.map(load_image, [image_path for _, _, image_path in work_items]))

        requests = []
        for (item_id, text, image_path), image in zip(work_items, images):
            image_ref = None
            if image is not None and image_path is not None:
                image_ref = f"file://{image_path.resolve()}"
            elif image_path is not None and not text:
                raise RuntimeError(f"Failed to load image-only item {item_id} from {image_path}")

            prompt = build_vllm_prompt(
                self.apply_chat_template,
                instruction=system_instruction,
                text=text or None,
                image_ref=image_ref,
            )
            request = {"prompt": prompt}
            if image is not None:
                request["multi_modal_data"] = {"image": image}

            requests.append((item_id, request))

        return requests

    def encode(
        self,
        *,
        records: list[dict],
        mbeir_root: Path,
        is_query: bool,
        cache_dir: Path,
        prefix: str,
        embedding_dim: int,
        overwrite: bool,
        system_instruction: str,
        text_prefix: str,
        expected_ids: list[str],
    ) -> tuple[np.memmap, list[str], bool]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_path = cache_dir / f"{prefix}_embeddings.npy"
        ids_path = cache_dir / f"{prefix}_ids.json"
        meta_path = cache_dir / f"{prefix}_meta.json"

        expected_meta = build_cache_metadata(
            model_name=self.model_name,
            count=len(records),
            embedding_dim=embedding_dim,
            batch_size=self.batch_size,
            system_instruction=system_instruction,
            text_prefix=text_prefix,
            kind=prefix,
            ids=expected_ids,
        )

        if not overwrite:
            cache_valid, reasons = inspect_dense_cache(cache_dir, prefix, expected_meta)
            if cache_valid:
                embeddings, ids, _ = load_cached_dense(cache_dir, prefix)
                print(f"Reusing cached {prefix} embeddings from {emb_path}")
                return embeddings, ids, True
            if reasons:
                joined = "; ".join(reasons)
                print(f"Cached {prefix} embeddings are incompatible; re-encoding because {joined}")

        embeddings = np.memmap(
            emb_path,
            dtype=np.float32,
            mode="w+",
            shape=(len(records), embedding_dim),
        )
        ids: list[str] = []
        progress_desc = "queries" if is_query else "docs"

        from tqdm.auto import tqdm

        progress = tqdm(total=len(records), desc=f"encode {progress_desc}", unit="item", dynamic_ncols=True)
        write_index = 0
        start_time = time.time()

        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            request_items = self._build_requests(
                batch,
                mbeir_root,
                is_query,
                system_instruction,
                text_prefix,
            )
            requests = [request for _, request in request_items]

            try:
                outputs = self.llm.embed(requests, use_tqdm=False)
            except Exception as exc:
                print(f"\nWarning: batch encode failed ({exc}); falling back to per-item requests.")
                outputs = []
                for request in requests:
                    outputs.extend(self.llm.embed([request], use_tqdm=False))

            if len(outputs) != len(request_items):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} embeddings for {len(request_items)} {prefix} items"
                )

            for (item_id, _), output in zip(request_items, outputs):
                embeddings[write_index] = embedding_from_vllm_output(output)
                ids.append(item_id)
                write_index += 1

            embeddings.flush()
            progress.update(len(outputs))
            elapsed = max(time.time() - start_time, 1e-6)
            progress.set_postfix_str(f"{write_index / elapsed:.1f} item/s")

        progress.close()

        if ids != expected_ids:
            raise RuntimeError(f"{prefix} ids changed during encoding; cache would be inconsistent")

        metadata = {
            **expected_meta,
            "count": write_index,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        with ids_path.open("w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2)

        return embeddings, ids, False


def build_relevance(
    query_ids: list[str],
    doc_ids: list[str],
    qrels: dict[str, set[str]],
) -> tuple[np.ndarray, list[str], list[set[int]]]:
    doc_lookup = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    selected_query_indices = []
    selected_query_ids = []
    relevance = []

    for idx, query_id in enumerate(query_ids):
        relevant_docs = {doc_lookup[doc_id] for doc_id in qrels.get(query_id, set()) if doc_id in doc_lookup}
        if not relevant_docs:
            continue
        selected_query_indices.append(idx)
        selected_query_ids.append(query_id)
        relevance.append(relevant_docs)

    if not selected_query_indices:
        raise RuntimeError("No queries have relevant documents in the selected candidate pool.")

    return np.asarray(selected_query_indices, dtype=np.int64), selected_query_ids, relevance


def exact_dense_rankings(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    *,
    top_k: int,
    device: str,
    query_batch_size: int,
    doc_chunk_size: int,
) -> list[list[int]]:
    device = resolve_device(device)
    rankings: list[list[int]] = []
    k = min(top_k, len(doc_embeddings))

    for query_start in range(0, len(query_embeddings), query_batch_size):
        query_batch_np = np.array(
            query_embeddings[query_start : query_start + query_batch_size],
            dtype=np.float32,
            copy=True,
        )
        query_batch = torch.from_numpy(query_batch_np).to(device)
        best_scores = None
        best_indices = None

        for doc_start in range(0, len(doc_embeddings), doc_chunk_size):
            doc_batch_np = np.array(
                doc_embeddings[doc_start : doc_start + doc_chunk_size],
                dtype=np.float32,
                copy=True,
            )
            doc_batch = torch.from_numpy(doc_batch_np).to(device)
            scores = torch.mm(query_batch, doc_batch.t())
            chunk_k = min(k, scores.shape[1])
            chunk_scores, chunk_indices = scores.topk(chunk_k, dim=-1)
            chunk_indices = chunk_indices + doc_start

            if best_scores is None:
                best_scores = chunk_scores
                best_indices = chunk_indices
                continue

            merged_scores = torch.cat([best_scores, chunk_scores], dim=-1)
            merged_indices = torch.cat([best_indices, chunk_indices], dim=-1)
            best_scores, merge_order = merged_scores.topk(k, dim=-1)
            best_indices = torch.gather(merged_indices, 1, merge_order)

        rankings.extend(best_indices.cpu().tolist())

    return rankings


def select_pca_fit_indices(
    num_rows: int,
    fit_count: int,
    *,
    seed: int,
) -> np.ndarray:
    if fit_count >= num_rows:
        return np.arange(num_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = rng.choice(num_rows, size=fit_count, replace=False)
    indices.sort()
    return indices.astype(np.int64, copy=False)


def fit_pca_projection(
    doc_embeddings: np.ndarray,
    *,
    output_dim: int,
    fit_max_docs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    num_docs, input_dim = doc_embeddings.shape
    output_dim = min(int(output_dim), int(input_dim))
    fit_count = min(max(1, int(fit_max_docs)), num_docs)
    fit_indices = select_pca_fit_indices(num_docs, fit_count, seed=seed)
    sample_np = np.array(doc_embeddings[fit_indices], dtype=np.float32, copy=True)

    if output_dim >= input_dim:
        mean = sample_np.mean(axis=0, dtype=np.float32)
        components = np.eye(input_dim, dtype=np.float32)
        meta = {
            "fit_docs": int(fit_count),
            "fit_seed": int(seed),
            "input_dim": int(input_dim),
            "output_dim": int(output_dim),
            "method": "identity",
        }
        return mean, components[:, :output_dim], meta

    sample = torch.from_numpy(sample_np)
    mean = sample.mean(dim=0, keepdim=True)
    centered = sample - mean
    q = min(output_dim + 8, centered.shape[0], centered.shape[1])
    _, _, v = torch.pca_lowrank(centered, q=q, center=False, niter=4)
    components = v[:, :output_dim].contiguous().cpu().numpy().astype(np.float32, copy=False)
    meta = {
        "fit_docs": int(fit_count),
        "fit_seed": int(seed),
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "method": "pca_lowrank",
    }
    return mean.squeeze(0).cpu().numpy().astype(np.float32, copy=False), components, meta


def project_dense_embeddings(
    embeddings: np.ndarray,
    *,
    mean: np.ndarray,
    components: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    output_dim = int(components.shape[1])
    projected = np.empty((len(embeddings), output_dim), dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    components = np.asarray(components, dtype=np.float32)

    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        batch = np.array(embeddings[start:end], dtype=np.float32, copy=True)
        projected[start:end] = (batch - mean) @ components

    return projected


def evaluate_matched_memory_dense(
    *,
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    matched_k: int,
    top_k: int,
    device: str,
    query_batch_size: int,
    doc_chunk_size: int,
    fit_max_docs: int,
    seed: int,
) -> tuple[list[list[int]], dict[str, float], dict]:
    compressed_dim = min(int(2 * matched_k), int(doc_embeddings.shape[1]))

    fit_start = time.time()
    mean, components, fit_meta = fit_pca_projection(
        doc_embeddings,
        output_dim=compressed_dim,
        fit_max_docs=fit_max_docs,
        seed=seed,
    )
    fit_time = time.time() - fit_start

    doc_project_start = time.time()
    projected_docs = project_dense_embeddings(doc_embeddings, mean=mean, components=components)
    doc_project_time = time.time() - doc_project_start

    query_project_start = time.time()
    projected_queries = project_dense_embeddings(query_embeddings, mean=mean, components=components)
    query_project_time = time.time() - query_project_start

    search_start = time.time()
    rankings = exact_dense_rankings(
        projected_queries,
        projected_docs,
        top_k=top_k,
        device=device,
        query_batch_size=query_batch_size,
        doc_chunk_size=doc_chunk_size,
    )
    search_time = time.time() - search_start

    timings = {
        "fit_time_s": fit_time,
        "doc_project_time_s": doc_project_time,
        "query_project_time_s": query_project_time,
        "search_time_s": search_time,
        "total_time_s": fit_time + doc_project_time + query_project_time + search_time,
    }
    meta = {
        "matched_k": int(matched_k),
        "compressed_dim": int(compressed_dim),
        **fit_meta,
    }
    return rankings, timings, meta


class DenseBottleneckAutoencoder(torch.nn.Module):
    """Dense AE used as a learned matched-memory compression baseline."""

    def __init__(self, input_dim: int, bottleneck_dim: int, hidden_dim: int, architecture: str):
        super().__init__()
        self.input_dim = int(input_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.architecture = str(architecture).lower()
        if self.architecture not in {"linear", "mlp"}:
            raise ValueError(f"Unsupported dense AE architecture: {self.architecture}")
        if self.architecture == "linear":
            self.hidden_dim = 0
            self.encoder = torch.nn.Linear(self.input_dim, self.bottleneck_dim)
            self.decoder = torch.nn.Linear(self.bottleneck_dim, self.input_dim)
        else:
            self.hidden_dim = int(hidden_dim)
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.hidden_dim, self.bottleneck_dim),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(self.bottleneck_dim, self.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.hidden_dim, self.input_dim),
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decoder(z)


def resolve_dense_ae_hidden_dim(input_dim: int, bottleneck_dim: int, requested_hidden_dim: int) -> int:
    if requested_hidden_dim > 0:
        return int(requested_hidden_dim)
    return min(int(input_dim), max(512, 4 * int(bottleneck_dim)))


def fit_dense_autoencoder(
    doc_embeddings: np.ndarray,
    *,
    output_dim: int,
    fit_max_docs: int,
    seed: int,
    device: str,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    architecture: str,
) -> tuple[DenseBottleneckAutoencoder, dict]:
    num_docs, input_dim = doc_embeddings.shape
    output_dim = min(int(output_dim), int(input_dim))
    fit_count = min(max(1, int(fit_max_docs)), num_docs)
    fit_indices = select_pca_fit_indices(num_docs, fit_count, seed=seed)
    sample_np = np.array(doc_embeddings[fit_indices], dtype=np.float32, copy=True)

    resolved_device = resolve_device(device)
    torch.manual_seed(int(seed))
    if resolved_device == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    model = DenseBottleneckAutoencoder(
        input_dim=input_dim,
        bottleneck_dim=output_dim,
        hidden_dim=resolve_dense_ae_hidden_dim(input_dim, output_dim, hidden_dim),
        architecture=architecture,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sample = torch.from_numpy(sample_np).to(resolved_device)
    effective_batch_size = min(max(1, int(batch_size)), fit_count)

    model.train()
    final_loss = float("nan")
    log_every = max(int(steps) // 5, 1)
    for step in range(max(1, int(steps))):
        batch_indices = torch.randint(
            low=0,
            high=fit_count,
            size=(effective_batch_size,),
            device=resolved_device,
        )
        batch = sample[batch_indices]
        _, recon = model(batch)
        cosine_loss = (1.0 - F.cosine_similarity(recon, batch, dim=-1)).mean()
        mse_loss = F.mse_loss(recon, batch)
        loss = cosine_loss + 0.1 * mse_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

        if (step + 1) % log_every == 0 or step == 0 or step + 1 == int(steps):
            print(
                f"    AE step {step + 1:>5d}/{int(steps)} | "
                f"loss={final_loss:.4f} | cos={float(cosine_loss.detach().cpu().item()):.4f}"
            )

    model.eval()
    meta = {
        "fit_docs": int(fit_count),
        "fit_seed": int(seed),
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "method": "dense_autoencoder",
        "architecture": str(model.architecture),
        "hidden_dim": int(model.hidden_dim),
        "train_steps": int(steps),
        "train_batch_size": int(effective_batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "final_train_loss": final_loss,
    }
    return model, meta


@torch.no_grad()
def encode_dense_autoencoder_embeddings(
    model: DenseBottleneckAutoencoder,
    embeddings: np.ndarray,
    *,
    device: str,
    batch_size: int = 8192,
    normalize: bool = True,
) -> np.ndarray:
    resolved_device = resolve_device(device)
    output_dim = int(model.bottleneck_dim)
    projected = np.empty((len(embeddings), output_dim), dtype=np.float32)
    model = model.to(resolved_device)
    model.eval()

    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        batch_np = np.array(embeddings[start:end], dtype=np.float32, copy=True)
        batch = torch.from_numpy(batch_np).to(resolved_device)
        z = model.encode(batch)
        if normalize:
            z = F.normalize(z, dim=-1, eps=1e-8)
        projected[start:end] = z.detach().cpu().numpy().astype(np.float32, copy=False)

    return projected


def evaluate_matched_memory_autoencoder(
    *,
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    matched_k: int,
    top_k: int,
    device: str,
    query_batch_size: int,
    doc_chunk_size: int,
    fit_max_docs: int,
    seed: int,
    steps: int,
    train_batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    architecture: str,
    normalize_latents: bool,
) -> tuple[list[list[int]], dict[str, float], dict]:
    compressed_dim = min(int(2 * matched_k), int(doc_embeddings.shape[1]))

    fit_start = time.time()
    model, fit_meta = fit_dense_autoencoder(
        doc_embeddings,
        output_dim=compressed_dim,
        fit_max_docs=fit_max_docs,
        seed=seed,
        device=device,
        steps=steps,
        batch_size=train_batch_size,
        lr=lr,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        architecture=architecture,
    )
    fit_time = time.time() - fit_start

    doc_project_start = time.time()
    projected_docs = encode_dense_autoencoder_embeddings(
        model,
        doc_embeddings,
        device=device,
        normalize=normalize_latents,
    )
    doc_project_time = time.time() - doc_project_start

    query_project_start = time.time()
    projected_queries = encode_dense_autoencoder_embeddings(
        model,
        query_embeddings,
        device=device,
        normalize=normalize_latents,
    )
    query_project_time = time.time() - query_project_start

    search_start = time.time()
    rankings = exact_dense_rankings(
        projected_queries,
        projected_docs,
        top_k=top_k,
        device=device,
        query_batch_size=query_batch_size,
        doc_chunk_size=doc_chunk_size,
    )
    search_time = time.time() - search_start

    timings = {
        "fit_time_s": fit_time,
        "doc_project_time_s": doc_project_time,
        "query_project_time_s": query_project_time,
        "search_time_s": search_time,
        "total_time_s": fit_time + doc_project_time + query_project_time + search_time,
    }
    meta = {
        "matched_k": int(matched_k),
        "compressed_dim": int(compressed_dim),
        "normalize_latents": bool(normalize_latents),
        **fit_meta,
    }
    return rankings, timings, meta


@torch.no_grad()
def encode_trained_dense_autoencoder_embeddings(
    model,
    embeddings: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    output_dim = int(model.bottleneck_dim)
    projected = np.empty((len(embeddings), output_dim), dtype=np.float32)
    model = model.to(device)
    model.eval()

    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        batch_np = np.array(embeddings[start:end], dtype=np.float32, copy=True)
        batch = torch.from_numpy(batch_np).to(device)
        z = model.encode(batch)
        projected[start:end] = z.detach().cpu().numpy().astype(np.float32, copy=False)

    return projected


def evaluate_trained_dense_autoencoder(
    *,
    checkpoint_path: str,
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    top_k: int,
    device: str,
    query_batch_size: int,
    doc_chunk_size: int,
    encode_batch_size: int,
) -> tuple[list[list[int]], dict[str, float], dict]:
    from dense_autoencoder import load_dense_autoencoder_checkpoint

    load_start = time.time()
    model, checkpoint = load_dense_autoencoder_checkpoint(checkpoint_path, device=device)
    model_load_time = time.time() - load_start

    doc_project_start = time.time()
    projected_docs = encode_trained_dense_autoencoder_embeddings(
        model,
        doc_embeddings,
        device=device,
        batch_size=encode_batch_size,
    )
    doc_project_time = time.time() - doc_project_start

    query_project_start = time.time()
    projected_queries = encode_trained_dense_autoencoder_embeddings(
        model,
        query_embeddings,
        device=device,
        batch_size=encode_batch_size,
    )
    query_project_time = time.time() - query_project_start

    search_start = time.time()
    rankings = exact_dense_rankings(
        projected_queries,
        projected_docs,
        top_k=top_k,
        device=device,
        query_batch_size=query_batch_size,
        doc_chunk_size=doc_chunk_size,
    )
    search_time = time.time() - search_start

    timings = {
        "model_load_time_s": model_load_time,
        "doc_project_time_s": doc_project_time,
        "query_project_time_s": query_project_time,
        "search_time_s": search_time,
        "total_time_s": model_load_time + doc_project_time + query_project_time + search_time,
    }
    raw_config = checkpoint.get("dense_ae_config", {})
    bottleneck_dim = int(getattr(model, "bottleneck_dim"))
    meta = {
        "checkpoint_path": str(checkpoint_path),
        "input_dim": int(getattr(model, "input_dim")),
        "bottleneck_dim": bottleneck_dim,
        "matched_sparse_k": bottleneck_dim // 2 if bottleneck_dim % 2 == 0 else None,
        "memory_rule": "dense_bottleneck_dim == 2 * sparse_k",
        "hidden_dim": int(getattr(model, "hidden_dim")),
        "normalize_latents": bool(getattr(model, "normalize_latents", True)),
        "training_stage": checkpoint.get("training_stage"),
        "training_source": checkpoint.get("training_source", {}),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "config": raw_config,
    }
    return rankings, timings, meta


def compute_metrics(rankings: list[list[int]], relevance: list[set[int]]) -> dict[str, float]:
    return {
        "recall@1": recall_at_k(rankings, relevance, 1),
        "recall@5": recall_at_k(rankings, relevance, 5),
        "recall@10": recall_at_k(rankings, relevance, 10),
        "recall@100": recall_at_k(rankings, relevance, 100),
        "hit_rate@1": hit_rate_at_k(rankings, relevance, 1),
        "hit_rate@5": hit_rate_at_k(rankings, relevance, 5),
        "hit_rate@10": hit_rate_at_k(rankings, relevance, 10),
        "hit_rate@100": hit_rate_at_k(rankings, relevance, 100),
        "mrr": mrr(rankings, relevance),
        "ndcg@10": ndcg_at_k(rankings, relevance, 10),
    }


def evaluate_sparse(
    *,
    checkpoint_path: str,
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    top_k: int,
    device: str,
    batch_size: int,
    scoring: str,
    sparse_k: Optional[int] = None,
) -> tuple[
    list[dict[int, float]],
    list[dict[int, float]],
    list[list[int]],
    dict,
    dict[str, float],
    dict[str, int],
]:
    load_start = time.time()
    encoder = SparseEncoder(
        checkpoint_path,
        device=device,
        batch_size=batch_size,
        override_k=sparse_k,
    )
    model_load_time = time.time() - load_start

    doc_encode_start = time.time()
    sparse_docs = encoder.encode(doc_embeddings, batch_size=batch_size)
    doc_encode_time = time.time() - doc_encode_start

    query_encode_start = time.time()
    sparse_queries = encoder.encode(query_embeddings, batch_size=batch_size)
    query_encode_time = time.time() - query_encode_start

    index_start = time.time()
    index = SimpleInvertedIndex()
    index.add_batch(sparse_docs)
    index_build_time = time.time() - index_start

    search_start = time.time()
    rankings = []
    for query in sparse_queries:
        results = index.search(query, top_k=top_k, scoring=scoring)
        rankings.append([doc_id for doc_id, _ in results])
    search_time = time.time() - search_start

    timings = {
        "model_load_time_s": model_load_time,
        "doc_encode_time_s": doc_encode_time,
        "query_encode_time_s": query_encode_time,
        "index_build_time_s": index_build_time,
        "search_time_s": search_time,
        "total_time_s": model_load_time + doc_encode_time + query_encode_time + index_build_time + search_time,
    }

    sparse_meta = {
        "k": encoder.active_k,
        "checkpoint_default_k": encoder.checkpoint_k,
    }

    return sparse_queries, sparse_docs, rankings, index.stats(), timings, sparse_meta


def build_torch_sparse_doc_matrix(
    sparse_docs: list[dict[int, float]],
    *,
    vocab_size: int,
    device: str,
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
    device: str,
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


def retrieve_with_torch_sparse(
    *,
    sparse_queries: list[dict[int, float]],
    sparse_docs: list[dict[int, float]],
    vocab_size: int,
    top_k: int,
    device: str,
    query_batch_size: int,
    scoring: str,
    filter_positive_scores: bool = True,
) -> tuple[list[list[int]], dict[str, float], dict]:
    if scoring not in {"dot", "cosine"}:
        raise ValueError("torch_sparse backend currently supports only dot and cosine scoring.")

    index_start = time.time()
    doc_matrix, doc_norms = build_torch_sparse_doc_matrix(
        sparse_docs,
        vocab_size=vocab_size,
        device=device,
    )
    index_build_time = time.time() - index_start

    search_start = time.time()
    rankings: list[list[int]] = []
    k = min(top_k, len(sparse_docs))
    for start in range(0, len(sparse_queries), query_batch_size):
        end = min(start + query_batch_size, len(sparse_queries))
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
            if filter_positive_scores:
                positive_mask = col_scores > 0
                if not torch.any(positive_mask):
                    rankings.append([])
                    continue
                candidate_indices = torch.nonzero(positive_mask, as_tuple=False).squeeze(-1)
                candidate_scores = col_scores[candidate_indices]
                take_k = min(k, candidate_scores.shape[0])
                _, order = torch.topk(candidate_scores, k=take_k, dim=0)
                rankings.append(candidate_indices[order].cpu().tolist())
            else:
                take_k = min(k, col_scores.shape[0])
                _, order = torch.topk(col_scores, k=take_k, dim=0)
                rankings.append(order.cpu().tolist())
    search_time = time.time() - search_start

    timings = {
        "index_build_time_s": index_build_time,
        "search_time_s": search_time,
    }
    return rankings, timings, compute_sparse_index_stats(sparse_docs)


def evaluate_sparse_with_torch_sparse(
    *,
    checkpoint_path: str,
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    top_k: int,
    device: str,
    batch_size: int,
    query_batch_size: int,
    scoring: str,
    sparse_k: Optional[int] = None,
) -> tuple[
    list[dict[int, float]],
    list[dict[int, float]],
    list[list[int]],
    dict,
    dict[str, float],
    dict[str, int],
]:
    load_start = time.time()
    encoder = SparseEncoder(
        checkpoint_path,
        device=device,
        batch_size=batch_size,
        override_k=sparse_k,
    )
    model_load_time = time.time() - load_start

    doc_encode_start = time.time()
    sparse_docs = encoder.encode(doc_embeddings, batch_size=batch_size)
    doc_encode_time = time.time() - doc_encode_start

    query_encode_start = time.time()
    sparse_queries = encoder.encode(query_embeddings, batch_size=batch_size)
    query_encode_time = time.time() - query_encode_start

    rankings, backend_timings, index_stats = retrieve_with_torch_sparse(
        sparse_queries=sparse_queries,
        sparse_docs=sparse_docs,
        vocab_size=int(encoder.sae.config.dict_size),
        top_k=top_k,
        device=device,
        query_batch_size=query_batch_size,
        scoring=scoring,
        filter_positive_scores=True,
    )

    timings = {
        "model_load_time_s": model_load_time,
        "doc_encode_time_s": doc_encode_time,
        "query_encode_time_s": query_encode_time,
        "index_build_time_s": backend_timings["index_build_time_s"],
        "search_time_s": backend_timings["search_time_s"],
        "total_time_s": (
            model_load_time
            + doc_encode_time
            + query_encode_time
            + backend_timings["index_build_time_s"]
            + backend_timings["search_time_s"]
        ),
    }
    sparse_meta = {
        "k": encoder.active_k,
        "checkpoint_default_k": encoder.checkpoint_k,
    }
    return sparse_queries, sparse_docs, rankings, index_stats, timings, sparse_meta


def dense_embeddings_to_topk_sparse(
    embeddings: np.ndarray,
    *,
    k: int,
) -> list[dict[int, float]]:
    if k <= 0:
        raise ValueError("Top-k must be positive.")

    num_rows, dim = embeddings.shape
    take_k = min(k, dim)
    sparse_vecs: list[dict[int, float]] = []
    for row_idx in range(num_rows):
        row = np.asarray(embeddings[row_idx], dtype=np.float32)
        abs_row = np.abs(row)
        topk_idx = np.argpartition(abs_row, -take_k)[-take_k:]
        topk_idx = topk_idx[np.argsort(abs_row[topk_idx])[::-1]]
        vec = {}
        for feature_id in topk_idx.tolist():
            weight = float(row[feature_id])
            if weight != 0.0:
                vec[int(feature_id)] = weight
        sparse_vecs.append(vec)
    return sparse_vecs


def evaluate_raw_topk_dense_with_torch_sparse(
    *,
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    raw_k: int,
    top_k: int,
    device: str,
    query_batch_size: int,
    scoring: str,
) -> tuple[
    list[dict[int, float]],
    list[dict[int, float]],
    list[list[int]],
    dict,
    dict[str, float],
    dict[str, int],
]:
    if scoring not in {"dot", "cosine"}:
        raise ValueError("Raw TopK dense baseline currently supports only dot and cosine scoring.")

    doc_encode_start = time.time()
    sparse_docs = dense_embeddings_to_topk_sparse(doc_embeddings, k=raw_k)
    doc_encode_time = time.time() - doc_encode_start

    query_encode_start = time.time()
    sparse_queries = dense_embeddings_to_topk_sparse(query_embeddings, k=raw_k)
    query_encode_time = time.time() - query_encode_start

    rankings, backend_timings, index_stats = retrieve_with_torch_sparse(
        sparse_queries=sparse_queries,
        sparse_docs=sparse_docs,
        vocab_size=int(doc_embeddings.shape[1]),
        top_k=top_k,
        device=device,
        query_batch_size=query_batch_size,
        scoring=scoring,
        filter_positive_scores=False,
    )

    timings = {
        "model_load_time_s": 0.0,
        "doc_encode_time_s": doc_encode_time,
        "query_encode_time_s": query_encode_time,
        "index_build_time_s": backend_timings["index_build_time_s"],
        "search_time_s": backend_timings["search_time_s"],
        "total_time_s": (
            doc_encode_time
            + query_encode_time
            + backend_timings["index_build_time_s"]
            + backend_timings["search_time_s"]
        ),
    }
    sparse_meta = {
        "k": int(raw_k),
        "checkpoint_default_k": int(raw_k),
    }
    return sparse_queries, sparse_docs, rankings, index_stats, timings, sparse_meta


def hit_rate_at_k(rankings: list[list[int]], relevance: list[set[int]], k: int) -> float:
    if not rankings:
        return 0.0
    hits = 0
    for ranked, rel in zip(rankings, relevance):
        top_k = ranked[:k]
        if any(doc_id in rel for doc_id in top_k):
            hits += 1
    return hits / len(rankings)


def print_metrics(label: str, metrics: dict[str, float]):
    print(f"\n{label}:")
    for key in METRIC_KEYS:
        print(f"  {key:<10s} {metrics[key]:.4f}")


def parse_positive_int_values(
    raw_values: list[int],
    *,
    parser: argparse.ArgumentParser,
    label: str,
) -> list[int]:
    sparse_k_values = []
    seen = set()
    for value in raw_values:
        if value <= 0:
            parser.error(f"{label} values must be positive integers.")
        if value not in seen:
            sparse_k_values.append(value)
            seen.add(value)
    return sparse_k_values


def parse_sparse_k_values(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[int]:
    if args.sparse_k is not None and args.sparse_k_values:
        parser.error("Use either --sparse_k or --sparse_k_values, not both.")

    raw_values: list[int] = []
    if args.sparse_k is not None:
        raw_values = [args.sparse_k]
    elif args.sparse_k_values:
        raw_values = args.sparse_k_values

    return parse_positive_int_values(raw_values, parser=parser, label="Sparse k")


def inspect_trained_dense_autoencoder_memory(checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("dense_ae_config") or checkpoint.get("config") or {}
    bottleneck_dim = config.get("bottleneck_dim")
    if bottleneck_dim is None:
        state_dict = checkpoint.get("dense_ae_state_dict") or checkpoint.get("model_state_dict") or {}
        if "encoder.weight" in state_dict:
            bottleneck_dim = int(state_dict["encoder.weight"].shape[0])
        elif "encoder.2.weight" in state_dict:
            bottleneck_dim = int(state_dict["encoder.2.weight"].shape[0])
    if bottleneck_dim is None:
        raise KeyError(f"Could not infer bottleneck_dim from dense AE checkpoint: {checkpoint_path}")

    bottleneck_dim = int(bottleneck_dim)
    matched_sparse_k = bottleneck_dim // 2 if bottleneck_dim % 2 == 0 else None
    return {
        "checkpoint_path": str(checkpoint_path),
        "bottleneck_dim": bottleneck_dim,
        "matched_sparse_k": matched_sparse_k,
        "memory_rule": "dense_bottleneck_dim == 2 * sparse_k",
    }


def select_matched_memory_k_values(
    *,
    source: str,
    metric: str,
    sparse_results_by_k: dict[str, dict],
    sparse_primary_meta: dict[str, int],
) -> tuple[list[int], dict]:
    available_ks = [int(k) for k in sparse_results_by_k.keys()]
    primary_k = int(sparse_primary_meta["k"])

    if source == "all_sparse":
        target_ks = available_ks or [primary_k]
        return target_ks, {
            "source": source,
            "target_k_values": target_ks,
            "available_sparse_k_values": available_ks,
            "primary_sparse_k": primary_k,
        }

    if source == "primary_sparse":
        return [primary_k], {
            "source": source,
            "target_k_values": [primary_k],
            "available_sparse_k_values": available_ks,
            "primary_sparse_k": primary_k,
        }

    if source == "best_sparse":
        if not sparse_results_by_k:
            raise RuntimeError("Cannot select best sparse k because no sparse results are available.")
        missing_metric_ks = [k for k, result in sparse_results_by_k.items() if metric not in result]
        if missing_metric_ks:
            raise RuntimeError(
                f"Cannot select best sparse k by {metric}; missing metric for k={missing_metric_ks}."
            )
        best_key, best_result = max(
            sparse_results_by_k.items(),
            key=lambda item: (float(item[1][metric]), -int(item[0])),
        )
        best_k = int(best_key)
        best_score = float(best_result[metric])
        return [best_k], {
            "source": source,
            "metric": metric,
            "target_k_values": [best_k],
            "available_sparse_k_values": available_ks,
            "primary_sparse_k": primary_k,
            "selected_sparse_k": best_k,
            "selected_sparse_metric": metric,
            "selected_sparse_metric_value": best_score,
        }

    raise ValueError(f"Unknown matched-memory k source: {source}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark a trained SAE on local M-BEIR data")
    parser.add_argument("--dataset", required=True, help="Dataset/task name, e.g. cirr_task7")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--mbeir_root", type=str, default="./M-BEIR")
    parser.add_argument("--candidate_scope", choices=["local", "global"], default="local")
    parser.add_argument("--query_path", type=str, default=None)
    parser.add_argument("--cand_pool_path", type=str, default=None)
    parser.add_argument("--qrels_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None, help="Qwen model name/path for vLLM encoding")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/sae_final.pt")
    parser.add_argument("--dense_cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--embedding_dim", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64, help="vLLM embedding batch size")
    parser.add_argument(
        "--skip_cache_validation",
        action="store_true",
        help="Trust the dense cache as-is; skip metadata checks and never re-encode (read-only).",
    )
    parser.add_argument("--image_workers", type=int, default=16)
    parser.add_argument("--sparse_batch_size", type=int, default=256)
    parser.add_argument(
        "--sparse_k",
        type=int,
        default=None,
        help="Override the checkpoint inference sparsity with a single active-feature count.",
    )
    parser.add_argument(
        "--sparse_k_values",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate sparse retrieval at multiple inference k values in one run.",
    )
    parser.add_argument(
        "--raw_topk_values",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate a no-training raw TopK dense baseline by keeping the top-k dense dimensions by absolute value.",
    )
    parser.add_argument(
        "--matched_memory_dense",
        action="store_true",
        help="Evaluate a PCA dense-compression baseline at matched memory budget, using compressed_dim = 2*k for each sparse k.",
    )
    parser.add_argument(
        "--matched_memory_autoencoder",
        action="store_true",
        help="Evaluate a learned dense autoencoder baseline at matched memory budget, using bottleneck_dim = 2*k for each sparse k.",
    )
    parser.add_argument(
        "--matched_memory_k_source",
        choices=["all_sparse", "primary_sparse", "best_sparse"],
        default="all_sparse",
        help=(
            "Which sparse k values to use for matched-memory dense baselines. "
            "'all_sparse' mirrors the sparse sweep, 'primary_sparse' uses the first sparse result, "
            "and 'best_sparse' selects the best sparse k for this dataset/run."
        ),
    )
    parser.add_argument(
        "--matched_memory_best_metric",
        choices=METRIC_KEYS,
        default="ndcg@10",
        help="Metric used when --matched_memory_k_source=best_sparse.",
    )
    parser.add_argument(
        "--matched_memory_fit_docs",
        type=int,
        default=20000,
        help="Maximum number of candidate embeddings used to fit matched-memory dense baselines.",
    )
    parser.add_argument(
        "--matched_memory_seed",
        type=int,
        default=0,
        help="Random seed for candidate sampling and AE initialization in matched-memory dense baselines.",
    )
    parser.add_argument(
        "--matched_memory_ae_steps",
        type=int,
        default=2000,
        help="Training steps for the matched-memory dense autoencoder baseline.",
    )
    parser.add_argument(
        "--matched_memory_ae_batch_size",
        type=int,
        default=1024,
        help="Training batch size for the matched-memory dense autoencoder baseline.",
    )
    parser.add_argument(
        "--matched_memory_ae_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the matched-memory dense autoencoder baseline.",
    )
    parser.add_argument(
        "--matched_memory_ae_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for the matched-memory dense autoencoder baseline.",
    )
    parser.add_argument(
        "--matched_memory_ae_architecture",
        choices=["linear", "mlp"],
        default="linear",
        help="Architecture for the fitted matched-memory AE. 'linear' is 2048 -> 2k -> 2048; 'mlp' keeps the older hidden-layer variant.",
    )
    parser.add_argument(
        "--matched_memory_ae_hidden_dim",
        type=int,
        default=0,
        help="Hidden dimension for --matched_memory_ae_architecture=mlp; 0 chooses a size from the bottleneck.",
    )
    parser.add_argument(
        "--matched_memory_ae_no_normalize",
        action="store_true",
        help="Do not L2-normalize AE bottleneck vectors before dense retrieval.",
    )
    parser.add_argument(
        "--trained_dense_autoencoder_checkpoint",
        type=str,
        default=None,
        help="Evaluate a separately trained dense-AE checkpoint from run_dense_ae_pipeline.py.",
    )
    parser.add_argument(
        "--trained_dense_autoencoder_batch_size",
        type=int,
        default=8192,
        help="Encoding batch size for --trained_dense_autoencoder_checkpoint.",
    )
    parser.add_argument(
        "--trained_dense_autoencoder_match_sparse_k",
        action="store_true",
        help=(
            "Force sparse and matched-memory baselines to use the sparse k implied by "
            "the trained dense AE bottleneck, using bottleneck_dim = 2*k."
        ),
    )
    parser.add_argument("--query_batch_size", type=int, default=128)
    parser.add_argument("--doc_chunk_size", type=int, default=8192)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--scoring", choices=["dot", "cosine"], default="dot")
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Legacy alias applied to both query and document system prompts when the specific flags are unset.",
    )
    parser.add_argument("--query_instruction", type=str, default=None)
    parser.add_argument("--doc_instruction", type=str, default=None)
    parser.add_argument(
        "--query_text_prefix",
        type=str,
        default=None,
        help="Prefix prepended to query text before embedding. Defaults to a dataset-specific retrieval hint.",
    )
    parser.add_argument("--doc_text_prefix", type=str, default=None)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--overwrite_dense", action="store_true")
    parser.add_argument(
        "--sparse_backend",
        choices=["python", "torch_sparse"],
        default="python",
        help=(
            "Exact sparse retrieval backend. 'python' uses the reference inverted "
            "index; 'torch_sparse' uses native PyTorch CSR tensors and torch.sparse.mm "
            "(the third-party torch-sparse package is not required)."
        ),
    )

    args = parser.parse_args()
    requested_sparse_ks = parse_sparse_k_values(args, parser)
    requested_raw_topk_ks = parse_positive_int_values(
        args.raw_topk_values or [],
        parser=parser,
        label="Raw TopK k",
    )
    trained_dense_autoencoder_memory: Optional[dict] = None
    if args.matched_memory_fit_docs <= 0:
        parser.error("--matched_memory_fit_docs must be positive.")
    if args.matched_memory_autoencoder:
        if args.matched_memory_ae_steps <= 0:
            parser.error("--matched_memory_ae_steps must be positive.")
        if args.matched_memory_ae_batch_size <= 0:
            parser.error("--matched_memory_ae_batch_size must be positive.")
        if args.matched_memory_ae_lr <= 0:
            parser.error("--matched_memory_ae_lr must be positive.")
        if args.matched_memory_ae_weight_decay < 0:
            parser.error("--matched_memory_ae_weight_decay must be non-negative.")
        if args.matched_memory_ae_hidden_dim < 0:
            parser.error("--matched_memory_ae_hidden_dim must be non-negative.")
    if args.trained_dense_autoencoder_checkpoint is not None:
        if args.trained_dense_autoencoder_batch_size <= 0:
            parser.error("--trained_dense_autoencoder_batch_size must be positive.")
        if not Path(args.trained_dense_autoencoder_checkpoint).exists():
            parser.error(
                f"--trained_dense_autoencoder_checkpoint not found: "
                f"{args.trained_dense_autoencoder_checkpoint}"
            )
        trained_dense_autoencoder_memory = inspect_trained_dense_autoencoder_memory(
            args.trained_dense_autoencoder_checkpoint
        )
        trained_match_k = trained_dense_autoencoder_memory["matched_sparse_k"]
        if args.trained_dense_autoencoder_match_sparse_k:
            if trained_match_k is None:
                parser.error(
                    "Cannot memory-match sparse k because the trained dense AE bottleneck "
                    f"dimension is odd: {trained_dense_autoencoder_memory['bottleneck_dim']}"
                )
            requested_sparse_ks = [int(trained_match_k)]
            args.matched_memory_k_source = "all_sparse"
        elif trained_match_k is not None and requested_sparse_ks and int(trained_match_k) not in requested_sparse_ks:
            print(
                "Warning: trained dense AE bottleneck implies sparse "
                f"k={int(trained_match_k)} under dim=2*k, but requested sparse k values are "
                f"{requested_sparse_ks}. Add --trained_dense_autoencoder_match_sparse_k for a strict "
                "same-memory comparison."
            )

    run_name = f"{args.dataset}_{args.split}"
    mbeir_root = Path(args.mbeir_root)
    dense_cache_dir = Path(args.dense_cache_dir) if args.dense_cache_dir else Path("./benchmark_cache") / run_name
    output_dir = Path(args.output_dir) if args.output_dir else Path("./benchmark_runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    query_file, cand_file, qrels_file = resolve_mbeir_paths(
        mbeir_root=mbeir_root,
        dataset=args.dataset,
        split=args.split,
        candidate_scope=args.candidate_scope,
        query_path=args.query_path,
        cand_pool_path=args.cand_pool_path,
        qrels_path=args.qrels_path,
    )

    print("M-BEIR inputs")
    print(f"  Queries:    {query_file}")
    print(f"  Candidates: {cand_file}")
    print(f"  Qrels:      {qrels_file}")

    query_rows = load_jsonl(query_file, max_items=args.max_queries)
    doc_rows = load_jsonl(cand_file, max_items=args.max_docs)
    qrels = load_qrels(qrels_file)
    expected_query_ids = [record_id(record, is_query=True) for record in query_rows]
    expected_doc_ids = [record_id(record, is_query=False) for record in doc_rows]

    print(f"  Loaded {len(query_rows):,} queries and {len(doc_rows):,} candidates")

    if not Path(args.checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    query_instruction = args.query_instruction or args.instruction or DEFAULT_EMBED_SYSTEM_INSTRUCTION
    doc_instruction = args.doc_instruction or args.instruction or DEFAULT_EMBED_SYSTEM_INSTRUCTION
    query_text_prefix = (
        default_query_text_prefix(args.dataset)
        if args.query_text_prefix is None
        else args.query_text_prefix
    )
    doc_text_prefix = "" if args.doc_text_prefix is None else args.doc_text_prefix

    expected_query_meta = build_cache_metadata(
        model_name=args.model_path,
        count=len(query_rows),
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        system_instruction=query_instruction,
        text_prefix=query_text_prefix,
        kind="query",
        ids=expected_query_ids,
    )
    expected_doc_meta = build_cache_metadata(
        model_name=args.model_path,
        count=len(doc_rows),
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        system_instruction=doc_instruction,
        text_prefix=doc_text_prefix,
        kind="doc",
        ids=expected_doc_ids,
    )

    query_cache_valid, query_cache_errors = inspect_dense_cache(dense_cache_dir, "query", expected_query_meta)
    doc_cache_valid, doc_cache_errors = inspect_dense_cache(dense_cache_dir, "doc", expected_doc_meta)
    need_dense = args.overwrite_dense or not (query_cache_valid and doc_cache_valid)
    if args.skip_cache_validation:
        if args.overwrite_dense:
            raise ValueError("--skip_cache_validation is incompatible with --overwrite_dense")
        need_dense = False
        print(f"\n[skip_cache_validation] Trusting cache at {dense_cache_dir} as-is (read-only).")

    if need_dense and args.model_path is None:
        reasons = []
        if query_cache_errors:
            reasons.append(f"query cache: {'; '.join(query_cache_errors)}")
        if doc_cache_errors:
            reasons.append(f"doc cache: {'; '.join(doc_cache_errors)}")
        reason_text = " | ".join(reasons) if reasons else "cache is missing"
        parser.error(
            "--model_path is required because dense embeddings are missing or incompatible "
            f"({reason_text})"
        )

    dense_query_encode_time = 0.0
    dense_doc_encode_time = 0.0
    query_cache_reused = False
    doc_cache_reused = False

    if need_dense or args.overwrite_dense:
        print(f"\nLoading {args.model_path} via vLLM for benchmark encoding...")
        encoder = VLLMEncoder(
            model_name=args.model_path,
            batch_size=args.batch_size,
            image_workers=args.image_workers,
        )
        query_encode_start = time.time()
        query_embeddings, query_ids, query_cache_reused = encoder.encode(
            records=query_rows,
            mbeir_root=mbeir_root,
            is_query=True,
            cache_dir=dense_cache_dir,
            prefix="query",
            embedding_dim=args.embedding_dim,
            overwrite=args.overwrite_dense,
            system_instruction=query_instruction,
            text_prefix=query_text_prefix,
            expected_ids=expected_query_ids,
        )
        dense_query_encode_time = 0.0 if query_cache_reused else time.time() - query_encode_start
        doc_encode_start = time.time()
        doc_embeddings, doc_ids, doc_cache_reused = encoder.encode(
            records=doc_rows,
            mbeir_root=mbeir_root,
            is_query=False,
            cache_dir=dense_cache_dir,
            prefix="doc",
            embedding_dim=args.embedding_dim,
            overwrite=args.overwrite_dense,
            system_instruction=doc_instruction,
            text_prefix=doc_text_prefix,
            expected_ids=expected_doc_ids,
        )
        dense_doc_encode_time = 0.0 if doc_cache_reused else time.time() - doc_encode_start
    else:
        print(f"\nReusing dense cache from {dense_cache_dir}")
        query_embeddings, query_ids, _ = load_cached_dense(dense_cache_dir, "query")
        doc_embeddings, doc_ids, _ = load_cached_dense(dense_cache_dir, "doc")
        query_cache_reused = True
        doc_cache_reused = True

    selected_query_indices, eval_query_ids, relevance = build_relevance(query_ids, doc_ids, qrels)
    query_eval_embeddings = query_embeddings[selected_query_indices]

    print(f"  Evaluating {len(eval_query_ids):,} queries with qrels coverage")

    device = resolve_device(args.device)

    print("\nRunning dense baseline...")
    dense_start = time.time()
    dense_rankings = exact_dense_rankings(
        query_eval_embeddings,
        doc_embeddings,
        top_k=args.top_k,
        device=device,
        query_batch_size=args.query_batch_size,
        doc_chunk_size=args.doc_chunk_size,
    )
    dense_time = time.time() - dense_start
    dense_metrics = compute_metrics(dense_rankings, relevance)
    dense_total_time = dense_query_encode_time + dense_doc_encode_time + dense_time

    print("\nRunning sparse evaluation...")
    sparse_results_by_k: dict[str, dict] = {}
    sparse_index_stats_by_k: dict[str, dict] = {}
    sparse_run_paths: dict[str, str] = {}
    sparse_primary_metrics: Optional[dict[str, float]] = None
    sparse_primary_timings: Optional[dict[str, float]] = None
    sparse_primary_index_stats: Optional[dict] = None
    sparse_primary_meta: Optional[dict[str, int]] = None
    sparse_primary_run_path: Optional[Path] = None

    requested_labels = requested_sparse_ks if requested_sparse_ks else ["checkpoint default"]
    print(f"  Requested sparse k values: {requested_labels}")
    print(f"  Sparse backend: {args.sparse_backend}")
    sparse_implementation = (
        "python_inverted_index"
        if args.sparse_backend == "python"
        else "torch_sparse_exact"
    )
    if trained_dense_autoencoder_memory is not None:
        print(
            "  Trained dense AE memory: "
            f"bottleneck_dim={trained_dense_autoencoder_memory['bottleneck_dim']} "
            f"-> matched sparse k={trained_dense_autoencoder_memory['matched_sparse_k']}"
        )

    sparse_eval_plan: list[Optional[int]] = requested_sparse_ks if requested_sparse_ks else [None]
    for sparse_k in sparse_eval_plan:
        label = f"k={sparse_k}" if sparse_k is not None else "checkpoint default"
        print(f"\n  Sparse evaluation ({label})...")
        if args.sparse_backend == "python":
            (
                _,
                _,
                sparse_rankings,
                sparse_index_stats,
                sparse_timings,
                sparse_meta,
            ) = evaluate_sparse(
                checkpoint_path=args.checkpoint_path,
                query_embeddings=query_eval_embeddings,
                doc_embeddings=doc_embeddings,
                top_k=args.top_k,
                device=device,
                batch_size=args.sparse_batch_size,
                scoring=args.scoring,
                sparse_k=sparse_k,
            )
        else:
            (
                _,
                _,
                sparse_rankings,
                sparse_index_stats,
                sparse_timings,
                sparse_meta,
            ) = evaluate_sparse_with_torch_sparse(
                checkpoint_path=args.checkpoint_path,
                query_embeddings=query_eval_embeddings,
                doc_embeddings=doc_embeddings,
                top_k=args.top_k,
                device=device,
                batch_size=args.sparse_batch_size,
                query_batch_size=args.query_batch_size,
                scoring=args.scoring,
                sparse_k=sparse_k,
            )
        sparse_metrics = compute_metrics(sparse_rankings, relevance)
        actual_k = int(sparse_meta["k"])
        actual_k_key = str(actual_k)
        run_path = output_dir / (
            "sparse.run"
            if sparse_primary_metrics is None
            else f"sparse_k{actual_k}.run"
        )
        run_tag = "sparse" if args.sparse_backend == "python" else "torch_sparse"
        write_trec_run(
            run_path,
            eval_query_ids,
            sparse_rankings,
            doc_ids,
            tag=f"{run_tag}_k{actual_k}",
        )

        sparse_result = {
            **sparse_metrics,
            **sparse_timings,
            **sparse_meta,
            "implementation": sparse_implementation,
            "run_path": str(run_path),
        }
        if sparse_k is not None:
            sparse_result["requested_k"] = int(sparse_k)

        sparse_results_by_k[actual_k_key] = sparse_result
        sparse_index_stats_by_k[actual_k_key] = sparse_index_stats
        sparse_run_paths[actual_k_key] = str(run_path)

        if sparse_primary_metrics is None:
            sparse_primary_metrics = sparse_metrics
            sparse_primary_timings = sparse_timings
            sparse_primary_index_stats = sparse_index_stats
            sparse_primary_meta = sparse_meta
            sparse_primary_run_path = run_path

    raw_topk_results_by_k: dict[str, dict] = {}
    raw_topk_index_stats_by_k: dict[str, dict] = {}
    raw_topk_run_paths: dict[str, str] = {}
    raw_topk_primary_metrics: Optional[dict[str, float]] = None
    raw_topk_primary_timings: Optional[dict[str, float]] = None
    raw_topk_primary_index_stats: Optional[dict] = None
    raw_topk_primary_meta: Optional[dict[str, int]] = None
    raw_topk_primary_run_path: Optional[Path] = None

    matched_memory_dense_results_by_k: dict[str, dict] = {}
    matched_memory_dense_primary_metrics: Optional[dict[str, float]] = None
    matched_memory_dense_primary_timings: Optional[dict[str, float]] = None
    matched_memory_dense_primary_meta: Optional[dict] = None
    matched_memory_dense_primary_run_path: Optional[Path] = None

    matched_memory_autoencoder_results_by_k: dict[str, dict] = {}
    matched_memory_autoencoder_primary_metrics: Optional[dict[str, float]] = None
    matched_memory_autoencoder_primary_timings: Optional[dict[str, float]] = None
    matched_memory_autoencoder_primary_meta: Optional[dict] = None
    matched_memory_autoencoder_primary_run_path: Optional[Path] = None

    trained_dense_autoencoder_metrics: Optional[dict[str, float]] = None
    trained_dense_autoencoder_timings: Optional[dict[str, float]] = None
    trained_dense_autoencoder_meta: Optional[dict] = None
    trained_dense_autoencoder_run_path: Optional[Path] = None

    if sparse_primary_meta is None:
        raise RuntimeError("Sparse evaluation did not produce any results.")

    matched_memory_target_ks, matched_memory_k_selection = select_matched_memory_k_values(
        source=args.matched_memory_k_source,
        metric=args.matched_memory_best_metric,
        sparse_results_by_k=sparse_results_by_k,
        sparse_primary_meta=sparse_primary_meta,
    )
    if args.matched_memory_dense or args.matched_memory_autoencoder:
        print("\nMatched-memory k selection")
        print(f"  Source: {args.matched_memory_k_source}")
        print(f"  Target k values: {matched_memory_target_ks}")
        if args.matched_memory_k_source == "best_sparse":
            print(
                f"  Selected sparse k={matched_memory_k_selection['selected_sparse_k']} "
                f"by {matched_memory_k_selection['selected_sparse_metric']}="
                f"{matched_memory_k_selection['selected_sparse_metric_value']:.4f}"
            )

    if args.matched_memory_dense:
        print("\nRunning matched-memory dense baseline...")
        for matched_k in matched_memory_target_ks:
            print(f"\n  Matched-memory dense evaluation (k={matched_k}, dim={2 * matched_k})...")
            dense_mm_rankings, dense_mm_timings, dense_mm_meta = evaluate_matched_memory_dense(
                query_embeddings=query_eval_embeddings,
                doc_embeddings=doc_embeddings,
                matched_k=int(matched_k),
                top_k=args.top_k,
                device=device,
                query_batch_size=args.query_batch_size,
                doc_chunk_size=args.doc_chunk_size,
                fit_max_docs=args.matched_memory_fit_docs,
                seed=args.matched_memory_seed,
            )
            dense_mm_metrics = compute_metrics(dense_mm_rankings, relevance)
            matched_k_key = str(int(matched_k))
            run_path = output_dir / (
                "matched_memory_dense.run"
                if matched_memory_dense_primary_metrics is None
                else f"matched_memory_dense_k{matched_k}.run"
            )
            write_trec_run(
                run_path,
                eval_query_ids,
                dense_mm_rankings,
                doc_ids,
                tag=f"matched_memory_dense_k{matched_k}",
            )
            dense_mm_result = {
                **dense_mm_metrics,
                **dense_mm_timings,
                **dense_mm_meta,
                "implementation": "dense_pca_matched_memory",
                "run_path": str(run_path),
                "source": "dense_embedding",
                "k_selection_source": args.matched_memory_k_source,
            }
            matched_memory_dense_results_by_k[matched_k_key] = dense_mm_result
            if matched_memory_dense_primary_metrics is None:
                matched_memory_dense_primary_metrics = dense_mm_metrics
                matched_memory_dense_primary_timings = dense_mm_timings
                matched_memory_dense_primary_meta = dense_mm_meta
                matched_memory_dense_primary_run_path = run_path

    if args.matched_memory_autoencoder:
        print("\nRunning matched-memory dense autoencoder baseline...")
        for matched_k in matched_memory_target_ks:
            print(f"\n  Matched-memory AE evaluation (k={matched_k}, bottleneck={2 * matched_k})...")
            dense_ae_rankings, dense_ae_timings, dense_ae_meta = evaluate_matched_memory_autoencoder(
                query_embeddings=query_eval_embeddings,
                doc_embeddings=doc_embeddings,
                matched_k=int(matched_k),
                top_k=args.top_k,
                device=device,
                query_batch_size=args.query_batch_size,
                doc_chunk_size=args.doc_chunk_size,
                fit_max_docs=args.matched_memory_fit_docs,
                seed=args.matched_memory_seed,
                steps=args.matched_memory_ae_steps,
                train_batch_size=args.matched_memory_ae_batch_size,
                lr=args.matched_memory_ae_lr,
                weight_decay=args.matched_memory_ae_weight_decay,
                hidden_dim=args.matched_memory_ae_hidden_dim,
                architecture=args.matched_memory_ae_architecture,
                normalize_latents=not args.matched_memory_ae_no_normalize,
            )
            dense_ae_metrics = compute_metrics(dense_ae_rankings, relevance)
            matched_k_key = str(int(matched_k))
            run_path = output_dir / (
                "matched_memory_autoencoder.run"
                if matched_memory_autoencoder_primary_metrics is None
                else f"matched_memory_autoencoder_k{matched_k}.run"
            )
            write_trec_run(
                run_path,
                eval_query_ids,
                dense_ae_rankings,
                doc_ids,
                tag=f"matched_memory_autoencoder_k{matched_k}",
            )
            dense_ae_result = {
                **dense_ae_metrics,
                **dense_ae_timings,
                **dense_ae_meta,
                "implementation": "dense_autoencoder_matched_memory",
                "run_path": str(run_path),
                "source": "dense_embedding",
                "k_selection_source": args.matched_memory_k_source,
            }
            matched_memory_autoencoder_results_by_k[matched_k_key] = dense_ae_result
            if matched_memory_autoencoder_primary_metrics is None:
                matched_memory_autoencoder_primary_metrics = dense_ae_metrics
                matched_memory_autoencoder_primary_timings = dense_ae_timings
                matched_memory_autoencoder_primary_meta = dense_ae_meta
                matched_memory_autoencoder_primary_run_path = run_path

    if args.trained_dense_autoencoder_checkpoint is not None:
        print("\nRunning trained dense autoencoder checkpoint baseline...")
        dense_trained_rankings, dense_trained_timings, dense_trained_meta = evaluate_trained_dense_autoencoder(
            checkpoint_path=args.trained_dense_autoencoder_checkpoint,
            query_embeddings=query_eval_embeddings,
            doc_embeddings=doc_embeddings,
            top_k=args.top_k,
            device=device,
            query_batch_size=args.query_batch_size,
            doc_chunk_size=args.doc_chunk_size,
            encode_batch_size=args.trained_dense_autoencoder_batch_size,
        )
        trained_dense_autoencoder_metrics = compute_metrics(dense_trained_rankings, relevance)
        trained_dense_autoencoder_timings = dense_trained_timings
        trained_dense_autoencoder_meta = dense_trained_meta
        trained_dense_autoencoder_run_path = output_dir / "trained_dense_autoencoder.run"
        write_trec_run(
            trained_dense_autoencoder_run_path,
            eval_query_ids,
            dense_trained_rankings,
            doc_ids,
            tag="trained_dense_autoencoder",
        )

    if requested_raw_topk_ks:
        print("\nRunning raw TopK dense baseline...")
        print(f"  Requested raw TopK k values: {requested_raw_topk_ks}")
        for raw_k in requested_raw_topk_ks:
            print(f"\n  Raw TopK dense evaluation (k={raw_k})...")
            (
                raw_queries,
                raw_docs,
                raw_rankings,
                raw_index_stats,
                raw_timings,
                raw_meta,
            ) = evaluate_raw_topk_dense_with_torch_sparse(
                query_embeddings=query_eval_embeddings,
                doc_embeddings=doc_embeddings,
                raw_k=raw_k,
                top_k=args.top_k,
                device=device,
                query_batch_size=args.query_batch_size,
                scoring=args.scoring,
            )
            raw_metrics = compute_metrics(raw_rankings, relevance)
            raw_k_key = str(int(raw_meta["k"]))
            run_path = output_dir / (
                "raw_topk_dense.run"
                if raw_topk_primary_metrics is None
                else f"raw_topk_dense_k{raw_k}.run"
            )
            write_trec_run(run_path, eval_query_ids, raw_rankings, doc_ids, tag=f"raw_topk_dense_k{raw_k}")

            raw_result = {
                **raw_metrics,
                **raw_timings,
                **raw_meta,
                "requested_k": int(raw_k),
                "implementation": "raw_topk_dense_torch_sparse_exact",
                "run_path": str(run_path),
                "selection": "topk_abs_signed",
                "source": "dense_embedding",
            }
            raw_topk_results_by_k[raw_k_key] = raw_result
            raw_topk_index_stats_by_k[raw_k_key] = raw_index_stats
            raw_topk_run_paths[raw_k_key] = str(run_path)

            if raw_topk_primary_metrics is None:
                raw_topk_primary_metrics = raw_metrics
                raw_topk_primary_timings = raw_timings
                raw_topk_primary_index_stats = raw_index_stats
                raw_topk_primary_meta = raw_meta
                raw_topk_primary_run_path = run_path

    write_trec_run(output_dir / "dense.run", eval_query_ids, dense_rankings, doc_ids, tag="dense")

    if (
        sparse_primary_metrics is None
        or sparse_primary_timings is None
        or sparse_primary_index_stats is None
        or sparse_primary_meta is None
        or sparse_primary_run_path is None
    ):
        raise RuntimeError("Sparse evaluation did not produce any results.")

    results = {
        "dataset": args.dataset,
        "split": args.split,
        "query_file": str(query_file),
        "cand_pool_file": str(cand_file),
        "qrels_file": str(qrels_file),
        "num_queries_total": len(query_ids),
        "num_queries_evaluated": len(eval_query_ids),
        "num_docs": len(doc_ids),
        "prompting": {
            "query_system_instruction": query_instruction,
            "doc_system_instruction": doc_instruction,
            "query_text_prefix": query_text_prefix,
            "doc_text_prefix": doc_text_prefix,
        },
        "dense": {
            **dense_metrics,
            "query_encode_time_s": dense_query_encode_time,
            "doc_encode_time_s": dense_doc_encode_time,
            "encode_time_s": dense_query_encode_time + dense_doc_encode_time,
            "search_time_s": dense_time,
            "total_time_s": dense_total_time,
            "query_cache_reused": query_cache_reused,
            "doc_cache_reused": doc_cache_reused,
        },
        "sparse": {
            **sparse_primary_metrics,
            **sparse_primary_timings,
            **sparse_primary_meta,
            "implementation": sparse_implementation,
            "run_path": str(sparse_primary_run_path),
        },
        "sparse_by_k": sparse_results_by_k,
        "index_stats": sparse_primary_index_stats,
        "index_stats_by_k": sparse_index_stats_by_k,
        "sparse_suite": {
            "requested_k_values": requested_sparse_ks,
            "primary_k": int(sparse_primary_meta["k"]),
            "checkpoint_default_k": int(sparse_primary_meta["checkpoint_default_k"]),
        },
        "raw_topk_dense": (
            {
                **raw_topk_primary_metrics,
                **raw_topk_primary_timings,
                **raw_topk_primary_meta,
                "implementation": "raw_topk_dense_torch_sparse_exact",
                "run_path": str(raw_topk_primary_run_path),
                "selection": "topk_abs_signed",
                "source": "dense_embedding",
            }
            if raw_topk_primary_metrics is not None
            and raw_topk_primary_timings is not None
            and raw_topk_primary_meta is not None
            and raw_topk_primary_run_path is not None
            else None
        ),
        "raw_topk_dense_by_k": raw_topk_results_by_k,
        "raw_topk_index_stats": raw_topk_primary_index_stats,
        "raw_topk_index_stats_by_k": raw_topk_index_stats_by_k,
        "raw_topk_suite": {
            "requested_k_values": requested_raw_topk_ks,
            "primary_k": (
                int(raw_topk_primary_meta["k"])
                if raw_topk_primary_meta is not None
                else None
            ),
        },
        "matched_memory_suite": {
            **matched_memory_k_selection,
            "pca_enabled": bool(args.matched_memory_dense),
            "autoencoder_enabled": bool(args.matched_memory_autoencoder),
        },
        "trained_dense_autoencoder_memory": trained_dense_autoencoder_memory,
        "matched_memory_dense": (
            {
                **matched_memory_dense_primary_metrics,
                **matched_memory_dense_primary_timings,
                **matched_memory_dense_primary_meta,
                "implementation": "dense_pca_matched_memory",
                "run_path": str(matched_memory_dense_primary_run_path),
                "source": "dense_embedding",
                "k_selection_source": args.matched_memory_k_source,
            }
            if matched_memory_dense_primary_metrics is not None
            and matched_memory_dense_primary_timings is not None
            and matched_memory_dense_primary_meta is not None
            and matched_memory_dense_primary_run_path is not None
            else None
        ),
        "matched_memory_dense_by_k": matched_memory_dense_results_by_k,
        "matched_memory_autoencoder": (
            {
                **matched_memory_autoencoder_primary_metrics,
                **matched_memory_autoencoder_primary_timings,
                **matched_memory_autoencoder_primary_meta,
                "implementation": "dense_autoencoder_matched_memory",
                "run_path": str(matched_memory_autoencoder_primary_run_path),
                "source": "dense_embedding",
                "k_selection_source": args.matched_memory_k_source,
            }
            if matched_memory_autoencoder_primary_metrics is not None
            and matched_memory_autoencoder_primary_timings is not None
            and matched_memory_autoencoder_primary_meta is not None
            and matched_memory_autoencoder_primary_run_path is not None
            else None
        ),
        "matched_memory_autoencoder_by_k": matched_memory_autoencoder_results_by_k,
        "trained_dense_autoencoder": (
            {
                **trained_dense_autoencoder_metrics,
                **trained_dense_autoencoder_timings,
                **trained_dense_autoencoder_meta,
                "implementation": "trained_dense_autoencoder_pipeline",
                "run_path": str(trained_dense_autoencoder_run_path),
                "source": "dense_autoencoder_checkpoint",
            }
            if trained_dense_autoencoder_metrics is not None
            and trained_dense_autoencoder_timings is not None
            and trained_dense_autoencoder_meta is not None
            and trained_dense_autoencoder_run_path is not None
            else None
        ),
        "paths": {
            "dense_cache_dir": str(dense_cache_dir),
            "output_dir": str(output_dir),
            "dense_run": str(output_dir / "dense.run"),
            "sparse_run": str(sparse_primary_run_path),
            "sparse_run_by_k": sparse_run_paths,
            "raw_topk_dense_run": str(raw_topk_primary_run_path) if raw_topk_primary_run_path else None,
            "raw_topk_dense_run_by_k": raw_topk_run_paths,
            "matched_memory_dense_run": (
                str(matched_memory_dense_primary_run_path)
                if matched_memory_dense_primary_run_path
                else None
            ),
            "matched_memory_autoencoder_run": (
                str(matched_memory_autoencoder_primary_run_path)
                if matched_memory_autoencoder_primary_run_path
                else None
            ),
            "trained_dense_autoencoder_run": (
                str(trained_dense_autoencoder_run_path)
                if trained_dense_autoencoder_run_path
                else None
            ),
        },
    }

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_metrics("Dense", dense_metrics)
    print_metrics(f"Sparse (k={sparse_primary_meta['k']})", sparse_primary_metrics)
    if len(sparse_results_by_k) > 1:
        print("\nSparse k sweep:")
        for sparse_k_key, sparse_result in sparse_results_by_k.items():
            print(
                f"  k={int(sparse_k_key):>3d} "
                f"hit_rate@5={sparse_result['hit_rate@5']:.4f} "
                f"recall@10={sparse_result['recall@10']:.4f} "
                f"ndcg@10={sparse_result['ndcg@10']:.4f} "
                f"search={sparse_result['search_time_s']:.1f}s"
            )
    if raw_topk_primary_metrics is not None and raw_topk_primary_meta is not None:
        print_metrics(f"Raw TopK Dense (k={raw_topk_primary_meta['k']})", raw_topk_primary_metrics)
        if len(raw_topk_results_by_k) > 1:
            print("\nRaw TopK dense k sweep:")
            for raw_k_key, raw_result in raw_topk_results_by_k.items():
                print(
                    f"  k={int(raw_k_key):>3d} "
                    f"hit_rate@5={raw_result['hit_rate@5']:.4f} "
                    f"recall@10={raw_result['recall@10']:.4f} "
                    f"ndcg@10={raw_result['ndcg@10']:.4f} "
                    f"search={raw_result['search_time_s']:.1f}s"
                )
    if matched_memory_dense_primary_metrics is not None and matched_memory_dense_primary_meta is not None:
        print_metrics(
            f"Matched-memory Dense (k={matched_memory_dense_primary_meta['matched_k']}, "
            f"dim={matched_memory_dense_primary_meta['compressed_dim']})",
            matched_memory_dense_primary_metrics,
        )
        if len(matched_memory_dense_results_by_k) > 1:
            print("\nMatched-memory dense sweep:")
            for matched_k_key, dense_mm_result in matched_memory_dense_results_by_k.items():
                print(
                    f"  k={int(matched_k_key):>3d} "
                    f"dim={int(dense_mm_result['compressed_dim']):>3d} "
                    f"hit_rate@5={dense_mm_result['hit_rate@5']:.4f} "
                    f"recall@10={dense_mm_result['recall@10']:.4f} "
                    f"ndcg@10={dense_mm_result['ndcg@10']:.4f} "
                    f"search={dense_mm_result['search_time_s']:.1f}s"
                )
    if matched_memory_autoencoder_primary_metrics is not None and matched_memory_autoencoder_primary_meta is not None:
        print_metrics(
            f"Matched-memory Dense AE (k={matched_memory_autoencoder_primary_meta['matched_k']}, "
            f"dim={matched_memory_autoencoder_primary_meta['compressed_dim']})",
            matched_memory_autoencoder_primary_metrics,
        )
        if len(matched_memory_autoencoder_results_by_k) > 1:
            print("\nMatched-memory dense AE sweep:")
            for matched_k_key, dense_ae_result in matched_memory_autoencoder_results_by_k.items():
                print(
                    f"  k={int(matched_k_key):>3d} "
                    f"dim={int(dense_ae_result['compressed_dim']):>3d} "
                    f"hit_rate@5={dense_ae_result['hit_rate@5']:.4f} "
                    f"recall@10={dense_ae_result['recall@10']:.4f} "
                    f"ndcg@10={dense_ae_result['ndcg@10']:.4f} "
                    f"search={dense_ae_result['search_time_s']:.1f}s"
                )
    if trained_dense_autoencoder_metrics is not None and trained_dense_autoencoder_meta is not None:
        trained_label = f"Trained Dense AE (dim={trained_dense_autoencoder_meta['bottleneck_dim']}"
        if trained_dense_autoencoder_meta.get("matched_sparse_k") is not None:
            trained_label += f", matched k={trained_dense_autoencoder_meta['matched_sparse_k']}"
        trained_label += ")"
        print_metrics(
            trained_label,
            trained_dense_autoencoder_metrics,
        )
    print(
        f"\nDense cache reused: query={query_cache_reused} doc={doc_cache_reused}"
    )
    print(
        "Dense timing: "
        f"encode={dense_query_encode_time + dense_doc_encode_time:.1f}s "
        f"search={dense_time:.1f}s total={dense_total_time:.1f}s"
    )
    print(
        "Sparse timing: "
        f"load={sparse_primary_timings['model_load_time_s']:.1f}s "
        f"doc_encode={sparse_primary_timings['doc_encode_time_s']:.1f}s "
        f"query_encode={sparse_primary_timings['query_encode_time_s']:.1f}s "
        f"index={sparse_primary_timings['index_build_time_s']:.1f}s "
        f"search={sparse_primary_timings['search_time_s']:.1f}s "
        f"total={sparse_primary_timings['total_time_s']:.1f}s"
    )
    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
