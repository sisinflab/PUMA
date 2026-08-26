"""
Stage 1: Extract retrieval-format dense embeddings from RZen for local M-BEIR.

This mirrors the query/doc cache layout produced by extract_qwen.py so the
existing PUMA training and benchmarking pipeline can reuse it unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


RZEN_REPO = os.environ.get("RZEN_EMBED_PATH", "/tmp/RzenEmbed")
# Optional: path to an external RZen provider module exposing a `Provider` class.
# Set RZEN_PROVIDER_PATH to enable the provider backend; otherwise the upstream
# `rzen_embed_inference.RzenEmbed` is used (requires RZEN_EMBED_PATH to point at
# a local clone of the RZen repository).
RZEN_PROVIDER_PATH = os.environ.get("RZEN_PROVIDER_PATH", "")
if os.path.isdir(RZEN_REPO):
    sys.path.insert(0, RZEN_REPO)

from benchmark_mbeir import (  # noqa: E402
    DEFAULT_EMBED_SYSTEM_INSTRUCTION,
    build_cache_metadata,
    default_query_text_prefix,
    inspect_dense_cache,
    item_id_text_image,
    load_cached_dense,
    load_jsonl,
    record_id,
    resolve_mbeir_paths,
)


def load_rzen_backend():
    if RZEN_PROVIDER_PATH:
        provider_path = Path(RZEN_PROVIDER_PATH)
        if provider_path.is_file():
            spec = importlib.util.spec_from_file_location("rzen_provider", provider_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load RZen provider from {provider_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module.Provider, getattr(module, "ItemKey", None), "provider"
        raise FileNotFoundError(
            f"RZEN_PROVIDER_PATH was set to '{RZEN_PROVIDER_PATH}' but the file does not exist. "
            "Unset the variable to fall back to the upstream `rzen_embed_inference.RzenEmbed`."
        )

    try:
        from rzen_embed_inference import RzenEmbed  # noqa: E402
    except ImportError as exc:
        raise ImportError(
            "Could not import `rzen_embed_inference.RzenEmbed`. Either install the upstream "
            "RZen repository (and point RZEN_EMBED_PATH at it) or set RZEN_PROVIDER_PATH to "
            "a local provider module exposing a `Provider` class."
        ) from exc

    return RzenEmbed, None, "upstream"


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return x / norms


class RZenEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int,
        device: str,
        attn_implementation: str = "sdpa",
    ):
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self.attn_implementation = attn_implementation
        backend_cls, item_key_cls, backend_kind = load_rzen_backend()
        self.backend_kind = backend_kind
        self.item_key_cls = item_key_cls
        if backend_kind == "provider":
            self.model = backend_cls(
                device=device,
                model_id=model_name,
                attn_implementation=attn_implementation,
            )
            hidden_size = getattr(getattr(self.model, "model", None), "config", None)
            self.embedding_dim = int(getattr(hidden_size, "hidden_size"))
        else:
            self.model = backend_cls(
                model_name=model_name,
                device=device,
                attn_implementation=attn_implementation,
            )
            hidden_size = getattr(getattr(self.model, "base", None), "config", None)
            self.embedding_dim = int(getattr(hidden_size, "hidden_size"))

    def encode_records(
        self,
        *,
        records: list[dict],
        ids: list[str],
        mbeir_root: Path,
        is_query: bool,
        system_instruction: str,
        text_prefix: str,
    ) -> tuple[np.ndarray, list[str]]:
        embeddings = np.memmap(
            Path("/tmp") / f"rzen_tmp_{os.getpid()}_{'query' if is_query else 'doc'}.npy",
            dtype=np.float32,
            mode="w+",
            shape=(len(records), self.embedding_dim),
        )

        for start in range(0, len(records), self.batch_size):
            end = min(start + self.batch_size, len(records))
            batch = records[start:end]
            if self.backend_kind == "provider":
                batch_keys = []
                for record in batch:
                    _, text, image_path = item_id_text_image(
                        record,
                        is_query=is_query,
                        mbeir_root=mbeir_root,
                        text_prefix=text_prefix,
                    )
                    image = None
                    if image_path is not None:
                        with Image.open(image_path) as img:
                            image = img.convert("RGB").copy()
                    batch_keys.append(
                        self.item_key_cls(
                            text=text or "",
                            img_path=str(image_path) if image_path is not None else "",
                            image=image,
                        )
                    )

                with torch.no_grad():
                    use_target_path = (
                        not is_query
                        and all(key.image is not None and not (key.text or "").strip() for key in batch_keys)
                    )
                    if use_target_path:
                        if hasattr(self.model, "target_instruction"):
                            self.model.target_instruction = system_instruction
                        batch_embeddings = self.model.embed_targets(batch_keys)
                    else:
                        if hasattr(self.model, "default_instruction"):
                            self.model.default_instruction = system_instruction
                        batch_embeddings = self.model.embed_queries(batch_keys)
            else:
                texts: list[Optional[str]] = []
                images: list[Optional[str]] = []
                for record in batch:
                    _, text, image_path = item_id_text_image(
                        record,
                        is_query=is_query,
                        mbeir_root=mbeir_root,
                        text_prefix=text_prefix,
                    )
                    texts.append(text or None)
                    images.append(str(image_path) if image_path is not None else None)

                with torch.no_grad():
                    batch_embeddings = self.model.get_fused_embeddings(
                        texts=texts,
                        images=images,
                        instruction=system_instruction + "\n",
                        batch_size=min(self.batch_size, 32),
                        show_progress_bar=False,
                    )
            batch_np = batch_embeddings.detach().cpu().float().numpy()
            embeddings[start:end] = normalize_rows(batch_np)

        result = np.array(embeddings, copy=True)
        del embeddings
        return result, ids


def extract_mbeir_with_rzen(
    model_name: str,
    output_dir: str,
    *,
    mbeir_root: str,
    dataset: str,
    split: str,
    candidate_scope: str = "local",
    query_path: Optional[str] = None,
    cand_pool_path: Optional[str] = None,
    qrels_path: Optional[str] = None,
    batch_size: int = 16,
    max_queries: int = 0,
    max_docs: int = 0,
    overwrite: bool = False,
    query_instruction: Optional[str] = None,
    doc_instruction: Optional[str] = None,
    query_text_prefix: Optional[str] = None,
    doc_text_prefix: Optional[str] = None,
    device: str = "cuda",
    attn_implementation: str = "sdpa",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mbeir_root_path = Path(mbeir_root)

    query_file, cand_file, resolved_qrels_file = resolve_mbeir_paths(
        mbeir_root=mbeir_root_path,
        dataset=dataset,
        split=split,
        candidate_scope=candidate_scope,
        query_path=query_path,
        cand_pool_path=cand_pool_path,
        qrels_path=qrels_path,
    )

    print("M-BEIR inputs")
    print(f"  Queries:    {query_file}")
    print(f"  Candidates: {cand_file}")
    print(f"  Qrels:      {resolved_qrels_file}")

    query_rows = load_jsonl(query_file, max_items=max_queries)
    doc_rows = load_jsonl(cand_file, max_items=max_docs)
    expected_query_ids = [record_id(record, is_query=True) for record in query_rows]
    expected_doc_ids = [record_id(record, is_query=False) for record in doc_rows]
    print(f"  Loaded {len(query_rows):,} queries and {len(doc_rows):,} candidates")

    query_instruction = query_instruction or DEFAULT_EMBED_SYSTEM_INSTRUCTION
    doc_instruction = doc_instruction or DEFAULT_EMBED_SYSTEM_INSTRUCTION
    query_text_prefix = (
        default_query_text_prefix(dataset)
        if query_text_prefix is None
        else query_text_prefix
    )
    doc_text_prefix = "" if doc_text_prefix is None else doc_text_prefix

    print(f"Loading {model_name} via RZen...")
    encoder = RZenEncoder(
        model_name,
        batch_size=batch_size,
        device=device,
        attn_implementation=attn_implementation,
    )
    embedding_dim = encoder.embedding_dim
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Attention backend:   {attn_implementation}")
    print(f"  Loader backend:      {encoder.backend_kind}")

    expected_query_meta = build_cache_metadata(
        model_name=model_name,
        count=len(query_rows),
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        system_instruction=query_instruction,
        text_prefix=query_text_prefix,
        kind="query",
        ids=expected_query_ids,
    )
    expected_doc_meta = build_cache_metadata(
        model_name=model_name,
        count=len(doc_rows),
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        system_instruction=doc_instruction,
        text_prefix=doc_text_prefix,
        kind="doc",
        ids=expected_doc_ids,
    )

    query_cache_valid, query_cache_errors = inspect_dense_cache(output_dir, "query", expected_query_meta)
    doc_cache_valid, doc_cache_errors = inspect_dense_cache(output_dir, "doc", expected_doc_meta)
    need_dense = overwrite or not (query_cache_valid and doc_cache_valid)

    if need_dense:
        if query_cache_errors:
            print(f"  Query cache refresh: {'; '.join(query_cache_errors)}")
        if doc_cache_errors:
            print(f"  Doc cache refresh:   {'; '.join(doc_cache_errors)}")

        query_embeddings, query_ids = encoder.encode_records(
            records=query_rows,
            ids=expected_query_ids,
            mbeir_root=mbeir_root_path,
            is_query=True,
            system_instruction=query_instruction,
            text_prefix=query_text_prefix,
        )
        doc_embeddings, doc_ids = encoder.encode_records(
            records=doc_rows,
            ids=expected_doc_ids,
            mbeir_root=mbeir_root_path,
            is_query=False,
            system_instruction=doc_instruction,
            text_prefix=doc_text_prefix,
        )

        np.memmap(
            output_dir / "query_embeddings.npy",
            dtype=np.float32,
            mode="w+",
            shape=query_embeddings.shape,
        )[:] = query_embeddings
        np.memmap(
            output_dir / "doc_embeddings.npy",
            dtype=np.float32,
            mode="w+",
            shape=doc_embeddings.shape,
        )[:] = doc_embeddings
        (output_dir / "query_ids.json").write_text(json.dumps(query_ids, indent=2), encoding="utf-8")
        (output_dir / "doc_ids.json").write_text(json.dumps(doc_ids, indent=2), encoding="utf-8")
        (output_dir / "query_meta.json").write_text(json.dumps(expected_query_meta, indent=2), encoding="utf-8")
        (output_dir / "doc_meta.json").write_text(json.dumps(expected_doc_meta, indent=2), encoding="utf-8")
        query_cache_reused = False
        doc_cache_reused = False
    else:
        print(f"Reusing dense cache from {output_dir}")
        _, query_ids, _ = load_cached_dense(output_dir, "query")
        _, doc_ids, _ = load_cached_dense(output_dir, "doc")
        query_cache_reused = True
        doc_cache_reused = True

    metadata = {
        "cache_layout": "retrieval",
        "dataset_type": "mbeir",
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "mbeir_root": str(mbeir_root_path),
        "dataset": dataset,
        "split": split,
        "candidate_scope": candidate_scope,
        "query_file": str(query_file),
        "cand_pool_file": str(cand_file),
        "qrels_file": str(resolved_qrels_file),
        "num_queries": len(expected_query_ids),
        "num_docs": len(expected_doc_ids),
        "batch_size": batch_size,
        "query_system_instruction": query_instruction,
        "doc_system_instruction": doc_instruction,
        "query_text_prefix": query_text_prefix,
        "doc_text_prefix": doc_text_prefix,
        "query_cache_reused": query_cache_reused,
        "doc_cache_reused": doc_cache_reused,
        "extraction_method": "rzen",
        "attn_implementation": attn_implementation,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nDone! query/doc retrieval cache ready at {output_dir}")
    print(f"  Queries: {len(expected_query_ids):,}")
    print(f"  Docs:    {len(expected_doc_ids):,}")
    print(f"  Dim:     {embedding_dim}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RZen dense retrieval caches from M-BEIR")
    parser.add_argument("--model_path", type=str, default="qihoo360/RzenEmbed")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mbeir_root", type=str, default="./M-BEIR")
    parser.add_argument("--mbeir_dataset", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--candidate_scope", choices=["local", "global"], default="local")
    parser.add_argument("--query_path", type=str, default=None)
    parser.add_argument("--cand_pool_path", type=str, default=None)
    parser.add_argument("--qrels_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--query_instruction", type=str, default=None)
    parser.add_argument("--doc_instruction", type=str, default=None)
    parser.add_argument("--query_text_prefix", type=str, default=None)
    parser.add_argument("--doc_text_prefix", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    args = parser.parse_args()

    extract_mbeir_with_rzen(
        model_name=args.model_path,
        output_dir=args.output_dir,
        mbeir_root=args.mbeir_root,
        dataset=args.mbeir_dataset,
        split=args.dataset_split,
        candidate_scope=args.candidate_scope,
        query_path=args.query_path,
        cand_pool_path=args.cand_pool_path,
        qrels_path=args.qrels_path,
        batch_size=args.batch_size,
        max_queries=args.max_queries,
        max_docs=args.max_docs,
        overwrite=args.overwrite,
        query_instruction=args.query_instruction,
        doc_instruction=args.doc_instruction,
        query_text_prefix=args.query_text_prefix,
        doc_text_prefix=args.doc_text_prefix,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )


if __name__ == "__main__":
    main()
