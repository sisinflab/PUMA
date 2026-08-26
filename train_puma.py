"""
train_puma.py -- Full pipeline for sparsifying Qwen3-VL-Embedding.

This is the main entry point. It runs all stages sequentially:
  Stage 1: Extract embeddings (or use synthetic data for testing)
  Stage 2: Train SAE with cross-modal alignment + k-annealing + AuxK
  Stage 3: Contrastive fine-tuning for retrieval
  Eval:    Compare sparse vs dense retrieval quality

Usage:
    # Quick test with synthetic data (no GPU needed):
    python train_puma.py --test

    # Full pipeline with Qwen3-VL-Embedding-2B:
    python train_puma.py --model_size 2b --model_path ./models/Qwen3-VL-Embedding-2B

    # Full pipeline with 8B:
    python train_puma.py --model_size 8b --model_path ./models/Qwen3-VL-Embedding-8B

    # Just train SAE on existing cached embeddings:
    python train_puma.py --skip_extraction --cache_dir ./cached_embeddings

    # Adjust sparsity target:
    python train_puma.py --test --k_final 16 --expansion 16
"""

import argparse
import json
import math
import time
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def build_lr_scheduler(optimizer, total_steps, warmup_steps=1000):
    """Linear warmup then cosine decay to 0."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class ConstantLRScheduler:
    """Minimal no-op scheduler used when LR schedules are disabled."""

    def step(self):
        return None


def resolve_k_initial(dict_size: int, k_final: int) -> int:
    
    upper_cap = max(1, dict_size // 4)
    proposed = max(256, 2 * k_final)
    return max(k_final, min(proposed, upper_cap))


def run_test_pipeline(args):
    """
    Generate synthetic embeddings, then reuse the normal cached-data pipeline.
    """
    # --- Stage 1: Synthetic data ---
    print("\n" + "-" * 70)
    print("STAGE 1: Generating synthetic Qwen3-VL-Embedding-style data")
    print("-" * 70)

    embedding_dim = 2048 if args.model_size == "2b" else 4096

    from extract_qwen import generate_synthetic
    generate_synthetic(
        output_dir=args.cache_dir,
        num_samples=args.num_samples,
        embedding_dim=embedding_dim,
        cross_modal_correlation=0.7,
    )

    train_puma_from_cache(args, source_label="synthetic")


def resolve_device(device_arg: str) -> str:
    """Resolve the requested runtime device."""
    import torch

    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device_arg


def inspect_cache_source(cache_dir: str, cache_format: str) -> dict:
    """Load metadata for either paired or benchmark-style retrieval caches."""
    from data import infer_cache_layout

    cache_path = Path(cache_dir)
    resolved_format = infer_cache_layout(cache_path) if cache_format == "auto" else cache_format

    if resolved_format == "paired":
        metadata_path = cache_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No paired cache metadata found at {metadata_path}"
            )
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return {
            "cache_format": "paired",
            "embedding_dim": metadata["embedding_dim"],
            "metadata": metadata,
            "num_queries": metadata["num_samples"],
            "num_docs": metadata["num_samples"],
        }

    if resolved_format == "retrieval":
        query_meta_path = cache_path / "query_meta.json"
        doc_meta_path = cache_path / "doc_meta.json"
        if not query_meta_path.exists() or not doc_meta_path.exists():
            raise FileNotFoundError(
                f"Expected query_meta.json and doc_meta.json under {cache_path}"
            )
        with open(query_meta_path, "r", encoding="utf-8") as f:
            query_meta = json.load(f)
        with open(doc_meta_path, "r", encoding="utf-8") as f:
            doc_meta = json.load(f)

        if query_meta["embedding_dim"] != doc_meta["embedding_dim"]:
            raise ValueError(
                "Query/doc benchmark caches use different embedding dimensions: "
                f"{query_meta['embedding_dim']} vs {doc_meta['embedding_dim']}"
            )

        return {
            "cache_format": "retrieval",
            "embedding_dim": query_meta["embedding_dim"],
            "metadata": {
                "query_meta": query_meta,
                "doc_meta": doc_meta,
            },
            "num_queries": query_meta["count"],
            "num_docs": doc_meta["count"],
        }

    raise ValueError(f"Unsupported cache_format: {resolved_format}")


def train_puma_from_cache(args, source_label: str):
    """
    Train and evaluate the SAE using embeddings already present in cache_dir.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from sae import (
        SparseAutoencoder,
        EncoderOnlySparseEncoder,
        ContrastiveSparseEncoder,
        SAEConfig,
        SAELoss,
        SparseContrastiveLoss,
        KAnnealingScheduler,
    )
    from data import PairedEmbeddingDataset, RetrievalEmbeddingDataset, get_dataloader
    from evaluate import evaluate_retrieval, recall_at_k, mrr, ndcg_at_k

    def resolve_stage2_contrastive_weight() -> float:
        if args.stage2_contrastive_weight is not None:
            return float(args.stage2_contrastive_weight)
        return 0.05 if cache_format == "retrieval" else 0.0

    def resolve_stage2_lr() -> float:
        return float(args.lr_stage2)

    class ActivityTracker:
        """Recent-activity tracker for encoder-only sparse models."""

        def __init__(self, dead_threshold: int = 1000):
            self.dead_threshold = dead_threshold
            self.steps_since_active = None

        def update(self, z: torch.Tensor):
            active_mask = (z.abs() > 0).any(dim=0)
            if self.steps_since_active is None:
                self.steps_since_active = torch.zeros(
                    z.shape[-1], device=z.device, dtype=torch.long
                )
            self.steps_since_active[active_mask] = 0
            self.steps_since_active[~active_mask] += 1

        @property
        def dead_fraction(self) -> float:
            if self.steps_since_active is None:
                return 0.0
            return float(
                (self.steps_since_active > self.dead_threshold).float().mean().item()
            )

    def supports_reconstruction(model) -> bool:
        return bool(getattr(model, "supports_reconstruction", True))

    def model_dead_fraction(model) -> float:
        if loss_fn is not None:
            return loss_fn.dead_fraction
        if activity_tracker is not None:
            return activity_tracker.dead_fraction
        if hasattr(model, "stats_last_nonzero") and hasattr(model, "dead_threshold"):
            return float(
                (model.stats_last_nonzero > model.dead_threshold).float().mean().item()
            )
        return 0.0

    def csr_normalized_mse(recon: torch.Tensor, xs: torch.Tensor, criterion):
        xs_mu = xs.mean(dim=0)
        loss = criterion(recon, xs) / criterion(
            xs_mu[None, :].broadcast_to(xs.shape), xs
        )
        return loss

    def retrieval_cl(latents_q: torch.Tensor, latents_d: torch.Tensor, temperature: float = 0.2):
        q = F.normalize(latents_q, dim=-1)
        d = F.normalize(latents_d, dim=-1)
        logits = q @ d.t() / temperature
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

    def compute_stage2_losses(
        model,
        out_query,
        out_positive,
        query_dense,
        positive_dense,
    ):
        if args.sparse_model == "sae":
            if cache_format == "paired":
                l_query = loss_fn(out_query, out_positive, sae=model)
                l_positive = loss_fn(out_positive, out_query, sae=model)
                recon_loss = (l_query["recon"] + l_positive["recon"]) / 2
                auxk_loss = (l_query["auxk"] + l_positive["auxk"]) / 2
                align_loss = (l_query["align"] + l_positive["align"]) / 2
                loss = (l_query["total"] + l_positive["total"]) / 2
            else:
                l_query = loss_fn(out_query, sae=model)
                l_positive = loss_fn(out_positive, sae=model)
                recon_loss = (l_query["recon"] + l_positive["recon"]) / 2
                auxk_loss = (l_query["auxk"] + l_positive["auxk"]) / 2
                align_loss = retrieval_similarity_distillation_loss(
                    query_dense,
                    positive_dense,
                    out_query["z"],
                    out_positive["z"],
                )
                base_loss = (l_query["total"] + l_positive["total"]) / 2
                loss = base_loss + args.alpha_align * align_loss
            return loss, recon_loss, align_loss, auxk_loss

        if args.sparse_model == "encoder_only":
            if cache_format != "retrieval":
                raise ValueError("Encoder-only training is currently supported only for retrieval caches.")

            activity_tracker.update(out_query["z"])
            activity_tracker.update(out_positive["z"])
            align_loss = retrieval_similarity_distillation_loss(
                query_dense,
                positive_dense,
                out_query["z"],
                out_positive["z"],
            )
            recon_loss = torch.tensor(0.0, device=device)
            auxk_loss = torch.tensor(0.0, device=device)
            loss = args.alpha_align * align_loss
            return loss, recon_loss, align_loss, auxk_loss

        criterion = nn.MSELoss().to(device)
        topk_4 = min(4 * k_final, dict_size)
        out_query_4k = model(query_dense, topk=topk_4)
        out_positive_4k = model(positive_dense, topk=topk_4)

        recon_loss = (
            criterion(query_dense, out_query["x_hat"])
            + criterion(positive_dense, out_positive["x_hat"])
        ) * 0.5
        recon_4k_loss = (
            criterion(query_dense, out_query_4k["x_hat"])
            + criterion(positive_dense, out_positive_4k["x_hat"])
        ) * 0.5
        auxk_loss_q = csr_normalized_mse(
            out_query["x_hat_aux"],
            query_dense - out_query["x_hat"].detach() + model.pre_bias.detach(),
            criterion,
        ).nan_to_num(0)
        auxk_loss_p = csr_normalized_mse(
            out_positive["x_hat_aux"],
            positive_dense - out_positive["x_hat"].detach() + model.pre_bias.detach(),
            criterion,
        ).nan_to_num(0)
        auxk_loss = (auxk_loss_q + auxk_loss_p) * 0.5
        align_loss = retrieval_cl(out_query["z"], out_positive["z"])
        loss = recon_loss + 0.125 * recon_4k_loss + args.alpha_auxk * auxk_loss + stage2_contrastive_weight * align_loss
        return loss, recon_loss, align_loss, auxk_loss

    def save_checkpoint(path: Path, training_stage: str):
        training_source = {
            "label": source_label,
            "cache_format": cache_format,
            "cache_dir": str(Path(args.cache_dir).resolve()),
            "training_stage": training_stage,
        }
        if cache_format == "paired":
            training_source["metadata"] = metadata
        else:
            training_source["qrels_path"] = str(Path(args.qrels_path).resolve())
            if args.query_jsonl:
                training_source["query_jsonl"] = str(Path(args.query_jsonl).resolve())
            training_source["query_meta"] = metadata["query_meta"]
            training_source["doc_meta"] = metadata["doc_meta"]
            training_source["val_fraction"] = args.val_fraction
            training_source["split_seed"] = args.split_seed
            training_source["num_hard_negatives"] = args.num_hard_negatives
        training_source["optimization"] = {
            "alpha_align": args.alpha_align,
            "alpha_auxk": args.alpha_auxk,
            "lr_stage2": stage2_lr,
            "lr_stage3": args.lr_stage3,
            "warmup_steps_stage2": args.warmup_steps_stage2,
            "warmup_steps_stage3": args.warmup_steps_stage3,
            "stage2_contrastive_weight": stage2_contrastive_weight,
            "validation_selection_enabled": not args.disable_validation_selection,
            "lr_schedules_enabled": not args.disable_lr_schedules,
            "sparse_model": args.sparse_model,
        }

        torch.save({
            "sae_state_dict": sae.state_dict(),
            "sae_config": {
                "input_dim": embedding_dim,
                "dict_size": dict_size,
                "k_initial": sae_config.k_initial,
                "k_final": k_final,
            },
            "current_k": int(sae.current_k),
            "model_type": args.sparse_model,
            "training_source": training_source,
        }, path)

    def retrieval_similarity_distillation_loss(
        query_dense: torch.Tensor,
        positive_dense: torch.Tensor,
        z_query: torch.Tensor,
        z_positive: torch.Tensor,
    ) -> torch.Tensor:
        """Distill dense similarity structure into sparse codes using dot product."""
        dense_sim = (query_dense * positive_dense).sum(dim=-1)
        sparse_sim = (z_query * z_positive).sum(dim=-1)
        return F.mse_loss(sparse_sim, dense_sim)

    def build_eval_bundle():
        if len(val_dataset) == 0:
            return None
        n_eval = len(val_dataset)
        if args.eval_max_queries > 0:
            n_eval = min(n_eval, args.eval_max_queries)

        if cache_format == "paired":
            queries = np.stack([val_dataset[i]["text_emb"].numpy() for i in range(n_eval)])
            docs = np.stack([val_dataset[i]["image_emb"].numpy() for i in range(n_eval)])
            relevance = [{i} for i in range(n_eval)]
        else:
            queries = val_dataset.get_query_embeddings(limit=n_eval)
            docs = val_dataset.get_doc_embeddings()
            relevance = val_dataset.get_relevance_sets(limit=n_eval)

        return queries, docs, relevance

    def evaluate_selection_metric():
        if eval_bundle is None:
            return None
        queries, docs, relevance = eval_bundle
        was_training = sae.training
        sae.eval()
        with torch.no_grad():
            retrieval_results = evaluate_retrieval(
                sae,
                query_embeddings=queries,
                doc_embeddings=docs,
                relevance=relevance,
                batch_size=min(max(args.batch_size, 1), 512),
                sparse_backend=args.validation_sparse_backend,
            )
        if was_training:
            sae.train()
        return retrieval_results

    cache_info = inspect_cache_source(args.cache_dir, args.cache_format)
    cache_format = cache_info["cache_format"]
    metadata = cache_info["metadata"]
    embedding_dim = cache_info["embedding_dim"]
    expected_dim = None
    if args.model_size == "2b":
        expected_dim = 2048
    elif args.model_size == "8b":
        expected_dim = 4096
    if expected_dim is not None and embedding_dim != expected_dim:
        raise ValueError(
            f"Cached embeddings are {embedding_dim}-d but model_size={args.model_size} expects {expected_dim}-d"
        )
    if cache_format == "retrieval" and not args.qrels_path:
        raise ValueError(
            "--qrels_path is required when training from retrieval benchmark caches"
        )

    stage2_contrastive_weight = resolve_stage2_contrastive_weight()
    stage2_lr = resolve_stage2_lr()
    dict_size = embedding_dim * args.expansion
    k_final = args.k_final
    k_initial = resolve_k_initial(dict_size, k_final)
    device = resolve_device(args.device)

    sparse_model_label = {
        "sae": "TopK SAE",
        "encoder_only": "Encoder-only sparse baseline",
    }[args.sparse_model]
    model_label = (
        f"Qwen3-VL-Embedding-{args.model_size.upper()}"
        if args.model_size in {"2b", "8b"}
        else (args.model_path or "Custom-Embedding")
    )
    print("=" * 70)
    print(f"SPARSE EMBEDDING PIPELINE ({source_label})")
    print("=" * 70)
    print(f"  Model:         {model_label}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Dictionary:    {dict_size:,} ({effective_expansion}x expansion)")
    print(f"  Target k:      {k_final}")
    print(f"  K schedule:    {k_initial} -> {k_final}")
    print(f"  Sparsity:      {1 - k_final / dict_size:.4%}")
    print(f"  Cache format:  {cache_format}")
    if cache_format == "paired":
        print(f"  Samples:       {metadata['num_samples']:,} ({source_label})")
    else:
        print(f"  Queries:       {cache_info['num_queries']:,}")
        print(f"  Documents:     {cache_info['num_docs']:,}")
        print(f"  Qrels:         {args.qrels_path}")
        if args.query_jsonl:
            print(f"  Query JSONL:   {args.query_jsonl}")
        print(f"  Hard negs:     {args.num_hard_negatives}")
    print(f"  Device:        {device}")
    print(f"  Sparse model:  {sparse_model_label}")
    print(f"  Stage2 c-loss: {stage2_contrastive_weight:.4f}")
    print(f"  LR stage2/3:   {stage2_lr:.2e} / {args.lr_stage3:.2e}")
    print(f"  LR schedules:  {'on' if not args.disable_lr_schedules else 'off'}")
    print(f"  Val backend:   {args.validation_sparse_backend}")

    # --- Stage 2: Train SAE ---
    if args.sparse_model == "sae":
        stage2_label = (
            "Training TopK SAE with cross-modal alignment + AuxK"
            if cache_format == "paired"
            else "Training TopK SAE with retrieval distillation + AuxK"
        )
    else:
        stage2_label = "Training encoder-only sparse baseline with retrieval distillation"
    print("\n" + "-" * 70)
    print(f"STAGE 2: {stage2_label}")
    print("-" * 70)

    sae_config = SAEConfig(
        input_dim=embedding_dim,
        dict_size=dict_size,
        k_initial=k_initial,
        k_final=k_final,
        k_annealing_steps=args.stage2_steps,
        k_annealing_schedule="cosine",
    )
    if args.sparse_model == "sae":
        sae = SparseAutoencoder(sae_config).to(device)
    else:
        if cache_format != "retrieval":
            raise ValueError("Encoder-only sparse training currently requires retrieval-format caches.")
        sae = EncoderOnlySparseEncoder(sae_config).to(device)

    param_count = sum(p.numel() for p in sae.parameters())
    print(f"  Sparse parameters: {param_count:,}")
    if args.sparse_model == "sae":
        print(f"  Architecture:      {embedding_dim} -> {dict_size} -> {embedding_dim}")
    else:
        print(f"  Architecture:      {embedding_dim} -> {dict_size}")

    if cache_format == "paired":
        dataset = PairedEmbeddingDataset(
            args.cache_dir,
            split="train",
            val_fraction=args.val_fraction,
        )
        val_dataset = PairedEmbeddingDataset(
            args.cache_dir,
            split="val",
            val_fraction=args.val_fraction,
        )
    else:
        dataset = RetrievalEmbeddingDataset(
            args.cache_dir,
            qrels_path=args.qrels_path,
            query_jsonl=args.query_jsonl,
            split="train",
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            num_hard_negatives=args.num_hard_negatives,
            negative_sampling_strategy=args.negative_sampling_strategy,
        )
        val_dataset = RetrievalEmbeddingDataset(
            args.cache_dir,
            qrels_path=args.qrels_path,
            query_jsonl=args.query_jsonl,
            split="val",
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            num_hard_negatives=args.num_hard_negatives,
            negative_sampling_strategy=args.negative_sampling_strategy,
        )
    if len(dataset) == 0:
        raise ValueError("Training split is empty; reduce --val_fraction or check qrels")

    eval_bundle = build_eval_bundle()

    loader = get_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    loss_fn = None
    activity_tracker = None
    if args.sparse_model == "sae":
        loss_fn = SAELoss(
            alpha_align=args.alpha_align,
            alpha_auxk=args.alpha_auxk,
            dead_threshold=1000,
            auxk_k=512,
        ).to(device)
    elif args.sparse_model == "encoder_only":
        activity_tracker = ActivityTracker(dead_threshold=1000)
    k_scheduler = KAnnealingScheduler(sae_config)

    # Contrastive loss for optional Stage 2 blending + Stage 3
    contrastive = SparseContrastiveLoss(temperature=0.05)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stage2_ckpt_path = ckpt_dir / "sae_stage2.pt"
    resume_stage3_checkpoint = Path(args.resume_stage3_checkpoint) if args.resume_stage3_checkpoint else None

    if resume_stage3_checkpoint is not None:
        if not resume_stage3_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found at {resume_stage3_checkpoint}")
        checkpoint = torch.load(resume_stage3_checkpoint, map_location=device)
        sae.load_state_dict(checkpoint["sae_state_dict"])
        sae.current_k = int(checkpoint.get("current_k", k_final))
        stage2_time = 0.0
        best_loss = float("nan")
        save_checkpoint(stage2_ckpt_path, training_stage="stage2_resumed")
        print(f"  Loaded Stage 2 checkpoint from {resume_stage3_checkpoint}")
        print(f"  Re-saved Stage 2 checkpoint to {stage2_ckpt_path}")
    else:
        optimizer = torch.optim.AdamW(sae.parameters(), lr=stage2_lr, weight_decay=1e-5)
        lr_scheduler = (
            build_lr_scheduler(optimizer, args.stage2_steps, args.warmup_steps_stage2)
            if not args.disable_lr_schedules
            else ConstantLRScheduler()
        )
        sae.train()
        step = 0
        t0 = time.time()
        best_loss = float("inf")

        while step < args.stage2_steps:
            for batch in loader:
                if step >= args.stage2_steps:
                    break

                query = batch["query_emb"].to(device, non_blocking=True)
                positive = batch["positive_emb"].to(device, non_blocking=True)

                sae.current_k = k_scheduler.get_k(step)

                out_query = sae(query)
                out_positive = sae(positive)

                loss, recon_loss, align_loss, auxk_loss = compute_stage2_losses(
                    sae,
                    out_query,
                    out_positive,
                    query,
                    positive,
                )

                if stage2_contrastive_weight > 0 and args.sparse_model in {"sae", "encoder_only"}:
                    l_contrastive = contrastive(out_query["z"], out_positive["z"])
                    loss = loss + stage2_contrastive_weight * l_contrastive

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                sae.normalize_decoder_()

                if step % max(args.stage2_steps // 10, 1) == 0:
                    dead_frac = model_dead_fraction(sae)
                    active_mean = (out_query["active_dims"].item() + out_positive["active_dims"].item()) / 2
                    positive_pre_mean = (
                        (out_query["h_pre"] > 0).float().sum(dim=-1).mean().item()
                        + (out_positive["h_pre"] > 0).float().sum(dim=-1).mean().item()
                    ) / 2
                    print(
                        f"  Step {step:>6d}/{args.stage2_steps} | k={sae.current_k:>3d} | "
                        f"loss={loss.item():.4f} | recon={recon_loss.item():.4f} | "
                        f"align={align_loss.item():.4f} | auxk={auxk_loss.item():.4f} | "
                        f"active={active_mean:.1f} | pospre={positive_pre_mean:.1f} | "
                        f"dead={dead_frac:.1%}"
                    )

                if loss.item() < best_loss:
                    best_loss = loss.item()

                step += 1

        stage2_time = time.time() - t0
        print(f"\n  Stage 2 done in {stage2_time:.1f}s | best loss: {best_loss:.4f}")
        save_checkpoint(stage2_ckpt_path, training_stage="stage2")
        print(f"  Saved Stage 2 checkpoint to {stage2_ckpt_path}")

    best_ckpt_path = ckpt_dir / "sae_best.pt"
    best_selection_score = None
    if cache_format == "retrieval" and eval_bundle is not None and not args.disable_validation_selection:
        stage2_retrieval = evaluate_selection_metric()
        if stage2_retrieval is not None:
            best_selection_score = float(stage2_retrieval["sparse"]["ndcg@10"])
            save_checkpoint(best_ckpt_path, training_stage="stage2_best")
            print(
                "  Stage 2 validation selection: "
                f"nDCG@10={best_selection_score:.4f} "
                f"(saved best checkpoint to {best_ckpt_path})"
            )
    elif cache_format == "retrieval" and args.disable_validation_selection:
        print("  Validation selection disabled; keeping the latest checkpoint in each stage.")

    # --- Stage 3: Contrastive fine-tuning ---
    print("\n" + "-" * 70)
    print("STAGE 3: Contrastive retrieval fine-tuning")
    print("-" * 70)

    sae.current_k = k_final
    optimizer = torch.optim.AdamW(sae.parameters(), lr=args.lr_stage3)
    lr_scheduler = (
        build_lr_scheduler(optimizer, args.stage3_steps, args.warmup_steps_stage3)
        if not args.disable_lr_schedules
        else ConstantLRScheduler()
    )

    sae.train()
    step = 0
    t0 = time.time()
    stage3_eval_interval = max(args.stage3_steps // 5, 1) if args.stage3_steps > 0 else 0

    while step < args.stage3_steps:
        for batch in loader:
            if step >= args.stage3_steps:
                break

            query = batch["query_emb"].to(device, non_blocking=True)
            positive = batch["positive_emb"].to(device, non_blocking=True)
            hard_negative_embs = batch.get("hard_negative_embs")

            out_q = sae(query)
            out_d = sae(positive)

            z_neg = None
            if hard_negative_embs is not None:
                hard_negative_embs = hard_negative_embs.to(device, non_blocking=True)
                bsz, num_negatives, _ = hard_negative_embs.shape
                out_neg = sae(hard_negative_embs.reshape(bsz * num_negatives, -1))
                z_neg = out_neg["z"].reshape(bsz, num_negatives, -1)

            l_c = contrastive(out_q["z"], out_d["z"], z_neg=z_neg)
            if args.sparse_model == "sae":
                l_r = (loss_fn(out_q, sae=sae)["recon"] + loss_fn(out_d, sae=sae)["recon"]) / 2
            else:
                activity_tracker.update(out_q["z"])
                activity_tracker.update(out_d["z"])
                l_r = torch.tensor(0.0, device=device)
            l_align = retrieval_similarity_distillation_loss(
                query,
                positive,
                out_q["z"],
                out_d["z"],
            ) if cache_format == "retrieval" else torch.tensor(0.0, device=device)
            loss = l_c + (0.1 * l_r if args.sparse_model == "sae" else 0.0) + (
                0.05 * l_align if cache_format == "retrieval" else 0.0
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            sae.normalize_decoder_()

            if step % max(args.stage3_steps // 5, 1) == 0:
                active_mean = (out_q["active_dims"].item() + out_d["active_dims"].item()) / 2
                positive_pre_mean = (
                    (out_q["h_pre"] > 0).float().sum(dim=-1).mean().item()
                    + (out_d["h_pre"] > 0).float().sum(dim=-1).mean().item()
                ) / 2
                print(
                    f"  Step {step:>6d}/{args.stage3_steps} | "
                    f"contrastive={l_c.item():.4f} | recon={l_r.item():.4f}"
                    + (f" | align={l_align.item():.4f}" if cache_format == "retrieval" else "")
                    + f" | active={active_mean:.1f} | pospre={positive_pre_mean:.1f}"
                    + f" | dead={model_dead_fraction(sae):.1%}"
                )

            should_select = (
                cache_format == "retrieval"
                and eval_bundle is not None
                and not args.disable_validation_selection
                and stage3_eval_interval > 0
                and (step % stage3_eval_interval == 0 or step + 1 == args.stage3_steps)
            )
            if should_select:
                selection = evaluate_selection_metric()
                if selection is not None:
                    selection_score = float(selection["sparse"]["ndcg@10"])
                    print(
                        "  Stage 3 validation selection: "
                        f"nDCG@10={selection_score:.4f}"
                        + (
                            f" | best={best_selection_score:.4f}"
                            if best_selection_score is not None
                            else ""
                        )
                    )
                    if best_selection_score is None or selection_score > best_selection_score:
                        best_selection_score = selection_score
                        save_checkpoint(best_ckpt_path, training_stage="stage3_best")
                        print(f"  Saved new best checkpoint to {best_ckpt_path}")
            step += 1

    stage3_time = time.time() - t0
    print(f"\n  Stage 3 done in {stage3_time:.1f}s")

    if cache_format == "retrieval" and not args.disable_validation_selection and best_ckpt_path.exists():
        best_checkpoint = torch.load(best_ckpt_path, map_location=device)
        sae.load_state_dict(best_checkpoint["sae_state_dict"])
        sae.current_k = int(best_checkpoint.get("current_k", k_final))
        print(
            f"  Restored best validation checkpoint from {best_ckpt_path}"
            + (
                f" (nDCG@10={best_selection_score:.4f})"
                if best_selection_score is not None
                else ""
            )
        )

    # --- Evaluation ---
    print("\n" + "-" * 70)
    print("EVALUATION: Sparse vs Dense retrieval")
    print("-" * 70)

    sae.eval()
    if len(val_dataset) == 0:
        print("  Skipping evaluation because the validation split is empty.")
    else:
        n_eval = len(val_dataset)
        if args.eval_max_queries > 0:
            n_eval = min(n_eval, args.eval_max_queries)

        if cache_format == "paired":
            queries = np.stack([val_dataset[i]["text_emb"].numpy() for i in range(n_eval)])
            docs = np.stack([val_dataset[i]["image_emb"].numpy() for i in range(n_eval)])
            relevance = [{i} for i in range(n_eval)]
        else:
            queries = val_dataset.get_query_embeddings(limit=n_eval)
            docs = val_dataset.get_doc_embeddings()
            relevance = val_dataset.get_relevance_sets(limit=n_eval)

        with torch.no_grad():
            if supports_reconstruction(sae):
                recon_docs = np.array(docs[:min(len(docs), 500)], copy=True)
                x = torch.from_numpy(recon_docs).to(device)
                out = sae(x)
                cos_sim = F.cosine_similarity(out["x"], out["x_hat"], dim=-1)
                print(f"\n  Reconstruction cosine: {cos_sim.mean():.4f} +/- {cos_sim.std():.4f}")
            else:
                print("\n  Reconstruction cosine: n/a (encoder-only baseline has no decoder)")

            if cache_format == "paired":
                pair_docs = docs[:min(len(docs), 500)]
                pair_queries = queries[:min(len(queries), 500)]
            else:
                pair_queries, pair_docs = val_dataset.get_alignment_pairs(limit=min(n_eval, 500))

            doc_t = torch.from_numpy(pair_docs).to(device)
            q_t = torch.from_numpy(pair_queries).to(device)
            out_i = sae(doc_t)
            out_t = sae(q_t)

            active_i = (out_i["z"] != 0).float()
            active_t = (out_t["z"] != 0).float()
            intersection = (active_i * active_t).sum(dim=-1)
            union = ((active_i + active_t) > 0).float().sum(dim=-1)
            jaccard = (intersection / (union + 1e-8)).mean()

            sparse_cos = float(F.cosine_similarity(out_i["z"], out_t["z"], dim=-1).mean().item())
            dense_cos = float(F.cosine_similarity(doc_t, q_t, dim=-1).mean().item())
            jaccard_value = float(jaccard.item())

            print(f"  Feature overlap (Jaccard): {jaccard_value:.4f}")
            print(f"  Sparse cosine:  {sparse_cos:.4f}")
            print(f"  Dense cosine:   {dense_cos:.4f}")
            print(f"  Retention:      {sparse_cos / max(dense_cos, 1e-8):.2%}")
            print(
                f"  Dead features:  "
                f"{model_dead_fraction(sae):.1%}"
            )

        if cache_format == "paired":
            from inference import SimpleInvertedIndex

            with torch.no_grad():
                doc_t = torch.from_numpy(docs).to(device)
                q_t = torch.from_numpy(queries).to(device)

                d_idx, d_val = sae.get_active_features(doc_t)
                q_idx, q_val = sae.get_active_features(q_t)

                index = SimpleInvertedIndex()
                for i in range(n_eval):
                    doc = {int(idx): float(val) for idx, val in
                           zip(d_idx[i].tolist(), d_val[i].tolist()) if val > 0}
                    index.add_document(i, doc)

                sparse_rankings = []
                for i in range(n_eval):
                    q = {int(idx): float(val) for idx, val in
                         zip(q_idx[i].tolist(), q_val[i].tolist()) if val > 0}
                    results = index.search(q, top_k=10, scoring="dot")
                    sparse_rankings.append([d for d, _ in results])

                scores = torch.mm(q_t, doc_t.t())
                _, dense_top = scores.topk(10, dim=-1)
                dense_rankings = [dense_top[i].tolist() for i in range(n_eval)]

            print(f"\n  {'Metric':<15s} {'Sparse':>10s} {'Dense':>10s} {'Retention':>10s}")
            print("  " + "-" * 47)

            for name, fn, kwargs in [
                ("Recall@1",  recall_at_k, {"k": 1}),
                ("Recall@5",  recall_at_k, {"k": 5}),
                ("Recall@10", recall_at_k, {"k": 10}),
                ("MRR",       mrr,         {}),
                ("nDCG@10",   ndcg_at_k,   {"k": 10}),
            ]:
                s = fn(sparse_rankings, relevance, **kwargs)
                d = fn(dense_rankings, relevance, **kwargs)
                r = s / max(d, 1e-8)
                print(f"  {name:<15s} {s:>10.4f} {d:>10.4f} {r:>10.2%}")

            print(f"\n  Sparsity-quality tradeoff:")
            print(f"  {'k':>6s}  {'R@1':>8s}  {'R@10':>8s}  {'nDCG@10':>8s}  {'Sparsity':>10s}")
            print("  " + "-" * 48)

            for test_k in [4, 8, 16, 32, 64, 128]:
                if test_k > dict_size:
                    continue
                sae.current_k = test_k

                with torch.no_grad():
                    di, dv = sae.get_active_features(doc_t)
                    qi, qv = sae.get_active_features(q_t)

                    idx2 = SimpleInvertedIndex()
                    for i in range(n_eval):
                        doc = {int(a): float(b) for a, b in zip(di[i].tolist(), dv[i].tolist()) if b > 0}
                        idx2.add_document(i, doc)

                    rankings2 = []
                    for i in range(n_eval):
                        q = {int(a): float(b) for a, b in zip(qi[i].tolist(), qv[i].tolist()) if b > 0}
                        res = idx2.search(q, top_k=10, scoring="dot")
                        rankings2.append([d for d, _ in res])

                r1 = recall_at_k(rankings2, relevance, k=1)
                r10 = recall_at_k(rankings2, relevance, k=10)
                ndcg = ndcg_at_k(rankings2, relevance, k=10)
                sparsity = 1 - test_k / dict_size
                print(f"  {test_k:>6d}  {r1:>8.4f}  {r10:>8.4f}  {ndcg:>8.4f}  {sparsity:>10.4%}")
        else:
            retrieval_results = evaluate_retrieval(
                sae,
                query_embeddings=queries,
                doc_embeddings=docs,
                relevance=relevance,
                batch_size=min(max(args.batch_size, 1), 512),
                sparse_backend=args.validation_sparse_backend,
            )

            print(f"\n  {'Metric':<15s} {'Sparse':>10s} {'Dense':>10s} {'Retention':>10s}")
            print("  " + "-" * 47)
            for metric_name in ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]:
                sparse_value = retrieval_results["sparse"][metric_name]
                dense_value = retrieval_results["dense"][metric_name]
                retention = sparse_value / max(dense_value, 1e-8)
                label = {
                    "recall@1": "Recall@1",
                    "recall@5": "Recall@5",
                    "recall@10": "Recall@10",
                    "mrr": "MRR",
                    "ndcg@10": "nDCG@10",
                }[metric_name]
                print(f"  {label:<15s} {sparse_value:>10.4f} {dense_value:>10.4f} {retention:>10.2%}")

            print(f"\n  Sparsity-quality tradeoff:")
            print(f"  {'k':>6s}  {'R@1':>8s}  {'R@10':>8s}  {'nDCG@10':>8s}  {'Sparsity':>10s}")
            print("  " + "-" * 48)
            for test_k in [4, 8, 16, 32, 64, 128]:
                if test_k > dict_size:
                    continue
                sae.current_k = test_k
                tradeoff = evaluate_retrieval(
                    sae,
                    query_embeddings=queries,
                    doc_embeddings=docs,
                    relevance=relevance,
                    batch_size=min(max(args.batch_size, 1), 512),
                    sparse_backend=args.validation_sparse_backend,
                )
                sparsity = 1 - test_k / dict_size
                print(
                    f"  {test_k:>6d}  {tradeoff['sparse']['recall@1']:>8.4f}  "
                    f"{tradeoff['sparse']['recall@10']:>8.4f}  "
                    f"{tradeoff['sparse']['ndcg@10']:>8.4f}  {sparsity:>10.4%}"
                )

        sae.current_k = k_final

    final_ckpt_path = ckpt_dir / "sae_final.pt"
    save_checkpoint(final_ckpt_path, training_stage="stage3")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"""
  Model:          {model_label} ({source_label})
  Sparse model:   {sparse_model_label}
  Embedding dim:  {embedding_dim}
  Dictionary:     {dict_size:,} features
  Target k:       {k_final} active features
  Sparsity:       {1 - k_final / dict_size:.4%}
  Training time:  {stage2_time + stage3_time:.1f}s
  Stage 2 ckpt:   {stage2_ckpt_path}
  Checkpoint:     {final_ckpt_path}
  Dead features:  {model_dead_fraction(sae):.1%}
""")


def main():
    parser = argparse.ArgumentParser(
        description="Sparsify dense embedding caches via TopK SAE or encoder-only baseline"
    )

    # Mode
    parser.add_argument("--test", action="store_true",
                       help="Test mode with synthetic data (no GPU needed)")
    parser.add_argument("--skip_extraction", action="store_true",
                       help="Skip Stage 1, use existing cached embeddings")

    # Model
    parser.add_argument("--model_size", type=str, default="2b",
                       choices=["2b", "8b", "custom"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--sparse_model", type=str, default="sae",
                       choices=["sae", "encoder_only"],
                       help="Sparse encoder family to train. 'sae' is the PUMA TopK SAE; 'encoder_only' is the decoder-less baseline.")

    # SAE
    parser.add_argument("--expansion", type=int, default=16,
                       help="Dictionary expansion factor (dict_size = dim * expansion)")
    parser.add_argument("--k_final", type=int, default=32,
                       help="Target number of active features")
    parser.add_argument("--alpha_align", type=float, default=0.3,
                       help="Cross-modal alignment loss weight")
    parser.add_argument("--alpha_auxk", type=float, default=1/32,
                       help="AuxK dead feature revival loss weight")

    # Training
    parser.add_argument("--stage2_steps", type=int, default=2000,
                       help="Stage 2 training steps (use 40000 for real data)")
    parser.add_argument("--stage3_steps", type=int, default=5000,
                       help="Stage 3 training steps (use 10000-20000 for real data)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader workers for cached embedding reads")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Training device; auto prefers CUDA when available")
    parser.add_argument("--val_fraction", type=float, default=0.02,
                       help="Held-out fraction for validation when training from cache")
    parser.add_argument("--split_seed", type=int, default=0,
                       help="Random seed for train/val split assignment")
    parser.add_argument("--eval_max_queries", type=int, default=500,
                       help="Cap evaluation queries; use 0 to evaluate all")
    parser.add_argument("--lr_stage2", type=float, default=3e-4,
                       help="Peak learning rate for Stage 2")
    parser.add_argument("--lr_stage3", type=float, default=1e-4,
                       help="Peak learning rate for Stage 3")
    parser.add_argument("--warmup_steps_stage2", type=int, default=1000,
                       help="Linear LR warmup steps for Stage 2")
    parser.add_argument("--warmup_steps_stage3", type=int, default=500,
                       help="Linear LR warmup steps for Stage 3")
    parser.add_argument("--stage2_contrastive_weight", type=float, default=None,
                       help="Weight for contrastive loss blended into Stage 2. Defaults to 0.05 for retrieval caches and 0.0 for paired caches.")
    parser.add_argument("--disable_validation_selection", action="store_true",
                       help="Skip validation-based checkpoint selection and keep the last-stage weights.")
    parser.add_argument("--disable_lr_schedules", action="store_true",
                       help="Disable warmup/cosine LR schedules and use constant learning rates.")
    parser.add_argument("--validation_sparse_backend", type=str, default="torch_sparse",
                       choices=["python", "torch_sparse"],
                       help="Exact sparse backend used for validation/evaluation inside train_puma.")
    parser.add_argument("--resume_stage3_checkpoint", type=str, default=None,
                       help="Load a saved Stage 2 checkpoint, skip Stage 2 training, and continue from validation selection into Stage 3.")

    # Paths
    parser.add_argument("--cache_dir", type=str, default="./cached_embeddings")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--cache_format", type=str, default="auto",
                       choices=["auto", "paired", "retrieval"],
                       help="Cache layout under --cache_dir")
    parser.add_argument("--qrels_path", type=str, default=None,
                       help="Required for retrieval cache training/eval")
    parser.add_argument("--query_jsonl", type=str, default=None,
                       help="Optional M-BEIR train query JSONL used to load pos_cand_list/neg_cand_list.")
    parser.add_argument("--num_hard_negatives", type=int, default=4,
                       help="Hard negatives per query for retrieval Stage 3; falls back to random negatives when unavailable.")
    parser.add_argument(
        "--negative_sampling_strategy",
        type=str,
        default="fast",
        choices=["fast", "legacy"],
        help="Hard-negative sampler for retrieval caches. 'fast' avoids scanning the full corpus per sample; 'legacy' preserves the older behavior.",
    )

    args = parser.parse_args()

    if args.test:
        run_test_pipeline(args)
        return

    if args.skip_extraction:
        train_puma_from_cache(args, source_label="cached")
        return

    if args.model_path is None:
        parser.error("--model_path is required unless using --test or --skip_extraction")

    from extract_qwen import extract_with_qwen3vl_embedder
    extract_with_qwen3vl_embedder(
        model_path=args.model_path,
        output_dir=args.cache_dir,
        max_samples=args.num_samples,
    )
    train_puma_from_cache(args, source_label="extracted")


if __name__ == "__main__":
    main()
