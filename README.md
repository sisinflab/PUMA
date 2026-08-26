# PUMA

Post-Hoc Sparsification of Universal Multimodal Embeddings for Efficient Retrieval.

PUMA trains a TopK Sparse Autoencoder (SAE) on top of cached dense embeddings
produced by a frozen multimodal embedder (Qwen3-VL-Embedding 2B / 8B, or
RZenEmbed). The resulting sparse codes replace the dense vectors at retrieval
time. The recipe is post-hoc: the base embedder is never fine-tuned.

The released code reproduces the three steps used in the paper:

1. **Extract** dense embeddings of M-BEIR queries and candidate pools.
2. **Train** the SAE (Stage 2: reconstruction + cross-modal alignment + AuxK
   feature revival + dot-product distillation + progressive k-annealing).
3. **Fine-tune and benchmark** (Stage 3: contrastive InfoNCE on sparse codes
   followed by retrieval evaluation).

Baselines used in the paper are also included: PCA matched-memory, Raw TopK,
trained dense autoencoder, and encoder-only TopK (decoder-less variant of the
SAE).

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional:

- `vllm>=0.13.0` enables the vLLM extraction path in `extract_qwen.py`. The
  Hugging Face `transformers` path is used as a fallback.

A CUDA-enabled PyTorch install is required for both extraction and training.

## Data setup

The pipeline expects an M-BEIR checkout following the official layout:

```
./M-BEIR/
    query/
    cand_pool/
    qrels/
    mbeir_images/
```

Download the dataset from <https://huggingface.co/datasets/TIGER-Lab/M-BEIR>.
If it lives elsewhere, pass `--mbeir_root /path/to/M-BEIR` or export
`MBEIR_ROOT` before invoking the scripts.

For RZenEmbed extraction, point `RZEN_EMBED_PATH` at a clone of the
RZenEmbed repository (provides `rzen_embed_inference.RzenEmbed`).

## Three-step pipeline

### (a) Extract dense embeddings

For Qwen3-VL-Embedding:

```bash
python extract_qwen.py \
    --method qwen3vl \
    --model_path Qwen/Qwen3-VL-Embedding-2B \
    --mbeir_root ./M-BEIR \
    --mbeir_dataset cirr_task7 \
    --dataset_split test \
    --output_dir ./cached_embeddings/cirr_task7_test
```

For RZenEmbed, run `extract_rzen.py` with the same flags.

This caches `query_embeddings.npy`, `doc_embeddings.npy` and metadata under
`--output_dir`. Repeat for every (dataset, split) you want to evaluate.

### (b) Train the SAE (Stage 2)

```bash
python train_puma.py \
    --cache_dir ./cached_embeddings/cirr_task7_train \
    --checkpoint_dir ./checkpoints/cirr_task7_k128 \
    --model_size 2b \
    --k_final 128 \
    --expansion 16 \
    --stage2_steps 40000 \
    --stage3_steps 0
```

Setting `--stage3_steps 0` stops after Stage 2 and writes
`sae_stage2.pt` into `--checkpoint_dir`.

Key Stage-2 flags (see `python train_puma.py --help` for the full list):

- `--k_final` -- target sparsity (number of active features at inference).
- `--expansion` -- dictionary expansion factor (`dict_size = embedding_dim *
  expansion`); 16 is the paper default.
- `--alpha_align`, `--alpha_auxk` -- weights of the cross-modal alignment and
  AuxK losses.
- `--stage2_contrastive_weight` -- weight of the small Stage-2 contrastive
  blend; defaults to `0.05` for retrieval caches and `0` for paired caches.
- `--sparse_model` -- `sae` (default, the PUMA model) or `encoder_only` (the
  decoder-less baseline used in the paper).

### (c) Stage 3 contrastive fine-tune + benchmark

Run Stage 3 starting from the Stage-2 checkpoint:

```bash
python train_puma.py \
    --cache_dir ./cached_embeddings/cirr_task7_train \
    --checkpoint_dir ./checkpoints/cirr_task7_k128 \
    --model_size 2b \
    --k_final 128 \
    --expansion 16 \
    --stage2_steps 0 \
    --stage3_steps 5000 \
    --resume_stage3_checkpoint ./checkpoints/cirr_task7_k128/sae_stage2.pt
```

This produces `sae_final.pt` (the contrastively fine-tuned checkpoint).

Then evaluate on a test split with `benchmark_mbeir.py`:

```bash
python benchmark_mbeir.py \
    --dataset cirr_task7 \
    --split test \
    --mbeir_root ./M-BEIR \
    --model_path Qwen/Qwen3-VL-Embedding-2B \
    --checkpoint_path ./checkpoints/cirr_task7_k128/sae_final.pt \
    --output_dir ./benchmark_runs/cirr_task7_test_k128
```

`benchmark_mbeir.py` reads (or rebuilds) the test-split dense cache, scores
both the dense baseline and the PUMA sparse codes, and writes `*.run` TREC
files plus a `results.json` with Recall@k, MRR, and nDCG@10. The script also
evaluates the matched-memory PCA, Raw TopK, dense-AE, and encoder-only
baselines when their flags are enabled (`--matched_memory_dense`,
`--raw_topk_values 128 160`, `--trained_dense_autoencoder_checkpoint`, ...).

A single command can also run Stages 2 and 3 back-to-back:

```bash
python train_puma.py \
    --cache_dir ./cached_embeddings/cirr_task7_train \
    --checkpoint_dir ./checkpoints/cirr_task7_k128 \
    --model_size 2b --k_final 128 --expansion 16 \
    --stage2_steps 40000 --stage3_steps 5000
```

## File map

- `sae.py` -- TopK SAE model, AuxK, alignment / contrastive losses, k-annealing.
- `dense_autoencoder.py` -- trained dense-AE baseline.
- `config.py` -- training configuration dataclasses.
- `data.py` -- paired and retrieval-format dataset loaders for cached embeddings.
- `train_sae.py` -- checkpoint loader used by `benchmark_mbeir.py`.
- `inference.py` -- sparse encoding utilities and a minimal inverted index.
- `evaluate.py` -- retrieval metrics (Recall@k, MRR, nDCG@10) and TREC writers.
- `extract_qwen.py` -- Stage-1 extraction for Qwen3-VL-Embedding (vLLM and
  Hugging Face paths).
- `extract_rzen.py` -- Stage-1 extraction for RZenEmbed.
- `train_puma.py` -- Stage-2 + Stage-3 training driver.
- `benchmark_mbeir.py` -- M-BEIR benchmarking driver (sparse PUMA, dense
  baseline, PCA, Raw TopK, dense-AE, encoder-only).
