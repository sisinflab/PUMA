"""
Stage 1: Extract paired embeddings from Qwen3-VL-Embedding.

Uses the official Qwen3VLEmbedder wrapper from the Qwen3-VL-Embedding repo.
Caches dense embeddings to disk as memory-mapped arrays.

Prerequisites:
    git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git
    cd Qwen3-VL-Embedding && bash scripts/setup_environment.sh
    huggingface-cli download Qwen/Qwen3-VL-Embedding-2B --local-dir ./models/Qwen3-VL-Embedding-2B

Usage:
    # Extract from COCO
    python extract_qwen.py --model_path ./models/Qwen3-VL-Embedding-2B --dataset coco

    # Extract CIRR train query/doc caches from M-BEIR
    python extract_qwen.py --method mbeir_vllm \
      --model_path Qwen/Qwen3-VL-Embedding-2B \
      --mbeir_dataset cirr_task7 \
      --dataset_split train \
      --output_dir ./benchmark_cache/cirr_task7_train_qwen2b

    # Extract from custom image folder + captions
    python extract_qwen.py --model_path ./models/Qwen3-VL-Embedding-2B --image_dir /path/to/images --captions /path/to/captions.jsonl

    # Generate synthetic data for testing (no VLM needed)
    python extract_qwen.py --synthetic --embedding_dim 2048
"""

import os
import sys
import json
import argparse
import time
import inspect
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import numpy as np
from pathlib import Path
from typing import Optional

QWEN_REPO = os.environ.get("QWEN3_VL_EMBEDDING_PATH", "./Qwen3-VL-Embedding")
if os.path.isdir(QWEN_REPO):
    sys.path.insert(0, QWEN_REPO)


def create_memmap(path: str, shape: tuple, dtype=np.float32) -> np.memmap:
    """Create a memory-mapped array."""
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def open_memmap(path: str, shape: tuple, *, resume: bool, dtype=np.float32) -> np.memmap:
    """Open a memmap for fresh writes or resuming an interrupted run."""
    mode = "r+" if resume and Path(path).exists() else "w+"
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


def save_json_atomic(path: Path, payload: dict):
    """Write JSON atomically to avoid corrupting progress on interruption."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def load_progress_state(
    progress_path: Path,
    *,
    model_name: str,
    dataset_source: str,
    dataset_split: str,
    embedding_dim: int,
    max_samples: int,
):
    """Load and validate extractor resume state."""
    with open(progress_path) as f:
        state = json.load(f)

    expected = {
        "model_name": model_name,
        "dataset_source": dataset_source,
        "dataset_split": dataset_split,
        "embedding_dim": embedding_dim,
        "max_samples": max_samples,
    }
    for key, expected_value in expected.items():
        actual_value = state.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Resume mismatch for {key}: expected {expected_value!r}, found {actual_value!r}"
            )
    return state


def normalize_embedding(embedding) -> np.ndarray:
    """Convert an embedding payload to normalized float32 numpy."""
    arr = np.asarray(embedding, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-8)


def embedding_from_vllm_output(output) -> np.ndarray:
    """Extract a single embedding vector from a vLLM embed response."""
    outputs = getattr(output, "outputs", None)
    if hasattr(outputs, "embedding"):
        return normalize_embedding(outputs.embedding)
    if isinstance(outputs, list) and outputs and hasattr(outputs[0], "embedding"):
        return normalize_embedding(outputs[0].embedding)
    raise TypeError(f"Unsupported vLLM embedding output type: {type(output)!r}")


def load_streaming_dataset(
    dataset_name: str,
    split: str,
    dataset_path: Optional[str] = None,
    trust_remote_code: bool = False,
    local_coco_root: Optional[str] = None,
    local_coco_annotations: Optional[str] = None,
):
    """
    Load a streaming dataset and surface a useful message for script-backed repos.
    """
    import datasets as hf_datasets
    from datasets import load_dataset

    if local_coco_root or local_coco_annotations:
        if not (local_coco_root and local_coco_annotations):
            raise ValueError(
                "Using local COCO requires both --local_coco_root and --local_coco_annotations"
            )
        return iter_local_coco_samples(
            coco_root=local_coco_root,
            annotations_path=local_coco_annotations,
            split=split,
        )

    source = dataset_path or dataset_name
    try:
        dataset = load_dataset(
            source,
            split=split,
            streaming=True,
            trust_remote_code=trust_remote_code,
        )
        # Avoid datasets.Image() trying to decode broken zip-backed paths
        # produced by some script-based streaming datasets such as COCO.
        if hasattr(dataset, "decode"):
            dataset = dataset.decode(False)
        return dataset
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

        version = getattr(hf_datasets, "__version__", "unknown")
        raise RuntimeError(
            f"Unable to load dataset {source!r} with datasets {version}. "
            "This dataset is script-backed, and datasets>=4 no longer loads Hub "
            "dataset scripts. Fix one of these ways:\n"
            "  1. Downgrade: python -m pip install \"datasets<4\"\n"
            "  2. Clone the dataset repo locally and pass --dataset_path /path/to/COCO\n"
            "  3. Use a parquet/native dataset repo via --dataset_name"
        ) from exc


def skip_iterable(dataset, n: int):
    """Skip n items for either HF streaming datasets or plain Python iterables."""
    if n <= 0:
        return dataset
    if hasattr(dataset, "skip"):
        return dataset.skip(n)
    return islice(dataset, n, None)


def iter_local_coco_samples(
    *,
    coco_root: str,
    annotations_path: str,
    split: str,
):
    """
    Yield COCO Karpathy-split samples from local files.

    Expected layout under coco_root:
    - train2014/
    - val2014/
    annotations_path should point to dataset_coco.json from Karpathy captions.
    """
    coco_root_path = Path(coco_root)
    annotations_file = Path(annotations_path)

    train_dir = coco_root_path / "train2014"
    val_dir = coco_root_path / "val2014"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected {train_dir} and {val_dir} to exist under --local_coco_root"
        )
    if not annotations_file.exists():
        raise FileNotFoundError(
            f"Karpathy annotations file not found: {annotations_file}"
        )

    with open(annotations_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    def split_matches(image_split: str) -> bool:
        if split == "train":
            return image_split in {"train", "restval"}
        if split == "validation":
            return image_split == "val"
        if split == "test":
            return image_split == "test"
        raise ValueError(f"Unsupported split: {split}")

    for image_metadata in annotations["images"]:
        if not split_matches(image_metadata["split"]):
            continue

        filename = image_metadata["filename"]
        image_dir = val_dir if "val2014" in filename else train_dir
        image_path = image_dir / filename
        if not image_path.exists():
            continue

        for caption in image_metadata["sentences"]:
            yield {
                "image": str(image_path),
                "filepath": filename,
                "filename": filename,
                "imgid": image_metadata["imgid"],
                "split": image_metadata["split"],
                "sentences": {
                    "tokens": caption["tokens"],
                    "raw": caption["raw"],
                    "imgid": caption["imgid"],
                    "sentid": caption["sentid"],
                },
                "cocoid": image_metadata.get("cocoid"),
            }


def extract_caption(sample: dict) -> str:
    """Pick the most useful caption/text field for a dataset sample."""
    caption = sample.get("caption") or sample.get("text")

    if not caption:
        sentences = sample.get("sentences")
        if isinstance(sentences, dict):
            caption = sentences.get("raw")

    if not caption:
        sentences_raw = sample.get("sentences_raw")
        if isinstance(sentences_raw, list) and sentences_raw:
            caption = sentences_raw[0]

    if isinstance(caption, list):
        caption = caption[0] if caption else ""

    return caption.strip() if isinstance(caption, str) else ""


def coco_image_url(sample: dict) -> Optional[str]:
    """Rebuild a fetchable COCO image URL from the filename."""
    filename = sample.get("filename") or sample.get("filepath")
    if not isinstance(filename, str) or not filename:
        return None

    split_dir = "val2014" if "val2014" in filename else "train2014"
    return f"http://images.cocodataset.org/{split_dir}/{filename}"


def open_image_source(source: str):
    """Open a local/remote image path into a PIL image."""
    import fsspec
    from PIL import Image

    if os.path.exists(source):
        image = Image.open(source)
    else:
        with fsspec.open(source, "rb") as f:
            image = Image.open(BytesIO(f.read()))

    image.load()
    return image.convert("RGB")


def resolve_image(sample: dict):
    """
    Resolve a sample's image payload into a PIL image.

    Handles raw bytes, local paths, and COCO's broken zip-backed streaming paths.
    """
    from PIL import Image

    image_value = sample.get("image")

    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    path = None
    bytes_ = None

    if isinstance(image_value, dict):
        path = image_value.get("path")
        bytes_ = image_value.get("bytes")
    elif isinstance(image_value, str):
        path = image_value

    if bytes_ is not None:
        image = Image.open(BytesIO(bytes_))
        image.load()
        return image.convert("RGB")

    if isinstance(path, str) and path:
        # COCO's streaming script converts zip://... paths through pathlib,
        # producing unusable local-looking paths such as '/cwd/zip:/val2014/...'.
        if "zip:" in path:
            rebuilt = coco_image_url(sample)
            if rebuilt:
                return open_image_source(rebuilt)
        return open_image_source(path)

    rebuilt = coco_image_url(sample)
    if rebuilt:
        return open_image_source(rebuilt)

    return None


def image_reference(sample: dict) -> str:
    """Get a string image reference for chat-template rendering."""
    image_value = sample.get("image")

    path = None
    if isinstance(image_value, dict):
        path = image_value.get("path")
    elif isinstance(image_value, str):
        path = image_value

    if isinstance(path, str) and path and "zip:" not in path:
        if path.startswith(("http://", "https://", "oss://", "file://")):
            return path

        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return f"file://{abs_path}"

    rebuilt = coco_image_url(sample)
    if rebuilt:
        return rebuilt

    return "file:///image.jpg"


def build_vllm_prompt(
    apply_chat_template,
    *,
    instruction: str,
    text: Optional[str] = None,
    image_ref: Optional[str] = None,
) -> str:
    """Render a Qwen3-VL-compatible prompt for vLLM embedding."""
    content = []

    if image_ref:
        content.append({"type": "image", "image": image_ref})
    if text:
        content.append({"type": "text", "text": text})
    if not content:
        content.append({"type": "text", "text": ""})

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]
    return apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


def resolve_image_safe(sample: dict):
    """Resolve an image and return None on failure so batches can continue."""
    try:
        return resolve_image(sample)
    except Exception:
        return None


# Method 1: Using official Qwen3VLEmbedder

def extract_with_qwen3vl_embedder(
    model_path: str,
    output_dir: str,
    max_samples: int = 100_000,
    instruction: str = "Retrieve relevant content matching this input.",
    dataset_name: str = "HuggingFaceM4/COCO",
    dataset_split: str = "train",
    dataset_path: Optional[str] = None,
    trust_remote_code: bool = False,
    local_coco_root: Optional[str] = None,
    local_coco_annotations: Optional[str] = None,
):
    """
    Extract embeddings using the official Qwen3VLEmbedder class.
    
    This is the recommended approach — it handles all the multimodal
    input formatting, vision encoding, and EOS-token pooling internally.
    """
    import torch
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Qwen3VLEmbedder from {model_path}...")
    model = Qwen3VLEmbedder(
        model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    
    test_emb = model.encode([{"text": "hello"}])
    embedding_dim = test_emb.shape[1]
    print(f"  Embedding dimension: {embedding_dim}")
    
    dataset_source = dataset_path or dataset_name
    print(f"Loading dataset {dataset_source} [{dataset_split}]...")
    dataset = load_streaming_dataset(
        dataset_name,
        dataset_split,
        dataset_path,
        trust_remote_code=trust_remote_code,
        local_coco_root=local_coco_root,
        local_coco_annotations=local_coco_annotations,
    )
    
    n = max_samples
    img_embs = create_memmap(str(output_dir / "image_embeddings.npy"), (n, embedding_dim))
    txt_embs = create_memmap(str(output_dir / "text_embeddings.npy"), (n, embedding_dim))
    
    count = 0
    t0 = time.time()
    
    text_batch = []
    image_batch = []
    BATCH_SIZE = 32
    
    for sample in dataset:
        if count >= max_samples:
            break
        
        caption = extract_caption(sample)
        if not caption:
            continue
        try:
            image = resolve_image(sample)
        except Exception as exc:
            print(f"  Warning: failed to load image for sample {sample.get('filename', 'unknown')} ({exc})")
            continue
        if image is None:
            continue
        
        text_batch.append({
            "text": caption,
            "instruction": instruction,
        })
        image_batch.append({
            "image": image,  # PIL Image
            "instruction": instruction,
        })
        
        if len(text_batch) >= BATCH_SIZE:
            t_embs = model.encode(text_batch)  # (B, d) numpy
            txt_embs[count : count + len(text_batch)] = t_embs
            
            # Encode image batch (one by one — images have variable token counts)
            i_embs_list = []
            for img_input in image_batch:
                try:
                    emb = model.encode([img_input])  # (1, d) numpy
                    i_embs_list.append(emb[0])
                except Exception as e:
                    # Fallback: use text embedding if image fails
                    print(f"  Warning: image encode failed ({e}), using text fallback")
                    i_embs_list.append(t_embs[len(i_embs_list)])
            
            i_embs = np.stack(i_embs_list, axis=0)
            img_embs[count : count + len(image_batch)] = i_embs
            
            count += len(text_batch)
            text_batch = []
            image_batch = []
            
            if count % 500 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed
                print(f"  {count:,} pairs | {rate:.1f} pairs/s | "
                      f"ETA: {(max_samples - count) / rate / 60:.1f} min")
    
    if text_batch:
        t_embs = model.encode(text_batch)
        txt_embs[count : count + len(text_batch)] = t_embs
        for i, img_input in enumerate(image_batch):
            try:
                emb = model.encode([img_input])
                img_embs[count + i] = emb[0]
            except Exception:
                img_embs[count + i] = t_embs[i]
        count += len(text_batch)
    
    img_embs.flush()
    txt_embs.flush()
    
    metadata = {
        "model_name": model_path,
        "embedding_dim": embedding_dim,
        "num_samples": count,
        "instruction": instruction,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\nDone! {count:,} pairs in {elapsed / 60:.1f} min")
    print(f"  → {output_dir}")


# Method 2: Using raw transformers (no Qwen repo needed)

def extract_with_transformers(
    model_name: str,
    output_dir: str,
    max_samples: int = 100_000,
    embedding_dim: int = 2048,
    dataset_name: str = "HuggingFaceM4/COCO",
    dataset_split: str = "train",
    dataset_path: Optional[str] = None,
    trust_remote_code: bool = False,
    local_coco_root: Optional[str] = None,
    local_coco_annotations: Optional[str] = None,
):
    """
    Extract embeddings using raw HuggingFace transformers.
    
    Slightly more manual but doesn't require cloning the Qwen repo.
    Uses last-token pooling on the base model outputs.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer, AutoProcessor
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()
    
    def last_token_pool(hidden_states, attention_mask):
        """Extract the last non-padding token's hidden state."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths,
        ]
    
    def encode_texts(texts: list[str]) -> np.ndarray:
        """Encode text inputs with instruction prefix."""
        formatted = [
            f"Instruct: Retrieve relevant content matching this input.\nQuery: {t}"
            for t in texts
        ]
        batch_dict = tokenizer(
            formatted, padding=True, truncation=True,
            max_length=8192, return_tensors="pt",
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model(**batch_dict)
            embs = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1)
        
        return embs.cpu().float().numpy()
    
    dataset_source = dataset_path or dataset_name
    print(f"Loading dataset {dataset_source} [{dataset_split}]...")
    dataset = load_streaming_dataset(
        dataset_name,
        dataset_split,
        dataset_path,
        trust_remote_code=trust_remote_code,
        local_coco_root=local_coco_root,
        local_coco_annotations=local_coco_annotations,
    )
    
    n = max_samples
    img_embs = create_memmap(str(output_dir / "image_embeddings.npy"), (n, embedding_dim))
    txt_embs = create_memmap(str(output_dir / "text_embeddings.npy"), (n, embedding_dim))
    
    # For the raw transformers path, we encode text captions for both
    # (image encoding requires the VL processor — use Method 1 for images)
    # Here we extract TEXT embeddings only as a starting point
    print("NOTE: Raw transformers path extracts text-only embeddings.")
    print("      For image embeddings, use Method 1 (Qwen3VLEmbedder) or vLLM.")
    
    count = 0
    batch = []
    
    for sample in dataset:
        if count >= max_samples:
            break
        caption = extract_caption(sample)
        if not caption:
            continue
        batch.append(caption)
        
        if len(batch) >= 64:
            embs = encode_texts(batch)
            txt_embs[count : count + len(batch)] = embs
            # For text-only baseline: use same embedding for "image" side
            # (placeholder — replace with real image embeddings)
            img_embs[count : count + len(batch)] = embs
            count += len(batch)
            batch = []
            if count % 1000 == 0:
                print(f"  {count:,} samples...")
    
    if batch:
        embs = encode_texts(batch)
        txt_embs[count : count + len(batch)] = embs
        img_embs[count : count + len(batch)] = embs
        count += len(batch)
    
    img_embs.flush()
    txt_embs.flush()
    
    metadata = {
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "num_samples": count,
        "note": "text-only extraction via raw transformers",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDone! {count:,} text embeddings cached to {output_dir}")


# Method 3: Using vLLM (fastest for large-scale extraction)

def extract_with_vllm(
    model_name: str,
    output_dir: str,
    max_samples: int = 500_000,
    embedding_dim: int = 2048,
    dataset_name: str = "HuggingFaceM4/COCO",
    dataset_split: str = "train",
    dataset_path: Optional[str] = None,
    batch_size: int = 64,
    trust_remote_code: bool = False,
    image_workers: int = 8,
    resume: bool = False,
    local_coco_root: Optional[str] = None,
    local_coco_annotations: Optional[str] = None,
):
    """
    Extract embeddings using vLLM's embedding endpoint.
    
    vLLM is significantly faster than transformers for batch inference
    because it handles batching, memory, and scheduling automatically.
    
    Requires: pip install vllm>=0.8.5
    
    Supports both text and image inputs natively.
    """
    from tqdm.auto import tqdm
    from vllm import LLM
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    metadata_path = output_dir / "metadata.json"

    dataset_source = dataset_path or dataset_name
    print(f"Loading dataset {dataset_source} [{dataset_split}]...")
    dataset = load_streaming_dataset(
        dataset_name,
        dataset_split,
        dataset_path,
        trust_remote_code=trust_remote_code,
        local_coco_root=local_coco_root,
        local_coco_annotations=local_coco_annotations,
    )
    count = 0
    committed_seen_samples = 0

    if resume:
        if progress_path.exists():
            state = load_progress_state(
                progress_path,
                model_name=model_name,
                dataset_source=dataset_source,
                dataset_split=dataset_split,
                embedding_dim=embedding_dim,
                max_samples=max_samples,
            )
            count = int(state["count"])
            committed_seen_samples = int(state["seen_samples"])
            if committed_seen_samples:
                dataset = skip_iterable(dataset, committed_seen_samples)
            print(
                f"Resuming from {count:,} saved pairs after skipping "
                f"{committed_seen_samples:,} source samples."
            )
        elif metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            completed = int(metadata.get("num_samples", 0))
            if completed >= max_samples:
                print(f"Existing extraction already has {completed:,} pairs in {output_dir}.")
                return
            print("No progress file found; starting a fresh extraction run.")
    
    print(f"Loading {model_name} via vLLM...")
    llm_init = inspect.signature(LLM.__init__)
    llm_kwargs = {
        "model": model_name,
        "max_model_len": 8192,
        "dtype": "bfloat16",
    }
    if "task" in llm_init.parameters:
        llm_kwargs["task"] = "embed"
    else:
        # vLLM 0.15.x uses runner/convert instead of task.
        llm_kwargs["runner"] = "pooling"
        llm_kwargs["convert"] = "embed"
    llm = LLM(**llm_kwargs)
    apply_chat_template = llm.llm_engine.tokenizer.apply_chat_template
    instruction = "Retrieve relevant content matching this input."

    n = max_samples
    img_embs = open_memmap(
        str(output_dir / "image_embeddings.npy"),
        (n, embedding_dim),
        resume=resume,
    )
    txt_embs = open_memmap(
        str(output_dir / "text_embeddings.npy"),
        (n, embedding_dim),
        resume=resume,
    )
    
    text_inputs = []
    image_inputs = []
    pending_samples = []
    image_pool = ThreadPoolExecutor(max_workers=max(1, image_workers))
    progress = tqdm(
        total=max_samples,
        initial=count,
        desc="pairs",
        unit="pair",
        dynamic_ncols=True,
    )

    def write_progress():
        save_json_atomic(
            progress_path,
            {
                "model_name": model_name,
                "dataset_source": dataset_source,
                "dataset_split": dataset_split,
                "embedding_dim": embedding_dim,
                "max_samples": max_samples,
                "count": count,
                "seen_samples": committed_seen_samples,
                "batch_size": batch_size,
                "updated_at": time.time(),
            },
        )

    def flush_batch():
        nonlocal count, text_inputs, image_inputs, pending_samples, committed_seen_samples
        if not pending_samples:
            return

        image_inputs = []
        text_inputs = []
        resolved_images = list(image_pool.map(resolve_image_safe, pending_samples))

        for sample, image in zip(pending_samples, resolved_images):
            caption = extract_caption(sample)
            if not caption:
                continue
            text_inputs.append({
                "prompt": build_vllm_prompt(
                    apply_chat_template,
                    instruction=instruction,
                    text=caption,
                )
            })
            image_inputs.append(
                None if image is None else {
                    "prompt": build_vllm_prompt(
                        apply_chat_template,
                        instruction=instruction,
                        image_ref=image_reference(sample),
                    ),
                    "multi_modal_data": {"image": image},
                }
            )

        if not text_inputs:
            pending_samples = []
            committed_seen_samples = seen_samples
            write_progress()
            return

        batch_start = count
        text_outputs = llm.embed(text_inputs, use_tqdm=False)
        image_embeddings = [None] * len(text_inputs)
        valid_image_positions = [i for i, request in enumerate(image_inputs) if request is not None]
        valid_image_requests = [image_inputs[i] for i in valid_image_positions]

        if valid_image_requests:
            try:
                image_outputs = llm.embed(valid_image_requests, use_tqdm=False)
                for pos, out in zip(valid_image_positions, image_outputs):
                    image_embeddings[pos] = embedding_from_vllm_output(out)
            except Exception as exc:
                print(f"\nWarning: batched image encode failed ({exc}); falling back to per-image requests.")
                for pos, request in zip(valid_image_positions, valid_image_requests):
                    try:
                        output = llm.embed([request], use_tqdm=False)
                        image_embeddings[pos] = embedding_from_vllm_output(output[0])
                    except Exception:
                        image_embeddings[pos] = None

        for i, text_output in enumerate(text_outputs):
            text_embedding = embedding_from_vllm_output(text_output)
            txt_embs[batch_start + i] = text_embedding
            img_embs[batch_start + i] = (
                image_embeddings[i] if image_embeddings[i] is not None else text_embedding
            )

        batch_count = len(text_outputs)
        count += batch_count
        committed_seen_samples = seen_samples
        img_embs.flush()
        txt_embs.flush()
        write_progress()
        progress.update(batch_count)
        elapsed = max(time.time() - t0, 1e-6)
        progress.set_postfix_str(f"{count / elapsed:.1f} pair/s")
        text_inputs = []
        image_inputs = []
        pending_samples = []

    t0 = time.time()
    seen_samples = committed_seen_samples
    interrupted = False

    try:
        for sample in dataset:
            if count >= max_samples:
                break

            seen_samples += 1
            caption = extract_caption(sample)
            if not caption:
                continue

            pending_samples.append(sample)

            if len(pending_samples) >= batch_size or count + len(pending_samples) >= max_samples:
                flush_batch()
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Saving committed progress for resume...")
    finally:
        if not interrupted:
            flush_batch()

        progress.close()
        image_pool.shutdown(wait=True)
        img_embs.flush()
        txt_embs.flush()

    if interrupted:
        return

    metadata = {
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "num_samples": count,
        "extraction_method": "vllm",
        "dataset_source": dataset_source,
        "dataset_split": dataset_split,
        "batch_size": batch_size,
        "image_workers": image_workers,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    if progress_path.exists():
        progress_path.unlink()

    print(f"\nDone! {count:,} pairs via vLLM → {output_dir}")


def extract_mbeir_with_vllm(
    model_name: str,
    output_dir: str,
    *,
    embedding_dim: int,
    mbeir_root: str,
    dataset: str,
    split: str,
    candidate_scope: str = "local",
    query_path: Optional[str] = None,
    cand_pool_path: Optional[str] = None,
    qrels_path: Optional[str] = None,
    batch_size: int = 64,
    image_workers: int = 8,
    max_queries: int = 0,
    max_docs: int = 0,
    overwrite: bool = False,
    query_instruction: Optional[str] = None,
    doc_instruction: Optional[str] = None,
    query_text_prefix: Optional[str] = None,
    doc_text_prefix: Optional[str] = None,
):
    """
    Extract benchmark-style query/doc caches from a local M-BEIR split.

    This writes:
    - query_embeddings.npy / query_ids.json / query_meta.json
    - doc_embeddings.npy   / doc_ids.json   / doc_meta.json

    The resulting directory can be consumed directly by the retrieval-mode
    training path in train_puma.py.
    """
    from benchmark_mbeir import (
        DEFAULT_EMBED_SYSTEM_INSTRUCTION,
        VLLMEncoder,
        build_cache_metadata,
        default_query_text_prefix,
        inspect_dense_cache,
        load_cached_dense,
        load_jsonl,
        record_id,
        resolve_mbeir_paths,
    )

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

    query_cache_valid, query_cache_errors = inspect_dense_cache(
        output_dir, "query", expected_query_meta
    )
    doc_cache_valid, doc_cache_errors = inspect_dense_cache(
        output_dir, "doc", expected_doc_meta
    )
    need_dense = overwrite or not (query_cache_valid and doc_cache_valid)

    if need_dense:
        if query_cache_errors:
            print(f"  Query cache refresh: {'; '.join(query_cache_errors)}")
        if doc_cache_errors:
            print(f"  Doc cache refresh:   {'; '.join(doc_cache_errors)}")

        print(f"Loading {model_name} via vLLM...")
        encoder = VLLMEncoder(
            model_name=model_name,
            batch_size=batch_size,
            image_workers=image_workers,
        )
        query_embeddings, query_ids, query_cache_reused = encoder.encode(
            records=query_rows,
            mbeir_root=mbeir_root_path,
            is_query=True,
            cache_dir=output_dir,
            prefix="query",
            embedding_dim=embedding_dim,
            overwrite=overwrite,
            system_instruction=query_instruction,
            text_prefix=query_text_prefix,
            expected_ids=expected_query_ids,
        )
        doc_embeddings, doc_ids, doc_cache_reused = encoder.encode(
            records=doc_rows,
            mbeir_root=mbeir_root_path,
            is_query=False,
            cache_dir=output_dir,
            prefix="doc",
            embedding_dim=embedding_dim,
            overwrite=overwrite,
            system_instruction=doc_instruction,
            text_prefix=doc_text_prefix,
            expected_ids=expected_doc_ids,
        )
    else:
        print(f"Reusing dense cache from {output_dir}")
        query_embeddings, query_ids, _ = load_cached_dense(output_dir, "query")
        doc_embeddings, doc_ids, _ = load_cached_dense(output_dir, "doc")
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
        "num_queries": len(query_ids),
        "num_docs": len(doc_ids),
        "batch_size": batch_size,
        "image_workers": image_workers,
        "query_system_instruction": query_instruction,
        "doc_system_instruction": doc_instruction,
        "query_text_prefix": query_text_prefix,
        "doc_text_prefix": doc_text_prefix,
        "query_cache_reused": query_cache_reused,
        "doc_cache_reused": doc_cache_reused,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! query/doc retrieval cache ready at {output_dir}")
    print(f"  Queries: {len(query_ids):,}")
    print(f"  Docs:    {len(doc_ids):,}")


# Method 4: Synthetic (for testing without GPU)

def generate_synthetic(
    output_dir: str = "./cached_embeddings",
    num_samples: int = 100_000,
    embedding_dim: int = 2048,
    cross_modal_correlation: float = 0.7,
):
    """Generate synthetic paired embeddings for pipeline testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples:,} synthetic pairs (dim={embedding_dim})...")
    
    rng = np.random.default_rng(42)
    
    shared = rng.standard_normal((num_samples, embedding_dim)).astype(np.float32)
    
    img_noise = rng.standard_normal((num_samples, embedding_dim)).astype(np.float32)
    txt_noise = rng.standard_normal((num_samples, embedding_dim)).astype(np.float32)
    
    r = cross_modal_correlation
    img_raw = r * shared + (1 - r) * img_noise
    txt_raw = r * shared + (1 - r) * txt_noise
    
    img_embs = img_raw / np.linalg.norm(img_raw, axis=-1, keepdims=True)
    txt_embs = txt_raw / np.linalg.norm(txt_raw, axis=-1, keepdims=True)
    
    img_mm = create_memmap(str(output_dir / "image_embeddings.npy"), (num_samples, embedding_dim))
    img_mm[:] = img_embs
    img_mm.flush()
    
    txt_mm = create_memmap(str(output_dir / "text_embeddings.npy"), (num_samples, embedding_dim))
    txt_mm[:] = txt_embs
    txt_mm.flush()
    
    metadata = {
        "model_name": "synthetic",
        "embedding_dim": embedding_dim,
        "num_samples": num_samples,
        "cross_modal_correlation": cross_modal_correlation,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    cos_sims = np.sum(img_embs[:1000] * txt_embs[:1000], axis=-1)
    print(f"  Mean paired cosine similarity: {cos_sims.mean():.3f}")
    print(f"  Saved to {output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Qwen3-VL-Embedding embeddings")
    parser.add_argument("--method", type=str, default="qwen3vl",
                       choices=["qwen3vl", "transformers", "vllm", "mbeir_vllm", "synthetic"],
                       help="Extraction method")
    parser.add_argument("--model_path", type=str, 
                       default="./models/Qwen3-VL-Embedding-2B",
                       help="Path to model weights")
    parser.add_argument("--output_dir", type=str, default="./cached_embeddings")
    parser.add_argument("--max_samples", type=int, default=100_000)
    parser.add_argument("--embedding_dim", type=int, default=2048,
                       help="2048 for 2B, 4096 for 8B")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceM4/COCO",
                       help="Dataset repo to stream from Hugging Face")
    parser.add_argument("--dataset_split", type=str, default="train",
                       help="Dataset split to read")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Local dataset path; useful for script-backed repos with datasets>=4")
    parser.add_argument("--local_coco_root", type=str, default=None,
                       help="Local COCO root containing train2014/ and val2014/")
    parser.add_argument("--local_coco_annotations", type=str, default=None,
                       help="Path to Karpathy dataset_coco.json for local COCO extraction")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for vLLM extraction")
    parser.add_argument("--image_workers", type=int, default=8,
                       help="Concurrent workers for image loading in the vLLM path")
    parser.add_argument("--resume", action="store_true",
                       help="Resume an interrupted extraction from output_dir/progress.json")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite an existing compatible cache")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Allow datasets to execute custom loading code from the dataset repo")
    parser.add_argument("--synthetic", action="store_true",
                       help="Generate synthetic data (no GPU needed)")
    parser.add_argument("--mbeir_root", type=str, default="./M-BEIR",
                       help="Root directory for local M-BEIR data")
    parser.add_argument("--mbeir_dataset", type=str, default=None,
                       help="M-BEIR dataset/task name, e.g. cirr_task7")
    parser.add_argument("--candidate_scope", choices=["local", "global"], default="local")
    parser.add_argument("--query_path", type=str, default=None)
    parser.add_argument("--cand_pool_path", type=str, default=None)
    parser.add_argument("--qrels_path", type=str, default=None)
    parser.add_argument("--query_instruction", type=str, default=None)
    parser.add_argument("--doc_instruction", type=str, default=None)
    parser.add_argument("--query_text_prefix", type=str, default=None)
    parser.add_argument("--doc_text_prefix", type=str, default=None)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--max_docs", type=int, default=0)
    args = parser.parse_args()
    
    if args.synthetic or args.method == "synthetic":
        generate_synthetic(
            output_dir=args.output_dir,
            num_samples=args.max_samples,
            embedding_dim=args.embedding_dim,
        )
    elif args.method == "qwen3vl":
        extract_with_qwen3vl_embedder(
            model_path=args.model_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            dataset_path=args.dataset_path,
            trust_remote_code=args.trust_remote_code,
            local_coco_root=args.local_coco_root,
            local_coco_annotations=args.local_coco_annotations,
        )
    elif args.method == "transformers":
        extract_with_transformers(
            model_name=args.model_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            embedding_dim=args.embedding_dim,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            dataset_path=args.dataset_path,
            trust_remote_code=args.trust_remote_code,
            local_coco_root=args.local_coco_root,
            local_coco_annotations=args.local_coco_annotations,
        )
    elif args.method == "vllm":
        extract_with_vllm(
            model_name=args.model_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            embedding_dim=args.embedding_dim,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            image_workers=args.image_workers,
            resume=args.resume,
            trust_remote_code=args.trust_remote_code,
            local_coco_root=args.local_coco_root,
            local_coco_annotations=args.local_coco_annotations,
        )
    elif args.method == "mbeir_vllm":
        if not args.mbeir_dataset:
            parser.error("--mbeir_dataset is required for --method mbeir_vllm")
        extract_mbeir_with_vllm(
            model_name=args.model_path,
            output_dir=args.output_dir,
            embedding_dim=args.embedding_dim,
            mbeir_root=args.mbeir_root,
            dataset=args.mbeir_dataset,
            split=args.dataset_split,
            candidate_scope=args.candidate_scope,
            query_path=args.query_path,
            cand_pool_path=args.cand_pool_path,
            qrels_path=args.qrels_path,
            batch_size=args.batch_size,
            image_workers=args.image_workers,
            max_queries=args.max_queries,
            max_docs=args.max_docs,
            overwrite=args.overwrite,
            query_instruction=args.query_instruction,
            doc_instruction=args.doc_instruction,
            query_text_prefix=args.query_text_prefix,
            doc_text_prefix=args.doc_text_prefix,
        )
