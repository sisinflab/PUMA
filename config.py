"""
Configuration for the full Qwen3-VL-Embedding → Sparse SAE pipeline.

Two model sizes supported:
  - Qwen3-VL-Embedding-2B: 2048-d embeddings, 28 layers
  - Qwen3-VL-Embedding-8B: 4096-d embeddings, 36 layers
"""

from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Qwen3-VL-Embedding model configuration."""
    model_name: str = "Qwen/Qwen3-VL-Embedding-2B"
    model_local_path: Optional[str] = None   # e.g., "./models/Qwen3-VL-Embedding-2B"
    embedding_dim: int = 2048                 # 2B→2048, 8B→4096
    num_layers: int = 28                      # 2B→28, 8B→36
    max_length: int = 8192
    # Image processing
    min_pixels: int = 4096
    max_pixels: int = 1843200                 # 1280×1440
    # Matryoshka dims (optional: can extract shorter embeddings)
    mrl_dim: Optional[int] = None             # None = use full dim
    
    @classmethod
    def qwen3_vl_2b(cls, local_path: Optional[str] = None) -> "ModelConfig":
        return cls(
            model_name="Qwen/Qwen3-VL-Embedding-2B",
            model_local_path=local_path or "./models/Qwen3-VL-Embedding-2B",
            embedding_dim=2048,
            num_layers=28,
        )
    
    @classmethod
    def qwen3_vl_8b(cls, local_path: Optional[str] = None) -> "ModelConfig":
        return cls(
            model_name="Qwen/Qwen3-VL-Embedding-8B",
            model_local_path=local_path or "./models/Qwen3-VL-Embedding-8B",
            embedding_dim=4096,
            num_layers=36,
        )


@dataclass
class SAEConfig:
    """Sparse Autoencoder configuration."""
    input_dim: int = 2048              # must match ModelConfig.embedding_dim
    expansion_factor: int = 16         # dict_size = input_dim * expansion_factor
    k_initial: int = 256               # starting sparsity
    k_final: int = 32                  # target sparsity
    k_annealing_steps: int = 40000     # steps for progressive annealing
    k_annealing_schedule: str = "cosine"  # "linear" or "cosine"
    tied_decoder: bool = False
    normalize_decoder: bool = True
    pre_encoder_bias: bool = True
    
    @property
    def dict_size(self) -> int:
        return self.input_dim * self.expansion_factor


@dataclass
class DataConfig:
    """Data extraction and loading configuration."""
    # Datasets for paired embedding extraction
    datasets: list[str] = field(default_factory=lambda: [
        "HuggingFaceM4/COCO",
    ])
    max_samples: int = 500_000
    cache_dir: str = "./cached_embeddings"
    val_fraction: float = 0.02
    # Extraction batch sizes
    text_batch_size: int = 64
    image_batch_size: int = 8   # smaller: VLM image processing is memory-heavy
    # Instructions for embedding extraction
    query_instruction: str = "Retrieve relevant content matching this input."
    doc_instruction: str = ""  # docs typically don't use instructions


@dataclass 
class Stage2Config:
    """SAE reconstruction + alignment training."""
    epochs: int = 10
    batch_size: int = 4096       # large: SAE is tiny, data throughput matters
    lr: float = 3e-4
    weight_decay: float = 1e-5
    alpha_align: float = 0.3     # cross-modal alignment weight
    alpha_dead: float = 0.01     # dead feature penalty
    normalize_decoder_every: int = 100
    gradient_clip: float = 1.0
    warmup_steps: int = 1000


@dataclass
class Stage3Config:
    """Contrastive retrieval fine-tuning."""
    epochs: int = 5
    batch_size: int = 512
    lr: float = 1e-4
    temperature: float = 0.05
    lambda_recon: float = 0.1    # reconstruction regularizer weight
    num_negatives: int = 7
    gradient_clip: float = 1.0
    # Optional: LoRA on the backbone (expensive but potentially better)
    use_backbone_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    
    checkpoint_dir: str = "./checkpoints"
    device: str = "cuda"
    seed: int = 42
    log_every: int = 100
    val_every: int = 1000
    
    def __post_init__(self):
        # Ensure SAE input dim matches model embedding dim
        self.sae.input_dim = self.model.embedding_dim
    
    @classmethod
    def for_2b(cls, **kwargs) -> "PipelineConfig":
        """Pre-configured for Qwen3-VL-Embedding-2B."""
        config = cls(
            model=ModelConfig.qwen3_vl_2b(),
            sae=SAEConfig(input_dim=2048, expansion_factor=16),  # 32768 dict
        )
        for k, v in kwargs.items():
            setattr(config, k, v)
        return config
    
    @classmethod
    def for_8b(cls, **kwargs) -> "PipelineConfig":
        """Pre-configured for Qwen3-VL-Embedding-8B."""
        config = cls(
            model=ModelConfig.qwen3_vl_8b(),
            sae=SAEConfig(input_dim=4096, expansion_factor=16),  # 65536 dict
        )
        for k, v in kwargs.items():
            setattr(config, k, v)
        return config
    
    def save(self, path: str):
        """Save config to JSON."""
        import dataclasses
        d = dataclasses.asdict(self)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Load config from JSON."""
        with open(path) as f:
            d = json.load(f)
        return cls(
            model=ModelConfig(**d["model"]),
            sae=SAEConfig(**d["sae"]),
            data=DataConfig(**d["data"]),
            stage2=Stage2Config(**d["stage2"]),
            stage3=Stage3Config(**d["stage3"]),
            checkpoint_dir=d["checkpoint_dir"],
            device=d["device"],
            seed=d["seed"],
        )


if __name__ == "__main__":
    print("=== Qwen3-VL-Embedding-2B config ===")
    cfg = PipelineConfig.for_2b()
    print(f"  Embedding dim:  {cfg.model.embedding_dim}")
    print(f"  SAE dict size:  {cfg.sae.dict_size}")
    print(f"  Sparsity:       k={cfg.sae.k_initial} → {cfg.sae.k_final}")
    print(f"  Sparsity ratio: {1 - cfg.sae.k_final / cfg.sae.dict_size:.4%}")
    
    print("\n=== Qwen3-VL-Embedding-8B config ===")
    cfg = PipelineConfig.for_8b()
    print(f"  Embedding dim:  {cfg.model.embedding_dim}")
    print(f"  SAE dict size:  {cfg.sae.dict_size}")
    print(f"  Sparsity:       k={cfg.sae.k_initial} → {cfg.sae.k_final}")
    print(f"  Sparsity ratio: {1 - cfg.sae.k_final / cfg.sae.dict_size:.4%}")
