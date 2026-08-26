from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DenseAEConfig:
    input_dim: int = 2048
    bottleneck_dim: int = 256
    hidden_dim: int = 0
    normalize_latents: bool = True
    architecture: str = "linear"


def resolve_hidden_dim(input_dim: int, bottleneck_dim: int, requested_hidden_dim: int = 0) -> int:
    if requested_hidden_dim > 0:
        return int(requested_hidden_dim)
    return min(int(input_dim), max(512, 4 * int(bottleneck_dim)))


class DenseAutoencoder(nn.Module):

    supports_reconstruction = True

    def __init__(self, config: DenseAEConfig):
        super().__init__()
        self.config = config
        self.input_dim = int(config.input_dim)
        self.bottleneck_dim = int(config.bottleneck_dim)
        self.normalize_latents = bool(config.normalize_latents)
        self.architecture = str(getattr(config, "architecture", "linear")).lower()
        if self.architecture not in {"linear", "mlp"}:
            raise ValueError(f"Unsupported dense AE architecture: {self.architecture}")

        if self.architecture == "linear":
            self.hidden_dim = 0
            self.encoder = nn.Linear(self.input_dim, self.bottleneck_dim)
            self.decoder = nn.Linear(self.bottleneck_dim, self.input_dim)
        else:
            self.hidden_dim = resolve_hidden_dim(
                self.input_dim,
                self.bottleneck_dim,
                int(config.hidden_dim),
            )
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.bottleneck_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.bottleneck_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.input_dim),
            )

    def encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode(self, x: torch.Tensor, normalize: bool | None = None) -> torch.Tensor:
        z = self.encode_raw(x)
        should_normalize = self.normalize_latents if normalize is None else bool(normalize)
        if should_normalize:
            z = F.normalize(z, dim=-1, eps=1e-8)
        return z

    def decode(self, z_raw: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_raw)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_raw = self.encode_raw(x)
        z = F.normalize(z_raw, dim=-1, eps=1e-8) if self.normalize_latents else z_raw
        x_hat = self.decode(z_raw)
        return {
            "x": x,
            "z": z,
            "z_raw": z_raw,
            "x_hat": x_hat,
            "active_dims": torch.tensor(float(self.bottleneck_dim), device=x.device),
        }

    def normalize_decoder_(self):
        """Compatibility no-op with sparse training loops."""
        return None


def dense_ae_config_to_dict(config: DenseAEConfig) -> dict[str, Any]:
    payload = asdict(config)
    architecture = str(getattr(config, "architecture", "linear")).lower()
    payload["architecture"] = architecture
    payload["hidden_dim"] = (
        0
        if architecture == "linear"
        else resolve_hidden_dim(
            int(config.input_dim),
            int(config.bottleneck_dim),
            int(config.hidden_dim),
        )
    )
    return payload


def _coerce_config(raw_config: Any) -> DenseAEConfig:
    if isinstance(raw_config, DenseAEConfig):
        return raw_config
    if not isinstance(raw_config, dict):
        raise TypeError(f"Unsupported DenseAE config payload: {type(raw_config)!r}")
    return DenseAEConfig(**raw_config)


def save_dense_autoencoder_checkpoint(
    path: str | Path,
    model: DenseAutoencoder,
    *,
    training_stage: str,
    training_source: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dense_ae_state_dict": model.state_dict(),
            "dense_ae_config": dense_ae_config_to_dict(model.config),
            "model_type": "dense_autoencoder",
            "training_stage": training_stage,
            "training_source": training_source or {},
            "metrics": metrics or {},
        },
        path,
    )


def load_dense_autoencoder_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[DenseAutoencoder, dict[str, Any]]:
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    state_dict = checkpoint.get("dense_ae_state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_file} does not contain a dense AE state dict")

    raw_config = checkpoint.get("dense_ae_config") or checkpoint.get("config")
    if raw_config is None:
        raise KeyError(f"Checkpoint {checkpoint_file} does not contain a dense AE config")

    if isinstance(raw_config, dict) and "architecture" not in raw_config:
        raw_config = dict(raw_config)
        raw_config["architecture"] = (
            "mlp"
            if any(key.startswith("encoder.0.") or key.startswith("decoder.0.") for key in state_dict)
            else "linear"
        )

    config = _coerce_config(raw_config)
    model = DenseAutoencoder(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint
