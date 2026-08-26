"""
Minimal checkpoint utilities for the sparse autoencoder.

The project currently trains inline inside ``train_puma.py`` and saves a
checkpoint that other modules expect to load via ``load_checkpoint``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sae import SAEConfig, SparseAutoencoder, EncoderOnlySparseEncoder


def _coerce_sae_config(raw_config: Any) -> SAEConfig:
    """Accept either a dict payload or an SAEConfig instance."""
    if isinstance(raw_config, SAEConfig):
        return raw_config
    if not isinstance(raw_config, dict):
        raise TypeError(f"Unsupported SAE config payload: {type(raw_config)!r}")
    return SAEConfig(**raw_config)


def _build_model(model_type: str, config: SAEConfig):
    if model_type == "sae":
        return SparseAutoencoder(config)
    if model_type == "encoder_only":
        return EncoderOnlySparseEncoder(config)
    raise ValueError(f"Unsupported sparse model type: {model_type}")


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """
    Restore a trained SAE from disk.

    Supports the checkpoint layout written by ``train_puma.py``:
    - ``sae_state_dict``
    - ``sae_config``
    - ``current_k``
    """
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location=device)

    state_dict = checkpoint.get("sae_state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_file} does not contain an SAE state dict")

    raw_config = checkpoint.get("sae_config") or checkpoint.get("config")
    if raw_config is None:
        raise KeyError(f"Checkpoint {checkpoint_file} does not contain an SAE config")

    config = _coerce_sae_config(raw_config)
    model_type = checkpoint.get("model_type", "sae")
    model = _build_model(model_type, config).to(device)
    model.load_state_dict(state_dict)
    model.current_k = checkpoint.get("current_k", config.k_final)
    return model
