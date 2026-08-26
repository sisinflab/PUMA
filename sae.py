"""
Sparse Autoencoder for multimodal embedding sparsification.

Key components:
- TopK activation for exact sparsity control
- Cosine reconstruction loss (stable for normalized embeddings)
- Group-sparse cross-modal alignment loss
- Progressive k-annealing scheduler
- AuxK dead feature revival
- Optional: tied/untied decoder weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SAEConfig:
    """Configuration for the Sparse Autoencoder."""
    input_dim: int = 4096          # d: VLM embedding dimension
    dict_size: int = 65536         # D: overcomplete dictionary size (16x expansion)
    k_initial: int = 256           # starting sparsity (easy)
    k_final: int = 32              # target sparsity (hard)
    k_annealing_steps: int = 50000 # steps to anneal from k_initial to k_final
    k_annealing_schedule: str = "linear"  # "linear" or "cosine"
    tied_decoder: bool = False     # tie W_dec = W_enc.T (saves params, sometimes worse)
    normalize_decoder: bool = True # unit-norm decoder columns (prevents scale drift)
    pre_encoder_bias: bool = True  # subtract decoder bias before encoding (Anthropic trick)
    dtype: torch.dtype = torch.float32


class TopKActivation(nn.Module):

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        # x shape: (batch, dict_size)
        topk_values, topk_indices = torch.topk(x, k=k, dim=-1)

        result = torch.zeros_like(x)
        result.scatter_(dim=-1, index=topk_indices, src=topk_values)

        # topk and scatter_ are differentiable: gradients flow only through
        # the selected top-k positions automatically.
        return result


class SparseAutoencoder(nn.Module):


    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        d, D = config.input_dim, config.dict_size

        self.W_enc = nn.Linear(d, D, bias=True, dtype=config.dtype)

        if config.tied_decoder:
            # W_dec shares weights with W_enc (transposed)
            self.W_dec_weight = None  # use W_enc.weight.T
        else:
            self.W_dec = nn.Linear(D, d, bias=True, dtype=config.dtype)

        self.topk = TopKActivation()

        self._current_k = config.k_initial

        self._init_weights()

    def _init_weights(self):

        nn.init.kaiming_uniform_(self.W_enc.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_enc.bias)

        if not self.config.tied_decoder:
            nn.init.kaiming_uniform_(self.W_dec.weight, a=math.sqrt(5))
            nn.init.zeros_(self.W_dec.bias)

            if self.config.normalize_decoder:
                with torch.no_grad():
                    # Normalize decoder columns to unit norm
                    # W_dec shape: (d, D) -- each column is a dictionary vector
                    self.W_dec.weight.data = F.normalize(
                        self.W_dec.weight.data, dim=0
                    )

    @property
    def current_k(self) -> int:
        return self._current_k

    @current_k.setter
    def current_k(self, value: int):
        self._current_k = max(1, min(value, self.config.dict_size))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode dense embedding -> sparse code.

        Args:
            x: (batch, input_dim) dense embeddings
        Returns:
            z: (batch, dict_size) sparse codes with exactly k non-zeros
        """
        # Pre-encoder bias subtraction (Anthropic trick)
        if self.config.pre_encoder_bias and not self.config.tied_decoder:
            x_centered = x - self.W_dec.bias
        else:
            x_centered = x

        h = self.W_enc(x_centered)
        h = F.relu(h)

        z = self.topk(h, k=self._current_k)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse code -> reconstructed dense embedding.

        Args:
            z: (batch, dict_size) sparse codes
        Returns:
            x_hat: (batch, input_dim) reconstructed embeddings
        """
        if self.config.tied_decoder:
            x_hat = F.linear(z, self.W_enc.weight.t()) + self.W_enc.bias
        else:
            x_hat = self.W_dec(z)

        return x_hat

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> decode.

        Returns dict with all intermediate values needed for loss computation,
        including h_pre (pre-ReLU activations) needed for AuxK dead feature revival.
        """
        if self.config.pre_encoder_bias and not self.config.tied_decoder:
            x_centered = x - self.W_dec.bias
        else:
            x_centered = x

        # Linear projection (pre-ReLU) -- keep for AuxK loss
        h_pre = self.W_enc(x_centered)
        h = F.relu(h_pre)

        z = self.topk(h, k=self._current_k)

        x_hat = self.decode(z)

        return {
            "x": x,             # original embedding
            "z": z,             # sparse code
            "x_hat": x_hat,     # reconstruction
            "h_pre": h_pre,     # pre-ReLU activations (for AuxK)
            "k": self._current_k,
            "active_dims": (z != 0).float().sum(dim=-1).mean(),  # sanity check
        }

    @torch.no_grad()
    def normalize_decoder_(self):
        
        if not self.config.tied_decoder and self.config.normalize_decoder:
            self.W_dec.weight.data = F.normalize(
                self.W_dec.weight.data, dim=0
            )

    @torch.no_grad()
    def get_active_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the indices and values of active features for a batch.
        Returns sparse representation suitable for inverted index.

        Returns:
            indices: (batch, k) -- which dictionary elements are active
            values:  (batch, k) -- their activation magnitudes
        """
        z = self.encode(x)
        # z has exactly k non-zeros per row
        values, indices = torch.topk(z, k=self._current_k, dim=-1)
        return indices, values




class EncoderOnlySparseEncoder(nn.Module):
    """
    Sparse encoder without a reconstruction decoder.

    This baseline keeps the same TopK encoder parameterization as the SAE but
    removes the decoder and reconstruction pathway entirely. Training is then
    driven only by retrieval-aligned objectives in ``train_puma.py``.
    """

    supports_reconstruction: bool = False

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        d, D = config.input_dim, config.dict_size

        self.W_enc = nn.Linear(d, D, bias=True, dtype=config.dtype)
        self.topk = TopKActivation()
        self._current_k = config.k_initial

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_enc.bias)

    @property
    def current_k(self) -> int:
        return self._current_k

    @current_k.setter
    def current_k(self, value: int):
        self._current_k = max(1, min(int(value), self.config.dict_size))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h_pre = self.W_enc(x)
        h = F.relu(h_pre)
        return self.topk(h, k=self._current_k)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h_pre = self.W_enc(x)
        h = F.relu(h_pre)
        z = self.topk(h, k=self._current_k)
        return {
            "x": x,
            "z": z,
            "h_pre": h_pre,
            "k": self._current_k,
            "active_dims": (z != 0).float().sum(dim=-1).mean(),
        }

    @torch.no_grad()
    def normalize_decoder_(self):
        """Encoder-only baseline has no decoder; keep the training loop interface consistent."""
        return None

    @torch.no_grad()
    def get_active_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        values, indices = torch.topk(z, k=self._current_k, dim=-1)
        return indices, values


# Backward-compatible alias for the earlier encoder-only baseline implementation.
ContrastiveSparseEncoder = EncoderOnlySparseEncoder


class KAnnealingScheduler:

    def __init__(self, config: SAEConfig):
        self.k_initial = config.k_initial
        self.k_final = config.k_final
        self.total_steps = config.k_annealing_steps
        self.schedule = config.k_annealing_schedule

    def get_k(self, step: int) -> int:
        """Get current k for a given training step."""
        if step >= self.total_steps:
            return self.k_final

        progress = step / self.total_steps  # 0 -> 1

        if self.schedule == "linear":
            k = self.k_initial + (self.k_final - self.k_initial) * progress
        elif self.schedule == "cosine":
            # Cosine annealing: slow start, fast middle, slow end
            k = self.k_final + (self.k_initial - self.k_final) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return max(self.k_final, int(round(k)))


class SAELoss(nn.Module):

    def __init__(
        self,
        alpha_align: float = 0.3,     # weight for cross-modal alignment
        alpha_auxk: float = 1/32,     # weight for AuxK dead feature loss
        dead_threshold: int = 1000,   # steps before a feature is "dead"
        auxk_k: int = 512,            # how many dead features to activate in aux pass
    ):
        super().__init__()
        self.alpha_align = alpha_align
        self.alpha_auxk = alpha_auxk
        self.dead_threshold = dead_threshold
        self.auxk_k = auxk_k

        self.register_buffer(
            "steps_since_active", None  # initialized lazily
        )
        self._step = 0

    def reconstruction_loss(
        self, x: torch.Tensor, x_hat: torch.Tensor
    ) -> torch.Tensor:
        cos_sim = F.cosine_similarity(x, x_hat, dim=-1)
        return (1 - cos_sim).mean()

    def alignment_loss(
        self,
        z_image: torch.Tensor,
        z_text: torch.Tensor,
    ) -> torch.Tensor:
        pattern_img = (z_image.abs() > 0).float()
        pattern_txt = (z_text.abs() > 0).float()

        pattern_diff = (pattern_img - pattern_txt).abs().sum(dim=-1).mean()

        co_active = (pattern_img * pattern_txt)  # 1 where both active
        magnitude_diff = (
            (z_image.abs() - z_text.abs()).abs() * co_active
        ).sum(dim=-1).mean()

        return pattern_diff + 0.5 * magnitude_diff

    def _update_activity_tracking(self, z: torch.Tensor):
        """Track which features have been active recently."""
        if self.steps_since_active is None:
            self.steps_since_active = torch.zeros(
                z.shape[-1], device=z.device, dtype=torch.long
            )

        active_mask = (z.abs() > 0).any(dim=0)  # (dict_size,)
        self.steps_since_active[active_mask] = 0
        self.steps_since_active[~active_mask] += 1
        self._step += 1

    def auxk_loss(
        self,
        h_pre: torch.Tensor,     # (batch, dict_size) pre-ReLU encoder activations
        z: torch.Tensor,          # (batch, dict_size) sparse codes from main TopK
        x: torch.Tensor,          # (batch, input_dim) original input
        sae: "SparseAutoencoder",
    ) -> torch.Tensor:
        
        self._update_activity_tracking(z)

        dead_mask = self.steps_since_active > self.dead_threshold  # (dict_size,)
        num_dead = dead_mask.sum().item()

        if num_dead == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # Get pre-activations at dead positions only; mask out alive features
        h_dead = h_pre.clone()
        h_dead[:, ~dead_mask] = float('-inf')

        k_aux = min(self.auxk_k, num_dead)
        topk_values, topk_indices = torch.topk(h_dead, k=k_aux, dim=-1)

        # Build sparse activation vector for dead features. Softplus keeps the
        # branch close to ReLU for positive values while still giving slightly
        # negative dead pre-activations a gradient to move upward.
        z_aux = torch.zeros_like(z)
        z_aux.scatter_(dim=-1, index=topk_indices, src=F.softplus(topk_values))

        # The auxiliary features should explain what the main features missed
        x_hat_main = sae.decode(z)
        residual = (x - x_hat_main).detach()  # detach so gradients only flow through aux path

        x_hat_aux = sae.decode(z_aux)
        return (x_hat_aux - residual).pow(2).sum(dim=-1).mean()

    @property
    def dead_fraction(self) -> float:
        if self.steps_since_active is None:
            return 0.0
        return float((self.steps_since_active > self.dead_threshold).float().mean().item())

    def forward(
        self,
        output: dict[str, torch.Tensor],
        output_paired: Optional[dict[str, torch.Tensor]] = None,
        sae: Optional["SparseAutoencoder"] = None,
    ) -> dict[str, torch.Tensor]:

        losses = {}

        # 1. Reconstruction loss (always)
        losses["recon"] = self.reconstruction_loss(output["x"], output["x_hat"])

        # 2. Cross-modal alignment loss (only when we have pairs)
        if output_paired is not None:
            losses["align"] = self.alignment_loss(
                output["z"], output_paired["z"]
            )
        else:
            losses["align"] = torch.tensor(0.0, device=output["x"].device)

        # 3. AuxK dead feature revival loss
        if sae is not None and "h_pre" in output:
            losses["auxk"] = self.auxk_loss(
                output["h_pre"], output["z"], output["x"], sae
            )
        else:
            # Still track activity even without AuxK
            self._update_activity_tracking(output["z"])
            losses["auxk"] = torch.tensor(0.0, device=output["x"].device)

        # Dead fraction for logging only (no gradient)
        losses["dead_frac"] = torch.tensor(self.dead_fraction, device=output["x"].device)

        losses["total"] = (
            losses["recon"]
            + self.alpha_align * losses["align"]
            + self.alpha_auxk * losses["auxk"]
        )

        return losses


class SparseContrastiveLoss(nn.Module):

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_query: torch.Tensor,   # (B, D) sparse query codes
        z_pos: torch.Tensor,     # (B, D) sparse positive document codes
        z_neg: Optional[torch.Tensor] = None,  # (B, N, D) hard negatives
    ) -> torch.Tensor:
        
        pos_scores = (z_query * z_pos).sum(dim=-1, keepdim=True)  # (B, 1)

        inbatch_scores = torch.mm(z_query, z_pos.t())  # (B, B)

        if z_neg is not None:
            # When explicit positives are prepended at column 0, the diagonal of
            # the in-batch matrix would duplicate the same positive and then be
            # treated as a negative class. Mask it out instead.
            inf_mask = torch.eye(
                z_query.size(0),
                dtype=torch.bool,
                device=z_query.device,
            )
            inbatch_scores = inbatch_scores.masked_fill(inf_mask, float("-inf"))

            neg_scores = torch.bmm(
                z_neg, z_query.unsqueeze(-1)
            ).squeeze(-1)
            all_scores = torch.cat([pos_scores, inbatch_scores, neg_scores], dim=-1)
        else:
            all_scores = inbatch_scores

        all_scores = all_scores / self.temperature

        # Labels: the positive is always at index 0 (for pos_scores path)
        # or on the diagonal (for inbatch path)
        labels = torch.arange(z_query.size(0), device=z_query.device)

        if z_neg is not None:
            # When using explicit pos + negatives, positive is at index 0
            labels = torch.zeros(z_query.size(0), dtype=torch.long, device=z_query.device)
            loss = F.cross_entropy(all_scores, labels)
        else:
            # In-batch negatives: positive is on the diagonal
            loss = F.cross_entropy(all_scores, labels)

        return loss



if __name__ == "__main__":
    config = SAEConfig(
        input_dim=4096,
        dict_size=65536,
        k_initial=256,
        k_final=32,
        k_annealing_steps=50000,
    )

    sae = SparseAutoencoder(config)
    scheduler = KAnnealingScheduler(config)
    loss_fn = SAELoss(alpha_align=0.3)

    print(f"SAE parameters: {sum(p.numel() for p in sae.parameters()):,}")
    print(f"  Encoder: {config.input_dim} -> {config.dict_size}")
    print(f"  Decoder: {config.dict_size} -> {config.input_dim}")
    print(f"  k schedule: {config.k_initial} -> {config.k_final} over {config.k_annealing_steps} steps")

    batch_size = 8
    x_img = F.normalize(torch.randn(batch_size, config.input_dim), dim=-1)
    x_txt = F.normalize(torch.randn(batch_size, config.input_dim), dim=-1)

    step = 1000
    sae.current_k = scheduler.get_k(step)
    print(f"\nStep {step}: k = {sae.current_k}")

    out_img = sae(x_img)
    out_txt = sae(x_txt)

    losses = loss_fn(out_img, out_txt, sae=sae)
    print(f"  Reconstruction loss: {losses['recon']:.4f}")
    print(f"  Alignment loss:      {losses['align']:.4f}")
    print(f"  AuxK loss:           {losses['auxk']:.4f}")
    print(f"  Dead fraction:       {losses['dead_frac']:.4f}")
    print(f"  Total loss:          {losses['total']:.4f}")
    print(f"  Active dims (avg):   {out_img['active_dims']:.1f}")

    step = 50000
    sae.current_k = scheduler.get_k(step)
    print(f"\nStep {step}: k = {sae.current_k}")

    out_img = sae(x_img)
    print(f"  Active dims (avg):   {out_img['active_dims']:.1f}")

    indices, values = sae.get_active_features(x_img)
    print(f"\n  Sparse output shape: indices={indices.shape}, values={values.shape}")
    print(f"  Sample indices[0]:   {indices[0][:8].tolist()}...")
    print(f"  Sample values[0]:    {values[0][:8].tolist()}")
