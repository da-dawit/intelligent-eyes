"""Frozen ViT-B encoders (never trained). Each returns, for a batch of 224x224 views:

    pooled  (B, D)        the image feature the linear classifier reads
    patches (B, n, n, D)  the final patch tokens (used for the glance's crop previews)

CLIP and DINOv2/v3 pool with the CLS token; SigLIP 2 has no CLS token and pools with
its attention head. CLS and register tokens come first in the sequence, so the patch
tokens are always the last n*n.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import VIEW


@dataclass(frozen=True)
class EncoderSpec:
    hf_name: str
    patch: int
    gated: bool = False


ENCODERS = {
    "clip": EncoderSpec("openai/clip-vit-base-patch16", 16),
    "siglip2": EncoderSpec("google/siglip2-base-patch16-224", 16),
    "dinov2": EncoderSpec("facebook/dinov2-base", 14),
    "dinov3": EncoderSpec("facebook/dinov3-vitb16-pretrain-lvd1689m", 16, gated=True),
}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FrozenEncoder(torch.nn.Module):
    def __init__(self, name: str, device: torch.device | None = None, dtype: torch.dtype = torch.float16):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.name = name
        self.spec = ENCODERS[name]
        self.device = device or pick_device()
        self.dtype = dtype if self.device.type != "cpu" else torch.float32
        model = AutoModel.from_pretrained(self.spec.hf_name, dtype=self.dtype)
        # CLIP and SigLIP checkpoints hold an image and a text tower; keep the image tower.
        self.model = getattr(model, "vision_model", model).to(self.device).eval()
        self.is_clip = name == "clip"
        proc = AutoImageProcessor.from_pretrained(self.spec.hf_name)
        self.register_buffer("mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(proc.image_std).view(1, 3, 1, 1))
        self.to(self.device)
        for p in self.parameters():
            p.requires_grad_(False)
        self.grid = VIEW // self.spec.patch

    @torch.no_grad()
    def cls_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Last-layer attention from the CLS token to each patch, averaged over heads: (B, n, n).

        For baseline B3 ("look where the model attends"). Attention maps are only returned by
        the slower 'eager' implementation, so a second eager copy runs on glances only.
        SigLIP 2 has no CLS token and is not supported.
        """
        if self.name == "siglip2":
            raise ValueError("SigLIP 2 has no CLS token")
        if not hasattr(self, "_eager"):
            from transformers import AutoModel
            m = AutoModel.from_pretrained(self.spec.hf_name, dtype=self.dtype, attn_implementation="eager")
            self._eager = getattr(m, "vision_model", m).to(self.device).eval()
        x = ((x - self.mean) / self.std).to(self.dtype)
        att = self._eager(pixel_values=x, output_attentions=True).attentions[-1]   # (B, heads, L, L)
        n = self.grid
        return att[:, :, 0, -n * n:].mean(dim=1).float().reshape(x.shape[0], n, n)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """x: (B, 3, H, H) floats in [0, 1] on self.device.

        H = 224 for the glance and the looks. H = 448 is baseline B1 (the whole frame at 2x
        resolution, 4x the tokens): the position embeddings are interpolated to the larger
        grid (DINOv2 does this by itself, CLIP and SigLIP are told to, DINOv3 uses RoPE).
        """
        x = ((x - self.mean) / self.std).to(self.dtype)
        resize = x.shape[-1] != VIEW and self.name in ("clip", "siglip2")
        out = self.model(pixel_values=x, **({"interpolate_pos_encoding": True} if resize else {}))
        tokens = out.last_hidden_state
        n = x.shape[-1] // self.spec.patch
        patches = tokens[:, -n * n:]
        if self.is_clip:
            # HF CLIP applies its final LayerNorm only to the pooled CLS token; apply it to
            # the patch tokens too so both live on the same scale.
            patches = self.model.post_layernorm(patches)
        pooled = out.pooler_output
        return pooled.float(), patches.float().reshape(x.shape[0], n, n, -1)
