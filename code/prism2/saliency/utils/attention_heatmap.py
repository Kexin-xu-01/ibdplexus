"""
Attention heatmap utilities for PRISM2.

Replaces flash_attn in the Perceiver's cross-attention with a pure PyTorch
(math_gqa) implementation that returns attention weights — not numerically
identical to flash attention due to floating-point order, but an accurate
approximation of the learned attention.

Public API:
    capture_perceiver_cross_attention(model)  context manager
    compute_attention_heatmap(model, tile_emb, attn_mask) -> (norm, raw)
    verify_math_gqa_vs_flash(model, tile_emb, attn_mask)
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# math_gqa forward
# ---------------------------------------------------------------------------

def _make_math_gqa_forward(storage: dict):
    """Build a CrossAttention.forward replacement using plain PyTorch matmuls.

    On the c-path (tile → latent), stores q, k, v, attn_weights, and
    attention_mask in *storage*.  Return signature matches the original:
    (output, kvt).
    """
    def _forward(self, x, c=None, kvt=None, attention_mask=None):
        B, Lq, _ = x.shape
        x_n = self.x_norm(x)
        q = self.q_proj(x_n).view(B, Lq, self.num_heads, self.d_head).transpose(1, 2)
        # q: (B, num_heads, Lq, d_head)

        if c is not None and kvt is None:
            assert attention_mask is not None
            c_n = self.c_norm(c)
            Lk = c_n.shape[1]
            k = self.k_proj(c_n).view(B, Lk, self.num_kv_heads, self.d_head).transpose(1, 2)
            v = self.v_proj(c_n).view(B, Lk, self.num_kv_heads, self.d_head).transpose(1, 2)
            # k, v: (B, num_kv_heads, Lk, d_head)
            kvt_out = (k, v, attention_mask)
            storage.update(
                q=q.detach(), k=k.detach(), v=v.detach(),
                attention_mask=attention_mask,
            )
        elif kvt is not None and c is None:
            k, v, _mask = kvt
            kvt_out = kvt
        else:
            raise ValueError("must pass exactly one of c or kvt")

        # GQA: expand kv-heads to match query-heads
        repeat = self.num_heads // self.num_kv_heads
        k_exp = k.repeat_interleave(repeat, dim=1) if repeat > 1 else k
        v_exp = v.repeat_interleave(repeat, dim=1) if repeat > 1 else v

        scale = self.d_head ** -0.5
        scores = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
        # scores: (B, num_heads, Lq, Lk)

        if attention_mask is not None:
            pad_val = torch.finfo(scores.dtype).min
            padding = (1 - attention_mask).bool()
            scores = scores.masked_fill(padding[:, None, None, :], pad_val)

        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, Lq, Lk)
        if c is not None:
            storage['attn_weights'] = attn_weights.detach()

        out = torch.matmul(attn_weights, v_exp)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.num_heads * self.d_head)
        return self.o_proj(out), kvt_out

    return _forward


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextmanager
def capture_perceiver_cross_attention(model) -> Iterator[dict]:
    """Patch the Perceiver's cross-attention to math_gqa and yield a storage dict.

    Restores the original flash_attn forward on exit.

    Example::

        with capture_perceiver_cross_attention(model) as stored:
            model.image_resampler.perceiver(tile_emb, attention_mask=mask)

        attn_weights = stored['attn_weights']  # (B, num_heads, Lq, Lk)
        v            = stored['v']             # (B, num_kv_heads, Lk, d_head)
    """
    storage: dict = {}
    xattn = model.image_resampler.perceiver.layers[0]["xattn"]["xattn"]
    original = xattn.forward
    xattn.forward = types.MethodType(_make_math_gqa_forward(storage), xattn)
    try:
        yield storage
    finally:
        xattn.forward = original


# ---------------------------------------------------------------------------
# Heatmap computation
# ---------------------------------------------------------------------------

def compute_attention_heatmap(
    model,
    tile_embeddings: Tensor,  # (B, N, context_dim)
    attention_mask: Tensor,   # (B, N) int, 1=real 0=pad
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (heatmap_norm, heatmap_raw) as (N,) float32 numpy arrays.

    heatmap[k] = (Σ_{h,q} attn[h,q,k]) × ‖v[k]‖₂

    heatmap_norm is min-max normalised to [0, 1] over real (unpadded) tiles.
    heatmap_raw is the unnormalised product.
    """
    model.eval()
    with torch.no_grad(), capture_perceiver_cross_attention(model) as stored:
        model.image_resampler.perceiver(tile_embeddings, attention_mask=attention_mask)

    attn = stored['attn_weights'].float()  # (B, num_heads, Lq, Lk)
    v    = stored['v'].float()             # (B, num_kv_heads, Lk, d_head)
    mask = stored['attention_mask']        # (B, Lk)

    B, _H, _Lq, Lk = attn.shape

    # Sum softmax scores over all heads and latent queries → (B, Lk)
    score_sum = attn.sum(dim=(1, 2))

    # L2 norm of value vector per tile (over kv_heads × d_head) → (B, Lk)
    v_norm = v.permute(0, 2, 1, 3).reshape(B, Lk, -1).norm(dim=-1)

    heatmap = (score_sum * v_norm) * mask.float().to(attn.device)

    raw  = heatmap[0].cpu().numpy().astype(np.float32)
    real = mask[0].bool().cpu().numpy()

    if normalize:
        norm = raw.copy()
        lo, hi = raw[real].min(), raw[real].max()
        norm[real] = (raw[real] - lo) / (hi - lo + 1e-8)
        return norm, raw

    return raw, raw


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_math_gqa_vs_flash(
    model,
    tile_embeddings: Tensor,
    attention_mask: Tensor,
) -> dict:
    """Run the same tensor through flash_attn and math_gqa, print diff stats.

    Q/K/V projections are bit-for-bit identical (same weights, same input).
    The perceiver output will differ by ~fp-precision (bf16 ≈ 1e-3, fp32 ≈ 1e-5).

    On CPU flash_attn is unavailable; only the math_gqa path runs and shapes
    are printed instead of a diff.
    """
    model.eval()

    on_cuda = tile_embeddings.is_cuda

    if on_cuda:
        with torch.no_grad():
            flash_out = model.image_resampler.perceiver(
                tile_embeddings, attention_mask=attention_mask
            )
    else:
        flash_out = None
        print("  [note] flash_attn requires CUDA — skipping flash comparison")

    with torch.no_grad(), capture_perceiver_cross_attention(model) as stored:
        mgqa_out = model.image_resampler.perceiver(
            tile_embeddings, attention_mask=attention_mask
        )

    attn = stored['attn_weights']

    if on_cuda:
        diff = (flash_out.float() - mgqa_out.float()).abs()
        ref  = flash_out.float().abs()
        print("flash_attn vs math_gqa — perceiver output diff:")
        print(f"  max  abs : {diff.max().item():.4e}")
        print(f"  mean abs : {diff.mean().item():.4e}")
        print(f"  max  rel : {(diff / (ref + 1e-8)).max().item():.4e}")
    else:
        diff = None

    print(f"attn_weights {attn.shape}  sum-over-keys={attn.sum(-1).mean().item():.6f}")
    print(f"q {stored['q'].shape}  k {stored['k'].shape}  v {stored['v'].shape}")

    return {
        "flash_out": flash_out, "mgqa_out": mgqa_out,
        "max_diff": diff.max().item() if diff is not None else None,
        "mean_diff": diff.mean().item() if diff is not None else None,
        "attn_weights": stored["attn_weights"],
        "v": stored["v"], "q": stored["q"], "k": stored["k"],
    }
