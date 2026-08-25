"""
Gradient × input saliency heatmap for PRISM2.

Backpropagates a scalar target through the Perceiver to the input tile
embeddings, then uses |grad * input| averaged across the feature dimension
as the per-tile importance score.

Targets:
  base    (default) L2 norm of the base (contrastive) embedding — pure WSI
          encoder, no text decoder needed.
  yesno   Log-odds P(Yes)/P(No) for a yes/no question, backpropagated through
          the full model including the Phi-3 decoder.

Usage:
    python prism2_saliency.py --slide 10407210HE1
    python prism2_saliency.py --slide 10407210HE1 --target yesno \\
        --question "Is there active inflammation?"
    python prism2_saliency.py --all

Outputs (per slide, written to <job_dir>/20x_.../prism2_saliency_<target>/):
    <slide>.h5   datasets: "saliency" (N,) float32, "coords" (N,2) int64
    <slide>.png  spatial heatmap (inferno colormap)
"""

import argparse
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

VIRCHOW2_DIM    = 2560
CLASS_TOKEN_DIM = 1280

DEFAULT_JOB_DIR = "/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed"
MODEL_PATH      = "/home/jovyan/shared-data/users/kexin/models/VLM/prism2"
CDIR            = "20x_224px_0px_overlap"


def _patch_cross_attention_for_cpu(model):
    """Replace flash_attn_varlen_func with F.scaled_dot_product_attention in all
    CrossAttention modules so the model can run (and differentiate) on CPU.

    The original kvt format is flat (sumLk, kvH, d_head); the patched version
    stores kvt as (k_padded (B,kvH,Lk,d), v_padded, attn_mask (B,Lk)) so all
    patched modules share a self-consistent format.
    """

    def _sdpa_forward(self, x, c=None, kvt=None, attention_mask=None):
        B, Lq, _ = x.shape
        x_n = self.x_norm(x)
        q = self.q_proj(x_n).view(B, Lq, self.num_heads, self.d_head).transpose(1, 2)

        if c is not None and kvt is None:
            assert attention_mask is not None
            c_n = self.c_norm(c)
            Lk = c_n.shape[1]
            k = self.k_proj(c_n).view(B, Lk, self.num_kv_heads, self.d_head).transpose(1, 2)
            v = self.v_proj(c_n).view(B, Lk, self.num_kv_heads, self.d_head).transpose(1, 2)
            kvt_out = (k, v, attention_mask)
        elif kvt is not None and c is None:
            assert attention_mask is not None
            k, v, _ = kvt
            kvt_out = (k, v, attention_mask)
        else:
            raise ValueError("must pass exactly one of c or kvt")

        # GQA: repeat kv heads to match query heads
        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Additive mask: 0 for real tokens, large negative for padding
        if attention_mask is not None:
            pad_val = -1e4 if x.dtype == torch.bfloat16 else -1e9
            amask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * pad_val
            amask = amask.to(x.dtype)
        else:
            amask = None

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=amask, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.num_heads * self.d_head)
        return self.o_proj(out), kvt_out

    patched = 0
    for module in model.modules():
        if type(module).__name__ == "CrossAttention":
            module.forward = types.MethodType(_sdpa_forward, module)
            patched += 1
    print(f"  CPU fallback: patched {patched} CrossAttention modules with SDPA.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job_dir", default=DEFAULT_JOB_DIR)
    p.add_argument("--slide", help="Slide stem (or .h5 path) to process")
    p.add_argument("--all", action="store_true", help="Process every slide")
    p.add_argument("--target", choices=["base", "yesno"], default="base")
    p.add_argument("--question",
                   help="Yes/No question string (required for --target yesno)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true", default=True)
    return p.parse_args()


def load_slide(path: Path):
    """Return (feats (N,1280) float32, coords (N,2) int64, meta dict)."""
    with h5py.File(path, "r") as f:
        feats  = f["features"][:]   # (N, 2560) or (N, 1280)
        coords = f["coords"][:]     # (N, 2) x,y in level-0 pixels
        meta   = dict(f["coords"].attrs)
    if feats.shape[1] == VIRCHOW2_DIM:
        feats = feats[:, :CLASS_TOKEN_DIM]
    return feats.astype(np.float32), coords, meta


def _load_processor_class():
    """Import Prism2Processor from the model directory."""
    sys.path.insert(0, MODEL_PATH)
    from processing_prism2 import Prism2Processor  # type: ignore
    return Prism2Processor


def compute_saliency(
    model,
    processor,
    feats_np: np.ndarray,   # (N, 1280) float32
    device: torch.device,
    target: str,
    question: str | None,
) -> np.ndarray:
    """Return per-tile |grad × input| importance, shape (N,) float32."""
    # Build padded batch — batch size 1
    model_dtype = next(model.parameters()).dtype
    batch = processor(tile_embeddings=[torch.from_numpy(feats_np)])
    tile_emb  = batch["tile_embeddings"].to(device=device, dtype=model_dtype)
    attn_mask = batch["attention_mask"].to(device=device)

    # Detach from any prior graph and mark as leaf requiring gradient.
    tile_emb = tile_emb.detach().requires_grad_(True)

    if target == "base":
        # Forward through Perceiver only (fast, no text decoder).
        out        = model._encode_images(tile_emb, attn_mask)
        token_name = model.config.pooler_tokens[0]
        score      = out[token_name][:, 0].norm()

    else:  # yesno
        assert question, "--question is required for --target yesno"
        Prism2Processor = _load_processor_class()
        messages = [[{"role": "user", "content": "<|image_1|>" + question}]]
        proc2    = Prism2Processor(
            tokenizer=model.tokenizer,
            num_img_tokens=model.config.num_img_tokens,
        )
        tok       = proc2.tokenize(messages, generation=True)
        input_ids = tok["input_ids"].to(device)

        resampler_out    = model._encode_images(tile_emb, attn_mask)
        image_embeddings = model._project(resampler_out["latents"])
        out              = model.text_decoder(
            input_ids, image_embeddings, output_hidden_states=False
        )

        last_logits = out["logits"][:, -1, :].float()   # (1, vocab)
        log_probs   = F.log_softmax(last_logits, dim=-1)
        yes_id      = model.tokenizer("Yes", add_special_tokens=False).input_ids[-1]
        no_id       = model.tokenizer("No",  add_special_tokens=False).input_ids[-1]
        # Scalar log-odds: high → model says Yes
        score = (log_probs[:, yes_id] - log_probs[:, no_id]).squeeze()

    score.backward()

    # |grad × input| averaged over feature dim → (N,) importance
    grad       = tile_emb.grad          # (1, N, 1280) bfloat16
    importance = (grad * tile_emb).abs().mean(-1).squeeze(0)  # (N,)
    return importance.detach().float().cpu().numpy()


def save_heatmap(
    out_path: Path,
    saliency: np.ndarray,   # (N,) float32, already normalized to [0,1]
    coords: np.ndarray,     # (N, 2) level-0 pixel coords
    meta: dict,
    slide_name: str,
    target_label: str,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip PNG] matplotlib not available")
        return

    patch_px = int(float(meta.get("patch_size_level0", 672)))

    # Rasterise into a 2D grid (much faster than N Rectangle patches).
    grid_col = (coords[:, 0] // patch_px).astype(int)
    grid_row = (coords[:, 1] // patch_px).astype(int)
    n_cols   = int(grid_col.max()) + 1
    n_rows   = int(grid_row.max()) + 1

    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    grid[grid_row, grid_col] = saliency

    fig, ax = plt.subplots(figsize=(10, 10 * n_rows / max(n_cols, 1)), dpi=120)
    im = ax.imshow(grid, cmap="inferno", vmin=0, vmax=1,
                   interpolation="nearest", aspect="equal")
    ax.set_title(f"{slide_name} — PRISM2 saliency ({target_label})", fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="importance (norm.)")
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  saved PNG  : {out_path}")


def process_slide(
    slide_path: Path,
    model,
    processor,
    device: torch.device,
    args,
    out_dir: Path,
    target_label: str,
):
    stem    = slide_path.stem
    out_h5  = out_dir / f"{stem}.h5"
    out_png = out_dir / f"{stem}.png"

    if args.skip_existing and out_h5.exists():
        print(f"  [skip] {stem}")
        return

    feats_np, coords, meta = load_slide(slide_path)
    N = feats_np.shape[0]
    print(f"  {stem}: {N} tiles", end="", flush=True)

    saliency_raw = compute_saliency(
        model, processor, feats_np, device,
        target=args.target, question=args.question,
    )

    # Normalise per-slide to [0, 1]
    vmin, vmax = saliency_raw.min(), saliency_raw.max()
    saliency = (saliency_raw - vmin) / (vmax - vmin + 1e-12)

    print(f"  min={vmin:.4g}  max={vmax:.4g}")

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("saliency", data=saliency)
        f.create_dataset("saliency_raw", data=saliency_raw)
        f.create_dataset("coords", data=coords)
        for k, v in meta.items():
            f["coords"].attrs[k] = v
        f.attrs["slide"]  = stem
        f.attrs["target"] = args.target
        if args.question:
            f.attrs["question"] = args.question
    print(f"  saved h5   : {out_h5}")

    save_heatmap(out_png, saliency, coords, meta, stem, target_label)


def main():
    args = parse_args()

    if not args.slide and not args.all:
        print("Specify --slide SLIDE_NAME or --all", file=sys.stderr)
        sys.exit(1)

    if args.target == "yesno" and not args.question:
        print("--target yesno requires --question", file=sys.stderr)
        sys.exit(1)

    device   = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cdir     = Path(args.job_dir) / CDIR
    feat_dir = cdir / "features_virchow2"

    target_label = args.target if args.target == "base" else f"yesno: {args.question}"
    out_dir_name = "prism2_saliency_base" if args.target == "base" else "prism2_saliency_yesno"
    out_dir      = cdir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PRISM2 from {MODEL_PATH} ...")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
    ).to(device=device, dtype=model_dtype).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
    )
    print("Model loaded.")
    if device.type == "cpu":
        _patch_cross_attention_for_cpu(model)
    print()

    if args.slide:
        p = Path(args.slide)
        if not p.exists():
            p = feat_dir / f"{args.slide}.h5"
        if not p.exists():
            print(f"Slide not found: {args.slide}", file=sys.stderr)
            sys.exit(1)
        slides = [p]
    else:
        slides = sorted(feat_dir.glob("*.h5"))
        print(f"Found {len(slides)} slides.\n")

    for i, slide_path in enumerate(slides, 1):
        print(f"[{i}/{len(slides)}]", end=" ")
        try:
            process_slide(slide_path, model, processor, device,
                          args, out_dir, target_label)
        except Exception as e:
            print(f"  [ERROR] {slide_path.name}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
