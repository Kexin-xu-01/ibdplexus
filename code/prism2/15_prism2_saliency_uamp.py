"""
Compute gradient saliency for all 11 UAMP questions on a single slide,
then produce per-question figures: heatmap overlay + top-N patches.

Usage:
    python prism2_saliency_umap.py --slide 10407210HE1
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

# openslide / matplotlib imported lazily — only needed for --visualize

VIRCHOW2_DIM    = 2560
CLASS_TOKEN_DIM = 1280
MODEL_PATH      = "/home/jovyan/shared-data/users/kexin/models/VLM/prism2"
FEAT_DIR        = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/20x_224px_0px_overlap/features_virchow2")
THUMB_DIR       = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/20x_224px_0px_overlap/visualization")
WSI_DIR         = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
OUT_ROOT        = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/20x_224px_0px_overlap/prism2_saliency_umap")

PATCH_PX = 672   # level-0 patch size
N_PATCHES = 8    # top patches per question

UAMP_TERMS = [
    "Inflammation involvement",
    "Crypt architectural distortion",
    "Neutrophil granulocytic infiltration",
    "Crypt abscesses",
    "Lymphoid aggregates",
    "Histiocytic granulomas",
    "Mucin depletion",
    "Pyloric gland metaplasia",
    "Paneth cell metaplasia",
    "Neuronal hyperplasia",
    "Muscular hypertrophy",
]


def _patch_cross_attention_for_cpu(model):
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
        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        pad_val = -1e4 if x.dtype == torch.bfloat16 else -1e9
        amask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * pad_val
        amask = amask.to(x.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=amask, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.num_heads * self.d_head)
        return self.o_proj(out), kvt_out

    patched = 0
    for module in model.modules():
        if type(module).__name__ == "CrossAttention":
            module.forward = types.MethodType(_sdpa_forward, module)
            patched += 1
    print(f"  CPU fallback: patched {patched} CrossAttention modules.")


def load_slide(h5_path):
    with h5py.File(h5_path) as f:
        feats  = f["features"][:]
        coords = f["coords"][:]
        meta   = dict(f["coords"].attrs)
    if feats.shape[1] == VIRCHOW2_DIM:
        feats = feats[:, :CLASS_TOKEN_DIM]
    return feats.astype(np.float32), coords, meta


def compute_saliency(model, processor, feats_np, device, question):
    sys.path.insert(0, MODEL_PATH)
    from processing_prism2 import Prism2Processor  # type: ignore

    model_dtype = next(model.parameters()).dtype
    batch = processor(tile_embeddings=[torch.from_numpy(feats_np)])
    tile_emb  = batch["tile_embeddings"].to(device=device, dtype=model_dtype)
    attn_mask = batch["attention_mask"].to(device=device)
    tile_emb  = tile_emb.detach().requires_grad_(True)

    messages = [[{"role": "user", "content": "<|image_1|>" + question}]]
    proc2    = Prism2Processor(tokenizer=model.tokenizer, num_img_tokens=model.config.num_img_tokens)
    tok      = proc2.tokenize(messages, generation=True)
    input_ids = tok["input_ids"].to(device)

    resampler_out    = model._encode_images(tile_emb, attn_mask)
    image_embeddings = model._project(resampler_out["latents"])
    out              = model.text_decoder(input_ids, image_embeddings, output_hidden_states=False)

    last_logits = out["logits"][:, -1, :].float()
    log_probs   = F.log_softmax(last_logits, dim=-1)
    yes_id = model.tokenizer("Yes", add_special_tokens=False).input_ids[-1]
    no_id  = model.tokenizer("No",  add_special_tokens=False).input_ids[-1]
    score  = (log_probs[:, yes_id] - log_probs[:, no_id]).squeeze()
    score.backward()

    grad       = tile_emb.grad
    importance = (grad * tile_emb).abs().mean(-1).squeeze(0)
    raw        = importance.detach().float().cpu().numpy()
    vmin, vmax = raw.min(), raw.max()
    return raw, (raw - vmin) / (vmax - vmin + 1e-12)


def make_overlay(thumb_np, sal, coords, sx, sy, pw, ph, alpha=0.4, cmap=None):
    import matplotlib.pyplot as plt
    if cmap is None:
        cmap = plt.cm.inferno
    H, W = thumb_np.shape[:2]
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    colors  = cmap(sal)
    for i, (x, y) in enumerate(coords):
        x0 = int(round(x * sx)); y0 = int(round(y * sy))
        x1 = min(W, x0 + int(round(pw)) + 1)
        y1 = min(H, y0 + int(round(ph)) + 1)
        overlay[y0:y1, x0:x1] = colors[i]
    tissue = (overlay[..., 3] > 0)[..., None]
    bg = thumb_np.astype(np.float32) / 255.0
    return np.clip(np.where(tissue, (1-alpha)*bg + alpha*overlay[..., :3], bg), 0, 1)


def extract_patches(sl, coords, patch_px=PATCH_PX):
    import openslide  # noqa: F401 — only called from visualize path
    patches = []
    for x, y in coords:
        r = sl.read_region((int(x), int(y)), 0, (patch_px, patch_px))
        patches.append(np.array(r.convert("RGB")))
    return patches


def plot_question(term, question, sal_norm, coords, thumb_np, sl,
                  meta, sx, sy, pw, ph, p_yes, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    cmap = plt.cm.inferno
    bg   = "#141414"

    overlay = make_overlay(thumb_np, sal_norm, coords, sx, sy, pw, ph, alpha=0.4, cmap=cmap)

    top_idx    = np.argsort(sal_norm)[::-1][:N_PATCHES]
    top_scores = sal_norm[top_idx]
    top_coords = coords[top_idx]
    patches    = extract_patches(sl, top_coords)

    fig = plt.figure(figsize=(2.2 + N_PATCHES * 1.9, 4.2), dpi=130, facecolor=bg)
    gs  = gridspec.GridSpec(
        1, N_PATCHES + 1,
        figure=fig,
        wspace=0.04,
        left=0.01, right=0.99, top=0.88, bottom=0.06,
        width_ratios=[2.2] + [1.9] * N_PATCHES,
    )

    # heatmap panel
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(overlay)
    ax0.axis("off")
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb  = plt.colorbar(sm, ax=ax0, fraction=0.05, pad=0.03, orientation="vertical")
    cb.set_label("saliency", color="white", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # patch panels
    for i, (patch, score) in enumerate(zip(patches, top_scores)):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(patch)
        ax.axis("off")
        ax.text(0.04, 0.96, f"#{i+1}", transform=ax.transAxes,
                color="white", fontsize=7, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc="#00000099", ec="none"))
        bar_w = max(0.02, min(score, 0.98))
        ax.add_patch(FancyBboxPatch((0,0), bar_w, 0.05,
                     transform=ax.transAxes, clip_on=True,
                     boxstyle="square,pad=0", fc=cmap(score), ec="none", zorder=5))
        ax.text(0.5, 0.018, f"{score:.2f}", transform=ax.transAxes,
                color="white", fontsize=6, ha="center", va="bottom",
                fontweight="bold", zorder=6)

    p_str = f"P(Yes)={p_yes:.3f}" if p_yes is not None else ""
    fig.suptitle(f"{term}  —  {p_str}", color="white", fontsize=10, y=0.97)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slide", default="10407210HE1")
    p.add_argument("--gpu",   type=int, default=0)
    p.add_argument("--compute-only", action="store_true",
                   help="Only compute + save saliency h5 files; skip visualization")
    p.add_argument("--viz-only", action="store_true",
                   help="Skip model loading; read cached h5 files and generate plots only")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = OUT_ROOT / args.slide
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.viz_only:
        # Load model once
        print("Loading PRISM2...")
        model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModel.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True,
        ).to(device=device, dtype=model_dtype).eval()
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True,
        )
        if device.type == "cpu":
            _patch_cross_attention_for_cpu(model)
        print("Model ready.\n")
    else:
        model = processor = model_dtype = None
        print("--viz-only: skipping model load, reading from cached h5 files.\n")

    # Load slide data once
    feats_np, coords, meta = load_slide(FEAT_DIR / f"{args.slide}.h5")

    if not args.compute_only:
        from PIL import Image
        import openslide
        thumb_np = np.array(Image.open(THUMB_DIR / f"{args.slide}.jpg"))
        sl       = openslide.OpenSlide(str(WSI_DIR / f"{args.slide}.tiff"))
    else:
        thumb_np = sl = None

    if not args.compute_only:
        th, tw = thumb_np.shape[:2]
        l0w    = int(meta["level0_width"])
        l0h    = int(meta["level0_height"])
        p_px   = float(meta["patch_size_level0"])
        sx, sy = tw / l0w, th / l0h
        pw, ph = p_px * sx, p_px * sy
    else:
        sx = sy = pw = ph = None

    # Score all terms for P(Yes) labels — read from h5 cache if viz-only
    p_yes_scores = {}
    if args.viz_only:
        print("Reading P(Yes) from cached h5 files...")
        for term in UAMP_TERMS:
            sal_h5 = out_dir / f"{term.replace(' ', '_')}.h5"
            with h5py.File(sal_h5) as f:
                p_yes_scores[term] = float(f.attrs.get("p_yes", float("nan")))
            print(f"  {term:40s}  P(Yes)={p_yes_scores[term]:.3f}")
        print()
    else:
        print("Scoring P(Yes) for all terms...")
        with torch.no_grad():
            batch     = processor(tile_embeddings=[torch.from_numpy(feats_np)])
            tile_emb  = batch["tile_embeddings"].to(device=device, dtype=model_dtype)
            attn_mask = batch["attention_mask"].to(device=device)
            for term in UAMP_TERMS:
                q = f"Is {term} present?"
                score = model.yes_no_score(tile_embeddings=tile_emb,
                                            attention_mask=attn_mask, question=q)
                p_yes_scores[term] = float(score.item())
                print(f"  {term:40s}  P(Yes)={p_yes_scores[term]:.3f}")
        print()

    # Compute saliency + plot for each term
    for i, term in enumerate(UAMP_TERMS, 1):
        question = f"Is {term} present?"
        print(f"[{i}/{len(UAMP_TERMS)}] {term}")

        sal_h5 = out_dir / f"{term.replace(' ', '_')}.h5"
        if sal_h5.exists():
            with h5py.File(sal_h5) as f:
                sal_raw  = f["saliency_raw"][:]
                sal_norm = f["saliency"][:]
            print("  (loaded from cache)")
        elif args.viz_only:
            print(f"  [SKIP] no cached h5 for {term} — run without --viz-only first")
            continue
        else:
            sal_raw, sal_norm = compute_saliency(model, processor, feats_np, device, question)
            with h5py.File(sal_h5, "w") as f:
                f.create_dataset("saliency",     data=sal_norm)
                f.create_dataset("saliency_raw", data=sal_raw)
                f.create_dataset("coords",       data=coords)
                f.attrs["slide"]    = args.slide
                f.attrs["term"]     = term
                f.attrs["question"] = question
                f.attrs["p_yes"]    = p_yes_scores[term]
            print(f"  saliency computed  raw range [{sal_raw.min():.3g}, {sal_raw.max():.3g}]")

        if not args.compute_only:
            out_png = out_dir / f"{i:02d}_{term.replace(' ', '_')}.png"
            plot_question(term, question, sal_norm, coords, thumb_np, sl,
                          meta, sx, sy, pw, ph, p_yes_scores[term], out_png)
            print(f"  saved: {out_png.name}")

    print(f"\nAll done. Output: {out_dir}")


if __name__ == "__main__":
    main()
