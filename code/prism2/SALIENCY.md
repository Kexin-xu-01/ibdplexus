# PRISM2 Gradient Saliency

Gradient × input attribution maps for PRISM2, showing which tiles drive the model's output for a given target.

## Method

For each slide, we backpropagate a scalar target through the Perceiver to the input tile embeddings and compute per-tile importance as:

```
importance_i = mean( |∂score/∂tile_i  ×  tile_i| )
```

Flash attention does not return attention weights, but it does support autograd — gradients flow back through `flash_attn_varlen_func` to the tile embeddings without any model surgery. On CPU (no GPU), `CrossAttention.forward` is monkey-patched to use `F.scaled_dot_product_attention` with a compatible padding mask.

### Targets

| Target | Scalar | Notes |
|--------|--------|-------|
| `base` | L2 norm of the base (contrastive) embedding | Perceiver only, no text decoder |
| `yesno` | log P(Yes) − log P(No) for a question | Full model incl. Phi-3 decoder |

## Scripts

### `prism2_saliency.py`
Single-slide, flexible target.
```bash
# base embedding saliency
python prism2_saliency.py --slide 10407210HE1

# question-conditioned saliency
python prism2_saliency.py --slide 10407210HE1 \
    --target yesno \
    --question "Is Inflammation involvement present?"

# all slides
python prism2_saliency.py --all --target base
```
Outputs to `20x_224px_0px_overlap/prism2_saliency_{base|yesno}/`:
- `<slide>.h5` — datasets: `saliency` (N, norm. 0–1), `saliency_raw` (N), `coords` (N×2)
- `<slide>.png` — spatial heatmap on WSI thumbnail

### `prism2_saliency_uamp.py`
All 11 UAMP histological questions for a single slide. Loads model once, loops over questions.
```bash
# GPU job (compute only — recommended)
python prism2_saliency_uamp.py --slide 10407210HE1 --gpu 0 --compute-only

# Visualization only (reads cached h5, no model needed — fast)
python prism2_saliency_uamp.py --slide 10407210HE1 --viz-only
```
Outputs to `20x_224px_0px_overlap/prism2_saliency_uamp/<slide>/`:
- `<N>_<Term>.h5` — saliency scores + p_yes in attrs
- `<N>_<Term>.png` — heatmap overlay + top-8 patches at level-0 (60×)
- `all_questions.png` — all 11 questions stacked

### Kubernetes GPU job
```bash
kubectl apply -f /home/jovyan/ibdplexus/code/prism2/job_prism2_saliency_uamp.yaml
kubectl logs -n ibd-plexus-research -l job-name=kgbk271-prism2-saliency-uamp -c prism2 -f
```
The job runs `--compute-only`; run `--viz-only` locally afterward for plotting.

## UAMP Results — 10407210HE1

| # | Term | P(Yes) | Notes |
|---|------|--------|-------|
| 1 | Inflammation involvement | 0.560 | Main mass dominates; dense inflamed core as #1 patch |
| 2 | Crypt architectural distortion | 0.193 | Similar spatial distribution to inflammation |
| 3 | Neutrophil granulocytic infiltration | 0.269 | Same core region; score drops off steeply |
| 4 | Crypt abscesses | 0.622 | Strong #1 dominance |
| 5 | Lymphoid aggregates | 0.777 | Well-distributed scores; multiple patches contribute |
| 6 | Histiocytic granulomas | 0.119 | More stromal tiles appear in top ranks |
| 7 | Mucin depletion | 0.438 | Patches #2–3 shift to pale/depleted crypts |
| 8 | Pyloric gland metaplasia | 0.731 | Clear shift toward mucin-rich glandular tiles |
| 9 | Paneth cell metaplasia | 0.321 | Subtle shift toward crypt-base morphology |
| 10 | Neuronal hyperplasia | 0.029 | Rank #1 switches to peripheral tissue edge |
| 11 | Muscular hypertrophy | 0.029 | Same as above; main mass deprioritized |

## Interpretation notes

- The **same densely cellular tile** is rank #1 for most inflammatory questions — it is the most morphologically complex region and dominates gradient flow regardless of question due to the Perceiver bottleneck (256 latents compress all N tiles).
- The **meaningful signal is in ranks #2–8**, where the question redirects attention to question-relevant morphology.
- For submucosal features (neuronal, muscular), the model shifts attention to **tissue edges** rather than mucosal patches — the most biologically interpretable difference across the question set.
- The bottleneck architecture means saliency maps are diffuse compared to patch-level classifiers. For sharper localization, two-step attribution (latent → tile) is a future direction.

## Output locations

```
.../trident_processed/20x_224px_0px_overlap/
  prism2_saliency_base/          # base embedding target
  prism2_saliency_yesno/         # single yesno question
  prism2_saliency_uamp/
    10407210HE1/
      01_Inflammation_involvement.png
      ...
      11_Muscular_hypertrophy.png
      all_questions.png          # all 11 stacked
      *.h5                       # saliency + coords + p_yes
```
