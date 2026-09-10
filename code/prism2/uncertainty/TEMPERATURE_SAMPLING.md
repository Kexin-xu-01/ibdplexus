# PRISM2 Temperature-Scaled Uncertainty Scoring

Script: `17_run_prism2_temperature_sampling.py`  
Job: `jobs/08_temperature_sampling/`  
Output: `results/prism2_manual_knn/temperature_sampling/prism2_temperature_scores.csv`

---

## Motivation

The standard PRISM2 yes/no scoring (`04_run_prism2_umap.py`) returns a single `P(Yes)` value per slide per concept. This deterministic score does not distinguish between a slide where the model is strongly confident and one where it is internally on the fence — both can produce a score of, say, 0.6.

This pipeline extracts a per-slide **uncertainty score** alongside the probability, enabling identification of slides where the model's answer is unreliable and pathologist review may be warranted.

---

## Method

### 1. Extract log-probabilities from the model

PRISM2's `yes_no_score` method is called with `return_log_probs=True`. This returns the full-vocabulary log-softmax values at the "Yes" and "No" token positions — referred to here as `z_yes` and `z_no`.

```python
log_pair = model.yes_no_score(
    tile_embeddings=batch["tile_embeddings"],
    attention_mask=batch["attention_mask"],
    question=f"Is {term} present?",
    return_log_probs=True,
)  # shape (B, 2): [log P(Yes), log P(No)]
```

### 2. Temperature scaling

A temperature parameter `T` is applied before re-normalising over the Yes/No pair:

```
p_yes = exp(z_yes / T) / (exp(z_yes / T) + exp(z_no / T))
```

In code:

```python
p_yes = F.softmax(log_pair / temperature, dim=-1)[:, 0]
```

**Effect of T:**

| T | Effect |
|---|--------|
| T = 1 | No change — raw model probabilities |
| T > 1 | Probabilities pushed towards 0.5 — uncertain cases become more ambiguous |
| T < 1 | Probabilities pushed towards 0 or 1 — everything becomes sharper |

Using `T = 2` makes the scoring more conservative: only cases where the model has a strong signal remain confidently above or below 0.5. Cases where Yes and No were similarly likely are pulled to the middle.

### 3. Stochastic sampling

`n_samples` Bernoulli draws are taken from `p_yes`:

```python
samples = torch.bernoulli(p_yes.unsqueeze(1).expand(-1, n_samples))  # (B, n_samples)
frac_yes = samples.mean(dim=1)
```

`frac_yes` converges to `p_yes_T` for large `n_samples`. It is included as a Monte Carlo sanity check and for downstream analyses that expect binary decisions.

### 4. Uncertainty: binary entropy

Binary entropy `H(p)` is computed as the primary uncertainty measure:

```
H(p) = -p × log₂(p) − (1−p) × log₂(1−p)
```

Scale: **0 to 1 bits**

| H(p) | Meaning |
|------|---------|
| 0.0 | Certain — model strongly says Yes or No |
| ~0.5 | Moderate uncertainty |
| ~0.9 | High uncertainty — model is close to the fence |
| 1.0 | Maximal uncertainty — p = 0.5 exactly |

H is maximised at `p = 0.5` and reaches 0 at `p = 0` or `p = 1`. It is a direct function of `p_yes_T` and does not depend on the number of samples.

### Limitation: calibration

The entropy score measures the model's *internal indecision*, not its empirical accuracy. A model that says `p = 0.9` but is only correct 60% of the time will produce low entropy despite being unreliable. Entropy is most useful as a relative measure — slides with high entropy are more uncertain *relative to other slides* — but should not be interpreted as an absolute confidence interval without calibration data (ground-truth pathologist labels).

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | 2.0 | Softmax temperature T |
| `--n_samples` | 50 | Bernoulli samples per slide × concept |
| `--batch_size` | 4 | Slides per GPU batch |
| `--feat_dir` | `…/manual_knn/…/features_virchow2` | Input Virchow2 features |
| `--out_dir` | `…/prism2_manual_knn/temperature_sampling` | Output directory |

---

## Output

**File:** `prism2_temperature_scores.csv`  
**Rows:** one per slide (3318 slides)  
**Columns:** for each of the 11 UAMP concepts, three columns:

| Column suffix | Description |
|---|---|
| `_p_yes_T` | Temperature-scaled P(Yes) — the primary score |
| `_frac_yes` | Fraction of 50 Bernoulli samples that were Yes |
| `_entropy` | Binary entropy H(p_yes_T) in bits — the uncertainty score |

Example: `inflammation_involvement_p_yes_T`, `inflammation_involvement_frac_yes`, `inflammation_involvement_entropy`

---

## Results (manual_knn dataset, T=2, n_samples=50)

Scored 3318 slides. Runtime: ~24 minutes on 1× GPU (semldgx04).

| Concept | Mean P(Yes) | Mean Entropy |
|---------|-------------|--------------|
| Inflammation involvement | 0.398 | 0.914 |
| Crypt architectural distortion | 0.397 | 0.936 |
| Neutrophil granulocytic infiltration | 0.240 | 0.762 |
| Crypt abscesses | 0.469 | 0.904 |
| Lymphoid aggregates | 0.594 | 0.880 |
| Histiocytic granulomas | 0.193 | 0.675 |
| Mucin depletion | 0.352 | 0.926 |
| Pyloric gland metaplasia | 0.421 | 0.906 |
| Paneth cell metaplasia | 0.377 | 0.917 |
| Neuronal hyperplasia | 0.242 | 0.778 |
| Muscular hypertrophy | 0.158 | 0.618 |

**Observations:**

- **Lymphoid aggregates** has the highest mean P(Yes) (0.59) — the model leans positive across much of the cohort.
- **Muscular hypertrophy** and **Histiocytic granulomas** have the lowest entropy and lowest P(Yes) — the model is relatively confident these are absent. Both are histologically rare findings, consistent with this.
- **Crypt architectural distortion** has the highest entropy (0.94) — the model is maximally uncertain across the cohort on this concept.
- High entropy overall reflects the T=2 temperature flattening, which is expected. Per-slide entropy values in the CSV will show much more spread than the cohort means.

**Suggested use:** filter the CSV for slides where `<concept>_entropy > 0.95` to identify cases where the model is near-maximally uncertain; these are the strongest candidates for manual pathologist review.

---

## Submitting the job

```bash
kubectl apply -f jobs/08_temperature_sampling/job_prism2_temperature_sampling_t15_filt_nodarkspot_manual_knn.yaml
kubectl logs -n ibd-plexus-research -l job-name=kgbk271-prism2-temp-sampling-t15-manual-knn -f
```
