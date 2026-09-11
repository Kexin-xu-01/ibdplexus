# Prompt-framing and internal uncertainty of a pathology vision–language model applied to inflammatory bowel disease biopsies

**Cohort:** IBD-PLEXUS SPARC, 3,318 H&E whole-slide images at 20× (0.17 µm/pixel)
**Model:** PRISM2 (Paige AI) built on Virchow2 patch embeddings
**Dataset used for headline results:** `tissue_threshold_15_filtered_no_darkspot_manual_knn`
**Compiled:** 2026-09-10

---

## Abstract

We evaluated PRISM2, a vision–language pathology foundation model, on 3,318 H&E biopsies from an inflammatory bowel disease (IBD) cohort. PRISM2 was queried for eleven UAMP histological concepts covering the standard IBD biopsy checklist (inflammation, crypt architecture, granulocytic infiltration, crypt abscesses, lymphoid aggregates, granulomas, mucin depletion, pyloric and Paneth cell metaplasia, neuronal hyperplasia, muscular hypertrophy). Two questions frame the study. First: how much does the model's answer depend on how the question is asked? Second: when the answer is fixed, how confident is the model internally? We ran five prompting experiments — temperature-scaled yes/no, three-way multiple choice with an explicit "not sure" option, ten-way single-prompt competition, five-way severity grading, and four-phrasing linguistic robustness — on the same 3,318 slides. As an ablation on the input side, we recomputed all eleven scores across five successive stages of patch-level QC filtering (raw TRIDENT output through manual KNN artefact curation) and quantified how much of the signal is driven by tissue selection. PRISM2 exhibits high native binary entropy (mean > 0.85 bits, max = 1 bit, for 7/11 concepts), and prevalence estimates for the same concept vary by up to a factor of three between prompt formats — most sharply for lymphoid aggregates and Paneth cell metaplasia. Severity grading is internally consistent: slides in the top severity quartile carry higher P(Yes) at T=1 than slides in the bottom quartile for 8/11 concepts, most cleanly for inflammation, crypt abscesses, lymphoid aggregates and pyloric gland metaplasia. Population-level scores are robust to QC filtering (nine of eleven concepts shift < 3 % end-to-end), but two metaplasia terms shift by 8–10 % — a shift driven almost entirely by the tissue-coverage threshold, consistent with non-tissue patches being systematically scored as metaplastic. We report scores at temperature T=2 as the primary prevalence estimate and flag three concepts (Paneth cell metaplasia, mucin depletion, crypt architectural distortion) whose scores are wording-sensitive and should be treated as unreliable readouts in downstream analyses without a fixed prompt.

---

## 1 Introduction

Foundation models for computational pathology now support both slide-level classification and free-text question answering from whole-slide images. PRISM2 (Paige AI) is a vision–language model that consumes precomputed Virchow2 patch embeddings, aggregates them into a slide-level embedding via a resampler and a projector, and answers natural-language prompts through a Phi-3 text decoder. A common way to use such models is to pose a yes/no question about a histological concept and take the softmax of the "Yes" and "No" logits as a slide-level prevalence score.

This mode of use conflates several distinct questions that pathologists are trained to separate:

- **Is the concept present?** — a categorical judgment.
- **How severe is it?** — an ordinal judgment.
- **How confident am I?** — a meta-cognitive judgment.

A raw P(Yes) collapses all three. Two failure modes then follow. First, if the model is asked the same semantic question with a different wording, the score may change; this is a prompt-framing effect and is well-known for large language models but not systematically quantified for pathology VLMs. Second, the model may output a "Yes" token whose supporting logit lies almost exactly at the "No" logit; the point estimate is then close to a coin flip in the model's own internal calculation, even though it yields a single-valued prediction — this is internal indecision, measurable via the entropy of the (yes, no) distribution.

The IBD setting sharpens both concerns. IBD biopsies are read against a well-defined biopsy checklist: architectural distortion, plasmacytosis and neutrophilic infiltration for activity, granulomas and pyloric/Paneth cell metaplasia for chronicity or Crohn-favouring features, and neuronal or muscular changes for stricturing disease. Each of these concepts is nameable, but their morphological correlates differ widely (a granuloma is a compact cellular structure; muscular hypertrophy is a diffuse quantitative change; lymphoid aggregates are common in normal terminal ileum). A model that scores all eleven with the same prompt template implicitly assumes prompt sensitivity is roughly comparable across concepts — an assumption we test directly.

We had two goals:

1. Quantify uncertainty in PRISM2's histological scoring using five complementary prompting strategies on the same 3,318 slides.
2. Ablate the upstream QC pipeline to determine how much of the score comes from the tissue selection rather than from the model itself.

We deliberately did not compare PRISM2 against pathologist-graded ground truth. IBD-PLEXUS biopsy slides do not carry per-slide UAMP labels, so the analyses below concern the model's internal consistency and reproducibility under perturbation, not its diagnostic accuracy. This is a real limitation, and Section 6 restates it.

---

## 2 Materials and methods

### 2.1 Dataset

IBD-PLEXUS SPARC contributes 3,318 H&E biopsy whole-slide images from patients with confirmed or suspected IBD (Crohn's disease, ulcerative colitis, IBD-U, or non-IBD controls). Slides are scanned at 20× (0.170868 µm/pixel), from multiple sites and both lesional and normal-appearing tissue as recorded in `results/metadata/slide_metadata.csv`. Ten anatomical sites are represented; sigmoid colon and ileum are most frequent. Age, disease location, macroscopic appearance, disease activity and (for Crohn's disease) phenotype are recorded per slide where available, but no per-slide UAMP annotation was performed for this cohort.

### 2.2 Patch feature extraction

Slides were tiled with TRIDENT at 20×, 224 px, 0 px overlap, and each patch was embedded with Virchow2 (2,560-dim feature vector per patch; only the 1,280-dim CLS token is used downstream by PRISM2). Features are saved as one HDF5 file per slide under `.../features_virchow2/`.

Five successive QC filter stages were prepared:

| # | Dataset directory suffix | Filters applied | Slides |
|---|---|---|---|
| 1 | `trident_processed` | none | 3,313 |
| 2 | `tissue_threshold_15` | ≥ 15 % tissue coverage per patch | 3,318 |
| 3 | `tissue_threshold_15_filtered` | + Laplacian blur (variance < 100) + faint patch removal (mean RGB ≥ p98 of 211.8) | 3,318 |
| 4 | `..._no_darkspot` | + GrandQC dark-spot removal (class 3, > 10 % pixels) | 3,318 |
| 5 | `..._no_darkspot_manual_knn` | + manual KNN artefact curation via patch UMAP | 3,318 |

The full patch-level filter methodology is documented in `results/patch_filtering_report.md`. All headline results in Section 3 use stage 5 (manual_knn); Section 4 is an ablation over stages 1–5.

### 2.3 PRISM2 model

PRISM2 weights are loaded from a local HuggingFace snapshot (`paige-ai/prism2`) with `trust_remote_code=True`, `torch_dtype=torch.bfloat16`, on a single NVIDIA GPU per job. Inputs are the Virchow2 CLS-token embeddings; the model applies its resampler, projection and text decoder internally.

For efficiency, all per-concept prompts share a single forward pass through the resampler and projector per batch; only the text decoder is re-invoked per concept. This is roughly a 10× reduction in cost versus repeating the full forward pass per concept, and is achieved by calling the model's internal methods (`_encode_images`, `_project`, `text_decoder`) directly rather than the public `yes_no_score()` wrapper. The scripts are therefore coupled to PRISM2's internal architecture; if the model is updated, the internal calls may need re-wiring.

### 2.4 Prompting experiments (Section 3)

All five experiments are run on the same 3,318 slides from the manual_knn dataset. Each experiment produces one row per slide and one or more numeric columns per UAMP concept.

**E1 — Temperature-scaled yes/no** (`uncertainty/13_run_prism2_temperature_sampling.py`).
For each concept, the standard yes/no prompt is posed. The full-vocabulary log-softmax at the next-token position is captured; only the `Yes` and `No` token logits are retained and re-softmaxed with temperature T. The native probability (T=1) is used to compute the binary entropy

    H = -p log2(p) - (1-p) log2(1-p),   H in [0, 1]

which is independent of T by construction. The flattened score at T=2 is used as the primary prevalence estimate: it preserves the sign of the model's preference while pulling confident predictions less far towards 1 or 0.

**E2 — Multiple choice A/B/C** (`uncertainty/19_run_prism2_multiple_choice.py`).
Each concept is scored with an explicit three-way prompt: (A) present, (B) normal, (C) not sure. Softmax over the token IDs of A, B, C gives three probabilities per concept per slide.

**E3 — Multi-feature 10-way competition** (`uncertainty/20_run_prism2_multi_feature.py`).
A single prompt lists ten concepts as options A–J and asks which best describes the image. Softmax over the ten letter tokens gives a mutually-exclusive probability distribution — one forward pass per slide. Muscular hypertrophy is excluded because it did not fit the ten-way template.

**E4 — Severity grading** (`uncertainty/21_run_prism2_severity.py`).
Each concept is scored with a five-way severity prompt: mild / moderate / severe / not sure / none. A weighted severity score

    severity = 1 * P(mild) + 2 * P(moderate) + 3 * P(severe),   severity in [0, 3]

is computed. P(not sure) and P(none) contribute zero to severity but are retained for interpretation.

**E5 — Linguistic robustness** (`uncertainty/22_run_prism2_question_robustness.py`).
Each concept is scored with four semantically equivalent but differently phrased yes/no questions (for example, for crypt architectural distortion: "Is crypt architectural distortion present?" / "Is crypt distorted?" / "Is crypt deformed?" / "Is crypt deformation present?"). The mean and SD of P(Yes) across the four phrasings are reported. The image embedding is computed once per batch and reused across the 44 phrasings (11 concepts × 4), keeping cost comparable to a single-concept run.

### 2.5 Ablation over QC filtering (Section 4)

The primary UAMP scoring pipeline (`scoring/01_run_prism2_uamp.py`, native yes/no) was re-run on each of the five dataset stages listed in Section 2.2, holding the model, prompt, batch size and random state fixed. Deltas at each transition were computed both at the population level (change in mean P(Yes) per concept) and at the individual slide level (max |Δ| per concept, and the fraction of slides with |Δ| > 0.05).

### 2.6 Slide-level embedding visualisation

For each slide, PRISM2 emits a base embedding (visual pooling only) and a diagnostic embedding (image conditioned on a diagnostic prompt). We reduce each embedding to 50 principal components and then to two UMAP dimensions with `n_neighbors=30`, `min_dist=0.25` (`umap/05_umap_embeddings.py`). UAMP scores and free-text report-derived features are overlaid on the same coordinates for visual inspection but are not the subject of quantitative claims in this article.

### 2.7 Reproducibility

Every step is a stand-alone Python script under `code/prism2/` with argparse defaults corresponding to the paths used here. Each script writes atomically to disk and is safe to interrupt: an interrupted run resumes from the existing CSV or `.jsonl` file. Kubernetes job specifications for GPU-bound steps are checked in under `code/prism2/jobs/`. The command that reproduces the primary yes/no score CSV used throughout this article is:

```bash
conda activate prism2
python scoring/01_run_prism2_uamp.py \
    --feat_dir /home/jovyan/kgbk271-ibd-volume/data/processed/\
tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/features_virchow2 \
    --results_root /home/jovyan/kgbk271-ibd-volume/results/prism2_manual_knn
```

The five uncertainty experiments are launched analogously from `code/prism2/uncertainty/`. The QC filter comparison (Section 4) is produced by `compare/14_compare_histological_scores.py` and its output is under `results/prism2_manual_knn/compare_histological_scores/`.

---

## 3 Results

Unless noted otherwise, all numbers in this section refer to the 3,318 slides in the manual_knn dataset.

### 3.1 Prevalence depends strongly on how the concept is framed

Figure 1A shows mean P(Yes) per concept under three prompt strategies. Temperature sampling at T=2, the three-way multiple choice, and the four-phrasing robustness mean disagree substantially on the same 3,318 slides:

- Lymphoid aggregates. Multiple choice: 0.955. Temperature sampling: 0.594. Robustness mean: 0.563. The MC framing — an explicit "lymphoid aggregates is present" option — elicits near-uniform assent, while a neutral yes/no is much lower.
- Paneth cell metaplasia. MC: 0.828. Robustness mean: 0.248. A factor of 3.3× between two ways of asking the same question.
- Pyloric gland metaplasia. MC: 0.714. Robustness mean: 0.552. Substantial but smaller gap.
- Crypt abscesses. Temperature T=2: 0.469. Robustness mean: 0.184. The plain yes/no is more permissive; generic paraphrases suppress detection.
- Muscular hypertrophy and neuronal hyperplasia are consistently low across all three methods (0.02–0.24), suggesting either genuine rarity in this biopsy cohort or a systematic model bias against these concepts.

The prevalence divergence is not run-to-run noise: it reflects how framing shifts the response distribution. We recommend the temperature-sampled P(Yes|T=2) as the primary readout because (i) it is the closest analog to the model's native probability, (ii) it is bounded away from 0/1 saturation for confident predictions, and (iii) unlike the MC prompt it does not textually prime the model with the concept name as an assertion.

**Biological reading.** Lymphoid aggregates and Paneth cells are common findings in normal terminal ileum, and pyloric-type mucous glands normally line the antrum and duodenum. When a foundation model trained on a broad histological corpus is asked "is X present?" for concepts that are near-baseline in normal GI mucosa, the answer depends on where the model draws the "abnormal enough to say yes" line — a boundary that shifts with prompt structure. The prevalence gap for these three concepts is consistent with them being on-the-fence findings rather than clearly abnormal ones.

### 3.2 Internal indecision is high for most concepts

Figure 1C, top row, shows the mean per-slide entropy of the (Yes, No) distribution at T=1 (native temperature). Entropy is bounded in [0, 1] bit; 1.0 corresponds to a 50/50 split.

- Seven of eleven concepts have mean entropy > 0.85 bits: crypt architectural distortion (0.936), mucin depletion (0.926), Paneth cell metaplasia (0.917), inflammation involvement (0.914), pyloric gland metaplasia (0.906), crypt abscesses (0.904), lymphoid aggregates (0.880).
- Two concepts are more decisive: muscular hypertrophy (0.618) and histiocytic granulomas (0.675). Consistent with these, muscular hypertrophy also has the lowest robustness SD (0.011) and a high P(not sure) (0.188), while lymphoid aggregates has the lowest P(not sure) (0.014).
- The remaining two (neuronal hyperplasia 0.778, neutrophil granulocytic infiltration 0.762) sit in between.

The three uncertainty axes we captured — entropy at T=1, P(not sure) from multiple choice, and SD across phrasings — are not redundant (Figure 1C). Neuronal hyperplasia has the highest P(not sure) (0.201) but only middling entropy and low SD; crypt architectural distortion has the highest entropy and the highest SD but only moderate P(not sure). This means that a slide can be internally uncertain in different ways, and no single metric captures all of them.

**Calibration caveat.** Entropy measures internal indecision, not accuracy. A model can be sharply confident and wrong, or 0.5/0.5 and land on the correct token by chance. PRISM2 has not been calibrated on this cohort, so entropy here should be read as a signal of the model's own indecision, not as a proxy for uncertainty about a hidden ground truth. This is a genuine limitation and we return to it in Section 6.

### 3.3 Severity grading is internally consistent

Figure 1B shows the mean severity score (0–3 scale) per concept. All eleven concepts cluster in the mild-to-moderate range (1.04–1.48), meaning that when the model does call the feature present, it tends to call it mild or moderate rather than severe. Mucin depletion (1.48) and crypt abscesses (1.44) lead; neuronal hyperplasia and pyloric gland metaplasia score lowest (~1.04). Severity does not simply follow prevalence: crypt abscesses is moderately prevalent but highly severe, whereas lymphoid aggregates is highly prevalent but moderately severe. This dissociation is expected — an aggregate is either present or not, but does not have a severe form, whereas an abscess when present is intrinsically an acute finding.

Figure 2 provides a more stringent internal-consistency test: for each concept, slides were split into quartiles by their severity score (E4), and the P(Yes) at native temperature (E1) was plotted per quartile. Two structural regimes appear:

- **Monotonic increase across quartiles**: inflammation involvement, crypt abscesses, lymphoid aggregates, pyloric gland metaplasia, Paneth cell metaplasia, histiocytic granulomas, neutrophil granulocytic infiltration, and crypt architectural distortion. The model's yes/no prevalence and its independent severity score agree on which slides carry more of the feature — Q4 medians are 3–4× higher than Q1 medians for inflammation, crypt abscesses and pyloric gland metaplasia, indicating that severity is not a spurious re-parameterisation of the same output but tracks a coherent signal.
- **Flat or inverted**: mucin depletion, neuronal hyperplasia, muscular hypertrophy. Mucin depletion has almost identical P(Yes) across all four severity quartiles; neuronal hyperplasia is actually higher in the lowest quartile. These are among the concepts that show the lowest severity scores overall; the severity axis carries little information for them, and stratifying by severity does not stratify by P(Yes).

Two experimental branches are agreeing where they should and disagreeing where the underlying signal is weak — a mild sign of internal validity for the strong-signal concepts.

### 3.4 Wording sensitivity is concept-specific

Figure 1D plots SD of P(Yes) across four semantically-equivalent phrasings against the mean of those phrasings. Two regimes emerge:

- Low prevalence, low SD (bottom-left). Muscular hypertrophy (SD 0.011), neutrophil granulocytic infiltration (0.046), crypt abscesses (0.060), histiocytic granulomas (0.094), inflammation involvement (0.095), neuronal hyperplasia (0.109). The model gives a consistent negative or low-P(Yes) answer regardless of phrasing.
- Moderate-to-high prevalence, high SD (upper cluster). Paneth cell metaplasia (SD 0.233), mucin depletion (0.220), crypt architectural distortion (0.224), pyloric gland metaplasia (0.176), lymphoid aggregates (0.113). These are the concepts whose scores flip most when the question is rephrased.

An SD of 0.22 across four semantically identical questions means the point estimate for that concept on a single slide is not stable to a paraphrase. Any downstream analysis that uses these scores as a numeric readout (correlation with clinical outcomes, clustering, etc.) should either (i) fix a validated question wording and treat the score as a template-specific measurement, or (ii) average across a defined set of paraphrases.

### 3.5 A single-prompt 10-way competition is dominated by three concepts

Figure 4 shows results from the ten-way single-prompt experiment (E3). When forced to pick one of ten features, the model concentrates 92 % of the probability mass on three options: pyloric gland metaplasia (0.410), Paneth cell metaplasia (0.283), lymphoid aggregates (0.230). Winner counts (highest probability per slide, over 3,318 slides) reinforce this: pyloric gland metaplasia wins in 1,724 slides, lymphoid aggregates in 815, Paneth cell metaplasia in 760. Inflammation involvement, crypt abscesses, mucin depletion and crypt architectural distortion each win in fewer than three slides.

This is an artefact of the mutually-exclusive framing: inflammation and architectural changes co-occur with the metaplasia and aggregate findings, and when the prompt forces the model to choose one dominant feature, chronic and normal-baseline findings out-compete active inflammatory findings that would score highly in a per-concept yes/no. The 10-way prompt is therefore useful as a dominant-feature readout but should not be interpreted as a prevalence estimator for concepts that co-occur with others.

---

## 4 Ablation: QC filtering has minimal effect on scores except for metaplasia

To distinguish signal that comes from the model from signal that comes from the tissue selection, we re-ran the primary yes/no scoring at every QC filter stage (2.5).

### 4.1 Population-level

End-to-end (`trident_processed` → `manual_knn`) changes in mean P(Yes):

| Concept | Δ mean | % change |
|---|---:|---:|
| Paneth cell metaplasia | −0.031 | −9.6 % |
| Pyloric gland metaplasia | −0.033 | −8.2 % |
| Muscular hypertrophy | −0.002 | −5.0 % |
| Mucin depletion | −0.008 | −3.3 % |
| Neuronal hyperplasia | −0.003 | −2.8 % |
| Crypt abscesses | −0.012 | −2.7 % |
| Crypt architectural distortion | −0.007 | −2.1 % |
| Inflammation involvement | −0.003 | −0.9 % |
| Neutrophil granulocytic infiltration | 0.000 | 0.0 % |
| Lymphoid aggregates | +0.004 | +0.6 % |
| Histiocytic granulomas | +0.004 | +5.8 % |

Nine of eleven concepts change by less than 3 %. Two metaplasia concepts (Paneth cell, pyloric gland) drop by 8–10 %. Score standard deviations across slides are essentially unchanged at every stage (0.030–0.279), so the filters shift the mean without altering inter-slide spread.

### 4.2 Which filter causes the shift

Mean absolute per-slide change, averaged across all eleven concepts, at each transition:

| Transition | Filter added | Mean \|Δ\| | Max \|Δ\| |
|---|---|---:|---:|
| 1 → 2 | ≥ 15 % tissue coverage | 0.0189 | 0.302 |
| 2 → 3 | Blur + faint patch | 0.0118 | 0.241 |
| 3 → 4 | Dark-spot removal | 0.0006 | 0.179 |
| 4 → 5 | Manual KNN artefact curation | 0.0074 | 0.595 |

The tissue coverage filter (Stage 1 → 2) is responsible for the majority of the population-level shift. For Paneth cell metaplasia and pyloric gland metaplasia, 34 % and 34 % of slides individually shift by more than 0.05 P(Yes) at this stage alone. All other concepts have fewer than 27 % of slides shifting by that magnitude at this stage.

**Biological reading.** Non-tissue patches (mucus, connective tissue, near-empty background) apparently look "metaplastic" to PRISM2 by default; removing them deflates the Paneth and pyloric gland metaplasia scores. Inflammatory features (crypts, inflammation, neutrophils) are recognisable only on genuine tissue and are essentially unaffected by tissue thresholding. This is the expected direction if metaplasia is being confused with low-cellularity, low-nuclear-density regions.

### 4.3 The manual KNN step

The manual KNN artefact curation (Stage 4 → 5) shifts population means by ≤ 0.002 for every concept, but has a long tail of individual slides: two slides shift by |Δ| > 0.4 (max: 10801140HE101 at 0.595 for lymphoid aggregates; 11007122HE1 at 0.497). These outliers are candidates for visual inspection — they represent slides where a cluster of artefact patches had been co-locating with a high-scoring region, and their removal reveals the score of the underlying tissue.

The KNN curation therefore does not appreciably move the cohort-level signal but does clean up individual scores where artefact-driven false positives dominated the aggregate. This justifies its inclusion as the default dataset for downstream analysis while confirming that the numerical results reported here would look nearly identical on the `no_darkspot` dataset.

---

## 5 Figures

- **Figure 1** — 4-panel uncertainty comparison across the five prompting experiments (Sections 3.1–3.4).
  Source: `results/prism2_manual_knn/uncertainty_estimation/uncertainty_comparison.{png,pdf}` (generated by `uncertainty/24_plot_uncertainty_comparison.py`).

- **Figure 2** — P(Yes|T=1) stratified by concept-specific severity quartile (Section 3.3).
  Source: `results/prism2_manual_knn/uncertainty_estimation/severity_vs_pyes_boxplot.{png,pdf}` (generated by `uncertainty/25_plot_severity_vs_pyes_boxplot.py`).

- **Figure 3** — Temperature-sampled entropy and prevalence at T=2, per concept (Sections 3.1–3.2).
  Source: `results/prism2_manual_knn/uncertainty_estimation/temperature_sampling/temperature_scores_summary.{png,pdf}` (generated by `uncertainty/15_plot_temperature_scores.py`).

- **Figure 4** — Ten-way multi-feature competition (Section 3.5).
  Source: `results/prism2_manual_knn/uncertainty_estimation/multi_feature/multi_feature_summary.{png,pdf}` (generated by `uncertainty/23_plot_multi_feature_scores.py`).

- **Supplementary S1** — QC filter comparison plots (Section 4): `results/prism2_manual_knn/compare_histological_scores/{score_distributions,score_means,per_slide_delta}.html`, and the tabular summary `comparison_summary.csv`.

---

## 6 Discussion

**What we showed.** PRISM2's per-concept yes/no output is prompt-framing dependent to a degree that matters practically. Between multiple choice, temperature sampling and phrasing robustness, prevalence estimates for the same eleven UAMP concepts on the same 3,318 slides can differ by up to a factor of three. Native binary entropy is high for most concepts, meaning that most single-slide P(Yes) values sit close to 0.5 in the model's internal calculation. Where the underlying signal is strong (inflammation involvement, crypt abscesses, lymphoid aggregates, pyloric gland metaplasia), an independent severity readout stratifies the yes/no score cleanly across quartiles; where the signal is weak (mucin depletion, neuronal hyperplasia, muscular hypertrophy) the two readouts do not co-vary.

**What is robust.** Population-level scores are stable across the QC filter stack for nine of eleven concepts, with mean shifts under 3 %. Adding progressively more aggressive artefact removal narrows the mean without changing rank order or inter-slide variance. This is reassuring for downstream use of the numeric scores.

**What is not robust.** Two metaplasia concepts (Paneth cell, pyloric gland) shift by 8–10 % from raw to filtered, driven overwhelmingly by the single ≥ 15 % tissue coverage step. Non-tissue patches are systematically scored as metaplastic. Any analysis that pools scores across cohorts with different QC pipelines will conflate real biology with tissue selection for these two features. Investigators should either apply a matched QC pipeline to every cohort under comparison, or subtract a same-tissue-fraction control per slide.

Three concepts — Paneth cell metaplasia, mucin depletion, crypt architectural distortion — have between-phrasing SDs above 0.20. On a per-slide basis, this is enough that the score can change from "unlikely" to "likely" by rewording the question. These concepts should not be used as numeric features in downstream analyses without a fixed, validated question wording, or a paraphrase-averaged score with the phrasing set fixed in advance.

**Biological plausibility of the pattern.** The concepts the model is most confident about, and least prompt-sensitive on (muscular hypertrophy, histiocytic granulomas, neutrophil granulocytic infiltration), are histologically discrete: a granuloma is a compact structure, a neutrophil is a nucleated cell of characteristic size and morphology, muscular hypertrophy is a bulk change of a spatially bounded layer. The concepts the model is most uncertain about, and most prompt-sensitive on (Paneth cell metaplasia, mucin depletion, crypt architectural distortion), are context-dependent judgments in real practice: they require comparison against the expected mucosal architecture at that anatomical site. That the model finds the same distinction difficult that human readers find difficult is consistent with a real morphological ambiguity, not just a model quirk.

**Limitations.**

1. **No ground truth.** IBD-PLEXUS SPARC slides do not carry per-slide UAMP annotations. Every uncertainty measure reported here — entropy, MC "not sure", phrasing SD, severity — is a measure of the model's own consistency, not of its accuracy. A well-calibrated wrong model is not distinguishable from a well-calibrated right model with these data alone. Pathologist adjudication on a stratified sub-sample is the next step.
2. **Single model.** All five experiments use one model checkpoint (`paige-ai/prism2`) and one patch backbone (Virchow2). Whether the framing effects are a property of PRISM2 specifically, of vision–language models with this decoder design, or of prompted concept scoring in general, is not resolved by this study.
3. **Prompt space is discrete.** The four phrasings in E5 were chosen manually; they cover a plausible but small slice of the phrasing space. A larger phrasing panel could raise or lower the SD estimate.
4. **Deterministic decoding.** The yes/no probabilities are read from the logits at a single decoding step. Sampling from the full decoder (as in the free-text report generation not included in the headline analysis) may show larger or smaller variance depending on the concept.
5. **Cohort composition confound.** The prevalence estimates in Figure 1A depend on the composition of the cohort: 3,318 biopsies from an IBD registry are enriched for lesional colon and ileum. Cohort prevalence is not the same as slide-level probability of the feature, and the two should not be substituted for each other in clinical claims.
6. **Internal API coupling.** For efficiency we call PRISM2's `_encode_images`, `_project`, and `text_decoder` directly rather than the public wrappers. Model updates may break the scripts.

**Recommendations for downstream use.**

- Primary prevalence score: temperature-sampled P(Yes|T=2) from E1.
- Uncertainty flag: entropy from E1 combined with P(not sure) from E2. Slides with entropy > 0.9 and P(not sure) > 0.1 should be flagged.
- Severity readout: the weighted 0–3 score from E4, restricted to concepts where the severity-vs-P(Yes) quartile plot shows monotonic increase (Section 3.3).
- Avoid the 10-way competition (E3) as a per-concept prevalence estimator. It is useful only as a dominant-feature readout.
- Fix a validated phrasing before using any concept as a numeric feature, especially Paneth cell metaplasia, mucin depletion, or crypt architectural distortion.
- Match QC pipelines across cohorts, or restrict to concepts stable across QC (all except the two metaplasia terms).

---

## 7 Conclusion

PRISM2 is internally uncertain about most UAMP histological concepts on IBD biopsies, and its prevalence scores depend on how the question is asked. Both effects are reproducible across 3,318 slides, are unequal across concepts, and can be quantified with the five prompting experiments described here. Where an independent severity axis is available, it validates the yes/no scoring for the strong-signal concepts (inflammation, crypt abscesses, lymphoid aggregates, pyloric gland metaplasia). The upstream QC pipeline is orthogonal to the model's output for nine of eleven concepts; the two exceptions are the metaplasia terms, whose scores are inflated by non-tissue patches. We recommend reporting temperature-sampled P(Yes|T=2) with an entropy-based uncertainty flag, and treating scores for Paneth cell metaplasia, mucin depletion and crypt architectural distortion as prompt-conditional readouts until a fixed wording is validated against pathologist review.

---

## 8 Data and code availability

- Cohort: IBD-PLEXUS SPARC biopsies (access governed by the IBD-PLEXUS DUA).
- Model: `paige-ai/prism2` on HuggingFace, weights mirrored locally at `/home/jovyan/shared-data/users/kexin/models/VLM/prism2`.
- Features: Virchow2 CLS-token embeddings per patch, `/home/jovyan/kgbk271-ibd-volume/data/processed/...`.
- Pipeline code: `/home/jovyan/ibdplexus/code/prism2/`. Full pipeline described in [README.md](README.md). Uncertainty experiments in [uncertainty/](uncertainty/); QC-stage comparison in [compare/14_compare_histological_scores.py](compare/14_compare_histological_scores.py).
- Result artefacts referenced in this article: `/home/jovyan/kgbk271-ibd-volume/results/prism2_manual_knn/`.

No external citations are included in this article. Model, patch backbone, feature extractor and UMAP implementation are referenced by their canonical package identifiers so that the exact code paths are recoverable, and no bibliography is fabricated.
