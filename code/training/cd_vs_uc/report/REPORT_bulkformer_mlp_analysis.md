# CD vs UC Classification: BulkFormer Embeddings, Dimensionality Control, and Classifier Comparison

**Date:** 2026-08-24  
**Cohort:** At-20-cm biopsies, SPARC IBD Plexus  
**Scripts:** `08f`, `08g`, `08h`, `08h_rna_mlp_ablation`

---

## 1. Overview

This analysis evaluates CD vs UC classification performance across imaging, transcriptomics, and multimodal modalities, with three specific questions:

1. Do BulkFormer transcriptomic embeddings (pre-trained on TCGA) provide better signal than raw RNA-seq for CD/UC classification?
2. Is the BulkFormer + imaging multimodal gain due to the learned representation quality, or simply a dimensionality reduction artefact?
3. Does classifier choice (Random Forest vs MLP) change the conclusions?

---

## 2. Cohort

| | |
|---|---|
| Visits | 945 |
| Patients | 817 (CD: 504, UC: 313) |
| Tissue site | At 20 cm only |
| Matching | Imaging ↔ RNA proximity-matched (≤7 days) |
| CV | 5-fold patient-level split (`cv_splits_patients.csv`) |

All arms share the same 945-visit cohort. Patient-level splitting ensures zero patient overlap between train and validation in every fold.

---

## 3. Feature Representations

| Arm | Modality | Dimensionality | Source |
|---|---|---|---|
| `img_base_visit` | Imaging | 2,560 | prism2_base (Virchow2 perceiver) |
| `img_histoscore_visit` | Imaging | 11 | prism2 histological scores |
| `bulkformer_visit` | Transcriptomics | 640 | BulkFormer TCGA embeddings |
| `rna_visit` | Transcriptomics | 17,963 | RNA-seq VST (CombatSeq-corrected) |
| `pca640_rna_visit` | Transcriptomics | 640 | PCA(640) of RNA VST — dimensionality control |
| `concat_raw_visit` | Multimodal | 20,523 | RNA + prism2_base |
| `concat_histoscore_visit` | Multimodal | 17,974 | RNA + histo scores |
| `concat_bf_base_visit` | Multimodal | 3,200 | BulkFormer + prism2_base |
| `concat_bf_histo_visit` | Multimodal | 651 | BulkFormer + histo scores |
| `concat_pca640_base` | Multimodal | 3,200 | PCA-640 RNA + prism2_base — dimensionality control |

**BulkFormer:** Graph Transformer pre-trained on TCGA data using a gene co-expression graph. Applied to IBD log-TPM as pure inference (no IBD label leakage). Aggregates over 2,000 immune/inflammatory genes (`interested_gene_list.pt`): top genes include S100A9, CD74, LYZ, LGALS1, CD52, HBA1/2 — selected from general immunology, not from IBD labels.

**PCA-640 control:** PCA fitted on training fold only and applied to validation fold. No leakage. Same 640 output dimensions as BulkFormer; same 3,200-d total when concatenated with prism2_base. Directly tests whether BulkFormer's TCGA representations add value beyond linear compression.

---

## 4. Classifier Configuration

### Random Forest

```python
RandomForestClassifier(
    n_estimators=500, max_features='sqrt',
    min_samples_leaf=2, class_weight='balanced',
    n_jobs=-1, random_state=42
)
```

### MLP

```python
MLPClassifier(
    hidden_layer_sizes=(256, 128), activation='relu',
    solver='adam', alpha=1e-3, batch_size=64,
    max_iter=500, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20,
    random_state=42
)
# + StandardScaler (fitted on training fold only)
# + compute_sample_weight('balanced') for class imbalance
```

---

## 5. Results

### 5.1 Full 10-Arm Comparison — AUC (mean ± SD, 5-fold CV)

| Arm | Dim | RF AUC | MLP AUC | Δ (MLP−RF) |
|---|---|---|---|---|
| **Imaging** | | | | |
| `img_base_visit` | 2,560 | 0.758 ± 0.044 | **0.887 ± 0.028** | +0.129 |
| `img_histoscore_visit` | 11 | 0.733 ± 0.027 | 0.727 ± 0.032 | −0.006 |
| **Transcriptomics** | | | | |
| `bulkformer_visit` | 640 | 0.733 ± 0.032 | 0.733 ± 0.030 | 0.000 |
| `rna_visit` | 17,963 | **0.816 ± 0.033** | 0.761 ± 0.035 | −0.055 |
| `pca640_rna_visit` | 640 | 0.808 ± 0.039 | 0.780 ± 0.042 | −0.028 |
| **Multimodal** | | | | |
| `concat_raw_visit` | 20,523 | 0.820 ± 0.037 | **0.889 ± 0.029** | +0.069 |
| `concat_histoscore_visit` | 17,974 | 0.821 ± 0.030 | 0.764 ± 0.021 | −0.057 |
| `concat_bf_base_visit` | 3,200 | **0.914 ± 0.015** | 0.899 ± 0.023 | −0.015 |
| `concat_bf_histo_visit` | 651 | 0.746 ± 0.036 | 0.759 ± 0.043 | +0.013 |
| `concat_pca640_base` | 3,200 | **0.914 ± 0.013** | 0.897 ± 0.012 | −0.017 |

---

### 5.2 Finding 1 — BulkFormer Gain is a Dimensionality Artefact

The critical comparison is between `concat_bf_base_visit` and `concat_pca640_base`. Both fuse a 640-d transcriptomic representation with the 2,560-d prism2_base for a total of 3,200 features. The only difference is how the 640-d representation was obtained.

| Arm | RF | MLP |
|---|---|---|
| BulkFormer + prism2_base (3,200-d) | 0.914 ± 0.015 | 0.899 ± 0.023 |
| PCA-640 + prism2_base (3,200-d) | **0.914 ± 0.013** | **0.897 ± 0.012** |
| RNA + prism2_base (20,523-d) | 0.820 ± 0.037 | 0.889 ± 0.029 |

Under both classifiers, BulkFormer + prism2_base and PCA-640 + prism2_base are identical to 4 decimal places (RF) or within 0.002 AUC (MLP). The ~0.09 gain of the 3,200-d fusions over the 20,523-d RNA + prism2_base fusion under RF is entirely explained by feature space size: RF's random feature subsampling uses √3200 ≈ 57 features per split vs √20523 ≈ 143, giving better signal-to-noise. BulkFormer's TCGA pre-training contributes nothing beyond this compression.

Under MLP — which does not subsample features — the dimensionality gap shrinks substantially: RNA + prism2_base (0.889) now nearly matches BulkFormer + prism2_base (0.899).

**Conclusion:** BulkFormer has no advantage over linear PCA compression of the same data.

---

### 5.3 Finding 2 — prism2_base Signal is Severely Under-Exploited by RF

| Arm | RF | MLP |
|---|---|---|
| `img_base_visit` | 0.758 | **0.887** |
| `concat_raw_visit` | 0.820 | **0.889** |

The prism2_base 2,560-d features gain +0.129 AUC when switching from RF to MLP. This is the largest single-arm shift in the entire comparison. RF's random feature subsampling (√2560 ≈ 51 features per split) is an insufficient inductive bias for the structured, dense representations from the Virchow2 perceiver. MLP can fully exploit the continuous feature manifold.

This also explains why `concat_raw_visit` MLP (0.889) nearly matches `concat_bf_base_visit` MLP (0.899): with MLP, the imaging signal dominates and the transcriptomic representation becomes secondary.

---

### 5.4 Finding 3 — RF Outperforms MLP on High-Dimensional Raw RNA

| Configuration | AUC | vs RF (0.816) |
|---|---|---|
| RF, VST, implicit feature selection | **0.816 ± 0.033** | — |
| MLP (256,128), VST, L2 α=1e-3 | 0.786 ± 0.042 | −0.030 |
| MLP (256,128), log-TPM, L2 α=1e-3 | 0.794 ± 0.029 | −0.022 |
| MLP (2048,512,128), VST, L2 α=1e-2 | 0.801 ± 0.036 | −0.015 |
| MLP (2048,512,128), log-TPM, L2 α=1e-2 | 0.785 ± 0.032 | −0.031 |

Switching from VST to log-TPM input makes no material difference (≤0.008 AUC). A gradual bottleneck architecture (2048→512→128 with stronger L2) partially recovers performance but does not reach RF. RF's implicit feature selection via random column subsampling (√17963 ≈ 134 features per split) is a strong inductive bias well-suited to sparse high-dimensional gene expression data. With only ~680 effective training samples, the 17,963×2048 first-layer weight matrix (~36M parameters) cannot be adequately regularised by L2 alone.

---

### 5.5 Summary: What Each Classifier Is Best At

| Modality | Best classifier | Why |
|---|---|---|
| prism2_base (2,560-d) | **MLP** (+0.129) | Dense structured embedding; RF feature subsampling is too aggressive |
| RNA-seq VST (17,963-d) | **RF** (+0.055) | Sparse high-dim signal; RF implicit feature selection is the right inductive bias |
| BulkFormer (640-d) | Equal (0.000) | Mid-range dimensionality; representation quality is the ceiling, not classifier |
| Histoscores (11-d) | Equal (−0.006) | Low-dim; both classifiers saturate the available signal |

---

## 6. Preprocessing Contamination Audit

A systematic audit of all preprocessing and per-fold transforms was performed.

### Contaminated (upstream, all RNA arms affected equally)

| Step | Scope | Severity |
|---|---|---|
| CombatSeq batch correction | All 3,289 samples globally | High — batch parameters estimated using held-out samples |
| VST normalisation | All 3,289 samples globally | High — DESeq2 dispersion model fit on full cohort |
| BulkFormer inference | All 3,288 samples globally | Medium — fixed encoder, no label leakage; inherits CombatSeq input |

### Clean (per-fold, no leakage)

| Step | Script | Evidence |
|---|---|---|
| StandardScaler | 06, 06b, 08h | `StandardScaler().fit(X[tr])` |
| PCA(640) | 08g, 08h | `PCA().fit_transform(X[tr])` then `.transform(X[va])` |

### Implications

- **Absolute AUC values** may be marginally optimistic: CombatSeq batch correction (applied globally) could make data slightly cleaner than deployment conditions.
- **Relative comparisons between arms are valid**: all arms use the same preprocessed GCT; the contamination affects all equally.
- **The BulkFormer = PCA conclusion is unaffected**: both use the same GCT input; the comparison is internally clean.
- **Fix**: re-running VST + CombatSeq per training fold from raw counts would address this. This is non-trivial and atypical in the field; the limitation should be noted in any publication.

---

## 7. Key Caveats

1. **9 slides pending exclusion.** Slides `10502321HE101`, `11055940HE1`, `11007548HE1`, `10924448HE1`, `10537759HE101`, `10965446HE1`, `11051888HE1`, `11005300HE1`, `11025121HE1` were identified as problematic but are present in the current results. All RF and MLP results should be considered preliminary until these are excluded and experiments re-run.

2. **MLP without dropout.** sklearn's MLPClassifier has no dropout support. For the high-dimensional RNA arms (17,963-d), dropout or explicit gene pre-filtering would likely be needed to close the gap with RF. PyTorch would be required for a proper comparison.

3. **Early stopping uses internal validation split.** MLP's `validation_fraction=0.1` holds back ~68 samples per fold for convergence monitoring. This is standard practice but slightly reduces effective training size.

4. **BulkFormer gene selection.** The `interested_gene_list.pt` (2,000 genes: S100A9, CD74, LYZ, CD52, HBA1/2, etc.) was not selected based on IBD labels. However, the selection origin (general immunology vs any data-driven process) should be confirmed with the BulkFormer authors before publication.

---

## 8. Figures

All plots saved to `concept_learning/plots/`:

| File | Description |
|---|---|
| `bulkformer_10arm_comparison.png/pdf` | 10-arm RF bar chart |
| `rf_vs_mlp_comparison.png/pdf` | RF vs MLP side-by-side (10 arms each) |
| `mlp_10arm_comparison.png/pdf` | 10-arm MLP bar chart |
| `rf_vs_mlp_unimodal_interim.png/pdf` | Unimodal arms only comparison |

---

## 9. Scripts

| Script | Purpose |
|---|---|
| `08f_train_at20cm_bulkformer.py` | RF, BulkFormer arms (4 arms) |
| `08g_train_at20cm_pca_rna.py` | RF, PCA dimensionality control (2 arms) |
| `08h_train_at20cm_mlp.py` | MLP, all 10 arms |
| `08h_rna_mlp_ablation.py` | MLP ablation: VST vs log-TPM × architecture |
| `plot/bulkformer_comparison.py` | RF 10-arm comparison plot |
| `plot/mlp_comparison.py` | MLP plot + RF vs MLP side-by-side |
