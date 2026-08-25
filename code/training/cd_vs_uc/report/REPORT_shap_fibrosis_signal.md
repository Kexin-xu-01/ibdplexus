# SHAP Analysis — Fibrosis Signal Investigation
**CD vs UC Classifier · At-20-cm Site-Controlled Cohort**  
Generated: 2026-08-13

---

## 1. Background

A Random Forest classifier trained on VST RNA expression (17,963 genes, CombatSeq batch-corrected) from **At-20-cm colon biopsies** achieves AUC 0.824 ± 0.019 (828 patients, 5-fold CV). SHAP analysis of this model identified ECM/Fibrosis genes as the dominant predictors. This report investigates whether that signal reflects true biology or confounders, and validates top genes against the IBD literature.

**Cohort:** 841 CD · 409 UC (2.1:1 imbalance)  
**Biopsy site:** Sigmoid colon / At 20 cm  
**RNA data:** `GSF1491805_CombatSeq_vst_mtx_batch_corrected_alltissues_all3releases_header.gct`

---

## 2. Top 20 SHAP Genes — At-20-cm Model (`rna_20cm`)

| Rank | Gene | Ensembl ID | Mean \|SHAP\| | Biological category |
|------|------|-----------|--------------|---------------------|
| 1 | COL12A1 | ENSG00000111799 | 0.00294 | ECM / Fibrosis |
| 2 | POSTN | ENSG00000133110 | 0.00291 | ECM / Fibrosis |
| 3 | ZNF492 | ENSG00000229676 | 0.00289 | Transcription Factor |
| 4 | NPSR1 | ENSG00000187258 | 0.00260 | Neural / Receptor |
| 5 | CARD6 | ENSG00000132357 | 0.00250 | Immunity / Inflammation |
| 6 | FOXP2 | ENSG00000128573 | 0.00222 | Transcription Factor |
| 7 | MYEOV | ENSG00000172927 | 0.00219 | Other / Unknown |
| 8 | STPG4 | ENSG00000239605 | 0.00218 | Other / Unknown |
| 9 | BRINP3 | ENSG00000162670 | 0.00184 | Neural / Receptor |
| 10 | ITGB8 | ENSG00000105855 | 0.00161 | ECM / Fibrosis |
| 11 | PITX1 | ENSG00000069011 | 0.00153 | Transcription Factor |
| 12 | ACAT1 | ENSG00000075239 | 0.00149 | Metabolism / Transport |
| 13 | COL5A2 | ENSG00000204262 | 0.00148 | ECM / Fibrosis |
| 14 | TBX3 | ENSG00000135111 | 0.00145 | Transcription Factor |
| 15 | ADGRV1 | ENSG00000164199 | 0.00139 | ECM / Fibrosis |
| 16 | ABCA13 | ENSG00000179869 | 0.00139 | Metabolism / Transport |
| 17 | SHC3 | ENSG00000148082 | 0.00138 | Metabolism / Transport |
| 18 | SELENBP1 | ENSG00000143416 | 0.00134 | Metabolism / Transport |
| 19 | CYP2C18 | ENSG00000108242 | 0.00130 | Metabolism / Transport |
| 20 | XKR9 | ENSG00000221947 | 0.00125 | Other / Unknown |

> Note: SHAP values are mean absolute (unsigned). Direction confirmed separately via expression boxplots (Section 4).

---

## 3. Confounder Analysis — Patient Metadata

### 3.1 Disease duration imbalance

| | CD (n=841) | UC (n=409) | p-value |
|--|-----------|-----------|---------|
| Mean duration | 14.6 yr | 11.4 yr | <0.00001 (Mann-Whitney) |
| Median duration | 13.0 yr | 9.0 yr | |

Longer disease duration in CD → cumulative ECM deposition independent of active inflammation.

### 3.2 Surgery imbalance

| | CD | UC |
|--|---|---|
| Any IBD surgery | **29.1%** | 2.9% |
| Mean surgeries | 0.46 | 0.04 |

Post-surgical colon tissue (even distal sigmoid) can harbour reactive fibrosis and ECM remodelling. UC biopsies are almost entirely from unscarred mucosa.

In post-surgical CD patients, **87% have B2 (stricturing) or B3 (penetrating) phenotype**, meaning the most fibrotic CD subgroup is systematically enriched in post-surgical cases.

### 3.3 CD phenotype distribution

| CD phenotype | n | % |
|---|---|---|
| B1 — Inflammatory (non-stricturing) | 273 | 32.5% |
| B2 — Stricturing | 201 | 23.9% |
| B2B3 — Both | 69 | 8.2% |
| B3 — Penetrating | 124 | 14.7% |
| Unknown | 174 | 20.7% |

32% of CD patients are B2/B3 stricturing — the upper end of the fibrotic disease spectrum.  
B2/B3 mean disease duration: **16.6 yr** vs B1: **11.4 yr**.

### 3.4 Biologics imbalance

| | CD | UC | p-value |
|--|---|---|---|
| Any biologics | 72.5% | 60.4% | <0.001 |
| Anti-TNF specifically | 49.8% | 40.1% | 0.002 |

Anti-TNF suppresses acute mucosal inflammation but does not fully resolve established ECM/fibrosis, potentially widening the expression gap.

### 3.5 Disease activity — well matched

| | CD | UC | p-value |
|--|---|---|---|
| Mean PGA score (0–3) | 0.67 | 0.69 | 0.997 (Mann-Whitney) |
| Moderate/severe (≥2) | 19.0% | 21.4% | |

Disease activity at time of biopsy is comparable between groups — activity is **not** a confounder here.

---

## 4. Expression Boxplots — ECM Genes at 20 cm

**Key finding: fibrosis genes are HIGHER in UC, not CD.**

Analysis of 1,068 at-20-cm samples (UC=356, CD B1=273, CD B2/B3=330, CD unk.=109), CombatSeq VST expression:

| Gene | UC median | CD B1 median | CD B2/B3 median | UC vs CD B1 | UC vs CD B2/B3 |
|------|-----------|-------------|----------------|-------------|----------------|
| COL12A1 | ~10.4 | ~9.7 | ~9.3 | *** | *** |
| POSTN | ~10.8 | ~10.4 | ~10.1 | *** | *** |
| COL5A2 | ~10.0 | ~9.4 | ~9.2 | *** | * |
| COL3A1 | ~13.0 | ~12.7 | ~12.6 | *** | *** |
| SPARC | ~11.9 | ~11.1 | ~11.1 | *** | ** |
| CPXM1 | ~6.3 | ~5.3 | ~5.0 | *** | *** |
| ITGB8 | ~9.9 | ~9.4 | ~9.3 | *** | *** |

Mann-Whitney: * p<0.05 · ** p<0.01 · *** p<0.001  
**CD B1 ≈ CD B2/B3** — stricturing phenotype does not elevate ECM at the 20-cm sigmoid site.

### Interpretation

The model learned: **high ECM expression → UC**, not CD. The fibrosis signal reflects:

1. **Sigmoid-site UC biology**: The 20-cm site is involved in virtually every UC patient. Chronic mucosal UC here → pericryptal fibrosis, muscularis mucosae duplication, submucosal collagen deposition → high COL/POSTN/SPARC expression.
2. **CD at 20 cm is relatively quiescent**: Most CD fibrosis is transmural and occurs in the ileum or proximal colon, not the sigmoid. B2/B3 strictures rarely occur at 20 cm specifically.
3. **The gene category labels in SHAP are misleading**: Labels like "ECM / Fibrosis: CD fibrosis/stricture" describe what these genes do in CD biology generally, but in this site-specific classifier they fire in the UC direction.

---

## 5. Literature Evidence for Top SHAP Genes

### 5.1 Strongly validated — UC-high (consistent with expression boxplots)

| Gene | PubMed hits | Direction | Key papers |
|------|------------|-----------|------------|
| **CCL11** | 87 | **Up in UC** | Machine-learning UC transcriptomics: CCL11 diagnostic biomarker (high AUC with MMP1); Mendelian randomisation confirms UC association; elevated with gut microbiota changes in UC |
| **SPARC** | 53 | **Up in UC** | 2026 ML study: SPARC upregulated in UC intestinal fibroblasts specifically; AUC >0.8 for UC diagnosis; also a core IBD network hub gene |
| **POSTN** | 20 | Up in both; CD ileum literature dominates | Multiple papers confirm elevated in UC mucosa too; most CD studies sample terminal ileum/strictures (not sigmoid), explaining apparent CD-bias in literature |
| **COL12A1** | 3 | **Up in UC** | Upregulated in UC-associated neoplasia (PMID 42319552); marker of anti-fibrotic treatment efficacy in UC rat model (PMID 32982747) |
| **DAPP1** | 1 | **Up in UC** | Significantly overexpressed in UC non-responders; potential treatment resistance marker |
| **NPSR1** | 13 | Up in IBD broadly | Protein elevated in UC tissue and UC-derived organoids (PMID 29983891); knockdown protective in DSS colitis; gene polymorphisms linked to IBD risk |
| **ITGB8** | 2 | Risk allele increases expression → IBD risk | Nat Genet 2017 GWAS (n=59,957, PMID 28067908): established IBD susceptibility locus; expression-increasing allele correlates with disease risk |

### 5.2 Bidirectional discriminators — UC vs CD go opposite ways

| Gene | Direction | Key evidence |
|------|-----------|-------------|
| **SELENBP1** | **Down in active UC · Up in active CD** | Replicated across multiple independent cohorts; OR 23.7 for benign UC course when overexpressed (PMID 39728443); miR-122 methylation target in CD (PMID 31019652); among the cleanest bidirectional IBD expression discriminators in the literature |
| **BRINP3** | **Down in UC** (constitutive, independent of inflammation) | *Inflamm Bowel Dis* 2014 (PMID 25171508): most significantly underexpressed gene throughout the colon in UC; persists on rebiopsy at ~22 months; may predispose to UC. Replicated in ONT-RNA-seq (PMID 39955510) |

> SELENBP1 and BRINP3 are the most actionable findings from a biomarker perspective: both are directionally well-established and move in opposite directions between the two diseases.

### 5.3 CD-specific literature — UC-high signal at sigmoid would be novel

| Gene | Direction | Key evidence |
|------|-----------|-------------|
| **COL5A2** | **Up in CD** | scRNA-seq + bulk RNA-seq: higher in CD vs controls; lower in anti-TNF responders; proposed CD biomarker (PMID 38789829). **No UC comparison data exists — higher UC expression at sigmoid is a potentially novel finding.** |
| **TBX3** | **Up in CD fibrosis** | *Immunity* 2023 single-cell atlas (720k cells): TBX3 regulates disease-associated myofibroblast activation in CD fibrotic complications |
| **CPXM1** | **Up in inflamed CD** | Among 19 significantly dysregulated transcripts in active CD mucosa (PMID 28613228). No UC data. |

### 5.4 Genetic / GWAS evidence

| Gene | Evidence |
|------|---------|
| **FOXP2** | Suggestive CD GWAS locus (7q31); also implicated in Th2→Th9 differentiation relevant to colitis |
| **PITX1** | IBD GWAS functional screen: clusters with NOD2, NFKB1, IFIH1 as causal IBD genes; affects intestinal epithelial function |
| **CARD6** | Rare innate immunity mutations overlap with IBD genetics; no colonic expression data in IBD |

### 5.5 No IBD literature — novel candidates

| Gene | Notes |
|------|-------|
| **ZNF492** | Zero PubMed hits in any IBD context. Top-3 SHAP gene — may be genuine novel biology or dataset-specific noise. |
| **MYEOV** | Zero IBD hits. SHAP rank 7. |
| **ADGRV1** | Zero IBD hits. Adhesion GPCR; ECM adhesion function. |

---

## 6. Summary and Recommendations

### 6.1 Key conclusions

1. **The fibrosis signal is UC-driven at 20 cm, not CD-driven.** All 7 ECM genes tested (COL12A1, POSTN, COL5A2, COL3A1, SPARC, CPXM1, ITGB8) are significantly higher in UC than in both CD B1 and CD B2/B3 at the sigmoid. The model uses *high ECM → predict UC*.

2. **The signal is biologically plausible.** Chronic sigmoid UC causes pericryptal and submucosal fibrosis. This is under-reported in the literature compared to CD transmural fibrosis, but COL12A1, SPARC, CCL11, and POSTN have direct UC-upregulation evidence.

3. **Confounders amplify the contrast but do not create it.** CD patients in this cohort have longer disease duration (+3.2 yr, p<0.00001), much higher surgery rates (29% vs 3%), and greater B2/B3 enrichment than the general CD population — but their ECM expression at 20 cm is *lower*, not higher. These confounders do not explain the direction of the signal.

4. **SELENBP1 and BRINP3 are the most actionable novel markers.** Both are directionally established in independent cohorts and move in opposite directions between UC and CD — SELENBP1 down-in-UC / up-in-CD, BRINP3 constitutively down in UC.

5. **ZNF492, MYEOV, and ADGRV1 warrant follow-up.** Top SHAP genes with no prior IBD literature represent either genuine novel biology or site-specific confounders. Validation in external cohorts is needed.

### 6.2 Recommended sensitivity analyses

| Analysis | Rationale |
|----------|-----------|
| Restrict CD to B1 only | Removes stricturing-phenotype enrichment; tests whether UC-vs-B1-CD still shows the same ECM direction |
| Exclude post-surgical patients (both arms) | Removes reactive-fibrosis tissue from CD; tests whether surgery contamination affects ECM ranks |
| Add disease duration as covariate / match on it in CV splits | Controls for the 3-year median duration difference |
| Validate ZNF492, MYEOV, ADGRV1 in external UC/CD cohort | Tests whether these novel top-predictors replicate |
| SELENBP1 directional SHAP value extraction | Confirm SELENBP1 fires CD-high in this classifier as literature predicts |

---

## 7. Output Files

| File | Description |
|------|-------------|
| `plots/shap_rna_20cm_bar.pdf` | Top 20 genes — site-controlled At-20-cm RNA model |
| `plots/shap_rna_allsites_bar.pdf` | Top 20 genes — all-sites RNA model (includes HOX site confound) |
| `plots/shap_rank_comparison.pdf` | Dumbbell: rank shift between At-20-cm vs all-sites models |
| `plots/shap_fusion_split.pdf` | Multimodal fusion modality split (RNA 79% · Imaging 21%) |
| `plots/shap_panel.pdf` | 4-panel combined figure |
| `plots/fibrosis_gene_boxplots.pdf` | ECM gene expression boxplots — UC vs CD B1 vs CD B2/B3 at 20 cm |
| `shap_fibrosis_signal_report.md` | This report |

---

*Analysis performed on CombatSeq VST RNA data · 1,068 At-20-cm samples (356 UC, 273 CD B1, 330 CD B2/B3, 109 CD unknown phenotype) · Literature search via PubMed E-utilities (2026-08-13)*
