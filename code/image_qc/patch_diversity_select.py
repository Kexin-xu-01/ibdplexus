#!/usr/bin/env python3
"""
Patch diversity selection — adapted from pefim_utils for the PRISM2 patch pipeline.

Two modes
---------
Default (good-patch selection):
  Filter to top-quality patches, then cluster to deduplicate so the selected
  set is both high-quality AND diverse.  Output: selected_patches.csv

--invert (bad-patch clustering):
  Filter to the BOTTOM quality quantile, cluster the bad patches by artefact
  type, and output one representative per cluster.  Output can be fed directly
  into qc_find_nn.py as --bad_csv to automate the "seed bad patches" step —
  replacing the manual UMAP lasso selection in the prism2_knn workflow.
  With --out_all_csv you also get every bad patch labelled by cluster_id,
  useful for colouring the UMAP.

Quality score sources (CSV with slide_id, x, y, score):
  GrandQC tissue probability  →  higher = cleaner tissue
  Laplacian variance          →  higher = sharper
  Entropy                     →  higher = more texture
  If no --quality_csv, all patches are treated as equal (diversity only).

Usage
-----
  # Select diverse good patches (top 25% quality, deduplicated):
  python patch_diversity_select.py \\
      --quality_csv grandqc_scores.csv \\
      --exclusion_csv exclusion_list.csv \\
      --out_csv selected_patches.csv

  # Cluster bad patches → feed cluster reps into qc_find_nn:
  python patch_diversity_select.py --invert \\
      --quality_csv grandqc_scores.csv \\
      --quality_quantile 0.10 \\
      --distance_threshold 20 \\
      --out_csv bad_patch_reps.csv \\
      --out_all_csv bad_patches_all_clustered.csv

  # then:
  python prism2_knn/qc_find_nn.py \\
      --bad_csv bad_patch_reps.csv \\
      --dist_threshold 40 \\
      --html_out nn_gallery.html \\
      --out_csv exclusion_list.csv

  # Dry-run to calibrate --distance_threshold:
  python patch_diversity_select.py --invert --quality_csv ... --dry_run --max_slides 20
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FEAT_DIR = Path(
    "/home/jovyan/kgbk271-ibd-volume/data/processed/"
    "tissue_threshold_15/20x_224px_0px_overlap/features_virchow2"
)

# pefim_utils used DISTANCE_THRESHOLD=17 for endoscopy RGB features.
# Virchow2 2560-d L2 distances are larger; start around 20–40 and tune with --dry_run.
DEFAULT_DISTANCE_THRESHOLD = 30.0
MIN_CLUSTER_SIZE = 3   # mirrors pefim_utils MIN_NUM_FRAMES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir", default=str(FEAT_DIR))
    p.add_argument("--quality_csv", default=None,
                   help="CSV with slide_id,x,y,score (higher score = better quality).")
    p.add_argument("--quality_quantile", type=float, default=0.75,
                   help="Quality filter cutoff quantile (default 0.75). "
                        "Normal mode: keep patches above this (top 25%%). "
                        "--invert mode: keep patches below this (bottom 25%%).")
    p.add_argument("--invert", action="store_true",
                   help="Bad-patch mode: cluster the BOTTOM quality quantile instead "
                        "of the top. Output cluster representatives can be used as "
                        "--bad_csv for qc_find_nn.py.")
    p.add_argument("--exclusion_csv", default=None,
                   help="Exclusion list from prism2_knn — skipped before clustering.")
    p.add_argument("--distance_threshold", type=float, default=DEFAULT_DISTANCE_THRESHOLD,
                   help=f"L2 radius for greedy clustering (default {DEFAULT_DISTANCE_THRESHOLD}). "
                        "Patches within this distance → same cluster.")
    p.add_argument("--min_cluster_size", type=int, default=MIN_CLUSTER_SIZE,
                   help=f"Drop clusters smaller than this (default {MIN_CLUSTER_SIZE}).")
    p.add_argument("--out_csv", default="selected_patches.csv",
                   help="Output: cluster representatives "
                        "(good mode: diverse good patches; "
                        "invert mode: diverse bad-patch seeds for qc_find_nn).")
    p.add_argument("--out_all_csv", default=None,
                   help="(--invert only) Output all bad patches with cluster_id column "
                        "(useful for UMAP colouring).")
    p.add_argument("--bad_pool_csvs", nargs="*", default=None,
                   help="(--invert only) Extra bad-patch CSVs to merge into the bad pool "
                        "(e.g. exclusion lists from qc_find_nn.py). "
                        "Must have slide_id,x,y columns.")
    p.add_argument("--max_slides", type=int, default=None)
    p.add_argument("--slides_file", default=None,
                   help="Text file of slide IDs to process (one per line).")
    p.add_argument("--pca_components", type=int, default=50,
                   help="PCA dimensionality reduction before clustering (default 50). "
                        "Required for DBSCAN to work in high-d Virchow2 space. "
                        "Set to 0 to skip PCA (slow for d=2560).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print stats without writing output; use to calibrate thresholds.")
    return p.parse_args()


# ── Quality filtering (adapted from pefim_utils.get_topk_frames) ──────────────

def split_by_quality(
    feats: np.ndarray,
    coords: np.ndarray,
    slide_id: str,
    quality_map: dict | None,
    quantile: float,
    invert: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (feats_kept, coords_kept) filtered by quality score.

    Normal:  keep patches ABOVE quantile  (top quality, mirrors pefim_utils upper_quartile)
    Invert:  keep patches BELOW quantile  (bottom quality = likely artefacts)

    If quality_map is None: return all patches unchanged.
    """
    if quality_map is None or len(coords) == 0:
        return feats, coords

    scores = np.array([
        quality_map.get((slide_id, int(cx), int(cy)), np.nan)
        for cx, cy in coords
    ])
    valid = ~np.isnan(scores)
    if not valid.any():
        # No scored patches on this slide.
        # Normal mode: keep all (can't judge quality).
        # Invert mode: keep none (all patches presumed good, no known bad ones).
        if invert:
            return feats[:0], coords[:0]
        return feats, coords

    threshold = np.nanquantile(scores[valid], quantile)
    if invert:
        keep = valid & (scores <= threshold)
    else:
        keep = valid & (scores >= threshold)

    return feats[keep], coords[keep]


# ── Diversity clustering ───────────────────────────────────────────────────────

def cluster_patches(
    feats: np.ndarray,
    distance_threshold: float,
    min_cluster_size: int,
) -> tuple[list[int], list[dict], int, int]:
    """
    Cluster patches by feature-space distance and return one representative per cluster.

    Uses DBSCAN (sklearn, ball-tree backed) — O(N log N) vs the original pefim_utils
    greedy O(N·K) approach, which is too slow for >~2000 patches.

    DBSCAN parameters map naturally:
      eps         = distance_threshold  (radius for neighbourhood)
      min_samples = min_cluster_size    (min points to form a core point)
    Patches that are not within eps of min_cluster_size others are labelled -1 (noise).

    Representative per cluster = patch closest to the cluster median feature vector.

    Returns
    -------
    rep_indices   : local indices of the median representative per cluster
    cluster_info  : list of {cluster_id, member_indices, size, rep_idx}
    n_clusters    : number of clusters (excluding noise)
    n_noise       : patches labelled noise (-1) by DBSCAN
    """
    if len(feats) == 0:
        return [], [], 0, 0

    labels = DBSCAN(
        eps=distance_threshold,
        min_samples=min_cluster_size,
        algorithm="ball_tree",
        n_jobs=-1,
    ).fit_predict(feats)

    unique_labels = [l for l in set(labels) if l != -1]
    n_noise = int((labels == -1).sum())

    rep_indices = []
    cluster_info = []
    for cid in unique_labels:
        members = list(np.where(labels == cid)[0])
        cf = feats[members]
        med_local = int(np.argmin(
            np.linalg.norm(cf - np.median(cf, axis=0), axis=1)
        ))
        rep_idx = members[med_local]
        rep_indices.append(rep_idx)
        cluster_info.append({
            "cluster_id": int(cid),
            "member_indices": members,
            "size": len(members),
            "rep_idx": rep_idx,
        })

    return rep_indices, cluster_info, len(unique_labels), n_noise


# ── Main ───────────────────────────────────────────────────────────────────────

def _load_features_for_set(bad_set: set, feat_dir: Path, all_h5: list) -> tuple:
    """Load Virchow2 features for a specific set of (slide_id, x, y) tuples."""
    by_slide: dict[str, list] = {}
    for slide_id, x, y in bad_set:
        by_slide.setdefault(slide_id, []).append((x, y))

    feats_list, coords_list, slide_ids = [], [], []
    slides_with_features = {h.stem for h in all_h5}
    missing = 0
    for slide_id, coords_wanted in by_slide.items():
        if slide_id not in slides_with_features:
            missing += len(coords_wanted)
            continue
        h5_path = feat_dir / f"{slide_id}.h5"
        want_set = set(coords_wanted)
        with h5py.File(h5_path) as f:
            coords = f["coords"][:]
            # Find matching row indices (read coords first, then load only needed features)
            hit_idx = [j for j, (cx, cy) in enumerate(coords)
                       if (int(cx), int(cy)) in want_set]
            if not hit_idx:
                continue
            hit_idx_sorted = sorted(hit_idx)
            feats = f["features"][hit_idx_sorted]
        for k, j in enumerate(hit_idx_sorted):
            feats_list.append(feats[k])
            coords_list.append((slide_id, int(coords[j, 0]), int(coords[j, 1])))
    if missing:
        log.warning(f"  {missing} bad patches have no H5 feature file (skipped)")
    return (np.stack(feats_list) if feats_list else np.empty((0, 1)),
            coords_list)


def main():
    args = parse_args()
    feat_dir = Path(args.feat_dir)

    mode = "bad-patch clustering (cross-slide)" if args.invert else "good-patch selection (per-slide)"
    log.info(f"Mode: {mode}")

    # Load quality scores
    quality_map: dict | None = None
    if args.quality_csv:
        qdf = pd.read_csv(args.quality_csv)
        quality_map = {
            (str(r["slide_id"]), int(r["x"]), int(r["y"])): float(r["score"])
            for _, r in qdf.iterrows()
        }
        log.info(f"Quality CSV: {len(quality_map):,} scored patches from {args.quality_csv}")

    # Enumerate H5 files
    all_h5 = sorted(feat_dir.glob("*.h5"))
    if args.slides_file:
        allowed = set(Path(args.slides_file).read_text().split())
        all_h5 = [h for h in all_h5 if h.stem in allowed]
    if args.max_slides:
        all_h5 = all_h5[:args.max_slides]

    # ══════════════════════════════════════════════════════════════════════════
    # BAD-PATCH MODE (--invert): collect all bad patches across slides, cluster
    # globally to find artefact types. Per-slide clustering is wrong here
    # because most slides have only 1-2 bad patches — nothing to cluster within
    # a slide. Cross-slide clustering groups artefacts by type (pen mark, blur,
    # fold) regardless of which slide they came from.
    # ══════════════════════════════════════════════════════════════════════════
    if args.invert:
        # 1. Build bad patch set from quality CSV (score-0 patches)
        bad_set: set[tuple] = set()
        if quality_map:
            bad_set.update(
                (sid, x, y) for (sid, x, y), sc in quality_map.items() if sc == 0.0
            )
            log.info(f"  From quality CSV (score=0): {len(bad_set):,} patches")

        # 2. Merge extra bad-pool CSVs (exclusion lists from qc_find_nn.py)
        if args.bad_pool_csvs:
            for csv_path in args.bad_pool_csvs:
                df = pd.read_csv(csv_path)
                before = len(bad_set)
                for _, r in df.iterrows():
                    bad_set.add((str(r["slide_id"]), int(r["x"]), int(r["y"])))
                log.info(f"  From {Path(csv_path).name}: +{len(bad_set)-before:,} patches "
                         f"(total now {len(bad_set):,})")

        log.info(f"Total bad patches to cluster: {len(bad_set):,}  "
                 f"|  distance_threshold={args.distance_threshold}  "
                 f"|  min_cluster_size={args.min_cluster_size}")

        if len(bad_set) == 0:
            log.error("No bad patches found. Provide --quality_csv and/or --bad_pool_csvs.")
            return

        # 3. Load Virchow2 features for all bad patches
        log.info("Loading features for bad patches ...")
        pool_feats, pool_coords = _load_features_for_set(bad_set, feat_dir, all_h5)
        log.info(f"  Loaded {len(pool_feats):,} feature vectors")

        if len(pool_feats) == 0:
            log.error("No features loaded. Check feat_dir and slide IDs.")
            return

        # 4. Optionally reduce dimensionality before clustering
        feats_for_cluster = pool_feats
        if args.pca_components and args.pca_components < pool_feats.shape[1]:
            log.info(f"PCA: {pool_feats.shape[1]}d → {args.pca_components}d ...")
            pca = PCA(n_components=args.pca_components, random_state=0)
            feats_for_cluster = pca.fit_transform(pool_feats)
            log.info(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

        # 5. Global DBSCAN clustering
        log.info(f"DBSCAN clustering {len(feats_for_cluster):,} patches "
                 f"(d_threshold={args.distance_threshold}, min_samples={args.min_cluster_size}) ...")
        rep_idx, cluster_info, n_clusters, n_dropped = cluster_patches(
            feats_for_cluster, args.distance_threshold, args.min_cluster_size
        )

        # 5. Collect output rows
        all_reps = []
        all_members = []
        for ci in cluster_info:
            sid, rx, ry = pool_coords[ci["rep_idx"]]
            all_reps.append({
                "slide_id": sid, "x": rx, "y": ry,
                "cluster_id": ci["cluster_id"],
                "cluster_size": ci["size"],
                "reason": "auto_bad_cluster",
            })
            if args.out_all_csv or args.dry_run:
                for mi in ci["member_indices"]:
                    msid, mx, my = pool_coords[mi]
                    all_members.append({
                        "slide_id": msid, "x": mx, "y": my,
                        "cluster_id": ci["cluster_id"],
                        "is_rep": int(mi == ci["rep_idx"]),
                    })

        log.info("")
        log.info("── Summary " + "─" * 50)
        log.info(f"  Total bad patches           : {len(bad_set):>10,}")
        log.info(f"  Features loaded             : {len(pool_feats):>10,}")
        log.info(f"  Clusters formed             : {n_clusters:>10,}")
        log.info(f"  Noise patches (no cluster)  : {n_dropped:>10,}")
        log.info(f"  Cluster representatives out : {len(all_reps):>10,}")
        log.info("")
        log.info("  → Feed --out_csv into qc_find_nn.py --bad_csv to auto-seed NN search")

        if not args.dry_run:
            pd.DataFrame(all_reps).to_csv(args.out_csv, index=False)
            log.info(f"\nRepresentatives → {args.out_csv}  ({len(all_reps):,} rows)")
            if args.out_all_csv and all_members:
                pd.DataFrame(all_members).to_csv(args.out_all_csv, index=False)
                log.info(f"All clustered patches → {args.out_all_csv}  ({len(all_members):,} rows)")
        else:
            log.info("[dry_run] No output written.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # GOOD-PATCH MODE (default): per-slide quality filter + per-slide clustering
    # ══════════════════════════════════════════════════════════════════════════
    exclusion: set[tuple] = set()
    if args.exclusion_csv:
        excl_df = pd.read_csv(args.exclusion_csv)
        exclusion = set(zip(excl_df["slide_id"],
                            excl_df["x"].astype(int),
                            excl_df["y"].astype(int)))
        log.info(f"Exclusion list: {len(exclusion):,} patches skipped")

    log.info(f"Slides: {len(all_h5)}  |  distance_threshold={args.distance_threshold}  "
             f"|  min_cluster_size={args.min_cluster_size}")

    all_reps = []
    all_members = []
    total_input = total_after_excl = total_after_quality = total_selected = 0
    total_clusters = total_dropped = 0
    global_cluster_id = 0

    for i, h5_path in enumerate(all_h5):
        if i % 100 == 0:
            log.info(f"  [{i+1}/{len(all_h5)}] {h5_path.stem} ...")
        with h5py.File(h5_path) as f:
            feats  = f["features"][:]
            coords = f["coords"][:]
        slide_id = h5_path.stem
        total_input += len(feats)

        if exclusion:
            keep = np.array([
                (slide_id, int(cx), int(cy)) not in exclusion
                for cx, cy in coords
            ])
            feats, coords = feats[keep], coords[keep]
        total_after_excl += len(feats)
        if len(feats) == 0:
            continue

        feats_q, coords_q = split_by_quality(
            feats, coords, slide_id, quality_map, args.quality_quantile, False
        )
        total_after_quality += len(feats_q)
        if len(feats_q) == 0:
            continue

        rep_idx, cluster_info, n_cl, n_dr = cluster_patches(
            feats_q, args.distance_threshold, args.min_cluster_size
        )
        total_clusters += n_cl
        total_dropped  += n_dr
        total_selected += len(rep_idx)

        for ci in cluster_info:
            rx, ry = coords_q[ci["rep_idx"]]
            all_reps.append({
                "slide_id": slide_id, "x": int(rx), "y": int(ry),
                "cluster_id": global_cluster_id + ci["cluster_id"],
                "cluster_size": ci["size"],
                "reason": "auto_good_rep",
            })
            if args.out_all_csv or args.dry_run:
                for mi in ci["member_indices"]:
                    mx, my = coords_q[mi]
                    all_members.append({
                        "slide_id": slide_id, "x": int(mx), "y": int(my),
                        "cluster_id": global_cluster_id + ci["cluster_id"],
                        "is_rep": int(mi == ci["rep_idx"]),
                    })
        global_cluster_id += n_cl

    log.info("")
    log.info("── Summary " + "─" * 50)
    log.info(f"  Input patches               : {total_input:>10,}")
    log.info(f"  After exclusion list        : {total_after_excl:>10,}  "
             f"({100*total_after_excl/max(total_input,1):.1f}%)")
    if quality_map is not None:
        log.info(f"  After quality filter (top)  : {total_after_quality:>10,}  "
                 f"({100*total_after_quality/max(total_input,1):.1f}%)")
    log.info(f"  Clusters formed             : {total_clusters:>10,}")
    log.info(f"  Noise patches (no cluster)  : {total_dropped:>10,}")
    log.info(f"  Cluster representatives out : {total_selected:>10,}  "
             f"({100*total_selected/max(total_input,1):.3f}%)")

    if args.dry_run:
        log.info("\n[dry_run] No output written.")
        return

    pd.DataFrame(all_reps).to_csv(args.out_csv, index=False)
    log.info(f"\nRepresentatives → {args.out_csv}  ({len(all_reps):,} rows)")
    if args.out_all_csv and all_members:
        pd.DataFrame(all_members).to_csv(args.out_all_csv, index=False)
        log.info(f"All clustered patches → {args.out_all_csv}  ({len(all_members):,} rows)")


if __name__ == "__main__":
    main()
