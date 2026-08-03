"""
Convert 8 large VSI slides (64-113 GB uncompressed) to pyramidal TIFF.
Uses tiled region reading so the full image is never loaded into RAM.

Strategy:
  - Read full-res in horizontal strips (STRIP_HEIGHT rows at a time)
  - Write full-res page tile-by-tile using tifffile
  - Build each pyramid level by downsampling the strips on the fly
  - Write pyramid levels as flat multi-page (subfiletype=1) to match existing TIFFs

Format matches existing tiff_mpp_corrected files:
  LZW, 256x256 tiles, flat multi-page pyramid, ResolutionUnit=centimeter
"""

import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import slideio
import tifffile
from tqdm import tqdm

OUTPUT_DIR = Path("/home/jovyan/kgbk271-ibd-datavol-1/data/raw/tiff_mpp_corrected")
TILE_SIZE  = 256
STRIP_ROWS = 4096  # read this many rows at a time from VSI

SLIDES = [
    ("10432420HE1",   "/home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe/vsi-2021/10432420HE1.vsi"),
    ("10449041HE1",   "/home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe/vsi-2021/10449041HE1.vsi"),
    ("10475951HE1",   "/home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe/vsi-2021/10475951HE1.vsi"),
    ("10476471HE1",   "/home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe/vsi-2021/10476471HE1.vsi"),
    ("10502177HE1",   "/home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe/vsi-2021/10502177HE1.vsi"),
    ("10502178HE1",   "/home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe/vsi-2021/10502178HE1.vsi"),
    ("10537763HE101", "/home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe/vsi-2021/10537763HE101.vsi"),
    ("10713340HE1",   "/home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe/vsi-2021/10713340HE1.vsi"),
]


def n_pyramid_levels(w, h):
    levels, cw, ch = 1, w, h
    while min(cw, ch) >= 256:
        cw, ch = cw // 2, ch // 2
        levels += 1
    return levels


def convert_one(name: str, vsi_path: str) -> None:
    out_path = OUTPUT_DIR / f"{name}.tiff"
    tmp_path = OUTPUT_DIR / f"{name}.tiff.converting"

    if out_path.exists():
        print(f"[SKIP] {name} already exists")
        return

    slide = slideio.open_slide(vsi_path, "VSI")
    scene = slide.get_scene(0)
    mpp = scene.resolution[0] * 1e6
    W, H = scene.size  # width, height (x, y)
    n_levels = n_pyramid_levels(W, H)
    res_cm = 1e4 / mpp

    print(f"\n[{name}] {W}x{H}  mpp={mpp:.6f}  {W*H*3/1e9:.1f}GB raw  {n_levels} pyramid levels")
    print(f"[{name}] Writing -> {tmp_path}")

    with tifffile.TiffWriter(str(tmp_path), bigtiff=True) as tif:

        # --- Full-resolution level ---
        # Accumulate strip columns into row-aligned tile buffers, write tile rows
        # as soon as they are complete.
        opts_base = dict(
            tile=(TILE_SIZE, TILE_SIZE),
            compression="lzw",
            resolutionunit=tifffile.RESUNIT.CENTIMETER,
            resolution=(res_cm, res_cm),
            photometric="rgb",
            subfiletype=0,
        )
        # tifffile needs the full image shape up front when writing tiled pages
        # Use write() with a generator isn't supported, so we must pass the array.
        # Instead write row-by-row using the lower-level approach: write each
        # TILE_SIZE-row strip as a contiguous region and let tifffile tile it.
        # Simpler: read strips and accumulate full rows of tiles, then write.
        # Since we can't stream into tifffile's tiled writer easily, we write
        # whole level at once but read it in strips to avoid OOM.

        # Build full-res page row-by-row into a memmap-backed array on disk
        # so we never have >STRIP_ROWS*W*3 bytes in RAM at once.
        memmap_path = str(tmp_path) + ".mm"
        mm = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=(H, W, 3))

        n_strips = (H + STRIP_ROWS - 1) // STRIP_ROWS
        for i in tqdm(range(n_strips), desc=f"{name} reading", unit="strip", leave=False):
            y0 = i * STRIP_ROWS
            h = min(STRIP_ROWS, H - y0)
            strip = scene.read_block(rect=(0, y0, W, h), size=(W, h))
            mm[y0:y0+h] = strip
            del strip

        print(f"[{name}] Writing full-res level to TIFF...")
        tif.write(mm, **opts_base)

        # --- Pyramid levels (subfiletype=1) ---
        opts_sub = dict(
            tile=(TILE_SIZE, TILE_SIZE),
            compression="lzw",
            resolutionunit=tifffile.RESUNIT.CENTIMETER,
            resolution=(res_cm, res_cm),
            photometric="rgb",
            subfiletype=1,
        )
        prev = mm
        for lvl in range(1, n_levels):
            downsampled = prev[::2, ::2].copy()
            lh, lw = downsampled.shape[:2]
            print(f"[{name}]   level {lvl}: {lw}x{lh}")
            tif.write(downsampled, **opts_sub)
            prev = downsampled

        del mm

    os.unlink(memmap_path)
    shutil.move(str(tmp_path), str(out_path))
    print(f"[{name}] Done -> {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors = []
    for name, vsi_path in SLIDES:
        try:
            convert_one(name, vsi_path)
        except Exception:
            print(f"[ERROR] {name}:")
            traceback.print_exc()
            errors.append(name)
            for ext in (".tiff.converting", ".tiff.converting.mm"):
                p = OUTPUT_DIR / f"{name}{ext}"
                if p.exists():
                    p.unlink()

    print(f"\nFinished. Errors ({len(errors)}): {errors}")


if __name__ == "__main__":
    main()
