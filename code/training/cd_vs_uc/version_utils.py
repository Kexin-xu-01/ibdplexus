"""
Versioning helpers for report and slide generation scripts.

Usage in a generation script
-----------------------------
    from version_utils import next_versioned_path, log_version

    pdf_path, v = next_versioned_path(os.path.join(OUT_DIR, 'my_report.pdf'))
    # ... build report ...
    doc.build(story)
    log_version(pdf_path, description)   # appends to VERSIONS.md
"""

import os
import glob
import re

VERSIONS_MD = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/VERSIONS.md'


def next_versioned_path(base_path):
    """
    Return (versioned_path, version_number) for the next unused version of base_path.

    Example
    -------
    If report_v1.pdf and report_v2.pdf already exist, returns ('report_v3.pdf', 3).
    If no versioned files exist yet, returns ('report_v1.pdf', 1).
    """
    stem, ext = os.path.splitext(base_path)
    existing = glob.glob(f'{stem}_v*{ext}')
    nums = []
    for f in existing:
        m = re.search(r'_v(\d+)$', os.path.splitext(f)[0])
        if m:
            nums.append(int(m.group(1)))
    version = max(nums) + 1 if nums else 1
    return f'{stem}_v{version}{ext}', version


def log_version(file_path, description):
    """
    Append a version entry to VERSIONS.md.

    Parameters
    ----------
    file_path   : full path of the saved file (must contain _vN suffix)
    description : one-line summary of what changed in this version
    """
    basename = os.path.basename(file_path)
    m = re.search(r'_v(\d+)\.[^.]+$', basename)
    version = m.group(1) if m else '?'
    stem = re.sub(r'_v\d+$', '', os.path.splitext(basename)[0])

    entry = f'| {stem} | v{version} | {description} |\n'

    if not os.path.exists(VERSIONS_MD):
        with open(VERSIONS_MD, 'w') as f:
            f.write('# Output Versions\n\n'
                    '| File | Version | Description |\n'
                    '|------|---------|-------------|\n')

    with open(VERSIONS_MD, 'a') as f:
        f.write(entry)

    print(f'Version logged: {basename} — {description}')
