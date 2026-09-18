"""SEN12MS-CR & SEN12MS-CR-TS Dataset Downloader (WP0)
===========================================================
Official Landing Links:
- SEN12MS-CR (Mono-temporal): https://patricktum.github.io/cloud_removal/sen12mscr/
- SEN12MS-CR-TS (Multi-temporal): https://patricktum.github.io/cloud_removal/sen12mscrts/
- TUM Mediatum: https://mediatum.ub.tum.de/1554803

Downloads from the official TUM DataServ mirror (https://dataserv.ub.tum.de)
using the same archives as the maintainers' `dl_data.sh` (see
https://github.com/PatrickTUM/SEN12MS-CR-TS/blob/master/util/dl_data.sh).

Layout produced under the target root:
    <root>/
    ├── ROIs1158_spring_s2_clear/   (or ..._s2/)  -> 13-band clear Sentinel-2 .tif
    ├── ROIs1158_spring_s2_cloudy/                -> 13-band cloudy Sentinel-2 .tif
    ├── ROIs1158_spring_s1/                       -> 2-band (VV/VH) Sentinel-1 .tif
    └── ... (one folder set per ROI-season)

Features:
- Native HTTP Range resume (safe for interrupted downloads).
- Sequential download -> extract -> delete-archive per file (bounded disk usage).
"""

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

LANDING_PAGES = {
    "sen12ms_cr": "https://patricktum.github.io/cloud_removal/sen12mscr/",
    "sen12ms_cr_ts": "https://patricktum.github.io/cloud_removal/sen12mscrts/",
    "tum_mediatum": "https://mediatum.ub.tum.de/1554803",
}

# Official archives (URLs from SEN12MS-CR-TS util/dl_data.sh).
# size_approx_gb is the archive size, not the extracted size.
SEN12MS_CR_ARCHIVES: Dict[str, dict] = {
    "spring_s2_clear": {
        "url": "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1158_spring_s2.tar.gz",
        "size_approx_gb": 24.9,
        "modality": "s2_clear",
        "dataset": "sen12ms_cr",
    },
    "spring_s2_cloudy": {
        "url": "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1158_spring_s2_cloudy.tar.gz",
        "size_approx_gb": 24.9,
        "modality": "s2_cloudy",
        "dataset": "sen12ms_cr",
    },
    "spring_s1": {
        "url": "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1158_spring_s1.tar.gz",
        "size_approx_gb": 7.7,
        "modality": "s1",
        "dataset": "sen12ms_cr",
    },
    "summer_s2_clear": {
        "url": "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1868_summer_s2.tar.gz",
        "size_approx_gb": 28.9,
        "modality": "s2_clear",
        "dataset": "sen12ms_cr",
    },
    "summer_s2_cloudy": {
        "url": "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1868_summer_s2_cloudy.tar.gz",
        "size_approx_gb": 28.9,
        "modality": "s2_cloudy",
        "dataset": "sen12ms_cr",
    },
    "summer_s1": {
        "url": "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1868_summer_s1.tar.gz",
        "size_approx_gb": 8.9,
        "modality": "s1",
        "dataset": "sen12ms_cr",
    },
    "fall_s2_clear": {
        "url": "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1970_fall_s2.tar.gz",
        "size_approx_gb": 35.0,
        "modality": "s2_clear",
        "dataset": "sen12ms_cr",
    },
    "fall_s2_cloudy": {
        "url": "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1970_fall_s2_cloudy.tar.gz",
        "size_approx_gb": 35.0,
        "modality": "s2_cloudy",
        "dataset": "sen12ms_cr",
    },
    "fall_s1": {
        "url": "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1970_fall_s1.tar.gz",
        "size_approx_gb": 10.8,
        "modality": "s1",
        "dataset": "sen12ms_cr",
    },
    "winter_s2_clear": {
        "url": "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs2017_winter_s2.tar.gz",
        "size_approx_gb": 15.7,
        "modality": "s2_clear",
        "dataset": "sen12ms_cr",
    },
    "winter_s2_cloudy": {
        "url": "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs2017_winter_s2_cloudy.tar.gz",
        "size_approx_gb": 15.7,
        "modality": "s2_cloudy",
        "dataset": "sen12ms_cr",
    },
    "winter_s1": {
        "url": "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs2017_winter_s1.tar.gz",
        "size_approx_gb": 4.8,
        "modality": "s1",
        "dataset": "sen12ms_cr",
    },
}

# TLS is handled by the requests library; the server redirects (303) to the
# actual object store, which supports HTTP Range requests for resuming.


def download_with_resume(
    url: str,
    dest_path: Path,
    desc: Optional[str] = None,
    chunk_size: int = 1024 * 1024,
    timeout: int = 120,
) -> bool:
    """Downloads a remote file with native HTTP Range header resume support."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    initial_size = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {}
    if initial_size > 0:
        headers["Range"] = f"bytes={initial_size}-"

    print(f"\n[DOWNLOAD] Connecting to: {url}")
    if initial_size > 0:
        print(f"[RESUME] Resuming from byte {initial_size} ({initial_size / (1024 ** 3):.2f} GB)")

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True)
        if response.status_code == 416:
            temp_path.rename(dest_path)
            return True
        elif response.status_code not in (200, 206):
            print(f"[WARN] Server returned HTTP status {response.status_code}")
            return False

        total_size = int(response.headers.get("content-length", 0)) + initial_size
        mode = "ab" if initial_size > 0 else "wb"

        with open(temp_path, mode) as f, tqdm(
            desc=desc or dest_path.name,
            total=total_size,
            initial=initial_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        temp_path.rename(dest_path)
        print(f"[SUCCESS] Download completed: {dest_path.name}")
        return True

    except Exception as e:
        print(f"[ERROR] Download error: {e}")
        print(f"[INFO] Saved partial download at {temp_path} for resumption.")
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """Extracts tar/tar.gz/zip archives."""
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXTRACT] Extracting {archive_path.name} to {extract_dir} ...")
    try:
        if archive_path.name.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=extract_dir)
        elif archive_path.name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            print(f"[WARN] Unrecognized archive extension for {archive_path.name}")
            return False
        print(f"[SUCCESS] Extracted {archive_path.name}")
        return True
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False


def organize_into_standard_structure(raw_dir: Path) -> dict:
    """Reports dataset structure: scans for s1 / s2_cloudy / s2_clear GeoTIFFs."""
    raw_dir = Path(raw_dir)
    s1_files = list(raw_dir.rglob("*s1*.tif*")) + list(raw_dir.rglob("*sigma0*.tif*"))
    s2_cloudy = list(raw_dir.rglob("*s2_cloudy*.tif*"))
    s2_clear = list(raw_dir.rglob("*s2_clear*.tif*")) + [
        f for f in raw_dir.rglob("*s2*.tif*")
        if "cloudy" not in f.name.lower()
    ]
    counts = {
        "sigma0": len(s1_files),
        "cloudy": len(s2_cloudy),
        "clear": len(s2_clear),
    }
    print(f"[ORGANIZE] Dataset structure check: {counts}")
    return counts


def download_sen12ms_subset(
    target_dir: Path,
    subset: bool = True,
    seasons: Optional[List[str]] = None,
    dataset_type: str = "sen12ms_cr",
    extract: bool = True,
    dry_run: bool = False,
    delete_archive: bool = True,
) -> dict:
    """Main downloader for SEN12MS-CR / SEN12MS-CR-TS archives.

    Downloads each archive sequentially, extracts it, then deletes the archive
    to keep peak disk usage bounded.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Work Package 0: Geospatial Data Ingestion")
    print(f"Target directory: {target_dir}")
    print(f"Dataset: {dataset_type}")
    print(f"Landing URL: {LANDING_PAGES.get(dataset_type, LANDING_PAGES['sen12ms_cr'])}")
    print("=" * 70)

    selected_keys = []
    for k, v in SEN12MS_CR_ARCHIVES.items():
        if v.get("dataset") == dataset_type or dataset_type == "all":
            if seasons is None:
                selected_keys.append(k)
            elif any(s.lower() in k.lower() for s in seasons):
                selected_keys.append(k)

    results = {}
    for key in selected_keys:
        info = SEN12MS_CR_ARCHIVES[key]
        dest_archive = target_dir / Path(info["url"].split("files=")[-1].split("&")[0])
        print(f"\n- Partition: {key}")
        print(f"  Modality: {info['modality']}")
        print(f"  Approx Archive Size: {info['size_approx_gb']} GB")
        print(f"  Archive Path: {dest_archive}")

        if dry_run:
            results[key] = "dry_run"
            continue

        if dest_archive.exists():
            print(f"  [SKIP] Archive already present: {dest_archive.name}")
            success = True
        else:
            success = download_with_resume(info["url"], dest_archive, desc=key)

        if success and extract and dest_archive.exists():
            extract_success = extract_archive(dest_archive, target_dir)
            if extract_success:
                results[key] = "downloaded_and_extracted"
                if delete_archive:
                    try:
                        dest_archive.unlink()
                        print(f"  [CLEAN] Removed archive {dest_archive.name}")
                    except OSError as e:
                        print(f"  [WARN] Could not delete archive: {e}")
            else:
                results[key] = "downloaded_extract_failed"
        elif success:
            results[key] = "success"
        else:
            results[key] = "failed"
            print(f"  [!] Failed to download {key}; resume later with the same command.")

    if not dry_run:
        organize_into_standard_structure(target_dir)

    return results


def main():
    parser = argparse.ArgumentParser(description="SEN12MS-CR / SEN12MS-CR-TS Ingestion Downloader (WP0)")
    parser.add_argument("--target-dir", type=str, default="data/raw/sen12ms_cr",
                        help="Root directory for the SEN12MS-CR dataset")
    parser.add_argument("--subset", action="store_true", default=False,
                        help="Compatibility flag (all archives are curated per-ROI)")
    parser.add_argument("--dataset", type=str, default="sen12ms_cr",
                        choices=["sen12ms_cr", "all"],
                        help="Target dataset")
    parser.add_argument("--seasons", nargs="+", default=None,
                        help="Seasons to download (spring, summer, fall, winter)")
    parser.add_argument("--modalities", nargs="+", default=None,
                        help="Modalities to download (s2_clear, s2_cloudy, s1)")
    parser.add_argument("--no-extract", action="store_true", help="Skip archive extraction")
    parser.add_argument("--keep-archive", action="store_true",
                        help="Keep archive after successful extraction (uses more disk)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without downloading")
    args = parser.parse_args()

    download_sen12ms_subset(
        target_dir=Path(args.target_dir),
        subset=args.subset,
        seasons=args.seasons,
        dataset_type=args.dataset,
        extract=not args.no_extract,
        dry_run=args.dry_run,
        delete_archive=not args.keep_archive,
    )


if __name__ == "__main__":
    main()