import requests
from pathlib import Path
from tqdm import tqdm

from src.config import LISS4_RAW, S1_RAW, S2_RAW, DEM_RAW


def download_file(url: str, dest: Path, desc: str = None) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[SKIP] {dest.name} already exists")
        return dest

    print(f"[DOWNLOAD] {desc or dest.name}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(desc=desc or dest.name, total=total, unit="B", unit_scale=True) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    return dest


def download_liss4(scene_id: str, out_dir: Path = None) -> Path:
    out_dir = Path(out_dir or LISS4_RAW)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"liss4_{scene_id}.tif"

    print(f"\n{'='*60}")
    print(f"LISS-IV Scene: {scene_id}")
    print(f"{'='*60}")
    print("Manual download required from Bhoonidhi portal:")
    print(f"  1. Go to: https://bhoonidhi.nrsc.gov.in")
    print(f"  2. Search for LISS-IV FMX product")
    print(f"  3. Select scene: {scene_id}")
    print(f"  4. Download and save to: {out_path}")
    print(f"{'='*60}\n")

    return out_path


def download_sentinel1(scene_id: str, out_dir: Path = None) -> Path:
    out_dir = Path(out_dir or S1_RAW)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"s1_{scene_id}.zip"

    print(f"\n{'='*60}")
    print(f"Sentinel-1 Scene: {scene_id}")
    print(f"{'='*60}")
    print("Manual download required from Copernicus Data Space:")
    print(f"  1. Go to: https://dataspace.copernicus.eu")
    print(f"  2. Search for Sentinel-1 GRD product")
    print(f"  3. Select scene matching LISS-IV AOI and date")
    print(f"  4. Download and save to: {out_path}")
    print(f"{'='*60}\n")

    return out_path


def download_sentinel2(scene_id: str, out_dir: Path = None) -> Path:
    out_dir = Path(out_dir or S2_RAW)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"s2_{scene_id}.zip"

    print(f"\n{'='*60}")
    print(f"Sentinel-2 Scene: {scene_id}")
    print(f"{'='*60}")
    print("Manual download required from Copernicus Data Space:")
    print(f"  1. Go to: https://dataspace.copernicus.eu")
    print(f"  2. Search for Sentinel-2 L2A product")
    print(f"  3. Select scene matching LISS-IV AOI and date")
    print(f"  4. Download and save to: {out_path}")
    print(f"{'='*60}\n")

    return out_path


def download_srtm(bbox: tuple, out_dir: Path = None) -> Path:
    out_dir = Path(out_dir or DEM_RAW)
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_min, lon_min, lat_max, lon_max = bbox
    out_path = out_dir / f"srtm_{lat_min:.1f}_{lon_min:.1f}_{lat_max:.1f}_{lon_max:.1f}.tif"

    url = (
        f"https://portal.opentopography.org/API/globaldem?"
        f"demtype=SRTMGL3&west={lon_min}&south={lat_min}"
        f"&east={lon_max}&north={lat_max}&output=GTiff"
    )

    try:
        return download_file(url, out_path, desc="SRTM DEM")
    except Exception as e:
        print(f"[WARN] OpenTopography API failed: {e}")
        print("Fallback: Download manually from https://earthdata.nasa.gov")
        return out_path


def list_available_scenes() -> dict:
    raw_dirs = {
        "LISS-IV": LISS4_RAW,
        "Sentinel-1": S1_RAW,
        "Sentinel-2": S2_RAW,
        "DEM": DEM_RAW,
    }

    summary = {}
    for name, path in raw_dirs.items():
        files = sorted(path.glob("*"))
        summary[name] = [f.name for f in files if f.is_file()]
        print(f"{name}: {len(summary[name])} files")

    return summary


if __name__ == "__main__":
    list_available_scenes()
