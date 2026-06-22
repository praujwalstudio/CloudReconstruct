import json
import numpy as np
from pathlib import Path
from datetime import datetime


def _make_geotiff_profile(profile: dict, height: int, width: int, count: int,
                          dtype: np.dtype) -> dict:
    out_profile = profile.copy() if profile else {}
    out_profile.update({
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
    })
    return out_profile


def write_geotiff(output_path: Path, corrected_image: np.ndarray,
                  confidence_map: np.ndarray = None, profile: dict = None,
                  metadata: dict = None) -> Path:
    import rasterio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if corrected_image.ndim == 3:
        h, w, c = corrected_image.shape
        bands = np.moveaxis(corrected_image, -1, 0)
    else:
        h, w = corrected_image.shape
        c = 1
        bands = corrected_image[np.newaxis, ...]

    count = c + (1 if confidence_map is not None else 0)
    dtype = corrected_image.dtype

    out_profile = _make_geotiff_profile(profile, h, w, count, dtype)

    meta = dict(metadata or {})
    meta.setdefault("processing_date", datetime.now().isoformat())
    meta.setdefault("source", "CloudReconstruct")
    meta.setdefault("bands", json.dumps(["green", "red", "nir"]))
    meta.setdefault("bit_depth", 10)

    with rasterio.open(output_path, "w", **out_profile) as dst:
        for i in range(c):
            dst.write(bands[i], i + 1)
        if confidence_map is not None:
            dst.write((confidence_map * 65535).astype(np.uint16), c + 1)
        dst.update_tags(**meta)
        if "ars" in meta:
            dst.update_tags(ars=str(meta["ars"]))

    return output_path


def write_analysis_ready_product(output_path: Path, corrected_image: np.ndarray,
                                 confidence_map: np.ndarray, ars_result: dict,
                                 profile: dict = None, metadata: dict = None) -> Path:
    meta = dict(metadata or {})
    meta["ars"] = ars_result.get("ars", 0.0)
    meta["ars_components"] = json.dumps(ars_result.get("components", {}))
    meta["ars_weights"] = json.dumps(ars_result.get("weights", {}))

    return write_geotiff(output_path, corrected_image, confidence_map, profile, meta)


def read_geotiff_analysis(path: Path) -> dict:
    import rasterio
    with rasterio.open(path) as src:
        image = np.moveaxis(src.read(), 0, -1)
        profile = src.profile
        tags = src.tags()

    result = {
        "image": image,
        "profile": profile,
        "metadata": tags,
        "bands": image.shape[-1],
    }

    if "ars" in tags:
        result["ars"] = float(tags["ars"])
    if "ars_components" in tags:
        result["ars_components"] = json.loads(tags["ars_components"])
    if "ars_weights" in tags:
        result["ars_weights"] = json.loads(tags["ars_weights"])

    return result
