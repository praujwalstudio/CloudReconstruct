import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import numpy as np
import rasterio
from tempfile import NamedTemporaryFile
from skimage.transform import resize

from src.evaluation.inference import CloudFreeInference
from src.evaluation.metrics import compute_all_metrics
from src.config import OUTPUTS

st.set_page_config(page_title="CloudReconstruct", layout="wide")


@st.cache_resource
def get_inference_model():
    return CloudFreeInference(device="cpu")


def load_tif(uploaded_file) -> tuple[np.ndarray, dict] | None:
    if uploaded_file is None:
        return None, None
    with NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    with rasterio.open(tmp_path) as src:
        image = src.read()
        profile = src.profile
    if image.ndim == 3:
        image = np.moveaxis(image, 0, -1)
    return image, profile


def normalize_display(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        img = (img / 65535.0).clip(0, 1)
    elif img.dtype == np.uint8:
        img = (img / 255.0).clip(0, 1)
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        img = (img - p2) / (p98 - p2)
    return img.clip(0, 1)


def main():
    st.title("CloudReconstruct")
    st.markdown("Adaptive Multi-Source Cloud Removal for LISS-IV Satellite Imagery — *Bharatiya Antariksh Hackathon*")

    model = get_inference_model()

    col1, col2 = st.columns(2)
    with col1:
        liss4_file = st.file_uploader("LISS-IV Scene (GeoTIFF)", type=["tif", "tiff"])
    with col2:
        sar_file = st.file_uploader("Sentinel-1 SAR (optional)", type=["tif", "tiff"])
        dem_file = st.file_uploader("SRTM DEM (optional)", type=["tif", "tiff"])

    if liss4_file is None:
        st.info("Upload a LISS-IV GeoTIFF to begin")
        return

    with st.spinner("Loading data..."):
        image, profile = load_tif(liss4_file)
        if image is None:
            st.error("Failed to load LISS-IV scene")
            return

        sar = None
        if sar_file is not None:
            sar, _ = load_tif(sar_file)

        dem_processor = None
        if dem_file is not None:
            from src.evaluation.dem_integration import TerrainProcessor
            dem_data, _ = load_tif(dem_file)
            if dem_data is not None:
                with NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    from rasterio.transform import from_origin
                    with rasterio.open(tmp.name, "w", driver="GTiff",
                                       height=dem_data.shape[0], width=dem_data.shape[1],
                                       count=1, dtype=np.float32,
                                       transform=from_origin(0, 0, 1, 1)) as dst:
                        dst.write(dem_data.astype(np.float32), 1)
                    dem_processor = TerrainProcessor(resolution=5.8)
                    dem_processor.load(Path(tmp.name))

    with st.spinner("Running cloud removal..."):
        result = model.correct(image, sar=sar, dem_processor=dem_processor)

    corrected = result["corrected"]
    density = result["density"]
    confidence = result["confidence"]
    ars = result["ars"]
    grade = model.readiness.grade(ars["ars"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Input", "Cloud Density", "Corrected", "Confidence", "Metrics"])

    with tab1:
        st.image(normalize_display(image), use_container_width=True,
                 caption=f"Input LISS-IV Scene ({image.shape[1]}×{image.shape[0]})")

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(normalize_display(image), use_container_width=True, caption="Input")
        with col_b:
            st.image(density, use_container_width=True, caption="Cloud Density (red = dense)")
        st.caption(f"Mean density: {density.mean():.3f}")

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(normalize_display(image), use_container_width=True, caption="Input (Cloudy)")
        with col_b:
            st.image(normalize_display(corrected), use_container_width=True, caption="Cloud-Free Output")
        st.markdown(f"**ARS Score: {ars['ars']:.4f}** &nbsp;—&nbsp; Grade: **{grade}**")

    with tab4:
        st.image(confidence, use_container_width=True, caption="Confidence Map (green = high confidence)",
                 clamp=True)
        st.caption(f"Mean confidence: {confidence.mean():.3f}")

    with tab5:
        ref_file = st.file_uploader("Reference clear image (for metrics)", type=["tif", "tiff"],
                                    key="ref_uploader")
        metrics = {"psnr": None, "sam": None, "ndvi_correlation": None}
        if ref_file is not None:
            ref_image, _ = load_tif(ref_file)
            if ref_image is not None:
                if ref_image.shape != corrected.shape:
                    h, w = corrected.shape[:2]
                    ref_image = resize(ref_image, (h, w), preserve_range=True, anti_aliasing=True)
                    if corrected.ndim == 3 and ref_image.ndim == 2:
                        ref_image = np.stack([ref_image] * corrected.shape[2], axis=-1)
                    elif ref_image.ndim == 3 and corrected.ndim == 3 and ref_image.shape[2] != corrected.shape[2]:
                        ref_image = ref_image[:, :, :corrected.shape[2]]
                    ref_image = ref_image.astype(corrected.dtype)
                    st.warning(f"Reference resized to {h}x{w} to match input")
                cloud_mask = (density > 0.3).astype(np.uint8)
                metrics = compute_all_metrics(corrected, ref_image, mask=cloud_mask)
            else:
                st.warning("Reference image could not be loaded")

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("ARS", f"{ars['ars']:.4f} ({grade})")
        mcol2.metric("PSNR", f"{metrics['psnr']:.2f} dB" if metrics.get('psnr') else "N/A")
        mcol3.metric("SAM", f"{metrics['sam']:.2f}°" if metrics.get('sam') else "N/A")
        mcol4.metric("NDVI Corr", f"{metrics['ndvi_correlation']:.4f}" if metrics.get('ndvi_correlation') else "N/A")

        with st.expander("ARS Components"):
            for k, v in ars["components"].items():
                st.metric(k.replace("_", " ").title(), f"{v:.4f}")
        with st.expander("Weighted Scores"):
            for k, v in ars.get("weights", {}).items():
                weighted = ars["components"].get(f"weighted_{k}", 0)
                st.metric(f"{k.title()} (weight: {v})", f"{weighted:.4f}")

    st.divider()
    output_dir = OUTPUTS / "app_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cloud_free_result.tif"

    if st.button("Save Analysis-Ready GeoTIFF"):
        with st.spinner("Saving..."):
            meta = {"ars": str(ars["ars"]), "grade": grade}
            model.correct_and_save(out_path, image, sar, None, dem_processor,
                                   profile, meta)
            st.success(f"Saved to {out_path}")
            with open(out_path, "rb") as f:
                st.download_button("Download GeoTIFF", f, file_name="cloud_free_result.tif")


if __name__ == "__main__":
    main()
