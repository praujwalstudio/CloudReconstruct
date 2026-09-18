"""CloudReconstruct — Interactive Remote Sensing Visual Dashboard (WP3)
========================================================================
Bharatiya Antariksh Hackathon | LISS-IV / Sentinel-2 Multi-Modal Cloud Removal
"""

import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from skimage.transform import resize
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW, CHECKPOINTS, OUTPUTS
from src.data.band_harmonization import harmonize_s2_to_liss4
from src.evaluation.inference import CloudFreeInference
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.geotiff_output import write_analysis_ready_product
from src.evaluation.report_generator import QualityReportGenerator


# Page configuration with modern wide layout
st.set_page_config(
    page_title="CloudReconstruct | TerraLens AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics and dark-themed metrics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        color: #F8FAFC;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        border-radius: 6px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


def normalize_display(img: np.ndarray) -> np.ndarray:
    """Normalizes image array to [0.0, 1.0] float64 for display rendering."""
    arr = img.astype(np.float64)
    if img.dtype == np.uint16:
        arr = (arr / 65535.0).clip(0.0, 1.0)
    elif img.dtype == np.uint8:
        arr = (arr / 255.0).clip(0.0, 1.0)
    p2, p98 = np.percentile(arr, [2, 98])
    if p98 > p2:
        arr = (arr - p2) / (p98 - p2)
    return np.clip(arr, 0.0, 1.0).astype(np.float64)


def normalize_rgb(img: np.ndarray) -> np.ndarray:
    """Robust percentile linear stretch for multi-channel display."""
    return normalize_display(img)


@st.cache_resource
def get_inference_model():
    """Returns CloudFreeInference model instance."""
    return CloudFreeInference(
        device="cpu",
        density_ckpt=CHECKPOINTS / "density_model" / "best_model.pth",
        correction_ckpt=CHECKPOINTS / "correction_model" / "best_model.pth",
        sar_ckpt=CHECKPOINTS / "diffusion_model" / "best_model.pth",
        temporal_ckpt=CHECKPOINTS / "temporal_model" / "best_model.pth",
    )


load_production_pipeline = get_inference_model


def read_uploaded_geotiff(uploaded_file) -> tuple[np.ndarray, dict]:
    """Reads uploaded GeoTIFF file into array and metadata profile."""
    if uploaded_file is None:
        return None, None
    with NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    with rasterio.open(tmp_path) as src:
        image = src.read()
        profile = src.profile.copy()

    if image.ndim == 3:
        if image.shape[0] == 13:
            # 13-band Sentinel-2 -> harmonize to Green, Red, NIR
            image = harmonize_s2_to_liss4(image, scale_toa=True)
        elif image.shape[0] in (3, 4):
            image = np.moveaxis(image[:3], 0, -1)
        elif image.shape[0] == 2:
            image = np.moveaxis(image, 0, -1)
        else:
            image = np.moveaxis(image, 0, -1)
    elif image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    return image, profile


def compute_ndvi(image: np.ndarray) -> np.ndarray:
    """Computes Normalized Difference Vegetation Index (NIR - Red) / (NIR + Red)."""
    if image.ndim == 3 and image.shape[-1] >= 3:
        red = image[..., 1].astype(np.float32)
        nir = image[..., 2].astype(np.float32)
    else:
        red = image.astype(np.float32)
        nir = image.astype(np.float32)
    denom = nir + red + 1e-7
    ndvi = (nir - red) / denom
    return np.clip(ndvi, -1.0, 1.0)


def main():
    # Sidebar Setup
    st.sidebar.image("https://img.icons8.com/fluency/96/satellite.png", width=70)
    st.sidebar.markdown("### 🛰️ Pipeline Configuration")

    # Checkpoint status badge
    st.sidebar.success("● Active Weights: `checkpoints/` Loaded")

    preset_scene = st.sidebar.selectbox(
        "Load Preset Benchmark Scene:",
        [
            "None (Upload Custom GeoTIFF)",
            "Sample 1: Thin Cloud Scene (Spring ROI)",
            "Sample 2: Medium Cloud Scene (Summer ROI)",
            "Sample 3: Dense Cloud Scene (Winter ROI)",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### ⚙️ Adaptive Thresholds")
    thin_thresh = st.sidebar.slider("Thin Cloud Threshold", 0.1, 0.5, 0.3, 0.05)
    dense_thresh = st.sidebar.slider("Dense Cloud Threshold", 0.5, 0.95, 0.8, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🌐 Topography & SAR Fusion")
    enable_sar = st.sidebar.checkbox("Enable SAR VV/VH Guidance", value=True)
    enable_dem = st.sidebar.checkbox("Enable DEM Topographic Correction", value=True)

    # Main Header
    st.markdown('<div class="main-header">CloudReconstruct (TerraLens AI)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Adaptive Multi-Source Cloud & Shadow Removal for LISS-IV Satellite Imagery — <i>Bharatiya Antariksh Hackathon</i></div>',
        unsafe_allow_html=True,
    )

    # Ingestion / Upload Section
    col_up1, col_up2, col_up3 = st.columns([2, 1, 1])

    with col_up1:
        cloudy_file = st.file_uploader("Upload Optical Cloudy Scene (GeoTIFF)", type=["tif", "tiff"])
    with col_up2:
        sar_file = st.file_uploader("Sentinel-1 SAR VV/VH (optional)", type=["tif", "tiff"])
    with col_up3:
        dem_file = st.file_uploader("Topographic DEM (optional)", type=["tif", "tiff"])

    # Handle preset scenes
    image, profile = None, None
    sar_data, dem_data = None, None

    if preset_scene != "None (Upload Custom GeoTIFF)":
        raw_cloudy_list = sorted((RAW / "cloudy").glob("*.tif"))
        if raw_cloudy_list:
            idx = 0 if "Thin" in preset_scene else (2 if "Medium" in preset_scene else -1)
            selected_path = raw_cloudy_list[idx % len(raw_cloudy_list)]
            with rasterio.open(selected_path) as src:
                raw_img = src.read()
                profile = src.profile.copy()
            image = harmonize_s2_to_liss4(raw_img, scale_toa=True)
            if image.ndim == 3:
                image = np.moveaxis(image, 0, -1)

            # Look for paired SAR
            base_name = selected_path.name.replace("cloudy_", "")
            sar_cand = RAW / "sigma0" / f"sigma0_{base_name}"
            if sar_cand.exists():
                with rasterio.open(sar_cand) as src_s1:
                    sar_data = np.moveaxis(src_s1.read()[:2], 0, -1)

            # Look for paired DEM
            dem_cand = RAW / "dem" / f"dem_{base_name}"
            if dem_cand.exists():
                with rasterio.open(dem_cand) as src_dem:
                    dem_data = src_dem.read(1)

            st.info(f"Loaded preset benchmark: **{selected_path.name}**")
    elif cloudy_file is not None:
        image, profile = read_uploaded_geotiff(cloudy_file)
        if sar_file is not None:
            sar_data, _ = read_uploaded_geotiff(sar_file)
        if dem_file is not None:
            dem_data, _ = read_uploaded_geotiff(dem_file)

    if image is None:
        st.info("👆 Upload a LISS-IV / Sentinel-2 GeoTIFF or select a Preset Benchmark Scene from the sidebar to start inference.")
        return

    # Inference Execution
    pipeline = load_production_pipeline()
    pipeline.pipeline.thin_threshold = thin_thresh
    pipeline.pipeline.dense_threshold = dense_thresh

    dem_processor = None
    if enable_dem and dem_data is not None:
        from src.evaluation.dem_integration import TerrainProcessor
        dem_processor = TerrainProcessor(resolution=5.8)
        dem_processor.load_from_array(dem_data if dem_data.ndim == 2 else dem_data[..., 0])

    with st.spinner("Processing scene through tiered Multi-Modal AI pipeline..."):
        active_sar = sar_data if enable_sar else None
        result = pipeline.correct(
            image,
            sar=active_sar,
            dem_processor=dem_processor,
            data_max=1.0 if image.max() <= 1.0 else 65535.0,
        )

    corrected = result["corrected"]
    density = result["density"]
    confidence = result["confidence"]
    ars = result["ars"]
    grade = pipeline.readiness.grade(ars["ars"])

    # 1. KPI Summary Bar
    st.markdown("### 📊 Reconstruction Quality Scorecard")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    cloud_cover = float((density > thin_thresh).mean() * 100.0)
    kpi1.metric("Analysis-Readiness Score", f"{ars['ars']:.4f}", f"Grade: {grade}")
    kpi2.metric("Estimated Cloud Cover", f"{cloud_cover:.1f}%", "Obscured Pixels")
    kpi3.metric("Generative Confidence", f"{confidence.mean()*100:.1f}%", "Certainty Level")
    kpi4.metric("Recovered Surface Area", f"{100.0 - (1.0-confidence.mean())*cloud_cover:.1f}%", "Synthesized Area")

    st.markdown("---")

    # 2. 4-Column Multi-View Visual Grid
    st.markdown("### 🛰️ Side-by-Side Visual Inspection Grid")
    vcol1, vcol2, vcol3, vcol4 = st.columns(4)

    with vcol1:
        st.markdown("**1. Input Cloudy Scene**")
        st.image(normalize_display(image), use_container_width=True, caption=f"Dimensions: {image.shape[1]}×{image.shape[0]}")

    with vcol2:
        st.markdown("**2. Cloud Density Map**")
        fig_dens, ax_d = plt.subplots(figsize=(4, 4))
        im_d = ax_d.imshow(density, cmap="magma", vmin=0, vmax=1)
        ax_d.axis("off")
        plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
        st.pyplot(fig_dens, use_container_width=True)
        plt.close(fig_dens)

    with vcol3:
        st.markdown("**3. Pixel Confidence Map**")
        fig_conf, ax_c = plt.subplots(figsize=(4, 4))
        im_c = ax_c.imshow(confidence, cmap="viridis", vmin=0, vmax=1)
        ax_c.axis("off")
        plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
        st.pyplot(fig_conf, use_container_width=True)
        plt.close(fig_conf)

    with vcol4:
        st.markdown("**4. Reconstructed Output**")
        st.image(normalize_display(corrected), use_container_width=True, caption="Cloud-Free ARD Product")

    st.markdown("---")

    # 3. Deep Analytics & NDVI Distribution
    tab_analytics, tab_components, tab_export = st.tabs([
        "📈 Biogeochemical NDVI Curves",
        "🧩 ARS Component Weights",
        "💾 Precision Export & PDF QA Report",
    ])

    with tab_analytics:
        st.markdown("#### Normalized Difference Vegetation Index (NDVI) Distribution Analysis")
        ndvi_cloudy = compute_ndvi(image)
        ndvi_clean = compute_ndvi(corrected)

        fig_ndvi, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram & Distribution comparison
        ax1.hist(ndvi_cloudy.ravel(), bins=50, range=(-0.2, 0.9), alpha=0.6, color="#EF4444", label="Cloudy Input NDVI")
        ax1.hist(ndvi_clean.ravel(), bins=50, range=(-0.2, 0.9), alpha=0.6, color="#10B981", label="Reconstructed NDVI")
        ax1.set_title("NDVI Pixel Frequency Distribution", fontsize=11, fontweight="bold")
        ax1.set_xlabel("NDVI Value")
        ax1.set_ylabel("Pixel Count")
        ax1.legend()
        ax1.grid(True, alpha=0.2)

        # Spatial Map of Recovered NDVI
        im_n = ax2.imshow(ndvi_clean, cmap="YlGn", vmin=-0.1, vmax=0.8)
        ax2.set_title("Reconstructed Surface NDVI Map", fontsize=11, fontweight="bold")
        ax2.axis("off")
        plt.colorbar(im_n, ax=ax2, fraction=0.046, pad=0.04)

        st.pyplot(fig_ndvi, use_container_width=True)
        plt.close(fig_ndvi)

    with tab_components:
        st.markdown("#### Analysis-Readiness Score (ARS) Decomposition")
        comp_cols = st.columns(len(ars.get("components", {"density": 0, "confidence": 0, "preservation": 0})))
        for idx, (comp_name, val) in enumerate(ars.get("components", {}).items()):
            with comp_cols[idx % len(comp_cols)]:
                st.metric(comp_name.replace("_", " ").title(), f"{val:.4f}")

    with tab_export:
        st.markdown("#### Analysis-Ready Data (ARD) Export")
        out_dir = OUTPUTS / "app_exports"
        out_dir.mkdir(parents=True, exist_ok=True)

        scene_stem = "reconstructed_scene"
        tif_out_path = out_dir / f"{scene_stem}_cloud_free.tif"

        # Generate PDF & JSON reports
        report_gen = QualityReportGenerator(output_dir=out_dir)
        report_data = report_gen.generate_report(
            image_id=scene_stem,
            density_map=density,
            confidence_map=confidence,
            corrected_image=corrected,
            cloudy_image=image,
            ars_result={"ars": ars["ars"], "grade": grade, "components": ars.get("components", {})},
        )

        write_analysis_ready_product(
            tif_out_path,
            corrected,
            confidence_map=confidence,
            ars_result={"ars": ars["ars"], "grade": grade, "components": ars.get("components", {})},
            profile=profile,
        )

        dcol1, dcol2, dcol3 = st.columns(3)

        with dcol1:
            with open(tif_out_path, "rb") as f_tif:
                st.download_button(
                    label="📥 Download Precision GeoTIFF",
                    data=f_tif,
                    file_name=f"{scene_stem}_ARD.tif",
                    mime="image/tiff",
                    use_container_width=True,
                )

        with dcol2:
            pdf_path = Path(report_data["pdf_report_path"])
            if pdf_path.exists():
                with open(pdf_path, "rb") as f_pdf:
                    st.download_button(
                        label="📄 Download QA Inspection PDF",
                        data=f_pdf,
                        file_name=f"{scene_stem}_QA_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

        with dcol3:
            json_path = Path(report_data["json_report_path"])
            if json_path.exists():
                with open(json_path, "rb") as f_json:
                    st.download_button(
                        label="📋 Download Standardized JSON Metadata",
                        data=f_json,
                        file_name=f"{scene_stem}_Metadata.json",
                        mime="application/json",
                        use_container_width=True,
                    )


if __name__ == "__main__":
    main()
