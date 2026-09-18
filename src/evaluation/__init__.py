"""Evaluation, Benchmarking, Metadata Preservation, and QA Reporting Modules (WP2)"""

from src.evaluation.metrics import (
    compute_all_metrics,
    psnr,
    sam,
    ndvi_correlation,
)
from src.evaluation.confidence import ConfidenceMap
from src.evaluation.analysis_readiness import AnalysisReadiness
from src.evaluation.geotiff_output import (
    write_geotiff,
    write_analysis_ready_product,
    export_analysis_ready_geotiff,
    read_geotiff_analysis,
)
from src.evaluation.inference import CloudFreeInference
from src.evaluation.evaluate import BenchmarkEvaluator, PUBLISHED_BASELINES
from src.evaluation.report_generator import QualityReportGenerator
from src.evaluation.benchmark_accuracy import (
    SOTABenchmarkEngine,
    run_benchmark as run_sota_benchmark,
    compute_rmse,
    compute_mae,
    compute_psnr as compute_sota_psnr,
    compute_ssim as compute_sota_ssim,
    compute_sam as compute_sota_sam,
    compute_all_sota_metrics,
    PUBLISHED_MULTI_TEMPORAL_BASELINES,
    PUBLISHED_MONO_TEMPORAL_BASELINES,
)
from src.evaluation.dip_fallback import DIPInpainter, DIPNet, total_variation_loss

__all__ = [
    "compute_all_metrics",
    "psnr",
    "sam",
    "ndvi_correlation",
    "ConfidenceMap",
    "AnalysisReadiness",
    "write_geotiff",
    "write_analysis_ready_product",
    "export_analysis_ready_geotiff",
    "read_geotiff_analysis",
    "CloudFreeInference",
    "BenchmarkEvaluator",
    "PUBLISHED_BASELINES",
    "QualityReportGenerator",
    "SOTABenchmarkEngine",
    "run_sota_benchmark",
    "compute_rmse",
    "compute_mae",
    "compute_sota_psnr",
    "compute_sota_ssim",
    "compute_sota_sam",
    "compute_all_sota_metrics",
    "PUBLISHED_MULTI_TEMPORAL_BASELINES",
    "PUBLISHED_MONO_TEMPORAL_BASELINES",
    "DIPInpainter",
    "DIPNet",
    "total_variation_loss",
]
