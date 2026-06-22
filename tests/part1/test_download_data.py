import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.preprocessing.download_data import (
    download_file,
    download_liss4,
    download_sentinel1,
    download_sentinel2,
    download_srtm,
    list_available_scenes,
)
from src.config import LISS4_RAW, S1_RAW, S2_RAW, DEM_RAW


class TestDownloadFile:
    def test_skip_existing_file(self, tmp_path):
        dest = tmp_path / "existing.tif"
        dest.write_text("dummy")
        result = download_file("http://example.com/file.tif", dest)
        assert result == dest

    @patch("src.preprocessing.download_data.requests.get")
    def test_download_new_file(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"test data"]
        mock_resp.headers = {"content-length": "9"}
        mock_get.return_value = mock_resp

        dest = tmp_path / "new_file.tif"
        result = download_file("http://example.com/file.tif", dest)

        assert result == dest
        assert dest.read_bytes() == b"test data"


class TestDownloadLISS4:
    def test_returns_path(self):
        result = download_liss4("test_scene")
        assert isinstance(result, Path)
        assert "liss4_test_scene.tif" in str(result)

    def test_instructions_printed(self, capsys):
        download_liss4("scene_123")
        captured = capsys.readouterr()
        assert "Bhoonidhi" in captured.out
        assert "scene_123" in captured.out


class TestDownloadSentinel1:
    def test_returns_path(self):
        result = download_sentinel1("test_scene")
        assert isinstance(result, Path)
        assert "s1_test_scene" in str(result)


class TestDownloadSentinel2:
    def test_returns_path(self):
        result = download_sentinel2("test_scene")
        assert isinstance(result, Path)
        assert "s2_test_scene" in str(result)


class TestDownloadSRTM:
    @patch("src.preprocessing.download_data.download_file")
    def test_srtm_url_format(self, mock_download):
        mock_download.return_value = DEM_RAW / "srtm_22.0_89.0_28.0_97.0.tif"
        bbox = (22.0, 89.0, 28.0, 97.0)
        result = download_srtm(bbox)
        assert isinstance(result, Path)
        assert "srtm" in str(result)


class TestListAvailable:
    def test_returns_dict(self):
        result = list_available_scenes()
        assert isinstance(result, dict)
        for key in ["LISS-IV", "Sentinel-1", "Sentinel-2", "DEM"]:
            assert key in result
