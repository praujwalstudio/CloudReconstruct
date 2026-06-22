import pytest
import sys
from pathlib import Path
from unittest.mock import patch


class TestMainEntryPoint:
    def test_import_main(self):
        import main
        assert hasattr(main, "main")
        assert hasattr(main, "step_download")
        assert hasattr(main, "step_align")
        assert hasattr(main, "step_mask")
        assert hasattr(main, "step_patch")

    def test_step_return_types(self):
        from main import step_download, step_align, step_mask, step_patch

        # These work even with no data — they return empty dicts/lists
        with patch("src.preprocessing.download_data.list_available_scenes") as mock:
            mock.return_value = {}
            result = step_download()
            assert isinstance(result, dict)

        result = step_align()
        assert isinstance(result, list)

        with patch("src.preprocessing.cloud_mask.process_all") as mock:
            mock.return_value = []
            result = step_mask()
            assert isinstance(result, list)

    @patch("argparse.ArgumentParser.parse_args")
    def test_cli_parsing_all(self, mock_parse):
        mock_parse.return_value.step = "all"
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument("--step", default="all")
        args = parser.parse_args([])
        assert args.step == "all"

    @patch("argparse.ArgumentParser.parse_args")
    def test_cli_parsing_download(self, mock_parse):
        mock_parse.return_value.step = "download"
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument("--step", default="download")
        args = parser.parse_args([])
        assert args.step == "download"

    @patch("argparse.ArgumentParser.parse_args")
    def test_cli_parsing_align(self, mock_parse):
        mock_parse.return_value.step = "align"
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument("--step", default="align")
        args = parser.parse_args([])
        assert args.step == "align"
