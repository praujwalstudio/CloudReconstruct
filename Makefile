# CloudReconstruct Makefile
# ========================

PYTHON := python
VENV_PYTHON := ./.venv/Scripts/python.exe
VENV_STREAMLIT := ./.venv/Scripts/streamlit.exe
VENV_PYTEST := ./.venv/Scripts/pytest.exe

.PHONY: all help demo app evaluate benchmark train test clean

help:
	@echo "CloudReconstruct (TerraLens AI) Commands:"
	@echo "  make demo       - Run automated reference demo on 3 scenes with GeoTIFF & PDF reports"
	@echo "  make app        - Launch Streamlit interactive visual dashboard"
	@echo "  make benchmark  - Run official SOTA accuracy benchmarking (SEN12MS-CR / SEN12MS-CR-TS)"
	@echo "  make evaluate   - Run quantitative benchmark evaluation against DSen2-CR & GLF-CR"
	@echo "  make train      - Execute full training pipeline on multi-modal dataset"
	@echo "  make test       - Run complete pytest test suite"
	@echo "  make clean      - Clean cache and temporary outputs"

demo:
	$(VENV_PYTHON) main.py --demo

app:
	$(VENV_STREAMLIT) run src/app/app.py

benchmark:
	$(VENV_PYTHON) src/evaluation/benchmark_accuracy.py

evaluate:
	$(VENV_PYTHON) src/evaluation/evaluate.py

train:
	$(VENV_PYTHON) scripts/run_full_pipeline.py

test:
	$(VENV_PYTHON) -m pytest

clean:
	rmdir /s /q .pytest_cache 2>nul || true
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul || true
