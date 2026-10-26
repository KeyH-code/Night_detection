#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .cache/training/{tmp,pip,ultralytics,torch,matplotlib}
export TMPDIR="$PWD/.cache/training/tmp"
export TEMP="$TMPDIR" TMP="$TMPDIR"
export PIP_CACHE_DIR="$PWD/.cache/training/pip"
export YOLO_CONFIG_DIR="$PWD/.cache/training/ultralytics"
export TORCH_HOME="$PWD/.cache/training/torch"
export MPLCONFIGDIR="$PWD/.cache/training/matplotlib"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8:surrogateescape
if [ -f PACKAGE_MANIFEST.json ]; then
  python3.12 scripts/verify_source_bundle.py
fi
python3.12 -m venv .venv-train
.venv-train/bin/python -m pip install --require-hashes -r environment/requirements-linux.lock --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130
.venv-train/bin/python -m pip check
.venv-train/bin/python scripts/train_source.py environment --output runs/check/environment
