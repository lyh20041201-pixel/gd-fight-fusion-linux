#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required"'
"$PYTHON_BIN" -m venv .venv-linux
.venv-linux/bin/python -m pip install --upgrade pip
.venv-linux/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv-linux/bin/python -m pip install -r scripts/linux_fight/requirements.txt
.venv-linux/bin/python -m pip check
.venv-linux/bin/python scripts/linux_fight/verify.py --full
printf '%s\n' 'Installation and transfer verification complete. Run preflight.py --gpu-smoke before starting the queue.'
