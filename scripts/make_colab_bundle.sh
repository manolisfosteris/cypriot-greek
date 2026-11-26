#!/usr/bin/env bash
# Build cyasr_bundle.zip (code + configs + TSV + audio) to upload to Google Drive.
# Usage: bash scripts/make_colab_bundle.sh
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f cyasr_bundle.zip
zip -rq cyasr_bundle.zip cyasr configs scripts requirements.txt \
    data/ss-corpus-el-CY.tsv data/audios -x '*/__pycache__/*'
ls -lh cyasr_bundle.zip
