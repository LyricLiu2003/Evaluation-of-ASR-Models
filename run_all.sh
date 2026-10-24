#!/usr/bin/env bash
# HG4052 A1 — run every step in order, from the folder that holds these scripts.
# Usage (with the conda environment that has torch + transformers activated):
#   bash run_all.sh
# Each step prints to the terminal and also saves its output to outputs/logs/<step>.log
set -euo pipefail
cd "$(dirname "$0")"

python -c "import torch, transformers, jiwer, num2words, soundfile, openpyxl, pandas, matplotlib" 2>/dev/null || {
  echo "Missing packages in this environment. Run:"
  echo '  pip install "jiwer>=3.0" num2words soundfile openpyxl matplotlib pandas'
  exit 1
}

python step1_build_testset.py   # choose speakers + sentences, copy clips
python step2_normaliser.py      # fix the metric before any system output exists
python step3_run_asr.py         # run Whisper and wav2vec2 on the same clips
python step4_score.py           # WER tables, figures, results CSV
python step5_errors.py          # alignments + error-log template for listening
python step6_prefill_categories.py  # draft error categories + spot-check sample
echo "All steps done. See outputs/"
