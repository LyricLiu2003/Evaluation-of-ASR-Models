# Evaluation-of-ASR-Models
HG4052 Assignment 1 of NTUsg

A worked example for HG4052 Assignment 1: does automatic speech recognition work equally well for Singapore
English speakers of different accent groups? Two open ASR systems transcribe the same read sentences from
18 speakers of the IMDA National Speech Corpus (Part 1), and their word error rates are compared across
Chinese, Indian and Malay speakers.

**Results and discussion: [REPORT.md](REPORT.md)**

| | System A | System B |
|---|---|---|
| Model | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) | [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) |
| Overall WER | 8.7 % | 29.8 % |

## Data (not included)

This repository contains **no corpus data**. The NSC is released under the Singapore Open Data Licence and
its audio and transcripts must not be redistributed. Download NSC Part 1 from IMDA yourself and place it so
that the folder looks like this:

```
HG4052_A1/
├── DOC/Speaker Information (Part 1).XLSX
├── NSC/PART1/DATA/CHANNEL0/SCRIPT/*.TXT
├── NSC/PART1/DATA/CHANNEL0/WAVE/SPEAKER<id>/...
└── (the scripts in this repository)
```

`.gitignore` keeps the corpus, the copied clips and every output file that contains transcripts out of git.
Only aggregate tables and figures from `outputs/` are committed.

## Setup

Python 3.11 with PyTorch and Hugging Face Transformers, for example in a conda environment:

```bash
conda create -n asr python=3.11
conda activate asr
pip install torch transformers "jiwer>=3.0" num2words soundfile openpyxl matplotlib pandas
```

On Apple silicon the models run on `mps`; otherwise CUDA or CPU is chosen automatically.

## Running

```bash
bash run_all.sh
```

or one step at a time:

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| — | `config.py` | Every design decision: groups, microphone, sample sizes, model IDs, seed | — |
| 1 | `step1_build_testset.py` | Picks 3 speakers per accent × sex cell (same microphone) and 12 sentences read by all of them | `test_set.csv`, `clips/` |
| 2 | `step2_normaliser.py` | Fixes the text normaliser before any system output exists | `normaliser_config.json` |
| 3 | `step3_run_asr.py` | Runs both systems on the same 216 clips (`python step3_run_asr.py A` runs one) | `hypotheses.csv`, `run_info.json` |
| 4 | `step4_score.py` | WER overall, by group, speaker and sentence, with speaker-bootstrap 95 % CIs; figures | `audit_table.csv`, `fig1`, `fig2`, results CSV |
| 5 | `step5_errors.py` | Splits every error into one row with context, for listening | `error_log_template.csv`, `alignments_<GROUP>.txt` |
| 6 | `step6_prefill_categories.py` | Draft error categories from text rules, a spot-check sample, and agreement with the listener | `error_log_prefilled.csv`, `spot_check_agreement.csv` |

All outputs go to `outputs/`; each step also writes its terminal output to `outputs/logs/<step>.log`.

### Listening check (step 6)

The draft categories come from the text alone. To check them, copy `outputs/spot_check_sample.csv` to
`outputs/spot_check_listened.csv`, listen to each clip (`afplay clips/<file_id>.wav`, or open it in Praat)
and fill in:

- `checked_by_listening`: `ok` (draft is right), `changed` (you corrected `category`) or `unsure` (cannot tell)
- `narrow_transcription`: IPA for the mis-recognised stretch only, e.g. `[mɛn]`
- `notes`: `pron: standard`, `pron: SgE <feature>` or `pron: unclear`

Re-running step 6 merges these rows back into the full log and prints the agreement table.

## Repository layout

```
config.py, step1–6_*.py, run_all.sh   pipeline
outputs/                               aggregate tables and figures only (see .gitignore)
REPORT.md                              write-up of the results
```

## Acknowledgement

Contains information from the National Speech Corpus accessed from IMDA, made available under the terms of
the Singapore Open Data Licence version 1.0.
