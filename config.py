"""HG4052 A1 — shared settings used by every step.

Every design decision of the audit lives in this file, so the other scripts contain no magic numbers.
Data: IMDA National Speech Corpus (NSC), Part 1, Singapore Open Data Licence v1.0.
Corpus audio and transcripts stay on this machine: never upload, share or post them.
"""
import os
import sys

import pandas as pd

# All paths below are relative to the folder that holds these scripts.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# This file holds every design decision of the audit in one place: the corpus paths, the comparison (accent
# group x sex), the microphone held constant, the number of speakers per cell and sentences per speaker, and
# the exact model IDs of the two systems. The other scripts import from here, so a setting changed here
# changes everywhere.
# ---------------------------------------------------------------------------
RESULTS_CSV = "HG4052_A1_results.csv"   # results file to submit, in outputs/
SEED = 42

SPEAKER_SHEET = os.path.join("DOC", "Speaker Information (Part 1).XLSX")
CHANNEL = "CHANNEL0"                                   # headset / standing microphone
DATA_DIR = os.path.join("NSC", "PART1", "DATA", CHANNEL)
SCRIPT_DIR = os.path.join(DATA_DIR, "SCRIPT")
WAVE_DIR = os.path.join(DATA_DIR, "WAVE")

GROUPS = ["CHINESE", "INDIAN", "MALAY"]   # values of the ACC column in the speaker sheet
SEXES = ["F", "M"]
MIC_C0 = "AKG C214"            # channel-0 microphone model held constant
N_SPEAKERS_PER_CELL = 3        # per ACC x SEX cell -> 18 speakers
N_BACKUP = 2                   # reserve speakers per cell, used only if a primary has no audio
N_PROMPTS = 12                 # sentences read by every selected speaker -> 18 x 12 = 216 utterances
TARGET_SR = 16_000

SYSTEMS = {
    "A": {"name": "whisper_small",      "model_id": "openai/whisper-small"},
    "B": {"name": "wav2vec2_base_960h", "model_id": "facebook/wav2vec2-base-960h"},
}
HYP = {k: f"hyp_{k}_{v['name']}" for k, v in SYSTEMS.items()}   # column names for raw hypotheses

OUT_DIR = "outputs"
CLIP_DIR = "clips"
LOG_DIR = os.path.join(OUT_DIR, "logs")
for d in (OUT_DIR, CLIP_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Small helpers shared by the steps
# ---------------------------------------------------------------------------
def out(name):
    """Path of a file inside outputs/."""
    return os.path.join(OUT_DIR, name)


def read_csv(name):
    """Read a CSV from outputs/ keeping ids as text (leading zeros) and empty hypotheses as ''."""
    return pd.read_csv(out(name), dtype={"file_id": str, "speaker": str}, keep_default_na=False)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()

    def isatty(self):
        # Libraries (e.g. transformers) check this to decide on colour output
        return False

    def __getattr__(self, name):
        # Fall back to the real terminal for anything else (fileno, encoding, ...)
        return getattr(self.streams[0], name)


def start_log(step):
    """Print to the terminal and save the same text to outputs/logs/<step>.log."""
    sys.stdout = _Tee(sys.__stdout__, open(os.path.join(LOG_DIR, f"{step}.log"), "w", encoding="utf-8"))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
