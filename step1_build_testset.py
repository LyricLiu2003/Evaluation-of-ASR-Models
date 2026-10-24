"""Step 1 — choose the speakers and the sentences, and write the test set.

Outputs: outputs/selected_speakers.csv, outputs/test_set.csv, outputs/prompts.csv, clips/<file_id>.wav
"""
import glob
import os
import random
import re
import shutil
import unicodedata

import openpyxl
import pandas as pd
import soundfile as sf

from config import (SEED, SPEAKER_SHEET, CHANNEL, SCRIPT_DIR, WAVE_DIR, GROUPS, SEXES, MIC_C0,
                    N_SPEAKERS_PER_CELL, N_BACKUP, N_PROMPTS, CLIP_DIR, out, start_log)

start_log("step1_build_testset")

# ---------------------------------------------------------------------------
# Read the NSC Part 1 speaker sheet. It records only sex, accent group (ACC) and the microphone model of each
# channel (there is no age column), so we print how many speakers fall in each accent x sex x microphone cell
# before sampling.
# ---------------------------------------------------------------------------
def load_speaker_sheet(path):
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() for h in rows[0]]
    records = []
    for r in rows[1:]:
        d = {h: (str(v).strip() if v is not None else "") for h, v in zip(header, r)}
        if d.get("SEX") in SEXES and d.get("SCD/PART1", "").isdigit():
            d["SCD"] = d["SCD/PART1"].zfill(4)        # 4-digit speaker number used in folder names
            records.append(d)
    return records


speakers_all = load_speaker_sheet(SPEAKER_SHEET)
sheet = pd.DataFrame(speakers_all)
print(len(sheet), "speakers in the sheet\n")
print(pd.crosstab([sheet["ACC"], sheet["SEX"]], sheet["DEVICE/C0"] == MIC_C0)
      .rename(columns={True: MIC_C0, False: "other mic"}), "\n")

# ---------------------------------------------------------------------------
# Within each accent x sex cell, only speakers recorded on the same channel-0 microphone are eligible, so the
# microphone is held constant. They are shuffled with the fixed seed: the first 3 are the primary sample and
# the next 2 are backups, used in order only when a primary speaker has no audio.
# ---------------------------------------------------------------------------
rng = random.Random(SEED)
selection = []
for acc in GROUPS:
    for sex in SEXES:
        candidates = sorted((d for d in speakers_all
                             if d["ACC"] == acc and d["SEX"] == sex and d["DEVICE/C0"] == MIC_C0),
                            key=lambda d: d["SCD"])
        order = rng.sample(candidates, len(candidates))
        for i, d in enumerate(order[: N_SPEAKERS_PER_CELL + N_BACKUP]):
            selection.append({"speaker": d["SCD"], "acc": acc, "sex": sex, "mic_c0": d["DEVICE/C0"],
                              "role": "primary" if i < N_SPEAKERS_PER_CELL else f"backup{i - N_SPEAKERS_PER_CELL + 1}"})
selection = pd.DataFrame(selection)
selection["has_audio"] = selection.speaker.map(lambda s: os.path.isdir(os.path.join(WAVE_DIR, f"SPEAKER{s}")))

used = []
for (acc, sex), g in selection.groupby(["acc", "sex"], sort=False):
    used += list(g.loc[g.has_audio, "speaker"][:N_SPEAKERS_PER_CELL])
selection["used"] = selection.speaker.isin(used)
selection.to_csv(out("selected_speakers.csv"), index=False)

speakers = selection[selection.used].reset_index(drop=True)
print(len(speakers), "speakers used")
print(speakers.pivot_table(index="acc", columns="sex", values="speaker", aggfunc=" ".join), "\n")

# ---------------------------------------------------------------------------
# Each speaker's SCRIPT file has two lines per item: the file id with the prompt the speaker was shown, then a
# transcription of what was actually said. The second line is the reference we score against. Each item is
# then matched to its WAV file by file id.
# ---------------------------------------------------------------------------
CH_DIGIT = CHANNEL[-1]


def read_script(spk):
    items = []
    for path in sorted(glob.glob(os.path.join(SCRIPT_DIR, f"{CH_DIGIT}{spk}?.TXT"))):
        lines = open(path, encoding="utf-8-sig").read().splitlines()
        for first, second in zip(lines, lines[1:]):
            m = re.match(r"^(\d{9})\t(.*)$", first)
            if m and second.startswith("\t"):
                items.append({"file_id": m.group(1), "speaker": spk,
                              "prompt": m.group(2).strip(), "reference": second.strip()})
    return items


rows = []
for spk in speakers.speaker:
    wavs = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(WAVE_DIR, f"SPEAKER{spk}", "*", "*.WAV"))}
    for it in read_script(spk):
        if it["file_id"] in wavs:
            it["wav_path"] = wavs[it["file_id"]]
            rows.append(it)
utts = pd.DataFrame(rows).merge(speakers[["speaker", "acc", "sex"]], on="speaker")
print(utts.groupby(["acc", "sex"]).agg(speakers=("speaker", "nunique"), utterances=("file_id", "size")), "\n")

# ---------------------------------------------------------------------------
# Keep only prompts read by all 18 speakers and with no '**' (unintelligible) in anyone's reference, then
# sample 12 with the fixed seed. Every speaker reads the same 12 sentences, so differences between groups
# cannot come from the text.
# ---------------------------------------------------------------------------
def prompt_key(p):
    p = unicodedata.normalize("NFKC", p).lower().replace("’", "'")
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", p).split())


utts["key"] = utts.prompt.map(prompt_key)
utts = utts.sort_values("file_id").drop_duplicates(["speaker", "key"], keep="first")   # repeated prompt: first take

per_key = utts.groupby("key").agg(n_speakers=("speaker", "nunique"),
                                  unintelligible=("reference", lambda s: s.str.contains(r"\*\*", regex=True).any()))
n_used = speakers.speaker.nunique()
shared = per_key.n_speakers == n_used
eligible = sorted(per_key[shared & ~per_key.unintelligible].index)
print(f"prompts read by all {n_used} speakers: {shared.sum()} | with no '**' in any reference: {len(eligible)}")

chosen = sorted(random.Random(SEED).sample(eligible, N_PROMPTS))
prompt_ids = {k: f"P{i + 1:02d}" for i, k in enumerate(chosen)}

test = utts[utts.key.isin(chosen)].copy()
test["prompt_id"] = test.key.map(prompt_ids)
test["condition"] = test.acc
test["duration_s"] = test.wav_path.map(lambda p: round(sf.info(p).duration, 3))
test["clip_path"] = [os.path.join(CLIP_DIR, f"{f}.wav") for f in test.file_id]
for src, dst in zip(test.wav_path, test.clip_path):
    if not os.path.exists(dst):
        shutil.copy(src, dst)                 # a copy per utterance, named by file id, for listening / Praat
test = test.sort_values(["acc", "sex", "speaker", "prompt_id"]).reset_index(drop=True)

test[["file_id", "speaker", "acc", "sex", "condition", "prompt_id", "duration_s",
      "prompt", "reference", "wav_path", "clip_path"]].to_csv(out("test_set.csv"), index=False)
pd.DataFrame({"prompt_id": list(prompt_ids.values()), "prompt_key": list(prompt_ids)}).to_csv(out("prompts.csv"), index=False)

test["raw_words"] = test.reference.str.split().str.len()
print("\ntest set:")
print(test.groupby(["acc", "sex"]).agg(speakers=("speaker", "nunique"), utterances=("file_id", "size"),
                                       total_min=("duration_s", lambda x: round(x.sum() / 60, 2)),
                                       mean_dur_s=("duration_s", "mean"), raw_words=("raw_words", "sum")).round(2))
print(f"\n{len(test)} utterances written to outputs/test_set.csv; clips in {CLIP_DIR}/")
