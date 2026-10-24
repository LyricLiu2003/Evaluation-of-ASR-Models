"""Step 3 — run the two ASR systems on exactly the same audio.

Usage:  python step3_run_asr.py          (both systems)
        python step3_run_asr.py A        (only system A; keeps B's existing output)
Outputs: outputs/hypotheses.csv, outputs/run_info.json
"""
import json
import os
import platform
import sys
import time
from importlib.metadata import version

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # fall back to CPU for any op MPS lacks

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from config import SEED, SYSTEMS, HYP, TARGET_SR, out, read_csv, start_log

start_log("step3_run_asr")

# ---------------------------------------------------------------------------
# Fix the random seeds, choose the compute device (mps on Apple silicon) and print the package versions so the
# run can be reproduced. Both systems read exactly the same audio: the WAV copies in clips/.
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
# If a model fails on "mps", change the line above to DEVICE = "cpu" and run again.

print("python", platform.python_version(), "| device:", DEVICE)
for pkg in ["torch", "transformers", "numpy", "soundfile"]:
    print(f"  {pkg:13s}", version(pkg))

test = read_csv("test_set.csv")
WAVS = [sf.read(p, dtype="float32")[0] for p in test.clip_path]
print(len(WAVS), "clips loaded\n")


def run_system(key, load_fn, transcribe_fn, batch_size):
    spec = SYSTEMS[key]
    t0 = time.time()
    proc, model = load_fn(spec["model_id"])
    hyps = []
    for i in range(0, len(WAVS), batch_size):
        hyps.extend(transcribe_fn(proc, model, WAVS[i:i + batch_size]))
        print(f"  {spec['name']}: {min(i + batch_size, len(WAVS))}/{len(WAVS)}", end="\r", file=sys.__stdout__)
    info = {**spec, "revision": getattr(model.config, "_commit_hash", None),
            "wall_clock_s": round(time.time() - t0, 1)}
    print(f"  {spec['name']}: done in {info['wall_clock_s']} s, revision {info['revision']}")
    return hyps, info


# ---------------------------------------------------------------------------
# System A: Whisper small. The language is forced to English and the task to transcribe (not translate), with
# greedy decoding (num_beams=1, no sampling), in batches of 8 clips.
# ---------------------------------------------------------------------------
WHISPER_DECODING = {"language": "en", "task": "transcribe", "num_beams": 1, "do_sample": False}


def load_whisper(model_id):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(DEVICE).eval()
    return proc, model


@torch.no_grad()
def transcribe_whisper(proc, model, wavs):
    feats = proc.feature_extractor(wavs, sampling_rate=TARGET_SR, return_tensors="pt").input_features
    ids = model.generate(feats.to(DEVICE), **WHISPER_DECODING)
    return [s.strip() for s in proc.tokenizer.batch_decode(ids, skip_special_tokens=True)]


# ---------------------------------------------------------------------------
# System B: wav2vec2-base-960h, a CTC acoustic model trained on LibriSpeech read speech. It takes the most
# likely character at every frame (greedy) with no language model, one clip at a time so that padding cannot
# change the output.
# ---------------------------------------------------------------------------
def load_wav2vec2(model_id):
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    proc = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(DEVICE).eval()
    return proc, model


@torch.no_grad()
def transcribe_wav2vec2(proc, model, wavs):
    texts = []
    for wav in wavs:   # one clip at a time: no padding
        x = proc.feature_extractor(wav, sampling_rate=TARGET_SR, return_tensors="pt").input_values
        ids = torch.argmax(model(x.to(DEVICE)).logits, dim=-1)   # greedy CTC, no language model
        texts.append(proc.tokenizer.batch_decode(ids.cpu())[0].strip())
    return texts


RUNNERS = {
    "A": (load_whisper, transcribe_whisper, 8, WHISPER_DECODING),
    "B": (load_wav2vec2, transcribe_wav2vec2, 8, {"ctc": "greedy argmax", "language_model": None}),
}

# ---------------------------------------------------------------------------
# Run the requested systems in turn, save each clip's raw output to hypotheses.csv, and record the model
# revision, decoding settings and run time in run_info.json.
# ---------------------------------------------------------------------------
which = [a for a in sys.argv[1:] if a in SYSTEMS] or list(SYSTEMS)
hyp_path, info_path = out("hypotheses.csv"), out("run_info.json")
hyps = read_csv("hypotheses.csv") if os.path.exists(hyp_path) else test[["file_id"]].copy()
hyps = test[["file_id"]].merge(hyps, on="file_id", how="left")
run_info = json.load(open(info_path)) if os.path.exists(info_path) else {"systems": {}}

for k in which:
    load_fn, transcribe_fn, bs, decoding = RUNNERS[k]
    hyps[HYP[k]], info = run_system(k, load_fn, transcribe_fn, bs)
    run_info["systems"][k] = {**info, "decoding": decoding}

run_info.update({"device": DEVICE, "python": platform.python_version(),
                 "packages": {p: version(p) for p in ["torch", "transformers"]}})
hyps.to_csv(hyp_path, index=False)
with open(info_path, "w") as f:
    json.dump(run_info, f, indent=2, default=str)
print("\nsaved outputs/hypotheses.csv and outputs/run_info.json")
