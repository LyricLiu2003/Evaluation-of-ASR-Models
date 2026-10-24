"""Step 5 — extract the errors as material for listening. Categorising them is done by hand.

Outputs: outputs/alignments_<GROUP>.txt, outputs/error_log_template.csv, outputs/word_level_errors.csv,
         outputs/per_token_error_rate_by_group.csv
Categories from the brief: phonetically plausible substitution, out-of-vocabulary word or proper name,
discourse particle, code-switch or borrowed word, disfluency, normalisation artefact, hallucination, translation.
"""
from collections import Counter

import jiwer
import pandas as pd

from config import GROUPS, SYSTEMS, RESULTS_CSV, out, read_csv, start_log

start_log("step5_errors")

scored = read_csv(RESULTS_CSV)
scored = scored[scored.ref_norm.str.len() > 0].copy()
for k in SYSTEMS:
    for c in ["N", "errors"]:
        scored[f"{k}_{c}"] = scored[f"{k}_{c}"].astype(int)
    scored[f"wer_{k}"] = pd.to_numeric(scored[f"wer_{k}"])


def corpus_wer(df, k):
    return df[f"{k}_errors"].sum() / df[f"{k}_N"].sum()


# ---------------------------------------------------------------------------
# Find the accent group with the highest WER for each system. For that group, write every utterance's
# alignment, worst first, with its WAV path beside it, so the errors can be read while listening or opened in
# Praat.
# ---------------------------------------------------------------------------
worst = {k: max(GROUPS, key=lambda a: corpus_wer(scored[scored.acc == a], k)) for k in SYSTEMS}
print("highest-WER group:", {SYSTEMS[k]["name"]: a for k, a in worst.items()})


def alignment_text(ref, hyp):
    if not ref.split() or not hyp.split():
        return f"REF: {ref}\nHYP: {hyp}\n"
    return jiwer.visualize_alignment(jiwer.process_words(ref, hyp), show_measures=False, skip_correct=False)


for acc in sorted(set(worst.values())):
    g = scored[scored.acc == acc].copy()
    g["mean_wer"] = g[[f"wer_{k}" for k in SYSTEMS]].mean(axis=1)
    path = out(f"alignments_{acc}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for _, r in g.sort_values("mean_wer", ascending=False).iterrows():
            f.write(f"=== {r.file_id} | speaker {r.speaker} {r.sex} | {r.prompt_id} | {float(r.duration_s):.1f}s | "
                    f"{r.clip_path} | " + " | ".join(f"WER {k} {100 * r[f'wer_{k}']:.0f}%" for k in SYSTEMS) + "\n")
            for k in SYSTEMS:
                f.write(f"--- {k} {SYSTEMS[k]['name']}\n{alignment_text(r.ref_norm, r[f'{k}_hyp_norm'])}\n")
    print("wrote", path)

# ---------------------------------------------------------------------------
# Split every substitution, deletion and insertion into its own row with a few words of context, giving the
# error-log template. The narrow transcription and category columns are left empty, to be filled in by hand
# after listening.
# ---------------------------------------------------------------------------
def error_rows(r, k):
    ref, hyp = r.ref_norm.split(), r[f"{k}_hyp_norm"].split()
    base = {"file_id": r.file_id, "speaker": r.speaker, "acc": r.acc, "sex": r.sex, "prompt_id": r.prompt_id,
            "system": SYSTEMS[k]["name"], "clip": r.clip_path}
    if not hyp:
        return [{**base, "op": "delete", "reference": " ".join(ref), "hypothesis": "", "context": ""}]
    rows = []
    for ch in jiwer.process_words(" ".join(ref), " ".join(hyp)).alignments[0]:
        if ch.type == "equal":
            continue
        rs, re_ = ch.ref_start_idx, ch.ref_end_idx
        rows.append({**base, "op": ch.type,
                     "reference": " ".join(ref[rs:re_]),
                     "hypothesis": " ".join(hyp[ch.hyp_start_idx:ch.hyp_end_idx]),
                     "context": " ".join(ref[max(0, rs - 4):rs]) + " [" + " ".join(ref[rs:re_]) + "] "
                                + " ".join(ref[re_:re_ + 4])})
    return rows


errors = pd.DataFrame([row for _, r in scored.iterrows() for k in SYSTEMS for row in error_rows(r, k)])
errors["narrow_transcription"] = ""
errors["category"] = ""
errors["notes"] = ""
errors.to_csv(out("error_log_template.csv"), index=False)
print(f"\n{len(errors)} error chunks written to outputs/error_log_template.csv")
print(errors.groupby(["acc", "system", "op"]).size().unstack(fill_value=0).to_string(), "\n")

# ---------------------------------------------------------------------------
# Errors shared by the two systems: how many reference words both systems got wrong, and which ones. Because
# every group read the same sentences, the error rate on each word can also be compared across accent groups.
# ---------------------------------------------------------------------------
def wrong_ref_positions(r, k):
    ref, hyp = r.ref_norm.split(), r[f"{k}_hyp_norm"].split()
    if not hyp:
        return set(range(len(ref)))
    pos = set()
    for ch in jiwer.process_words(r.ref_norm, r[f"{k}_hyp_norm"]).alignments[0]:
        if ch.type in ("substitute", "delete"):
            pos.update(range(ch.ref_start_idx, ch.ref_end_idx))
    return pos


overlap_rows, word_rows = [], []
for _, r in scored.iterrows():
    a, b = wrong_ref_positions(r, "A"), wrong_ref_positions(r, "B")
    ref = r.ref_norm.split()
    overlap_rows.append({"acc": r.acc, "ref_words": len(ref), "A_only": len(a - b), "B_only": len(b - a),
                         "both": len(a & b)})
    for i, w in enumerate(ref):
        word_rows.append({"acc": r.acc, "speaker": r.speaker, "prompt_id": r.prompt_id, "position": i,
                          "ref_word": w, "A_wrong": i in a, "B_wrong": i in b})

print("REFERENCE WORDS MISSED BY A ONLY / B ONLY / BOTH")
print(pd.DataFrame(overlap_rows).groupby("acc").sum().to_string(), "\n")

words = pd.DataFrame(word_rows)
words.to_csv(out("word_level_errors.csv"), index=False)
per_token = (words.assign(both=words.A_wrong & words.B_wrong)
             .groupby(["prompt_id", "position", "ref_word", "acc"])[["A_wrong", "B_wrong", "both"]].mean()
             .unstack("acc").round(2))
per_token.to_csv(out("per_token_error_rate_by_group.csv"))
print("reference tokens most often missed by BOTH systems (mean share of speakers across groups):")
print(per_token["both"].mean(axis=1).sort_values(ascending=False).head(25).round(2).to_string(), "\n")

# ---------------------------------------------------------------------------
# The most frequent reference -> output substitution pairs for each system in its worst group, as a guide to
# which examples to listen to first.
# ---------------------------------------------------------------------------
for k in SYSTEMS:
    sub = errors[(errors.system == SYSTEMS[k]["name"]) & (errors.op == "substitute") & (errors.acc == worst[k])]
    print(f"{SYSTEMS[k]['name']} — top substitutions in {worst[k]}:")
    for (ref, hyp), n in Counter(zip(sub.reference, sub.hypothesis)).most_common(20):
        print(f"  {n:3d}  {ref!r:28s} -> {hyp!r}")
    print()
