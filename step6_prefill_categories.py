"""Step 6 — pre-fill a DRAFT error category for every row of the error log, from the text alone.

The rules only look at the reference and the system output, never at the audio, so every label is a guess
to be checked by listening. The `confidence` column says how much to trust each rule, and spot_check_sample.csv
picks a small stratified sample (low-confidence rows first) to listen to.

Listening results are kept in outputs/spot_check_listened.csv (a copy of the sample, edited by hand); they are merged
back on every run, and the agreement between the draft labels and the listener is reported.

Outputs: outputs/error_log_prefilled.csv, outputs/spot_check_sample.csv, outputs/spot_check_agreement.csv
"""
import os
import re
from difflib import SequenceMatcher

import pandas as pd

from config import SEED, out, read_csv, start_log

start_log("step6_prefill_categories")

# ---------------------------------------------------------------------------
# The categories from the brief. Rows that no rule covers get "unclassified" and are always listened to.
# Proper names are taken from the capitalised words in the prompts (step 1, prompts read by everyone).
# ---------------------------------------------------------------------------
PHON = "phonetically plausible substitution"
OOV = "out-of-vocabulary word or proper name"
DISFL = "disfluency"
NORM_ART = "normalisation artefact"
HALLU = "hallucination"
UNCL = "unclassified"

PROPER_NAMES = {"james", "melissa", "singapore"}
FUNCTION_WORDS = {"a", "an", "the", "to", "of", "in", "is", "be", "and", "our", "her", "this", "that", "has"}

errors = read_csv("error_log_template.csv")


def compact(s):
    return re.sub(r"\s+", "", s)


def similarity(a, b):
    return SequenceMatcher(None, compact(a), compact(b)).ratio()


def after_bracket(context):
    """Reference words that follow the [error] in the context string."""
    return context.split("]", 1)[1].split() if "]" in context else []


def before_bracket(context):
    return context.split("[", 1)[0].split()


def classify(r):
    ref, hyp, op = r.reference, r.hypothesis, r.op
    nxt, prev = after_bracket(r.context), before_bracket(r.context)
    ref_words = set(ref.split())

    # 1. Spacing only: "pen pals" in the reference, "penpals" in the output (or the deleted half of it).
    if op == "substitute" and nxt and compact(hyp) == compact(ref + nxt[0]):
        return NORM_ART, "output joins this word and the next one", "high"
    if op == "delete" and prev and r.prev_hyp and compact(r.prev_hyp).endswith(ref):
        return NORM_ART, "deleted word was joined to the previous word in the output", "high"

    # 2. The reference itself records a false start or repetition (e.g. "unfold be before").
    if op == "delete" and len(ref.split()) == 1 and nxt and nxt[0] != ref:
        if nxt[0].startswith(ref):
            return DISFL, "deleted word is a false start of the next word", "high"
        if similarity(ref, nxt[0]) >= 0.6:
            return DISFL, "deleted word looks like a false start of the next word", "medium"

    # 3. Proper names.
    if ref_words & PROPER_NAMES:
        return OOV, f"reference contains a proper name ({', '.join(sorted(ref_words & PROPER_NAMES))})", "medium"

    if op == "insert":
        # A word split in two by the decoder: "sing" + "singapore", "chald" + "child".
        if nxt and (nxt[0].startswith(hyp[:3]) or similarity(hyp, nxt[0]) >= 0.5):
            return PHON, "inserted fragment resembles the start of the next word (word split)", "medium"
        if len(hyp.split()) >= 2 and r.system == "whisper_small":
            return HALLU, "several inserted words with no counterpart in the reference", "low"
        return UNCL, "single inserted word; listen for a hesitation or a misheard sound", "low"

    if op == "delete":
        if ref in FUNCTION_WORDS:
            return UNCL, "short function word deleted; listen for reduction or elision", "low"
        return UNCL, "content word deleted", "low"

    # Substitutions.
    sim = similarity(ref, hyp)
    # Only Whisper has a language model that can invent fluent text; wav2vec2 (greedy CTC) just garbles sounds.
    if len(hyp.split()) >= 3 and sim < 0.45:
        if r.system == "whisper_small":
            return HALLU, f"fluent output unrelated to the reference (similarity {sim:.2f})", "medium"
        return UNCL, f"garbled CTC output over several words (similarity {sim:.2f})", "low"
    if sim >= 0.5:
        if compact(ref).rstrip("ds") == compact(hyp).rstrip("ds") and ref != hyp:
            return PHON, "differs only in final -d/-s (possible final consonant reduction)", "medium"
        return PHON, f"spelling close to the reference (similarity {sim:.2f})", "high" if sim >= 0.7 else "medium"
    if sim >= 0.34:
        return PHON, f"partly similar spelling (similarity {sim:.2f})", "low"
    return UNCL, f"output not similar to the reference (similarity {sim:.2f})", "low"


# The hypothesis of the previous error chunk in the same utterance and system (for rule 1).
key = ["file_id", "system"]
errors["prev_hyp"] = errors.groupby(key).hypothesis.shift(1).fillna("")
labels = errors.apply(classify, axis=1, result_type="expand")
errors["category"], errors["auto_rule"], errors["confidence"] = labels[0], labels[1], labels[2]
errors["auto_category"] = errors.category
errors["checked_by_listening"] = ""
errors = errors.drop(columns="prev_hyp")

# ---------------------------------------------------------------------------
# Rows already listened to (outputs/spot_check_listened.csv) override the draft: their category, narrow
# transcription, notes and check result (ok / changed / unsure) are copied over, matched on the error itself.
# ---------------------------------------------------------------------------
LISTENED = out("spot_check_listened.csv")
MATCH = ["file_id", "system", "op", "reference", "hypothesis"]
HAND = ["category", "narrow_transcription", "notes", "checked_by_listening"]
if os.path.exists(LISTENED):
    heard = read_csv("spot_check_listened.csv").drop_duplicates(MATCH).set_index(MATCH)[HAND]
    errors = errors.set_index(MATCH)
    hit = errors.index.isin(heard.index)
    errors.loc[hit, HAND] = heard.loc[errors.index[hit], HAND].to_numpy()
    errors = errors.reset_index()[list(errors.reset_index().columns)]
    print(f"{hit.sum()} listened rows merged from outputs/spot_check_listened.csv\n")

errors.to_csv(out("error_log_prefilled.csv"), index=False)
print(f"{len(errors)} rows pre-filled -> outputs/error_log_prefilled.csv\n")
print(pd.crosstab(errors.category, errors.confidence, margins=True).to_string(), "\n")
print(pd.crosstab([errors.system, errors.acc], errors.category).to_string(), "\n")

# ---------------------------------------------------------------------------
# Spot-check sample: every low-confidence row would be too many, so take up to 2 rows per
# system x group x category (all of a cell when it is smaller), low confidence first.
# ---------------------------------------------------------------------------
rank = {"low": 0, "medium": 1, "high": 2}
sample = (errors.assign(_r=errors.confidence.map(rank))
          .sample(frac=1, random_state=SEED)
          .sort_values("_r", kind="stable")
          .groupby(["system", "acc", "auto_category"], sort=False).head(2)
          .drop(columns="_r")
          .sort_values(["system", "acc", "auto_category", "file_id"]))
sample.to_csv(out("spot_check_sample.csv"), index=False)
print(f"{len(sample)} rows to listen to -> outputs/spot_check_sample.csv")
print(sample.groupby(["system", "acc"]).size().unstack(fill_value=0).to_string())

# ---------------------------------------------------------------------------
# Agreement between the draft labels and the listener. "unsure" rows (the listener could not tell) count
# neither way, so agreement is ok / (ok + changed).
# ---------------------------------------------------------------------------
heard = errors[errors.checked_by_listening.isin(["ok", "changed", "unsure"])]
if len(heard):
    agree = (heard.groupby(["auto_category", "confidence"]).checked_by_listening.value_counts().unstack(fill_value=0)
             .reindex(columns=["ok", "changed", "unsure"], fill_value=0))
    agree.loc[("all", ""), :] = agree.sum()
    agree["agreement %"] = (100 * agree.ok / (agree.ok + agree.changed)).round(0)
    agree.to_csv(out("spot_check_agreement.csv"))
    print("\nDRAFT LABEL vs LISTENER (agreement = ok / (ok + changed); unsure counts neither way)")
    print(agree.to_string())
