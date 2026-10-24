"""Step 4 — score both systems with the same normaliser: overall, by group, by speaker, by sentence.

Outputs: outputs/HG4052_A1_results.csv (results file to submit), outputs/audit_table.csv,
         outputs/wer_by_speaker.csv, outputs/wer_by_prompt.csv, outputs/fig1_wer_by_group.png,
         outputs/fig2_wer_by_speaker.png
"""
import json

import matplotlib
matplotlib.use("Agg")                     # write figures to files, no window
import matplotlib.pyplot as plt
import jiwer
import numpy as np
import pandas as pd

from config import (SEED, RESULTS_CSV, GROUPS, SEXES, N_SPEAKERS_PER_CELL, SYSTEMS, HYP,
                    out, read_csv, start_log)
from step2_normaliser import normalise, NORM

start_log("step4_score")

# ---------------------------------------------------------------------------
# The references and both systems' output go through the same normaliser, and jiwer counts the substitutions,
# deletions and insertions in each utterance. A reference that is empty after normalisation cannot be scored
# and is excluded.
# ---------------------------------------------------------------------------
test = read_csv("test_set.csv").merge(read_csv("hypotheses.csv"), on="file_id", how="left")
test["ref_norm"] = test.reference.map(normalise)
test["ref_words"] = test.ref_norm.str.split().str.len()


def edit_counts(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r:
        return 0, 0, 0, len(h)
    if not h:
        return len(r), 0, len(r), 0
    o = jiwer.process_words(ref, hyp)
    return len(r), o.substitutions, o.deletions, o.insertions


for k in SYSTEMS:
    test[f"{k}_hyp_norm"] = test[HYP[k]].fillna("").astype(str).map(normalise)
    names = [f"{k}_N", f"{k}_S", f"{k}_D", f"{k}_I"]
    counts = [edit_counts(r, h) for r, h in zip(test.ref_norm, test[f"{k}_hyp_norm"])]
    test[names] = pd.DataFrame(counts, index=test.index, columns=names)
    test[f"{k}_errors"] = test[f"{k}_S"] + test[f"{k}_D"] + test[f"{k}_I"]
    test[f"wer_{k}"] = (test[f"{k}_errors"] / test[f"{k}_N"].replace(0, np.nan)).round(4)

scored = test[test.ref_words > 0].copy()
print(f"scored utterances: {len(scored)} / {len(test)}")
print("normaliser:", json.dumps(NORM))
print("A =", SYSTEMS["A"]["model_id"], "| B =", SYSTEMS["B"]["model_id"], "\n")

# ---------------------------------------------------------------------------
# Group WER is corpus-level: total errors divided by total reference words. Confidence intervals come from a
# bootstrap that resamples speakers rather than utterances, because sentences from the same speaker are not
# independent; the same resamples give an interval for the A-B difference.
# ---------------------------------------------------------------------------
def corpus_wer(df, k):
    return df[f"{k}_errors"].sum() / df[f"{k}_N"].sum()


def speaker_bootstrap(df, n_boot=2000):
    per = df.groupby("speaker")[["A_N"] + [f"{k}_errors" for k in SYSTEMS]].sum()
    r = np.random.default_rng(SEED)
    idx = r.integers(0, len(per), size=(n_boot, len(per)))
    n = per["A_N"].to_numpy()[idx].sum(1)
    return {k: per[f"{k}_errors"].to_numpy()[idx].sum(1) / n for k in SYSTEMS}


def summarise(name, level, g):
    bs = speaker_bootstrap(g)
    row = {"level": level, "group": name, "n_speakers": g.speaker.nunique(),
           "n_utts": len(g), "n_ref_words": int(g["A_N"].sum())}
    for k in SYSTEMS:
        lo, hi = np.percentile(bs[k], [2.5, 97.5])
        n = g[f"{k}_N"].sum()
        row[f"WER_{k} %"] = round(100 * corpus_wer(g, k), 1)
        row[f"CI95_{k}"] = f"{100 * lo:.1f}-{100 * hi:.1f}"
        for op in "SDI":
            row[f"{op}_{k} %"] = round(100 * g[f"{k}_{op}"].sum() / n, 1)
    d_lo, d_hi = np.percentile(bs["A"] - bs["B"], [2.5, 97.5])
    row["A-B diff %"] = round(row["WER_A %"] - row["WER_B %"], 1)
    row["CI95_diff"] = f"{100 * d_lo:.1f} to {100 * d_hi:.1f}"
    return row


rows = [summarise("all", "overall", scored)]
rows += [summarise(a, "acc", scored[scored.acc == a]) for a in GROUPS]
rows += [summarise(s, "sex", scored[scored.sex == s]) for s in SEXES]
rows += [summarise(f"{a}-{s}", "acc x sex", scored[(scored.acc == a) & (scored.sex == s)])
         for a in GROUPS for s in SEXES]
audit = pd.DataFrame(rows).set_index(["level", "group"])
audit.to_csv(out("audit_table.csv"))
print("AUDIT TABLE")
print(audit.to_string())
print(f"\nacc x sex cells have only {N_SPEAKERS_PER_CELL} speakers each: read them as descriptive, not as findings.\n")

# ---------------------------------------------------------------------------
# WER per speaker shows who drives each group's figure, and WER per sentence shows which sentences are hard
# for everyone.
# ---------------------------------------------------------------------------
def wer_row(g, extra):
    return {**extra, "n_utts": len(g), "n_ref_words": int(g["A_N"].sum()),
            **{f"WER_{k} %": round(100 * corpus_wer(g, k), 1) for k in SYSTEMS}}


by_speaker = pd.DataFrame([wer_row(g, {"acc": a, "sex": s, "speaker": sp})
                           for (a, s, sp), g in scored.groupby(["acc", "sex", "speaker"])])
by_prompt = pd.DataFrame([wer_row(g, {"prompt_id": p, "words_per_utt": round(g["A_N"].mean(), 1)})
                          for p, g in scored.groupby("prompt_id")])
by_speaker.to_csv(out("wer_by_speaker.csv"), index=False)
by_prompt.to_csv(out("wer_by_prompt.csv"), index=False)
print("WER BY SPEAKER")
print(by_speaker.sort_values("WER_A %", ascending=False).to_string(index=False), "\n")
print("WER BY SENTENCE")
print(by_prompt.sort_values("WER_A %", ascending=False).to_string(index=False), "\n")

# ---------------------------------------------------------------------------
# Figure 1: WER by accent group and overall, with speaker-bootstrap 95% CIs as error bars. Figure 2: one point
# per speaker, showing how much speakers vary within each group.
# ---------------------------------------------------------------------------
COLORS = {"A": "#2a78d6", "B": "#eb6834"}   # colour-blind-safe pair
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

sub = audit.loc[[("overall", "all")] + [("acc", a) for a in GROUPS]]
fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(len(sub))
w, ymax = 0.36, 0
for j, k in enumerate(SYSTEMS):
    vals = sub[f"WER_{k} %"].to_numpy()
    ci = np.array([[float(v) for v in s.split("-")] for s in sub[f"CI95_{k}"]])
    pos = x + (j - 0.5) * (w + 0.02)
    ax.bar(pos, vals, w, color=COLORS[k], label=f"{k}: {SYSTEMS[k]['model_id']}", zorder=2)
    ax.errorbar(pos, vals, yerr=np.vstack([vals - ci[:, 0], ci[:, 1] - vals]),
                fmt="none", ecolor="#444", elinewidth=1, capsize=3, zorder=3)
    for p, top, v in zip(pos, ci[:, 1], vals):
        ax.text(p, top + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color="#222")
    ymax = max(ymax, ci[:, 1].max())
ax.set_xticks(x, [f"{g}\n{r.n_speakers} spk · {r.n_utts} utts\n{r.n_ref_words} words"
                  for (_, g), r in sub.iterrows()], fontsize=9)
ax.set_ylim(0, ymax * 1.15 + 1)
ax.set_ylabel("WER (%)")
ax.set_title("Word error rate by accent group (error bars: speaker-bootstrap 95% CI)", pad=30)
ax.grid(axis="y", color="#e5e5e5", zorder=0)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout()
fig.savefig(out("fig1_wer_by_group.png"), dpi=200)

fig, ax = plt.subplots(figsize=(7.5, 4.4))
MARKERS = {"F": "o", "M": "^"}
for gi, acc in enumerate(GROUPS):
    d = by_speaker[by_speaker.acc == acc].reset_index(drop=True)
    offs = np.linspace(-0.12, 0.12, len(d)) if len(d) > 1 else np.zeros(1)
    for j, k in enumerate(SYSTEMS):
        base = gi + (j - 0.5) * 0.42
        for o, (_, r) in zip(offs, d.iterrows()):
            ax.scatter(base + o, r[f"WER_{k} %"], s=36, marker=MARKERS[r.sex], color=COLORS[k],
                       edgecolor="white", linewidth=0.8, zorder=3)
for k in SYSTEMS:
    ax.scatter([], [], s=36, marker="s", color=COLORS[k], label=f"{k}: {SYSTEMS[k]['name']}")
for s, m in MARKERS.items():
    ax.scatter([], [], s=36, marker=m, color="#777", label={"F": "female", "M": "male"}[s])
ax.set_xticks(range(len(GROUPS)), GROUPS)
ax.set_ylim(bottom=0)
ax.set_ylabel("speaker WER (%)")
ax.set_title("WER per speaker (each point = one speaker)", pad=30)
ax.grid(axis="y", color="#e5e5e5", zorder=0)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
fig.tight_layout()
fig.savefig(out("fig2_wer_by_speaker.png"), dpi=200)
print("saved outputs/fig1_wer_by_group.png and outputs/fig2_wer_by_speaker.png")

# ---------------------------------------------------------------------------
# Write the results CSV for submission: one row per utterance with file ID, speaker, condition (accent group),
# reference, system A output and system B output, followed by sex, prompt ID, the normalised texts and the
# per-utterance error counts.
# ---------------------------------------------------------------------------
cols = ["file_id", "speaker", "condition", "reference", HYP["A"], HYP["B"],
        "acc", "sex", "prompt_id", "duration_s", "clip_path", "ref_norm", "A_hyp_norm", "B_hyp_norm"]
cols += [f"{k}_{m}" for k in SYSTEMS for m in ["N", "S", "D", "I", "errors"]] + [f"wer_{k}" for k in SYSTEMS]
results_name = RESULTS_CSV
test[cols].to_csv(out(results_name), index=False)
print(f"saved outputs/{results_name}  ({len(test)} rows)")
