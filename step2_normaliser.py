"""Step 2 — fix the metric: the text normaliser, decided before any system output is looked at.

Imported by steps 4 and 5 (`from step2_normaliser import normalise`).
Run on its own (`python step2_normaliser.py`) it lists the markup found in the test-set references,
shows the normaliser on invented examples, and saves outputs/normaliser_config.json.
"""
import json
import re
import unicodedata
from collections import Counter

from num2words import num2words

from config import SYSTEMS, out, read_csv, start_log

# ---------------------------------------------------------------------------
# This is the scoring rule itself: the same function is applied to the references and to both systems' output.
# Each switch below is a decision reported with the results, and all of them are fixed before any system
# output is looked at.
# ---------------------------------------------------------------------------
NORM = {
    "drop_angle_tags":      True,   # <FIL/> <SPK/> <STA/> <NON/> <NPS/> (non-speech events in the references)
    "drop_unintelligible":  True,   # ** (none in the chosen prompts, kept for safety)
    "drop_bracketed":       True,   # (...) and [...] in system output, e.g. "[BLANK_AUDIO]", "(laughs)"
    "numbers_to_words":     True,   # references spell numbers out; systems may print digits
    "four_digit_years":     True,   # 1900-2099 read as years: 2019 -> twenty nineteen
    "expand_contractions":  True,   # I'll -> i will, can't -> can not (both sides)
    "join_spelled_letters": True,   # reference "G S T" and output "GST" both -> gst
    "drop_filler_words":    True,   # um, uh, er, hmm ... (references mark fillers only as <FIL/>)
    "uk_us_spelling":       True,   # colour -> color, via Whisper's published spelling map
}
FILLER_WORDS = {"um", "umm", "uh", "uhm", "uhh", "er", "erm", "hmm", "hm", "mm", "mmm", "mhm"}

CONTRACTIONS = {
    "won't": "will not", "can't": "can not", "cannot": "can not", "shan't": "shall not",
    "ain't": "is not", "let's": "let us", "y'all": "you all",
    "it's": "it is", "that's": "that is", "what's": "what is", "there's": "there is",
    "here's": "here is", "he's": "he is", "she's": "she is", "who's": "who is",
    "where's": "where is", "how's": "how is",
}
SUFFIXES = [("n't", " not"), ("'re", " are"), ("'m", " am"), ("'ll", " will"), ("'ve", " have"), ("'d", " would")]


def _load_spelling_map():
    if not NORM["uk_us_spelling"]:
        return {}
    try:
        from transformers import WhisperTokenizer
        return WhisperTokenizer.from_pretrained(SYSTEMS["A"]["model_id"]).english_spelling_normalizer or {}
    except Exception as e:
        print("spelling map not available, skipping:", e)
        return {}


SPELLING = _load_spelling_map()


# ---------------------------------------------------------------------------
# Helpers that make written forms comparable: numbers become English words (including percentages, currency,
# times, ordinals, decimals and years), contractions are expanded, and runs of single letters are joined (G S
# T -> gst).
# ---------------------------------------------------------------------------
def _year_or_cardinal(s):
    n = int(s)
    if NORM["four_digit_years"] and len(s) == 4 and 1900 <= n <= 2099:
        return num2words(n, to="year")
    return num2words(n)


def numbers_to_words(t):
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)                          # 1,000 -> 1000
    t = re.sub(r"\$\s?(\d+(?:\.\d+)?)", r"\1 dollars", t)               # $5 -> 5 dollars
    t = re.sub(r"(\d)\s?%", r"\1 percent", t)                           # 10% -> 10 percent
    t = re.sub(r"(\d):(\d)", r"\1 \2", t)                               # 3:30 -> 3 30
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: num2words(int(m.group(1)), to="ordinal"), t)
    t = re.sub(r"\d+\.\d+", lambda m: num2words(float(m.group(0))), t)  # 3.5 -> three point five
    t = re.sub(r"\d+", lambda m: _year_or_cardinal(m.group(0)), t)
    return t


def expand_contractions(t):
    words = []
    for w in t.split():
        w = w.strip("'")
        if w in CONTRACTIONS:
            w = CONTRACTIONS[w]
        else:
            for suf, rep in SUFFIXES:
                if w.endswith(suf) and len(w) > len(suf):
                    w = w[: -len(suf)] + rep
                    break
        words.append(w)
    return " ".join(words)


def join_letters(words):
    """Join runs of 2+ single letters (other than the words 'a' and 'i'): g s t -> gst."""
    result, run = [], []
    for w in words + [""]:
        if len(w) == 1 and w.isalpha() and w not in ("a", "i"):
            run.append(w)
            continue
        if run:
            result.extend(["".join(run)] if len(run) > 1 else run)
            run = []
        if w:
            result.append(w)
    return result


def normalise(text, cfg=NORM):
    t = unicodedata.normalize("NFKC", str(text))
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'")
    if cfg["drop_angle_tags"]:      t = re.sub(r"<[^>]*>", " ", t)
    if cfg["drop_unintelligible"]:  t = t.replace("**", " ")
    if cfg["drop_bracketed"]:       t = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", t)
    t = t.lower()
    if cfg["numbers_to_words"]:     t = numbers_to_words(t)
    t = re.sub(r"[-_/]", " ", t)                 # hyphens, underscores, slashes -> space
    t = re.sub(r"[^\w\s']", " ", t)              # all other punctuation -> space
    if cfg["expand_contractions"]:  t = expand_contractions(t)
    words = t.replace("'", "").split()           # possessives: mary's -> marys (same on both sides)
    if cfg["join_spelled_letters"]: words = join_letters(words)
    if cfg["drop_filler_words"]:    words = [w for w in words if w not in FILLER_WORDS]
    if cfg["uk_us_spelling"]:       words = [SPELLING.get(w, w) for w in words]
    return " ".join(words)


if __name__ == "__main__":
    start_log("step2_normaliser")
    test = read_csv("test_set.csv")

    # -----------------------------------------------------------------------
    # Take an inventory of the markup in the test-set references (<FIL/>-style tags, **, capitalised names and
    # spaced initialisms, digits, apostrophes, hyphens) to check that the rules above cover every case that
    # actually occurs.
    # -----------------------------------------------------------------------
    INVENTORY = {
        "<...> tags":                 r"<[^>]*>",
        "** unintelligible":          r"\*\*",
        "spaced initialisms (G S T)": r"\b(?:[A-Z] ){1,}[A-Z]\b",
        "capitalised words":          r"\b[A-Z][a-z]+\b",
        "digits":                     r"\S*\d\S*",
        "apostrophes":                r"\S*['’]\S*",
        "hyphenated":                 r"\w+-\w+",
    }
    for label, pat in INVENTORY.items():
        found = Counter(m for r in test.reference for m in re.findall(pat, r))
        print(f"{label:28s} n={sum(found.values()):4d}  types={len(found):3d}  e.g. {[k for k, _ in found.most_common(6)]}")
    print("\nnon-alphanumeric characters:",
          Counter(ch for r in test.reference for ch in r if not (ch.isalnum() or ch.isspace())).most_common())
    print(f"\n{len(SPELLING)} UK->US spelling pairs loaded\n")

    # Invented examples (not corpus data) showing what the normaliser does:
    for s in ["<FIL/> the G S T rate went up to nine percent in twenty twenty four <SPK/>",
              "The GST rate went up to 9% in 2024.",
              "THE G S T RATE WENT UP TO NINE PERCENT IN TWENTY TWENTY FOUR",
              "[BLANK_AUDIO] I'll meet Mary's friend at the colour run, um, at 3:30."]:
        print(f"{s}\n  -> {normalise(s)}\n")

    empty = test.reference.map(normalise).str.len() == 0
    print("references empty after normalisation:", int(empty.sum()), list(test.loc[empty, "file_id"]))

    with open(out("normaliser_config.json"), "w") as f:
        json.dump({"NORM": NORM, "FILLER_WORDS": sorted(FILLER_WORDS), "n_spelling_pairs": len(SPELLING)}, f, indent=2)
    print("saved outputs/normaliser_config.json")
