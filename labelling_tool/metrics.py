"""Text normalization, word-level diffs, and WER computation matching crm-asr-server/asr.py."""
import difflib
import re
import unicodedata


def norm(text: str) -> str:
    """Lowercase, NFC, punctuation and symbols to spaces. Keeps Indic vowel signs and virama."""
    t = unicodedata.normalize("NFC", text or "").lower()
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return re.sub(r"\s+", " ", t).strip()


def words(text: str) -> list[str]:
    return [w for w in (text or "").strip().split() if w]


def word_diff(old_text: str, new_text: str) -> list[dict]:
    """Compute word-level change operations between old_text and new_text."""
    a = words(old_text)
    b = words(new_text)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ops.append({
            "op": tag,
            "old": " ".join(a[i1:i2]),
            "new": " ".join(b[j1:j2]),
            "old_range": [i1, i2],
            "new_range": [j1, j2],
        })
    return ops


def edit_distance(ref_words: list[str], hyp_words: list[str]) -> tuple[int, int]:
    """Return (levenshtein_edits, ref_word_count) for exact WER aggregation."""
    n = len(ref_words)
    m = len(hyp_words)
    if n == 0:
        return (m, 0)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        rw = ref_words[i - 1]
        for j in range(1, m + 1):
            cur = dp[j]
            if rw == hyp_words[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m], n


def compute_wer(ref_text: str, hyp_text: str) -> tuple[float | None, int, int]:
    """Normalized WER (edits, ref_words) using the model scorer's norm()."""
    r = norm(ref_text).split()
    h = norm(hyp_text).split()
    edits, n = edit_distance(r, h)
    if n == 0:
        return (0.0 if edits == 0 else None), edits, 0
    return edits / n, edits, n
