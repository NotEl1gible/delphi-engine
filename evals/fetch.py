"""Build the question set, once, and commit it.

Run before any instrument exists, so that no question can have been chosen because of how the
engine scored on it. `questions.jsonl` is the committed artifact; this script is here so a
reader can see the filter rather than trust the file.

Three filters do real work:

- **Liquidity.** A resolved market with four traders is a coin flip somebody wrote down. The
  crowd baseline this engine is measured against is only a baseline if the crowd showed up.
- **Personal questions.** "Will I finish my thesis" has a true probability only its author
  knows. Scoring a forecaster on those measures nothing about forecasting.
- **Balance.** A set that resolves 80% YES is beaten by a constant, and every arm then looks
  competent. The selection targets an even split and the report prints what it achieved.

The split is **temporal**: the earlier half by resolution date is dev, the later half is test.
A random split would let the calibrator be fitted on questions that resolved after the ones it
is scored on, which is not a mistake a deployed forecaster is able to make.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

API = "https://api.manifold.markets/v0/search-markets"
OUT = Path(__file__).with_name("questions.jsonl")

PERSONAL = re.compile(
    r"^\s*(will|can|do|did|am)\s+(i|we|my|me|our)\b"
    r"|^\s*my\b"                              # "My current relationship lasts >6 months"
    r"|\bfriend of mine\b|\ba particular friend\b|\bmy (current|new|next)\b"
    r"|\bmy (thesis|girlfriend|boyfriend|job|cat|dog|mom|dad|wife|husband)\b"
    r"|\bthis market\b|\bmanifold\b|\bmana\b", re.I)

YEARS = re.compile(r"\b(20\d\d)\b")
# Questions whose answer is random by construction. Not filtered because they are hard --
# hard is the point -- but because their true probability is exactly 0.5 and no forecaster
# can beat that, so they add irreducible noise to every arm equally and to the comparison.
CHANCE = re.compile(r"\bcoin\s?flip|\bdice roll|\brandom number|\blottery draw\b", re.I)


def _shingles(text: str, k: int = 4) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}


def dedupe(rows: list[dict], threshold: float = 0.5) -> tuple[list[dict], int]:
    """Drop near-duplicate questions, keeping the earliest.

    Two markets asking "Will Russia gain more territory than it loses in June 2026" are ONE
    event with two prices. Scoring both counts a single outcome twice, and the paired
    bootstrap resamples questions as if they were independent draws -- so a duplicate does not
    merely add a row, it narrows the interval it has no right to narrow.
    """
    kept: list[dict] = []
    sigs: list[set[str]] = []
    dropped = 0
    for m in sorted(rows, key=lambda x: x["resolutionTime"]):
        s = _shingles(m.get("question") or "")
        if any(s and t and len(s & t) / len(s | t) >= threshold for t in sigs):
            dropped += 1
            continue
        kept.append(m)
        sigs.append(s)
    return kept, dropped


def stale(m: dict) -> bool:
    """The question's own deadline is already in the past when it resolves.

    "Will X happen by the end of 2025?", resolved in mid-2026, is not a forecasting question
    -- it is a recall question, and a model that has seen the news answers it from memory.
    Three of these survived the first pass of the filter, which is exactly the kind of thing
    that would have quietly inflated every arm at once.
    """
    ts = datetime.fromtimestamp(m["resolutionTime"] / 1000, tz=UTC)
    years = [int(y) for y in YEARS.findall(m.get("question") or "")]
    return bool(years) and max(years) < ts.year


def _get(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "delphi-engine/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def harvest(pages: int = 12, per_page: int = 500) -> list[dict]:
    seen: dict[str, dict] = {}
    for i in range(pages):
        url = (f"{API}?term=&filter=resolved&contractType=BINARY"
               f"&limit={per_page}&offset={i * per_page}&sort=resolve-date")
        try:
            batch = _get(url)
        except Exception as e:
            print(f"  page {i}: {type(e).__name__}: {e}")
            break
        if not batch:
            break
        for m in batch:
            seen.setdefault(m["id"], m)
        print(f"  page {i}: +{len(batch)}  (pool {len(seen)})")
    return list(seen.values())


def usable(m: dict, min_bettors: int, min_volume: float) -> bool:
    if m.get("resolution") not in ("YES", "NO"):
        return False                                   # MKT / CANCEL carry no ground truth
    if not m.get("isResolved") or not m.get("resolutionTime"):
        return False
    q = (m.get("question") or "").strip()
    if not (25 <= len(q) <= 220) or PERSONAL.search(q) or CHANCE.search(q):
        return False
    if stale(m):
        return False
    if (m.get("uniqueBettorCount") or 0) < min_bettors:
        return False
    if (m.get("volume") or 0) < min_volume:
        return False
    # The crowd's last price, kept as a BASELINE and never shown to the panel. A market that
    # closed at a certainty is one the crowd already knew; those make the baseline unbeatable
    # for reasons that have nothing to do with forecasting skill.
    p = m.get("probability")
    return isinstance(p, (int, float)) and 0.05 <= p <= 0.95


def select(rows: list[dict], n: int) -> list[dict]:
    """Alternate YES and NO by resolution date so the set is balanced AND temporally ordered."""
    rows = sorted(rows, key=lambda m: m["resolutionTime"])
    yes = [m for m in rows if m["resolution"] == "YES"]
    no = [m for m in rows if m["resolution"] == "NO"]
    take = min(n // 2, len(yes), len(no))
    step_y = max(1, len(yes) // take) if take else 1
    step_n = max(1, len(no) // take) if take else 1
    picked = yes[::step_y][:take] + no[::step_n][:take]
    return sorted(picked, key=lambda m: m["resolutionTime"])


def to_question(m: dict, split: str) -> dict:
    ts = datetime.fromtimestamp(m["resolutionTime"] / 1000, tz=UTC)
    return {
        "id": f"Q-{m['id'][:10]}",
        "text": m["question"].strip(),
        "resolution_date": ts.date().isoformat(),
        "outcome": 1 if m["resolution"] == "YES" else 0,
        "split": split,
        "source": "manifold",
        # baseline only; never rendered into a prompt
        "market_p": round(float(m["probability"]), 4),
        "bettors": int(m.get("uniqueBettorCount") or 0),
        "volume": round(float(m.get("volume") or 0.0), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--pages", type=int, default=3)     # the API rejects offsets past ~1500
    ap.add_argument("--min-bettors", type=int, default=18)
    ap.add_argument("--min-volume", type=float, default=200.0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    print("harvesting resolved binary markets")
    pool = harvest(a.pages)
    keep = [m for m in pool if usable(m, a.min_bettors, a.min_volume)]
    keep, dup = dedupe(keep)
    print(f"\npool {len(pool)}  ->  usable {len(keep) + dup}  ->  after dedupe {len(keep)} "
          f"({dup} near-duplicate{'s' if dup != 1 else ''} dropped)")
    if len(keep) < a.n:
        print(f"only {len(keep)} usable questions; raise --pages or lower the thresholds")
    picked = select(keep, a.n)

    half = len(picked) // 2
    rows = [to_question(m, "dev" if i < half else "test") for i, m in enumerate(picked)]

    yes = sum(r["outcome"] for r in rows)
    print(f"selected {len(rows)}   YES {yes}  NO {len(rows) - yes}")
    for s in ("dev", "test"):
        sub = [r for r in rows if r["split"] == s]
        if sub:
            k = sum(r["outcome"] for r in sub)
            print(f"  {s:<5} n={len(sub):<3} YES {k}/{len(sub)}   "
                  f"{sub[0]['resolution_date']} .. {sub[-1]['resolution_date']}")
    base = [r["market_p"] for r in rows]
    ys = [r["outcome"] for r in rows]
    brier = sum((p - y) ** 2 for p, y in zip(base, ys, strict=True)) / len(rows)
    print(f"\ncrowd baseline Brier on this set: {brier:.4f}   "
          f"(a constant 0.5 scores {sum((0.5 - y) ** 2 for y in ys) / len(rows):.4f})")

    if a.write:
        OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
