#!/usr/bin/env python3
"""Recompute the SAE-vs-transcoder auto-interp head-to-head from the saved delphi result JSONs.

Uses the SAME unpaired difference-of-means bootstrap as the ai-g2 head-to-head and the steering
control (``infra/modal_app.py::steer_eval``): independently resample the SAE per-latent accuracy
array and the transcoder per-latent accuracy array, take the difference of means, 10000 times,
``numpy.random.default_rng(seed)`` (seed 0), then the [2.5, 97.5] percentiles. A difference is only
called real if the CI excludes 0 (R3/R4/R5  -  every number here traces to a logged JSON on the
microscope-artifacts volume; this script fabricates nothing).

Run (after ``modal volume get`` pulls the JSONs into a local dir):
    python scripts/headtohead_autointerp.py --dir artifacts_pull

Prints the 3B and 7B scorer head-to-heads side by side so the scorer-strength question can be
answered honestly: did the stronger scorer (a) raise absolute scores above 3B near-chance, and
(b) change the head-to-head conclusion (CI still includes 0, or a coder now wins)?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_BOOT = 10_000
SEED = 0
SCORERS = ("detection", "fuzz")


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _per_latent(result: dict, scorer: str) -> np.ndarray:
    """Per-latent accuracy array for one scorer ('detection'|'fuzz')."""
    return np.asarray(result["scores"][scorer]["per_latent"], dtype=float)


def unpaired_diff_bootstrap(a: np.ndarray, b: np.ndarray, seed: int = SEED,
                            n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Unpaired difference-of-means bootstrap of (mean(a) - mean(b)).

    Identical method to ai-g2 / steer_eval: independently resample a and b with replacement to their
    own lengths, take the difference of means, n_boot times; return (point, ci_lo, ci_hi).
    """
    if a.size == 0 or b.size == 0:
        raise ValueError("empty per-latent array; cannot bootstrap")
    rng = np.random.default_rng(seed)
    boot = np.array([
        a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def head_to_head(sae: dict, tc: dict, scorer_label: str) -> dict:
    """SAE-vs-TC head-to-head for one scorer model. Returns a dict of per-metric stats."""
    out: dict = {"scorer": scorer_label, "metrics": {}}
    for metric in SCORERS:
        a = _per_latent(sae, metric)
        b = _per_latent(tc, metric)
        diff, lo, hi = unpaired_diff_bootstrap(a, b)
        out["metrics"][metric] = {
            "sae_mean": round(float(a.mean()), 4), "sae_n": int(a.size),
            "tc_mean": round(float(b.mean()), 4), "tc_n": int(b.size),
            "sae_minus_tc": round(diff, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
    return out


def _fmt(h: dict) -> str:
    lines = [f"  scorer = {h['scorer']}"]
    for metric, m in h["metrics"].items():
        verdict = "SIGNIFICANT" if m["excludes_zero"] else "CI incl 0 (no sig diff)"
        lines.append(
            f"    {metric:9s}: SAE {m['sae_mean']:.4f} (n={m['sae_n']}) vs "
            f"TC {m['tc_mean']:.4f} (n={m['tc_n']})  "
            f"d(SAE-TC)={m['sae_minus_tc']:+.4f}  CI95[{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}]  "
            f"-> {verdict}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="artifacts_pull",
                    help="dir with autointerp_{sae,tc}[_7b].json pulled from the volume")
    args = ap.parse_args()
    d = Path(args.dir)

    print("=" * 78)
    print("SAE-vs-TRANSCODER AUTO-INTERP HEAD-TO-HEAD (unpaired diff-of-means bootstrap,")
    print(f"seed={SEED}, n_boot={N_BOOT}; per-latent arrays from delphi JSONs on microscope-artifacts)")
    print("=" * 78)

    results = []
    # 3B (the historical ai-g2 result).
    sae3, tc3 = d / "autointerp_sae.json", d / "autointerp_tc.json"
    if sae3.exists() and tc3.exists():
        s, t = _load(sae3), _load(tc3)
        h = head_to_head(s, t, s.get("scorer", "3B"))
        results.append(("3B", h))
        print("\n[3B scorer  -  historical ai-g2]")
        print(_fmt(h))
    else:
        print("\n[3B] missing autointerp_sae.json / autointerp_tc.json  -  skipping")

    # 7B (this unit).
    sae7, tc7 = d / "autointerp_sae_7b.json", d / "autointerp_tc_7b.json"
    if sae7.exists() and tc7.exists():
        s, t = _load(sae7), _load(tc7)
        h = head_to_head(s, t, s.get("scorer", "7B"))
        results.append(("7B", h))
        print("\n[7B scorer  -  this unit, A100-40GB]")
        print(_fmt(h))
    else:
        print("\n[7B] missing autointerp_sae_7b.json / autointerp_tc_7b.json  -  run not complete")

    # Side-by-side absolute-score summary (did 7B raise scores above 3B near-chance?).
    if len(results) == 2:
        print("\n" + "-" * 78)
        print("3B vs 7B SIDE BY SIDE (absolute means; chance = 0.5):")
        print("-" * 78)
        hdr = f"  {'metric/coder':22s} {'3B':>10s} {'7B':>10s} {'d(7B-3B)':>12s}"
        print(hdr)
        b3 = dict(results)["3B"]["metrics"]
        b7 = dict(results)["7B"]["metrics"]
        for metric in SCORERS:
            for coder, key in (("SAE", "sae_mean"), ("TC", "tc_mean")):
                v3, v7 = b3[metric][key], b7[metric][key]
                print(f"  {metric+'/'+coder:22s} {v3:>10.4f} {v7:>10.4f} {v7 - v3:>+12.4f}")

    # Emit a machine-readable blob too (for the EXPERIMENTS rows / report numbers).
    print("\nJSON:", json.dumps([{"scorer_tag": tag, **h} for tag, h in results]))


if __name__ == "__main__":
    main()
