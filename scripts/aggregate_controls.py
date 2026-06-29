#!/usr/bin/env python3
"""Aggregate the multi-concept solidification runs into one honest cross-concept verdict.

Two questions are re-asked across the N profession contrasts (not a single n=1 result):

A. CIRCUIT (circuit_multi.json). Does a sparse SAE feature set still faithfully mediate the concept
   after the holdout leak-fix? For the pre-registered primary K (default 10) we show, per concept,
   faithfulness_holdout vs faithfulness_fullset (the gap between them IS the leak the holdout split
   removed), the holdout-minus-random gap with its paired CI, and the permutation p. Across concepts
   we report mean/min holdout faithfulness, mean gap, how many beat random, and a Holm-Bonferroni
   correction over the N permutation p-values (which concepts survive at alpha=0.05). Repeated for
   every K present.

B. CONTROL (probing_multi_real.json vs probing_multi_random.json). Does the real-model SAE probe beat
   the randomized-model SAE probe per concept? real and random share the identical seeded train/test
   split, so yte aligns index-for-index and the comparison is PAIRED. From the persisted per-example
   correctness we run a paired bootstrap (10000 iters, ONE shared resample index per iter, seed 0):
   gap_b = mean(real_correct[idx]) - mean(random_correct[idx]); CI = [2.5, 97.5] pct; one-sided
   bootstrap p = fraction of boot gaps <= 0. Across concepts: mean gap, how many CIs exclude 0, an
   exact two-sided sign test (binomial, p=0.5) over the N concepts, and Holm over the N one-sided p's.

C. A short reproduced/novel/inconclusive verdict (R4) on whether each result REPLICATES across
   concepts after the leak-fix and multiple-comparison correction.

Everything traces to the pulled JSONs; this script computes, it does not fabricate. Writes
artifacts_pull/solidification_summary.json alongside the printed report.

Run (after the volume is pulled into a local dir):
    python scripts/aggregate_controls.py --dir artifacts_pull
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

W = 78
N_BOOT = 10_000
ALPHA = 0.05


def _load_optional(path: Path) -> dict | None:
    """Load a JSON file, or print a clear message and return None if it is missing/unreadable."""
    if not path.exists():
        print(f"  [missing] {path}  -  skipping this section")
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [unreadable] {path}: {exc}  -  skipping this section")
        return None


def holm_bonferroni(pvals: list[float], alpha: float = ALPHA) -> list[bool]:
    """Holm-Bonferroni step-down. Returns a survive/reject flag per p-value in ORIGINAL order.

    Sort ascending; the i-th smallest (0-based rank r over m tests) is rejected iff p <= alpha/(m-r)
    AND every smaller p was also rejected. The step-down stops at the first failure.
    """
    m = len(pvals)
    survive = [False] * m
    for rank, idx in enumerate(sorted(range(m), key=lambda i: pvals[i])):
        if pvals[idx] <= alpha / (m - rank):
            survive[idx] = True
        else:
            break
    return survive


def sign_test_two_sided(k: int, n: int) -> float:
    """Exact two-sided sign test p-value: k successes of n nonzero trials under Binomial(n, 0.5)."""
    if n == 0:
        return 1.0
    probs = [comb(n, j) * 0.5 ** n for j in range(n + 1)]
    pk = probs[k]
    return float(min(1.0, sum(p for p in probs if p <= pk * (1.0 + 1e-9))))


def paired_bootstrap_gap(real_correct: np.ndarray, random_correct: np.ndarray,
                         seed: int, n_boot: int = N_BOOT) -> tuple[float, float, float, float]:
    """Paired bootstrap of mean(real_correct) - mean(random_correct).

    ONE shared resample index per iteration (the comparison is paired: same test examples both arms).
    Returns (point_gap, ci_lo, ci_hi, p_one_sided) where p_one_sided = fraction of boot gaps <= 0.
    """
    n = real_correct.size
    if n == 0 or random_correct.size != n:
        raise ValueError("empty or mismatched correctness arrays; cannot pair-bootstrap")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = real_correct[idx].mean(axis=1) - random_correct[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    point = float(real_correct.mean() - random_correct.mean())
    return point, float(lo), float(hi), float(np.mean(boot <= 0.0))


def _label(classes: list) -> str:
    return "/".join(str(c) for c in classes)


def analyse_circuit(data: dict, primary_k: int) -> dict:
    """Per-K cross-concept circuit aggregate, with Holm over the per-concept permutation p-values."""
    per_concept = data.get("per_concept", [])
    n = len(per_concept)
    ks = [row["K"] for row in per_concept[0]["by_K"]] if n else []
    by_k: dict[str, dict] = {}
    for K in ks:
        rows = []
        for c in per_concept:
            r = next(x for x in c["by_K"] if x["K"] == K)
            rows.append({
                "concept": _label(c["classes"]),
                "faithfulness_holdout": r["faithfulness_holdout"],
                "faithfulness_fullset": r["faithfulness_fullset"],
                "leak_delta": round(r["faithfulness_fullset"] - r["faithfulness_holdout"], 3),
                "gap_holdout_minus_random": r["gap_holdout_minus_random"],
                "gap_ci95_paired": r["gap_ci95_paired"],
                "permutation_p": r["permutation_p"],
                "circuit_beats_random": bool(r["circuit_beats_random"]),
            })
        survive = holm_bonferroni([row["permutation_p"] for row in rows])
        for row, s in zip(rows, survive):
            row["holm_survives"] = bool(s)
        faiths = [row["faithfulness_holdout"] for row in rows]
        by_k[str(K)] = {
            "K": K, "n_concepts": n, "per_concept": rows,
            "mean_faithfulness_holdout": round(float(np.mean(faiths)), 3),
            "min_faithfulness_holdout": round(float(np.min(faiths)), 3),
            "mean_gap": round(float(np.mean([row["gap_holdout_minus_random"] for row in rows])), 4),
            "n_beating_random": int(sum(row["circuit_beats_random"] for row in rows)),
            "n_holm_survivors": int(sum(survive)),
            "holm_survivors": [row["concept"] for row in rows if row["holm_survives"]],
        }
    return {"n_concepts": n, "ks": ks, "primary_k": primary_k, "by_k": by_k}


def analyse_control(real: dict, random: dict, seed: int) -> dict:
    """Paired real-vs-random aggregate, with sign test and Holm over the per-concept one-sided p's."""
    rc, ac = real.get("per_concept", []), random.get("per_concept", [])
    if len(rc) != len(ac):
        print(f"  [warn] concept count differs (real={len(rc)}, random={len(ac)}); zipping shorter")
    rows = []
    for r, a in zip(rc, ac):
        rco = (np.asarray(r["yte"], int) == np.asarray(r["pred"], int)).astype(float)
        aco = (np.asarray(a["yte"], int) == np.asarray(a["pred"], int)).astype(float)
        point, lo, hi, p1 = paired_bootstrap_gap(rco, aco, seed)
        rows.append({
            "concept": _label(r["classes"]),
            "real_acc": r["sae_probe_acc"], "random_acc": a["sae_probe_acc"],
            "gap": round(r["sae_probe_acc"] - a["sae_probe_acc"], 4),
            "gap_boot_point": round(point, 4), "gap_ci95_paired": [round(lo, 4), round(hi, 4)],
            "p_one_sided_boot": round(p1, 4), "ci_excludes_zero": bool(lo > 0 or hi < 0),
        })
    survive = holm_bonferroni([row["p_one_sided_boot"] for row in rows])
    for row, s in zip(rows, survive):
        row["holm_survives"] = bool(s)
    n = len(rows)
    n_pos = sum(1 for row in rows if row["gap"] > 0)
    n_nonzero = sum(1 for row in rows if row["gap"] != 0)
    return {
        "n_concepts": n, "per_concept": rows,
        "mean_gap": round(float(np.mean([row["gap"] for row in rows])), 4) if n else 0.0,
        "n_ci_excludes_zero": int(sum(row["ci_excludes_zero"] for row in rows)),
        "n_positive": n_pos,
        "sign_test_p_two_sided": round(sign_test_two_sided(n_pos, n_nonzero), 4),
        "n_holm_survivors": int(sum(survive)),
        "holm_survivors": [row["concept"] for row in rows if row["holm_survives"]],
    }


def _verdict(circuit: dict | None, control: dict | None) -> dict:
    """R4-style reproduced/novel/inconclusive language for replication across concepts."""
    out: dict = {}
    if circuit and str(circuit["primary_k"]) in circuit["by_k"]:
        a = circuit["by_k"][str(circuit["primary_k"])]
        n = a["n_concepts"]
        if a["n_beating_random"] == n and a["n_holm_survivors"] == n:
            v = (f"REPLICATES across all {n} concepts (conclusive): at K={a['K']} every concept's "
                 f"sparse SAE circuit beats random and survives Holm; min holdout faithfulness "
                 f"{a['min_faithfulness_holdout']}. The leak-fix lowers but does not erase the effect.")
        elif a["n_holm_survivors"] > 0:
            v = (f"PARTIALLY replicates ({a['n_holm_survivors']}/{n} survive Holm, "
                 f"{a['n_beating_random']}/{n} beat random at K={a['K']}). Real but not uniform.")
        else:
            v = f"INCONCLUSIVE: 0/{n} concepts survive Holm at K={a['K']} after the leak-fix."
        out["circuit"] = v
    if control:
        n = control["n_concepts"]
        if (control["n_ci_excludes_zero"] == n and control["n_holm_survivors"] == n
                and control["sign_test_p_two_sided"] < ALPHA):
            v = (f"REPLICATES across all {n} concepts (conclusive): real-model SAE > randomized-model "
                 f"SAE in every concept (mean gap {control['mean_gap']}, all paired CIs exclude 0, "
                 f"sign-test p={control['sign_test_p_two_sided']}, all survive Holm).")
        elif control["n_holm_survivors"] > 0 or control["n_ci_excludes_zero"] > 0:
            v = (f"PARTIALLY replicates ({control['n_ci_excludes_zero']}/{n} CIs exclude 0, "
                 f"{control['n_holm_survivors']}/{n} survive Holm; sign-test "
                 f"p={control['sign_test_p_two_sided']}).")
        else:
            v = (f"INCONCLUSIVE: real>random does not hold across concepts "
                 f"(0 CIs exclude 0; sign-test p={control['sign_test_p_two_sided']}).")
        out["control"] = v
    return out


def _print_circuit(circuit: dict) -> None:
    print("\n" + "=" * W)
    print(f"A. CIRCUIT  -  {circuit['n_concepts']} concepts, K present = {circuit['ks']}, "
          f"primary K = {circuit['primary_k']}")
    print("=" * W)
    pk = str(circuit["primary_k"])
    if pk in circuit["by_k"]:
        print(f"\n  Primary K={circuit['primary_k']} per concept "
              f"(faithfulness vs ceiling; leak_delta = fullset - holdout):")
        for r in circuit["by_k"][pk]["per_concept"]:
            print(f"    {r['concept']:>9s}: faith hold={r['faithfulness_holdout']:.3f} "
                  f"full={r['faithfulness_fullset']:.3f} (leak {r['leak_delta']:+.3f})  "
                  f"gap={r['gap_holdout_minus_random']:+.4f} "
                  f"CI[{r['gap_ci95_paired'][0]:+.4f},{r['gap_ci95_paired'][1]:+.4f}]  "
                  f"perm_p={r['permutation_p']:.4f}  beats={str(r['circuit_beats_random']):5s}  "
                  f"Holm={'survive' if r['holm_survives'] else 'no'}")
    else:
        print(f"\n  [note] primary K={circuit['primary_k']} not in {circuit['ks']}; "
              f"per-concept detail skipped, all-K summary still shown.")
    print("\n  All-K cross-concept summary:")
    for K in circuit["ks"]:
        a = circuit["by_k"][str(K)]
        print(f"    K={a['K']:<3d} faith mean={a['mean_faithfulness_holdout']:.3f} "
              f"min={a['min_faithfulness_holdout']:.3f}  gap mean={a['mean_gap']:+.4f}  "
              f"beats_random={a['n_beating_random']}/{a['n_concepts']}  "
              f"Holm survivors={a['n_holm_survivors']}/{a['n_concepts']} {a['holm_survivors']}")


def _print_control(control: dict) -> None:
    print("\n" + "=" * W)
    print(f"B. CONTROL real vs random  -  {control['n_concepts']} concepts, "
          f"paired bootstrap (seed-driven, {N_BOOT} iters)")
    print("=" * W)
    for r in control["per_concept"]:
        print(f"    {r['concept']:>9s}: real={r['real_acc']:.4f} random={r['random_acc']:.4f}  "
              f"gap={r['gap']:+.4f} CI[{r['gap_ci95_paired'][0]:+.4f},{r['gap_ci95_paired'][1]:+.4f}]  "
              f"p1={r['p_one_sided_boot']:.4f}  CI_excl_0={str(r['ci_excludes_zero']):5s}  "
              f"Holm={'survive' if r['holm_survives'] else 'no'}")
    print(f"\n  Across concepts: mean gap={control['mean_gap']:+.4f}  "
          f"CIs excluding 0={control['n_ci_excludes_zero']}/{control['n_concepts']}  "
          f"positive={control['n_positive']}/{control['n_concepts']}  "
          f"sign-test p={control['sign_test_p_two_sided']:.4f}")
    print(f"  Holm survivors={control['n_holm_survivors']}/{control['n_concepts']} "
          f"{control['holm_survivors']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="artifacts_pull", help="dir with the pulled *_multi*.json")
    ap.add_argument("--primary-k", type=int, default=10, help="pre-registered primary K (default 10)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed (default 0)")
    args = ap.parse_args()
    d = Path(args.dir)

    print("=" * W)
    print("SOLIDIFICATION SUMMARY  -  multi-concept circuit + real-vs-random control")
    print(f"dir={d}  primary_k={args.primary_k}  seed={args.seed}  (numbers from pulled JSONs)")
    print("=" * W)

    summary: dict = {"dir": str(d), "primary_k": args.primary_k, "seed": args.seed}

    circuit_data = _load_optional(d / "circuit_multi.json")
    circuit = analyse_circuit(circuit_data, args.primary_k) if circuit_data else None
    if circuit:
        _print_circuit(circuit)
        summary["circuit"] = circuit

    real = _load_optional(d / "probing_multi_real.json")
    random = _load_optional(d / "probing_multi_random.json")
    control = analyse_control(real, random, args.seed) if (real and random) else None
    if control:
        _print_control(control)
        summary["control"] = control

    verdict = _verdict(circuit, control)
    summary["verdict"] = verdict
    print("\n" + "=" * W)
    print("C. VERDICT (R4: reproduced / novel / inconclusive, replication across concepts)")
    print("=" * W)
    if "circuit" in verdict:
        print(f"  Sparse circuit : {verdict['circuit']}")
    if "control" in verdict:
        print(f"  real > random  : {verdict['control']}")
    if not verdict:
        print("  Nothing to judge; both sections were skipped (see [missing] notes above).")

    out_path = d / "solidification_summary.json"
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote {out_path}")
    except OSError as exc:
        print(f"\n[warn] could not write {out_path}: {exc}")


if __name__ == "__main__":
    main()
