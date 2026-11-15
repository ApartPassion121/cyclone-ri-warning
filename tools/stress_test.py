#!/usr/bin/env python3
"""Stress test for the shipped rapid-intensification model.

Rerun this after any retrain. It needs no dependencies beyond the standard
library and the JSON files that sit next to it:

    python3 stress_test.py                 # report + compare to the baseline
    python3 stress_test.py --json out.json # also write the machine-readable summary
    python3 stress_test.py --update-baseline   # accept the current numbers as the new baseline

What it checks
  1. single-input sweeps  - does one input move RI the way physics says it should
  2. intensity            - wind and pressure moved together (a consistent storm)
  3. latitude edge        - the known extrapolation below the 5-40 N training corridor
  4. custom storms        - hand-built cyclones and impossible pairings
  5. numerical extremes   - no NaN, probabilities always sum to 100
  6. full dataset         - class mix and the alarm rate on every cell x month default row

The alarm rate (share of default rows that cross the warning threshold) is
tracked against stress_baseline.json. If a retrain moves it by more than the
recorded tolerance the run fails, so a threshold or calibration change cannot
slip through unnoticed.

Exit status: 0 all checks pass, 1 a check failed or the alarm rate drifted.
"""

import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The exported model data lives at the repo root; the baseline sits beside this file.
ROOT = HERE if os.path.exists(os.path.join(HERE, "model_weights.json")) else os.path.dirname(HERE)
CORRIDOR = (5.0, 40.0)          # training corridor latitude bounds (README: 5-40 N, 110-145 E)
DEFAULT_TOLERANCE_PP = 3.0      # allowed drift of the alarm rate, in percentage points

# A benchmark storm: mid-intensity, central West Pacific, September, typical trends.
BASE = dict(WIND=65, PRES=990, DWIND_24=5, DWIND_12=5, DWIND_6=2, DPRES_6=-2,
            DPRES_24=-5, LAT=16.25, LON=128.75, DIST2LAND=300, MONTH=9, SPEED=10, SST=29.0)


# --------------------------------------------------------------------------- model
def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def predict(w, row):
    """Softmax of the exported linear model. Returns (probabilities, predicted class)."""
    x = []
    for i, f in enumerate(w["features"]):
        v = row.get(f)
        x.append(None if v is None else (float(v) - w["scaler_mean"][i]) / w["scaler_scale"][i])
    z = []
    for ci in range(len(w["classes"])):
        s = w["intercept"][ci]
        for j, xi in enumerate(x):
            if xi is not None:
                s += xi * w["coef"][ci][j]
        z.append(s)
    m = max(z)
    e = [math.exp(t - m) for t in z]
    total = sum(e)
    p = {c: 100.0 * e[i] / total for i, c in enumerate(w["classes"])}
    return p, w["classes"][z.index(m)]


def ri(w, row):
    return predict(w, row)[0]["RI"]


def changed(row, **kw):
    out = dict(row)
    out.update(kw)
    return out


def sweep(w, field, values):
    return [(v, ri(w, changed(BASE, **{field: v}))) for v in values]


def counter_expected(pairs, direction):
    bad = 0
    for (_, a), (_, b) in zip(pairs, pairs[1:]):
        if direction == "up" and b < a:
            bad += 1
        if direction == "down" and b > a:
            bad += 1
    return bad


# --------------------------------------------------------------------------- checks
SWEEPS = [
    # field, values, direction it must move, one-line justification
    # pressure rises across the list, so RI must fall with it: the weaker storm has the room
    ("PRES", [1015, 1005, 990, 975, 940, 905], "down", "weaker storm (higher pressure) has more room to intensify"),
    ("DWIND_6", [-20, -10, 0, 10, 25], "up", "recent strengthening is momentum"),
    ("DPRES_6", [8, 2, -2, -8, -15], "up", "falling pressure is momentum"),
    ("SST", [24, 26, 28, 29, 30, 31], "up", "warm water fuels intensification"),
    ("DIST2LAND", [0, 50, 150, 300, 800, 1500], "up", "offshore storms are not disrupted by land"),
    ("MONTH", [1, 3, 5, 7, 9, 11, 12], "up", "season term, linear by construction"),
]

CONSISTENT = [(25, 1004), (45, 996), (65, 985), (85, 970), (105, 952), (125, 930), (145, 905)]

CUSTOM = [
    ("Baby monsoon depression", "25 kt, 1004 hPa, +5/24h, 30C, 10N, Aug",
     dict(WIND=25, PRES=1004, DWIND_24=5, DWIND_12=3, DWIND_6=1, DPRES_6=-1, DPRES_24=-3,
          LAT=10, LON=133, DIST2LAND=800, MONTH=8, SPEED=5, SST=30)),
    ("Rapid intensifier", "45 kt, 998 hPa, +25/24h, 30.5C, 13N, Oct",
     dict(WIND=45, PRES=998, DWIND_24=25, DWIND_12=14, DWIND_6=8, DPRES_6=-6, DPRES_24=-14,
          LAT=13, LON=131, DIST2LAND=700, MONTH=10, SPEED=9, SST=30.5)),
    ("Already a monster", "150 kt, 905 hPa, +5/24h, 18N, Sep",
     dict(WIND=150, PRES=905, DWIND_24=5, DWIND_12=3, DWIND_6=2, DPRES_6=-2, DPRES_24=-4,
          LAT=18, LON=128, DIST2LAND=500, MONTH=9, SPEED=11, SST=30)),
    ("Cold-water wanderer", "55 kt, 995 hPa, flat, 22C, 33N, Feb",
     dict(WIND=55, PRES=995, DWIND_24=0, DWIND_12=0, DWIND_6=0, DPRES_6=0, DPRES_24=0,
          LAT=33, LON=140, DIST2LAND=900, MONTH=2, SPEED=16, SST=22)),
    ("Coastal grinder", "70 kt, 985 hPa, +2/24h, 15 km from land, 20N, Sep",
     dict(WIND=70, PRES=985, DWIND_24=2, DWIND_12=1, DWIND_6=0, DPRES_6=-1, DPRES_24=-2,
          LAT=20, LON=121, DIST2LAND=15, MONTH=9, SPEED=8, SST=28)),
    ("Equatorial system", "40 kt at 3N, warm water, open ocean - no Coriolis",
     dict(WIND=40, PRES=1000, DWIND_24=8, DWIND_12=5, DWIND_6=3, DPRES_6=-3, DPRES_24=-5,
          LAT=3, LON=130, DIST2LAND=900, MONTH=7, SPEED=7, SST=29.5)),
    ("Impossible A", "20 kt with 940 hPa - a pairing that cannot exist",
     changed(BASE, WIND=20, PRES=940)),
    ("Impossible B", "150 kt with 1010 hPa - a pairing that cannot exist",
     changed(BASE, WIND=150, PRES=1010)),
]

EXTREMES = [
    ("WIND 0", changed(BASE, WIND=0)), ("WIND 500", changed(BASE, WIND=500)),
    ("PRES 0", changed(BASE, PRES=0)), ("PRES 2000", changed(BASE, PRES=2000)),
    ("SST 0", changed(BASE, SST=0)), ("SST 60", changed(BASE, SST=60)),
    ("LAT 0", changed(BASE, LAT=0)), ("LAT 90", changed(BASE, LAT=90)),
    ("DIST2LAND -100", changed(BASE, DIST2LAND=-100)),
    ("MONTH 0", changed(BASE, MONTH=0)), ("MONTH 13", changed(BASE, MONTH=13)),
    ("SPEED -5", changed(BASE, SPEED=-5)),
    ("all zeroes", {f: 0 for f in ["WIND", "PRES", "DWIND_24", "DWIND_12", "DWIND_6",
                                   "DPRES_6", "DPRES_24", "LAT", "LON", "DIST2LAND",
                                   "MONTH", "SPEED", "SST"]}),
    ("all missing", {}),
]


def run(w, defs, stats, args):
    threshold = float(w.get("warning_threshold", 0.25))
    failures, warnings = [], []
    say = (lambda *a: None) if args.quiet else print

    say(f"RI model stress test  ·  {os.path.basename(args.weights)}  "
        f"sha256 {hashlib.sha256(open(args.weights, 'rb').read()).hexdigest()[:12]}")
    say(f"corridor {CORRIDOR[0]:g}-{CORRIDOR[1]:g} N  ·  warning threshold {threshold:.2f}")
    say("")

    # 1. single-input sweeps -------------------------------------------------
    say("1. single-input sweeps")
    for field, values, direction, why in SWEEPS:
        pairs = sweep(w, field, values)
        bad = counter_expected(pairs, direction)
        line = f"   {field:<11} {values[0]:>6g} -> {values[-1]:<7g} RI {pairs[0][1]:5.1f} -> {pairs[-1][1]:5.1f}%"
        status = "ok" if bad == 0 else f"{bad} COUNTER-EXPECTED"
        if bad:
            failures.append(f"{field}: {bad} step(s) move against '{why}'")
        say(f"{line}   {status}")
    say("")

    # 2. consistent intensity ------------------------------------------------
    say("2. intensity, wind and pressure moved together")
    prof = []
    for wind, pres in CONSISTENT:
        row = changed(BASE, WIND=wind, PRES=pres)
        prof.append((f"{wind}/{pres}", ri(w, row)))
    say("   " + "   ".join(f"{k}kt:{v:.1f}%" for k, v in prof))
    peak = max(range(len(prof)), key=lambda i: prof[i][1])
    if 0 < peak < len(prof) - 1:
        say(f"   peak at {CONSISTENT[peak][0]} kt, {prof[peak][1]:.1f}%   ok")
    else:
        say("   peak at an end of the range - expected a hump in the middle")
        warnings.append("intensity response has no interior peak")
    if prof[-1][1] > 30:
        failures.append("a 145 kt storm still scores above 30% RI")

    # 3. latitude edge -------------------------------------------------------
    say("3. latitude edge (page refuses below 5 N; the exported weights still extrapolate)")
    for lat in (0, 3, 5, 12, 30):
        row = changed(BASE, WIND=50, PRES=1000, DWIND_24=10, DWIND_6=3, DPRES_6=-3, LAT=lat)
        note = "" if CORRIDOR[0] <= lat <= CORRIDOR[1] else "   <- outside corridor, UI refuses"
        say(f"   {lat:>2}N  RI {ri(w, row):5.1f}%{note}")
    say("")

    # 4. custom storms -------------------------------------------------------
    say("4. custom storms")
    for name, note, row in CUSTOM:
        p, cls = predict(w, row)
        conds = sum([row.get("SST", 0) >= 26.5, abs(row.get("LAT", 0)) >= 5, row.get("WIND", 0) >= 34])
        say(f"   {name:<24} RI {p['RI']:5.1f}  WK {p['WEAKEN']:5.1f}  {cls:<7} conds {conds}/3   {note}")
    say("")

    # 5. numerical extremes --------------------------------------------------
    say("5. numerical extremes")
    bad_extremes = 0
    for name, row in EXTREMES:
        p, _ = predict(w, row)
        finite = all(math.isfinite(v) for v in p.values())
        total = sum(p.values())
        if not finite or abs(total - 100.0) > 0.2:
            bad_extremes += 1
            failures.append(f"extreme '{name}' -> non-finite or sum {total:.2f}")
            say(f"   {name:<16} *** FAIL *** {p}")
    say(f"   {len(EXTREMES) - bad_extremes}/{len(EXTREMES)} finite with probabilities summing to 100   "
        + ("ok" if bad_extremes == 0 else "FAILED"))
    say("")

    # 6. full dataset --------------------------------------------------------
    rows = []
    for key, d in defs.items():
        parts = key.split(",")
        row = dict(d)
        row["MONTH"] = int(parts[2])
        rows.append(row)
    preds = [predict(w, r) for r in rows]
    mix = {c: 0 for c in w["classes"]}
    for _, cls in preds:
        mix[cls] += 1
    ris = sorted(p["RI"] for p, _ in preds)
    fired = sum(1 for v in ris if v >= threshold * 100.0)
    alarm = 100.0 * fired / len(ris)
    base_num = sum(s["ri_pct"] * s["points"] for s in stats.values())
    base_den = sum(s["points"] for s in stats.values())
    base_rate = base_num / base_den if base_den else 0.0

    def pct(q):
        return ris[min(len(ris) - 1, int(len(ris) * q))]

    say("6. full dataset")
    say(f"   {len(rows)} cell x month default rows · class mix "
        + " / ".join(f"{c} {mix[c]}" for c in w["classes"]))
    say(f"   RI p50 {pct(0.50):.1f}%  p95 {pct(0.95):.1f}%  max {ris[-1]:.1f}%")
    say(f"   RI WATCH {fired}/{len(rows)} = {alarm:.1f}%   "
        f"observed base rate {base_rate:.1f}%   ratio {alarm / base_rate:.1f}x")
    say("")

    summary = {
        "weights_sha256": hashlib.sha256(open(args.weights, "rb").read()).hexdigest(),
        "warning_threshold": threshold,
        "rows": len(rows),
        "class_mix": mix,
        "ri_percentiles": {"p50": round(pct(0.50), 1), "p95": round(pct(0.95), 1), "max": round(ris[-1], 1)},
        "alarm_rate_pct": round(alarm, 2),
        "base_rate_pct": round(base_rate, 2),
    }

    # baseline comparison ----------------------------------------------------
    base_path = args.baseline
    if os.path.exists(base_path) and not args.update_baseline:
        base = load_json(base_path)
        tol = float(base.get("tolerance_pp", DEFAULT_TOLERANCE_PP))
        delta = alarm - float(base["alarm_rate_pct"])
        ok = abs(delta) <= tol
        say(f"alarm rate vs baseline {base['alarm_rate_pct']:.1f}% "
            f"(tolerance {tol:.1f} pp): {delta:+.1f} pp   " + ("ok" if ok else "DRIFTED"))
        summary["baseline_delta_pp"] = round(delta, 2)
        if not ok:
            failures.append(
                f"alarm rate drifted {delta:+.1f} pp from baseline {base['alarm_rate_pct']:.1f}% "
                f"(tolerance {tol:.1f} pp); rerun with --update-baseline if this is intended")
    elif args.update_baseline:
        base = dict(summary, tolerance_pp=float(args.tolerance))
        with open(base_path, "w", encoding="utf-8") as fh:
            json.dump(base, fh, indent=2, sort_keys=True)
            fh.write("\n")
        say(f"baseline written to {os.path.basename(base_path)}")
    else:
        say(f"no baseline at {os.path.basename(base_path)}; run with --update-baseline to create one")

    for warn in warnings:
        say(f"warning: {warn}")
    say("")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")      # printed even under --quiet: silence must never hide a failure
        print("RESULT: FAIL")
    else:
        print("RESULT: PASS")
    return summary, failures


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stress-test the shipped RI model and track its alarm rate.")
    ap.add_argument("--weights", default=os.path.join(ROOT, "model_weights.json"))
    ap.add_argument("--defaults", default=os.path.join(ROOT, "region_defaults.json"))
    ap.add_argument("--stats", default=os.path.join(ROOT, "region_stats.json"))
    ap.add_argument("--baseline", default=os.path.join(HERE, "stress_baseline.json"))
    ap.add_argument("--update-baseline", action="store_true",
                    help="write the current numbers to the baseline file and exit 0")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_PP,
                    help=f"alarm-rate drift allowed, in percentage points (default {DEFAULT_TOLERANCE_PP})")
    ap.add_argument("--json", metavar="PATH", help="also write the summary as JSON")
    ap.add_argument("--quiet", action="store_true", help="only print the result lines")
    args = ap.parse_args(argv)

    for path in (args.weights, args.defaults, args.stats):
        if not os.path.exists(path):
            sys.exit(f"missing {path} - run this from the project folder")

    summary, failures = run(load_json(args.weights), load_json(args.defaults), load_json(args.stats), args)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
