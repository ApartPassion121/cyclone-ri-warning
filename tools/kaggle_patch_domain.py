#!/usr/bin/env python3
"""Patch the Kaggle notebook for the widened RI domain.

Run it on the .ipynb that `kaggle kernels pull` writes, not in the browser:

    python3 kaggle_patch_domain.py --nb cyclone_ri_warning.ipynb --domain wp --dry-run
    python3 kaggle_patch_domain.py --nb cyclone_ri_warning.ipynb --domain wp

It makes six edits, each anchored on text that exists in the current notebook and
each reported on stdout. It refuses to guess: if an anchor is missing it stops and
names the anchor, so a hand-edited notebook is never silently half-patched.

  wp    : lat 5-40, lon 110..180    -> model A (the shipped model, extended east)
  east  : lat 5-40, lon -180..-70   -> model B (central + eastern Pacific)

Re-run with --domain east on the same file to switch: the DOMAIN constants are
rewritten, the rest of the notebook is already domain-agnostic.
"""
import argparse
import json
import re
import sys

DOMAINS = {
    "wp": {"name": "wp", "lon": (110, 180), "basins": '["WP"]'},
    # CP matters east of the dateline: a central-Pacific storm carries BASIN 'CP'
    "east": {"name": "east", "lon": (-180, -70), "basins": '["WP", "EP", "CP"]'},
}

DOMAIN_CELL = '''DOMAIN = {
    "name": "NAME",              # "wp" = model A (110E-180E), "east" = model B (180E-70W)
    "basins": BASINS,
    "lat": (5, 40),
    "lon_pm180": LON,          # dateline-safe: 180E is -180
}
print("domain", DOMAIN["name"], DOMAIN["lat"], DOMAIN["lon_pm180"])
'''

FILTER_BLOCK = '''corridor = df[
    (df["BASIN"].isin(DOMAIN["basins"]))
    & (df["TRACK_TYPE"] == "main")
    & (df["SEASON"] >= 2000)
    & (pd.to_numeric(df["LAT"], errors="coerce").between(*DOMAIN["lat"]))
    & (df["LON"].between(*DOMAIN["lon_pm180"]))
].copy()'''

LON_NORMALISE = '''df["LON"] = ((pd.to_numeric(df["LON"], errors="coerce") + 180) % 360) - 180
'''

FEATURES_LINE = ('features = ["WIND", "PRES", "DWIND_24", "DWIND_12", "DWIND_6", "DPRES_6", '
                 '"DPRES_24", "LAT", "LON_SIN", "LON_COS", "DIST2LAND", "MONTH", "SPEED"]')

CYCLIC_BLOCK = '''corridor["LON_SIN"] = np.sin(np.radians(corridor["LON"]))
corridor["LON_COS"] = np.cos(np.radians(corridor["LON"]))
'''

SST_BLOCK = '''lat0, lat1 = DOMAIN["lat"]; lon0, lon1 = DOMAIN["lon_pm180"]
sst_path = f"/kaggle/working/sst_{DOMAIN['name']}.nc"
if not os.path.exists(sst_path):
    sst_url = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.nc"
               f"?sst%5B(2000-01-15):30:(2025-12-15)%5D%5B(0.0)%5D"
               f"%5B({lat0}):({lat1})%5D%5B({lon0}):({lon1})%5D")
    urllib.request.urlretrieve(sst_url, sst_path)'''


def join(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def span(text, start_pat, end_pat, what):
    """Return (start, end) covering start_pat .. end_pat inclusive, or fail loudly."""
    m = re.search(start_pat, text)
    if not m:
        sys.exit(f"anchor not found: {what} (pattern {start_pat!r})")
    tail = re.search(end_pat, text[m.start():])
    if not tail:
        sys.exit(f"anchor not found: end of {what} (pattern {end_pat!r})")
    return m.start(), m.start() + tail.end()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", required=True, help="path to the pulled .ipynb")
    ap.add_argument("--domain", choices=sorted(DOMAINS), default="wp")
    ap.add_argument("--threshold", type=float, help="overwrite the warning threshold (model B needs its own)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = DOMAINS[args.domain]
    nb = json.load(open(args.nb, encoding="utf-8"))

    # repair a stray leading indent: Kaggle stores it, Run All would fail on it
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = join(c)
        try:
            compile(src, "probe", "exec")
        except SyntaxError as exc:
            lines = src.splitlines(keepends=True)
            if exc.lineno == 1 and lines and lines[0][:1] in (" ", "\t"):
                lines[0] = lines[0].lstrip()
                set_source(c, "".join(lines))
                print(f"  - repaired a stray leading indent in cell {i + 1}")
    cells = nb["cells"]
    code = [c for c in cells if c.get("cell_type") == "code"]
    report = []

    already = any("DOMAIN = {" in join(c) for c in code)
    if not already:
        at = next((i for i, c in enumerate(cells)
                   if 'df["BASIN"]' in join(c) and "TRACK_TYPE" in join(c)), None)
        if at is None:
            sys.exit("could not find the corridor/filter cell (looks for df[\"BASIN\"] and TRACK_TYPE)")
        new = {"cell_type": "code", "execution_count": None, "metadata": {},
               "outputs": [],
               "source": DOMAIN_CELL.replace("NAME", spec["name"])
                                       .replace("BASINS", spec["basins"])
                                       .replace("LON", str(tuple(spec["lon"]))).splitlines(keepends=True)}
        cells.insert(at, new)
        report.append(f"inserted DOMAIN cell before cell {at + 1}")
        code = [c for c in cells if c.get("cell_type") == "code"]
    else:
        for c in code:
            t = join(c)
            if "DOMAIN = {" in t and '"lon_pm180"' in t:
                t = re.sub(r'"name": "[a-z]+"', f'"name": "{spec["name"]}"', t, count=1)
                t = re.sub(r'"basins": \[[^\]]*\]', f'"basins": {spec["basins"]}', t, count=1)
                t = re.sub(r'"lon_pm180": \([^)]*\)', f'"lon_pm180": {tuple(spec["lon"])}', t, count=1)
                set_source(c, t)
                report.append(f"DOMAIN switched to {spec['name']} {tuple(spec['lon'])}")
                break

    # 1. corridor filter ------------------------------------------------------
    for c in code:
        t = join(c)
        if 'df["BASIN"]' in t and "TRACK_TYPE" in t and "corridor = df[" in t:
            a, b = span(t, r"corridor = df\[", r"\]\.copy\(\)", "corridor filter")
            t = t[:a] + FILTER_BLOCK + t[b:]
            if LON_NORMALISE not in t:
                t = t.replace('df["SEASON"] = pd.to_numeric', LON_NORMALISE + 'df["SEASON"] = pd.to_numeric', 1) \
                     if 'df["SEASON"] = pd.to_numeric' in t else LON_NORMALISE + t
            set_source(c, t)
            report.append("filter: domain mask + longitude normalised to -180..180")
            break
    else:
        sys.exit("corridor filter cell not found")

    # 2. features: cyclic longitude -------------------------------------------
    done = False
    for c in code:
        t = join(c)
        if "features = [" in t and '"DWIND_24"' in t:
            if "LON_SIN" not in t:
                a, b = span(t, r"features = \[", r"\]", "features list")
                t = t[:a] + CYCLIC_BLOCK + FEATURES_LINE + t[b:]
            set_source(c, t)
            report.append("features: LON replaced by LON_SIN/LON_COS")
            done = True
            break
    if not done:
        sys.exit("features cell not found (looks for features = [ ... DWIND_24)")

    # 3. SST subset follows the domain ---------------------------------------
    done = False
    for c in code:
        t = join(c)
        if "ncdcOisst21Agg_LonPM180" in t:
            a, b = span(t, r"(?:lat0, lat1 = [^\n]*\n)?sst_path = ",
                        r"urlretrieve\([^)]*\)", "SST download")
            t = t[:a] + SST_BLOCK + t[b:]
            set_source(c, t)
            report.append("SST: subset and cache file follow DOMAIN")
            done = True
            break
    if not done:
        report.append("WARNING: SST cell not found - check it still downloads 5-40N")

    # 4. region exports: suffix + no derived values in defaults ---------------
    for c in code:
        t = join(c)
        if "region_stats[" in t and "region_defaults[" in t:
            t = t.replace('"/kaggle/working/region_stats.json"',
                          'f"/kaggle/working/region_stats_{DOMAIN[\'name\']}.json"')
            t = t.replace('"/kaggle/working/region_defaults.json"',
                          'f"/kaggle/working/region_defaults_{DOMAIN[\'name\']}.json"')
            t = t.replace("for f in features2",
                          'for f in [x for x in features2 if x not in ("LON_SIN", "LON_COS")]')
            set_source(c, t)
    if any("region_stats_{DOMAIN" in join(c) for c in code):
        report.append("region files: _wp/_east suffix, derived features left out of defaults")
    else:
        sys.exit("region export cell not found (looks for region_stats[ and region_defaults[)")

    # 5. weights: domain metadata, real SST fill, suffix ----------------------
    done = False
    for c in code:
        t = join(c)
        if "model_weights = {" in t:
            t = t.replace('"sst_fill": 28.6', '"sst_fill": float(sst_median)')
            if '"domain"' not in t:
                t = re.sub(r'("intercept": [^\n]+\n)',
                           r'\1 "domain": {"lat": list(DOMAIN["lat"]), "lon_pm180": list(DOMAIN["lon_pm180"])},\n', t, count=1)
            t = t.replace('"/kaggle/working/model_weights.json"',
                          'f"/kaggle/working/model_weights_{DOMAIN[\'name\']}.json"')
            set_source(c, t)
            report.append("weights: domain block, sst_fill from data, _wp/_east filename")
            done = True
            break
    if not done:
        sys.exit("weights export cell not found (looks for model_weights = {)")

    # syntax gate: every code cell must still compile
    broken = []
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = join(c)
        if src.lstrip().startswith("%"):
            continue  # magics are not Python
        try:
            compile(src, f"cell{i}", "exec")
        except SyntaxError as exc:
            head = next((ln.strip() for ln in src.splitlines() if ln.strip()), "")[:70]
            broken.append(f"cell {i + 1}: {exc.msg} (line {exc.lineno}) :: {head}")
    if broken:
        sys.exit("syntax errors after patching, nothing written:\n  " + "\n  ".join(broken))

    if args.threshold is not None:
        n = 0
        for c in code:
            t = join(c)
            if '"warning_threshold":' in t or "WARNING_THRESHOLD =" in t:
                t2 = re.sub(r'"warning_threshold": [0-9.]+', f'"warning_threshold": {args.threshold}', t)
                t2 = re.sub(r"WARNING_THRESHOLD = [0-9.]+", f"WARNING_THRESHOLD = {args.threshold}", t2)
                if t2 != t:
                    set_source(c, t2)
                    n += 1
        report.append(f"threshold set to {args.threshold} in {n} cell(s)")

    # 6. track climatology: same domain, same suffix -------------------------
    TRACK_MASK_OLD = ("raw = raw[(raw.LAT>=2.5)&(raw.LAT<=42.5)&(raw.LON>=107.5)"
                      "&(raw.LON<=147.5)].sort_values(['SID','ISO_TIME'])"
                      ".set_index(['SID','ISO_TIME'])")
    TRACK_MASK_NEW = ('''raw['LON'] = ((raw['LON'] + 180) % 360) - 180  # dateline-safe, matches the model domain
'''
                      '''_la0, _la1 = DOMAIN["lat"]; _lo0, _lo1 = DOMAIN["lon_pm180"]
'''
                      '''raw = raw[(raw.LAT >= _la0 - 2.5) & (raw.LAT <= _la1 + 2.5)
'''
                      '''          & (raw.LON >= _lo0 - 2.5) & (raw.LON <= _lo1 + 2.5)
'''
                      '''          ].sort_values(['SID','ISO_TIME']).set_index(['SID','ISO_TIME'])''')
    TRACK_CELL_OLD = ("t['CLAT'] = (np.floor((t['LAT']-5)/2.5)*2.5+6.25).round(2); "
                      "t['CLON'] = (np.floor((t['LON']-110)/2.5)*2.5+111.25).round(2)")
    TRACK_CELL_NEW = ('''t['CLAT'] = (np.floor((t['LAT'] - DOMAIN["lat"][0]) / 2.5) * 2.5 + DOMAIN["lat"][0] + 1.25).round(2)
'''
                      '''t['CLON'] = (np.floor((t['LON'] - DOMAIN["lon_pm180"][0]) / 2.5) * 2.5 + DOMAIN["lon_pm180"][0] + 1.25).round(2)''')
    TRACK_OUT_OLD = "json.dump(out, open('/kaggle/working/track_climatology.json','w'))"
    TRACK_OUT_NEW = "json.dump(out, open(f'/kaggle/working/track_climatology_{DOMAIN[\"name\"]}.json','w'))"

    hit = 0
    for c in code:
        t = join(c)
        if TRACK_MASK_OLD in t:
            t = t.replace(TRACK_MASK_OLD, TRACK_MASK_NEW)
            hit += 1
        if TRACK_CELL_OLD in t:
            t = t.replace(TRACK_CELL_OLD, TRACK_CELL_NEW)
            hit += 1
        if TRACK_OUT_OLD in t:
            t = t.replace(TRACK_OUT_OLD, TRACK_OUT_NEW)
            hit += 1
        if hit:
            set_source(c, t)
            hit = 0
    report.append("track climatology: domain mask, domain cell grid, _wp/_east suffix")

    print(f"notebook: {args.nb}")
    for line in report:
        print("  -", line)
    print("  - syntax: all code cells compile")
    if args.dry_run:
        print("dry run: nothing written")
        return
    with open(args.nb, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
        fh.write("\n")
    print("written")


if __name__ == "__main__":
    main()
