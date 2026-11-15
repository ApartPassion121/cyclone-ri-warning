#!/usr/bin/env python3
"""Retrain the RI model on one Pacific domain.

    python3 ri_domain_model.py --domain wp   --out exported/
    python3 ri_domain_model.py --domain east --out exported/

Conventions match the shipped model: TRACK_TYPE 'main', SEASON >= 2000, wind and
pressure unified by forward-fill across the agency columns, labels from the exact
+24 h change (RI >= +30 kt, WEAKEN <= -30 kt), logistic regression on scaled
features with balanced class weights, SST from OISST nearest-neighbour.

The one deliberate difference: LON is replaced by LON_SIN/LON_COS. A linear
longitude term is only meaningful over a small box; across 190 degrees (or across
the dateline) it would encode distance-from-110E as if it were a physical driver.
"""
import argparse
import json
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DOMAINS = {
    "wp":   {"basins": ["WP"],       "lat": (5, 40), "lon": (110, 180)},
    "east": {"basins": ["WP", "EP"], "lat": (5, 40), "lon": (-180, -70)},
}

SST_URL = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.nc"
           "?sst%5B(2000-01-15):30:(2025-12-15)%5D%5B(0.0)%5D"
           "%5B({lat0}):({lat1})%5D%5B({lon0}):({lon1})%5D")


def load_corridor(path, dom):
    all_cols = pd.read_csv(path, nrows=0).columns.tolist()
    wind_cols = [c for c in all_cols if c.endswith("_WIND")]
    pres_cols = [c for c in all_cols if c.endswith("_PRES")]
    keep = (["SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "LAT", "LON",
             "TRACK_TYPE", "DIST2LAND"] + wind_cols + pres_cols)
    df = pd.read_csv(path, skiprows=[1], usecols=keep, keep_default_na=False,
                     low_memory=False)

    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    # dateline-safe: 180E and 180W become the same number, so a domain that
    # crosses it is one contiguous interval
    df["LON"] = ((pd.to_numeric(df["LON"], errors="coerce") + 180) % 360) - 180

    lat0, lat1 = dom["lat"]
    lon0, lon1 = dom["lon"]
    corridor = df[
        df["BASIN"].isin(dom["basins"])
        & (df["TRACK_TYPE"] == "main")
        & (df["SEASON"] >= 2000)
        & df["LAT"].between(lat0, lat1)
        & df["LON"].between(lon0, lon1)
    ].copy()

    for c in wind_cols + pres_cols:
        corridor[c] = pd.to_numeric(corridor[c], errors="coerce")
    corridor["WIND"] = corridor[wind_cols].bfill(axis=1).iloc[:, 0]
    corridor["PRES"] = corridor[pres_cols].bfill(axis=1).iloc[:, 0]
    corridor["DIST2LAND"] = pd.to_numeric(corridor["DIST2LAND"], errors="coerce")
    corridor["TIME"] = pd.to_datetime(corridor["ISO_TIME"])
    corridor["MONTH"] = corridor["TIME"].dt.month
    return corridor.sort_values(["SID", "TIME"]).reset_index(drop=True)


def add_labels(corridor):
    base = corridor[["SID", "TIME", "WIND", "PRES", "LAT", "LON"]]
    for hours, tag in [(24, "_p24"), (-24, "_m24"), (-12, "_m12"), (-6, "_m6")]:
        t = base.copy()
        t["TIME"] = t["TIME"] + pd.Timedelta(hours=hours)
        t = t.rename(columns={"WIND": "WIND" + tag, "PRES": "PRES" + tag,
                              "LAT": "LAT" + tag, "LON": "LON" + tag})
        corridor = corridor.merge(t[["SID", "TIME", "WIND" + tag, "PRES" + tag,
                                     "LAT" + tag, "LON" + tag]],
                                  on=["SID", "TIME"], how="left")

    corridor["DWIND_24"] = corridor["WIND"] - corridor["WIND_m24"]
    corridor["DWIND_12"] = corridor["WIND"] - corridor["WIND_m12"]
    corridor["DWIND_6"] = corridor["WIND"] - corridor["WIND_m6"]
    corridor["DPRES_6"] = corridor["PRES"] - corridor["PRES_m6"]
    corridor["DPRES_24"] = corridor["PRES"] - corridor["PRES_m24"]

    lat1, lat2 = np.radians(corridor["LAT_m6"]), np.radians(corridor["LAT"])
    dlat = lat2 - lat1
    dlon = np.radians(corridor["LON"] - corridor["LON_m6"])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    corridor["SPEED"] = (2 * 6371 * np.arcsin(np.sqrt(a)) / 6.0) * 0.539957

    corridor["FUTURE_CHANGE"] = corridor["WIND_p24"] - corridor["WIND"]
    corridor["LABEL"] = np.select(
        [corridor["FUTURE_CHANGE"] >= 30, corridor["FUTURE_CHANGE"] <= -30],
        ["RI", "WEAKEN"], default="STEADY")

    corridor["LON_SIN"] = np.sin(np.radians(corridor["LON"]))
    corridor["LON_COS"] = np.cos(np.radians(corridor["LON"]))
    return corridor


FEATURES = ["WIND", "PRES", "DWIND_24", "DWIND_12", "DWIND_6", "DPRES_6", "DPRES_24",
            "LAT", "LON_SIN", "LON_COS", "DIST2LAND", "MONTH", "SPEED"]


def fetch_sst(dom, path):
    if os.path.exists(path):
        return path
    lat0, lat1 = dom["lat"]
    lon0, lon1 = dom["lon"]
    url = SST_URL.format(lat0=lat0, lat1=lat1, lon0=lon0, lon1=lon1)
    urllib.request.urlretrieve(url, path)
    return path


def attach_sst(labelled, path):
    import xarray as xr
    grid = xr.open_dataset(path)["sst"].isel(zlev=0)
    labelled["SST"] = grid.sel(
        time=xr.DataArray(labelled["TIME"].values),
        latitude=xr.DataArray(labelled["LAT"].values),
        longitude=xr.DataArray(labelled["LON"].values),
        method="nearest").values.ravel()
    return labelled


def report(name, pred, true):
    print(f"\n{name} - accuracy {(pred == true).mean() * 100:.1f}%")
    for cls in ["RI", "WEAKEN", "STEADY"]:
        mask = true == cls
        if mask.sum():
            print(f"  {cls:7s} caught {(pred[mask] == cls).mean() * 100:5.1f}% of {mask.sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(DOMAINS), default="wp")
    ap.add_argument("--ib", default="ibtracs_since1980.list.v04r01.csv",
                    help="IBTrACS CSV (the since1980 export is enough)")
    ap.add_argument("--out", default="exported")
    ap.add_argument("--threshold", type=float, default=0.25)
    args = ap.parse_args()
    if not os.path.exists(args.ib):
        ap.error(f"IBTrACS CSV not found: {args.ib} — pass --ib /path/to/ibtracs.since1980.list.v04r01.csv")

    dom = DOMAINS[args.domain]
    os.makedirs(args.out, exist_ok=True)

    corridor = load_corridor(args.ib, dom)
    print(f"corridor rows {len(corridor)}  storms {corridor['SID'].nunique()}")
    corridor = add_labels(corridor)
    labelled = corridor.dropna(subset=FEATURES + ["FUTURE_CHANGE"]).copy()
    print("labelled", len(labelled))
    print(labelled["LABEL"].value_counts().to_string())

    sst_path = fetch_sst(dom, f"/tmp/sst_{args.domain}.nc")
    labelled = attach_sst(labelled, sst_path)
    print("SST coverage", round(labelled["SST"].notna().mean() * 100, 1), "%")

    train = labelled[labelled["SEASON"] <= 2020].copy()
    test = labelled[labelled["SEASON"] >= 2021].copy()
    sst_fill = float(train["SST"].median())
    for frame in (train, test):
        frame["SST"] = frame["SST"].fillna(sst_fill)

    features = FEATURES + ["SST"]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(train[features], train["LABEL"])
    report(f"LR + SST ({args.domain})", model.predict(test[features]), test["LABEL"])
    print(f"train {len(train)} ({(train.LABEL == 'RI').mean() * 100:.2f}% RI) | "
          f"test {len(test)} ({(test.LABEL == 'RI').mean() * 100:.2f}% RI)")

    proba = model.predict_proba(test[features])
    classes = list(model.classes_)
    p_ri, p_we = proba[:, classes.index("RI")], proba[:, classes.index("WEAKEN")]
    y = test["LABEL"].values
    print("\n  t    RI caught   WEAKEN caught   RI false alarms")
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        pred = np.where(p_ri >= t, "RI", np.where(p_we >= t, "WEAKEN", "STEADY"))
        print(f" {t:.2f}   {(pred[y == 'RI'] == 'RI').mean() * 100:6.1f}%      "
              f"{(pred[y == 'WEAKEN'] == 'WEAKEN').mean() * 100:6.1f}%        "
              f"{int(((pred == 'RI') & (y != 'RI')).sum())}")

    # per-sector spread, so the two models can be compared like for like
    test["PRED"] = np.where(p_ri >= args.threshold, "RI",
                            np.where(p_we >= args.threshold, "WEAKEN", "STEADY"))
    for lo, hi, label in [(110, 145, "core 110-145E"), (145, 180, "east 145-180E"),
                          (-180, -140, "CP 180-140W"), (-140, -70, "EP 140-70W")]:
        s = test[(test.LON >= lo) & (test.LON <= hi)]
        if len(s):
            ri = s[s.LABEL == "RI"]
            print(f"  {label:15s} n={len(s):5d}  RI={len(ri):3d}  "
                  f"warned {int((ri.PRED == 'RI').sum()):3d}")

    labelled["CELL_LAT"] = (labelled["LAT"] // 2.5) * 2.5
    labelled["CELL_LON"] = (labelled["LON"] // 2.5) * 2.5
    stats = {}
    for (clat, clon), d in labelled.groupby(["CELL_LAT", "CELL_LON"]):
        stats[f"{clat},{clon}"] = {
            "points": int(len(d)), "storms": int(d["SID"].nunique()),
            "ri_pct": round(float((d["LABEL"] == "RI").mean() * 100), 1),
            "weaken_pct": round(float((d["LABEL"] == "WEAKEN").mean() * 100), 1),
            # IBTrACS DIST2LAND is 0 while the centre is over land; a cell where
            # every report was inland is a landfall track, not ocean climatology
            "land": round(float((d["DIST2LAND"] == 0).mean()), 2),
        }
    defaults = {}
    keep_defaults = [f for f in features if f not in ("LON_SIN", "LON_COS")]
    # A cell-month where every report lacks a field has a NaN median, and NaN is
    # not JSON: the browser rejects the whole file. Fall back to the basin median.
    basin_med = labelled[keep_defaults].median()
    for (clat, clon, m), d in labelled.groupby(["CELL_LAT", "CELL_LON", "MONTH"]):
        row = {}
        for f in keep_defaults:
            v = d[f].median()
            if pd.isna(v):
                v = basin_med[f]
            row[f] = round(float(v), 2) if pd.notna(v) else sst_fill
        defaults[f"{clat},{clon},{int(m)}"] = row

    # 24 h movement climatology on the site's grid (cell centre = origin + 1.25),
    # training years only, so the page's track forecast reaches the new cells
    lat0, lon0 = dom["lat"][0], dom["lon"][0]
    tr = corridor.dropna(subset=["LAT_m6", "LON_m6", "LAT_p24", "LON_p24"]).copy()
    tr = tr[tr["SEASON"] <= 2020]
    tr["CLAT"] = (np.floor((tr["LAT"] - lat0) / 2.5) * 2.5 + lat0 + 1.25).round(2)
    tr["CLON"] = (np.floor((tr["LON"] - lon0) / 2.5) * 2.5 + lon0 + 1.25).round(2)
    tr["dLat"] = tr["LAT_p24"] - tr["LAT"]
    tr["dLon"] = tr["LON_p24"] - tr["LON"]
    track = {}
    for (a, b, m), g in tr.groupby(["CLAT", "CLON", "MONTH"]):
        track[f"{a},{b},{int(m)}"] = {"dLat": round(float(g.dLat.mean()), 3),
                                      "dLon": round(float(g.dLon.mean()), 3),
                                      "n": int(len(g))}

    scaler = model.named_steps["standardscaler"]
    logreg = model.named_steps["logisticregression"]
    weights = {
        "features": features,
        "classes": list(logreg.classes_),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": logreg.coef_.tolist(),
        "intercept": logreg.intercept_.tolist(),
        "warning_threshold": args.threshold,
        "sst_fill": sst_fill,
        "domain": {"lat": list(dom["lat"]), "lon_pm180": list(dom["lon"])},
        "trained_years": [2000, 2020],
        "n_train": int(len(train)),
        "ri_base_rate": round(float((train.LABEL == "RI").mean() * 100), 2),
    }

    for name, obj in [(f"model_weights_{args.domain}.json", weights),
                      (f"region_stats_{args.domain}.json", stats),
                      (f"region_defaults_{args.domain}.json", defaults),
                      (f"track_climatology_{args.domain}.json", track)]:
        with open(os.path.join(args.out, name), "w") as fh:
            json.dump(obj, fh, indent=1, allow_nan=False)
    print(f"\ncells {len(stats)}  cell-month defaults {len(defaults)}  "
          f"track cell-months {len(track)}  -> {args.out}")


if __name__ == "__main__":
    main()
