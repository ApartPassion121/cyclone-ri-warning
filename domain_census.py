#!/usr/bin/env python3
"""Census of the IBTrACS archive by Pacific sector, to size a domain widening.

Answers, before any retraining: how many storms, records, RI points and
storm-days each candidate domain brings, and how different their RI base rates
are. Point it at an IBTrACS CSV (the since1980 export is enough):

    python3 domain_census.py --csv /kaggle/input/.../ibtracs.since1980.list.v04r01.csv

Writes nothing; prints tables. Needs pandas only.

Conventions match the shipped model and the site: TRACK_TYPE 'main', extratropical
(NATURE 'ET') rows dropped, wind unified WMO > TOKYO > USA, RI = +30 kt in
exactly 24 h, WEAK = -30 kt, longitude normalised to -180..180 so the dateline
is not a special case.
"""

import argparse

import numpy as np
import pandas as pd

SECTORS = [  # (label, lat range, lon range) - lon may wrap past 180
    ("WP core 110-145E", (5, 40), (110, 145)),
    ("WP east 145-180E", (5, 40), (145, 180)),
    ("CP 180-140W", (5, 40), (-180, -140)),
    ("EP 140-70W", (5, 40), (-140, -70)),
]
DOMAINS = [
    ("current 5-40N 110-145E", (5, 40), (110, 145)),
    ("WP to 180 5-40N 110-180E", (5, 40), (110, 180)),
    ("model A 110E-180E", (5, 40), (110, 180)),
    ("model B 180E-70W", (5, 40), (180, -70)),
    ("N Pacific 5-45N 100E-70W", (5, 45), (100, -70)),
]


def load(path):
    df = pd.read_csv(path, low_memory=False, skiprows=[1], dtype={"LAT": str, "LON": str})
    for c in ("LAT", "LON", "WMO_WIND", "TOKYO_WIND", "USA_WIND"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df["WIND"] = (df.get("WMO_WIND").combine_first(df.get("TOKYO_WIND"))
                  .combine_first(df.get("USA_WIND")))
    df["LON"] = ((df["LON"] + 180) % 360) - 180
    keep = (df["TRACK_TYPE"] == "main") & df["ISO_TIME"].notna() & df["LAT"].notna()
    keep &= df["LON"].notna() & df["WIND"].notna() & (df.get("NATURE") != "ET")
    return df[keep].copy()


def label_24h(d):
    later = d.copy()
    later["ISO_TIME"] = later["ISO_TIME"] - pd.Timedelta(hours=24)
    m = d.merge(later[["SID", "ISO_TIME", "WIND"]], on=["SID", "ISO_TIME"],
                how="left", suffixes=("", "_t24"))
    m["D24"] = m["WIND_t24"] - m["WIND"]
    m["cls"] = np.where(m["D24"] >= 30, "RI", np.where(m["D24"] <= -30, "WEAK", "STEADY"))
    return m


def in_box(frame, lat_range, lon_range):
    lat_ok = (frame.LAT >= lat_range[0]) & (frame.LAT <= lat_range[1])
    lo, hi = lon_range
    lon_ok = ((frame.LON >= lo) & (frame.LON <= hi)) if lo < hi else ((frame.LON >= lo) | (frame.LON <= hi))
    return frame[lat_ok & lon_ok]


def cells(frame):
    lat = (np.floor(frame.LAT / 2.5) * 2.5).astype(int)
    lon = (np.floor(frame.LON / 2.5) * 2.5).astype(int)
    return (len(set(zip(lat, lon))),
            len(set(zip(lat, lon, frame.ISO_TIME.dt.month))))


def base_rate(frame):
    hit = frame[frame.D24.notna()]
    return 100.0 * (hit.cls == "RI").mean() if len(hit) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="ibtracs.since1980.list.v04r01.csv")
    ap.add_argument("--first-year", type=int, default=2000)
    ap.add_argument("--test-from", type=int, default=2021)
    ap.add_argument("--test-through", type=int, default=2025)
    args = ap.parse_args()

    df = load(args.csv)
    df["YEAR"] = df["ISO_TIME"].dt.year
    df = df[(df.YEAR >= args.first_year) & (df.YEAR <= args.test_through)]
    m = label_24h(df.sort_values(["SID", "ISO_TIME"]))
    m["split"] = np.where(m.YEAR <= args.test_from - 1, "train", "test")
    print(f"records {len(m)}  storms {m.SID.nunique()}  "
          f"+24 h match {100 * m.D24.notna().mean():.1f}%")

    print("\nsector                     storms  records     RI  base%   WEAK")
    for name, lat_r, lon_r in SECTORS:
        s = in_box(m, lat_r, lon_r)
        hit = s[s.D24.notna()]
        print(f"{name:26s} {s.SID.nunique():6d} {len(s):8d} "
              f"{int((hit.cls == 'RI').sum()):6d} {100 * (hit.cls == 'RI').mean():6.2f} "
              f"{int((hit.cls == 'WEAK').sum()):6d}")

    print("\ndomain                       storms  records  cells  cellxmonth  RI%train  RI%test")
    for name, lat_r, lon_r in DOMAINS:
        s = in_box(m, lat_r, lon_r)
        c, cm = cells(s)
        print(f"{name:28s} {s.SID.nunique():6d} {len(s):8d} {c:6d} {cm:11d} "
              f"{base_rate(s[s.split == 'train']):9.2f} {base_rate(s[s.split == 'test']):8.2f}")

    core = in_box(m, (5, 40), (110, 145))
    east = in_box(m, (5, 40), (145, 180))
    both = set(core.SID) & set(east.SID)
    print(f"\nstorms the shipped box never sees but 145-180E has: {len(set(east.SID) - set(core.SID))}")
    print(f"storms crossing 145E: {len(both)}  (records in the east: {len(east)})")


if __name__ == "__main__":
    main()
