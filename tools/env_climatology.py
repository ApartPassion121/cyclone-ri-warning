#!/usr/bin/env python3
"""Fill era5_env.json for every cell the page draws.

    python3 tools/env_climatology.py            # fill what is missing, keep ERA5

era5_env.json holds, per cell x month, the typical environment the CONDITIONS
panel shows: SHEAR (m/s, 200 hPa minus 850 hPa), RH (%, 700 hPa) and CAPE
(J/kg). The shipped file was an ERA5 export for the original 110-145 E corridor,
so every cell added by the widened fits had no rows at all.

ERA5 upper-air data needs a Copernicus CDS account, which this tool cannot use.
Missing cells are therefore filled from the NCEP/NCAR Reanalysis 1 monthly
climatology for 1991-2020 on the same 2.5 degree grid (NOAA PSL, free OPeNDAP),
and each entry records where it came from:

    {"SHEAR": 40.5, "RH": 39, "CAPE": 812, "SRC": "era5"}
    {"SHEAR": 41.2, "RH": 37, "SRC": "ncep"}

CAPE is not in the NCEP monthly means and is not invented here: an NCEP entry
carries no CAPE, and the page says so for that cell. Cross-check against the
ERA5 cells it can be compared with: shear 40.5 vs 39.1 and 63.4 vs 63.3 m/s on
the two strong-jet cases, 4.3 vs 6.4 on a weak one; NCEP's 700 hPa humidity
runs 5-10 points drier than ERA5, which matters only near the 55 % threshold.

Re-running is safe: entries already marked era5 are never overwritten.
"""
import json
import os
import warnings

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "era5_env.json")
PSL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/"
LEV_LO, LEV_HI, LEV_RH = 850, 200, 700


def load_grid():
    """One small slice per variable: 12 months, three levels, 5-40 N, all lons."""
    def open_var(name):
        ds = xr.open_dataset(PSL + name)
        var = next(v for v in ds.data_vars if "lat" in ds[v].dims)   # skips climatology_bounds
        return ds[var].sel(lat=slice(45, 0))          # NCEP latitudes run north to south

    u = open_var("uwnd.mon.ltm.1991-2020.nc").sel(level=[LEV_LO, LEV_HI])
    v = open_var("vwnd.mon.ltm.1991-2020.nc").sel(level=[LEV_LO, LEV_HI])
    r = open_var("rhum.mon.ltm.1991-2020.nc").sel(level=[LEV_RH])
    return u, v, r


def cells():
    """Every cell the page can draw: the keys of its own stats files."""
    out = []
    for tag in ("wp", "east"):
        path = os.path.join(ROOT, f"region_stats_{tag}.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for k in json.load(fh):
                lat, lon = (float(x) for x in k.split(","))
                out.append((tag, lat, lon))
    return out


def main():
    with open(OUT) as fh:
        env = json.load(fh)
    had = len(env)

    u, v, r = load_grid()
    lats = set(float(x) for x in u.lat.values)
    lons = set(float(x) for x in u.lon.values)

    def nearest(values, x):
        if x in values:
            return x
        return min(values, key=lambda c: abs(c - x))

    filled = kept = 0
    for _tag, lat, lon in cells():
        la, lo = nearest(lats, lat), nearest(lons, (lon + 360) % 360)
        du = u.sel(lat=la, lon=lo, method="nearest").values     # (time, level)
        dv = v.sel(lat=la, lon=lo, method="nearest").values
        rh = r.sel(lat=la, lon=lo, method="nearest").values[:, 0]   # (time,)
        for m in range(12):
            key = f"{lat},{lon},{m + 1}"
            if key in env and env[key].get("SRC", "era5") == "era5":
                kept += 1
                continue
            shear = float(np.hypot(du[m, 1] - du[m, 0], dv[m, 1] - dv[m, 0]))
            row = {"SHEAR": round(shear, 1), "RH": int(round(float(rh[m]))), "SRC": "ncep"}
            if key in env and env[key].get("CAPE") is not None:
                row["CAPE"] = env[key]["CAPE"]                  # keep an ERA5 CAPE if one was there
            env[key] = row
            filled += 1

    with open(OUT, "w") as fh:
        json.dump(dict(sorted(env.items())), fh, separators=(",", ":"))
    srcs = {}
    for row in env.values():
        srcs[row.get("SRC", "era5")] = srcs.get(row.get("SRC", "era5"), 0) + 1
    print(f"entries {had} -> {len(env)} | added {filled} | kept {kept} | by source {srcs}")


if __name__ == "__main__":
    main()
