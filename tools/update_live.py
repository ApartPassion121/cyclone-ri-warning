# Regenerates live_storms.json and wind_frames.json; track_climatology.json is
# static (exported by the model notebook).
# Run by .github/workflows/live.yml every 6 hours. Source: IBTrACS ACTIVE (NOAA NCEI).
# Storms: every North Pacific one in that feed (WP, EP and CP basins), each scored
# by the fit for the box it sits in - the same choice the page makes in regionFor().
import pandas as pd, numpy as np, json, math, urllib.request, io, os
# This file lives in tools/; the JSON it reads and writes sits at the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def at(name): return os.path.join(ROOT, name)
URL = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ACTIVE.list.v04r01.csv'
raw = urllib.request.urlopen(URL, timeout=120).read().decode('utf-8', 'replace')
df = pd.read_csv(io.StringIO(raw), low_memory=False)
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
for c in ['LAT','LON','WMO_WIND','TOKYO_WIND','USA_WIND','WMO_PRES','TOKYO_PRES','USA_PRES','DIST2LAND','STORM_SPEED']:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
df['WIND'] = df['WMO_WIND'].combine_first(df['TOKYO_WIND']).combine_first(df['USA_WIND'])
df['PRES'] = df['WMO_PRES'].combine_first(df['TOKYO_PRES']).combine_first(df['USA_PRES'])
# One fit per box, the same three the page chooses between: a storm east of the
# dateline must be scored by the fit that saw east Pacific storms, not by the
# 110-145 E corridor model extrapolated 60 degrees west of everything it learned.
FITS = {}
for name, (wf, rf, tf) in {
        'wp':   ('model_weights_wp.json',   'region_defaults_wp.json',   'track_climatology_wp.json'),
        'east': ('model_weights_east.json', 'region_defaults_east.json', 'track_climatology_east.json')}.items():
    try: FITS[name] = {'W': json.load(open(at(wf))), 'D': json.load(open(at(rf))), 'T': json.load(open(at(tf)))}
    except FileNotFoundError: pass   # a checkout without the widened pair still runs
FITS['base'] = {'W': json.load(open(at('model_weights.json'))), 'D': json.load(open(at('region_defaults.json'))), 'T': json.load(open(at('track_climatology.json')))}
def fit_for(lat, lon):
    if 5 <= lat <= 40 and 110 <= lon <= 180: return FITS.get('wp', FITS['base'])
    if 5 <= lat <= 40 and (lon <= -70 or lon >= 180): return FITS.get('east', FITS['base'])
    return FITS['base']   # outside both boxes: still scored, the page flags it as extrapolated
def cell_keys(lat, lon, month):   # centre key for track_climatology, SW corner for region_defaults
    la = math.floor(lat/2.5)*2.5; lo = math.floor(lon/2.5)*2.5
    return f"{la+1.25:.2f},{lo+1.25:.2f},{month}", f"{la:.1f},{lo:.1f},{month}"
def predict(W, f):
    vals = [(f[ft]-W['scaler_mean'][i])/W['scaler_scale'][i] for i, ft in enumerate(W['features'])]
    zs = {c: sum(vals[i]*W['coef'][ci][i] for i in range(len(vals)))+W['intercept'][ci] for ci, c in enumerate(W['classes'])}
    m = max(zs.values()); ex = {k: math.exp(v-m) for k, v in zs.items()}; s = sum(ex.values())
    return {k: round(v/s*100,1) for k, v in ex.items()}
import datetime
out = {'source': 'IBTrACS ACTIVE (NOAA NCEI), provisional agency rows', 'generated_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M'), 'storms': []}
for sid, g in df[df['BASIN'].isin(('WP','EP','CP'))].groupby('SID'):
    g = g.sort_values('ISO_TIME').dropna(subset=['WIND','LAT','LON'])
    if len(g) < 9: continue
    g = g.reset_index(drop=True); r = g.iloc[-1]; t = r['ISO_TIME']
    def past(hours, col):   # named past(), not at(): at() is the path helper above
        j = g[g['ISO_TIME'] == t - pd.Timedelta(hours=hours)]
        return float(j[col].iloc[0]) if len(j) and pd.notna(j[col].iloc[0]) else None
    w6, w12, w24, p6, p24 = past(6,'WIND'), past(12,'WIND'), past(24,'WIND'), past(6,'PRES'), past(24,'PRES')
    la6, lo6 = past(6,'LAT'), past(6,'LON')
    if None in (w6,w12,w24,p6,p24,la6,lo6): continue
    lat, lon = float(r['LAT']), float(r['LON'])
    fit = fit_for(lat, lon)
    key, swkey = cell_keys(lat, lon, t.month)
    dd = fit['D'].get(swkey, {}); clim = fit['T'].get(key, {'dLat': 0.5, 'dLon': -1.0})
    pres = float(r['PRES']) if pd.notna(r['PRES']) else dd.get('PRES', 1005.0)
    f = {'WIND': float(r['WIND']), 'PRES': pres, 'DWIND_24': float(r['WIND']-w24), 'DWIND_12': float(r['WIND']-w12),
         'DWIND_6': float(r['WIND']-w6), 'DPRES_6': pres-p6, 'DPRES_24': pres-p24,
         'LAT': lat, 'LON': lon, 'LON_SIN': math.sin(math.radians(lon)), 'LON_COS': math.cos(math.radians(lon)),
         'DIST2LAND': float(r['DIST2LAND']) if pd.notna(r['DIST2LAND']) else 1000.0,
         'MONTH': int(t.month), 'SPEED': float(r['STORM_SPEED']) if pd.notna(r['STORM_SPEED']) else 10.0, 'SST': dd.get('SST', 28.5)}
    bdlat = 0.5*(4*(lat-la6)) + 0.5*clim['dLat']; bdlon = 0.5*(4*(lon-lo6)) + 0.5*clim['dLon']
    hist = [{'time_utc': row['ISO_TIME'].strftime('%Y-%m-%d %H:%M'), 'lat': float(row['LAT']), 'lon': float(row['LON']),
             'wind_kt': float(row['WIND']), 'pres_hpa': float(row['PRES']) if pd.notna(row['PRES']) else None}
            for _, row in g.iterrows() if pd.notna(row['LAT']) and pd.notna(row['WIND'])]
    out['storms'].append({'name': str(r['NAME']).title(), 'sid': sid, 'time_utc': t.strftime('%Y-%m-%d %H:%M'), 'history': hist,
        'lat': lat, 'lon': lon, 'wind_kt': float(r['WIND']), 'pres_hpa': float(r['PRES']) if pd.notna(r['PRES']) else None,
        'features': f, 'prediction': predict(fit['W'], f),
        'track_24h': {'dLat': round(bdlat,2), 'dLon': round(bdlon,2), 'lat': round(lat+bdlat,2), 'lon': round(lon+bdlon,2),
                      'method': 'blend of storm motion + typical path (test mean error 358 km)'}})
# newest report first: the panel puts the storm that is still alive at the top
out['storms'].sort(key=lambda s: s['time_utc'], reverse=True)
json.dump(out, open(at('live_storms.json'),'w'), indent=1, allow_nan=False)
print('storms:', [s['name'] for s in out['storms']])

# --- wind_frames.json: whole-globe 10 m wind, 3-hourly frames -24 h..+48 h ---
# Open-Meteo hourly (free, no key), baked into a static file so the page itself
# makes no API calls. A failure here must never block the storms update.
import time
from datetime import datetime, timedelta, timezone
try:
    STEP = 2.5
    lats = [87.5-STEP*i for i in range(71)]
    lons = [STEP*i for i in range(144)]
    pts = [(la, lo) for la in lats for lo in lons]
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # 6-hourly frames back to the earliest storm history (so finished storms can
    # be replayed on the time bar), then 3-hourly for the last day + 48 h forecast.
    cutoff = now-timedelta(hours=24)
    back = cutoff
    for st_ in out['storms']:
        if st_.get('history'):
            h0 = datetime.strptime(st_['history'][0]['time_utc'], '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
            back = min(back, h0.replace(hour=(h0.hour//6)*6))
    # Open-Meteo only serves ~10 days of past hourly data; asking for older
    # frames made idx[w] raise KeyError, which skipped the whole wind update.
    back = max(back, now-timedelta(days=10))
    back = back.replace(hour=(back.hour//6)*6, minute=0, second=0, microsecond=0)   # 6 h grid
    # One sorted set of stamps: the 6-hourly replay back to the first storm
    # history, then the 3-hourly window. A set because the two grids overlap at
    # the join, and a duplicate timestamp is a dead step on the time bar.
    want = set()
    tcur = back
    while tcur < cutoff:
        want.add(tcur.strftime('%Y-%m-%dT%H:00')); tcur += timedelta(hours=6)
    tcur = cutoff.replace(hour=(cutoff.hour//3)*3)
    while tcur <= now+timedelta(hours=48):
        want.add(tcur.strftime('%Y-%m-%dT%H:00')); tcur += timedelta(hours=3)
    want = sorted(want)
    U = {w: [None]*len(pts) for w in want}; V = {w: [None]*len(pts) for w in want}
    # MSLP on every other point (5 deg): the cursor readout needs the smooth
    # field, not 2.5 deg detail, and a full-resolution array would add ~3 MB to
    # a file the Action commits every 6 hours.
    PS = 2; PNX = len(range(0, len(lons), PS)); PNY = len(range(0, len(lats), PS))
    P = {w: [None]*(PNX*PNY) for w in want}
    pruned = False
    for st in range(0, len(pts), 40):
        chunk = pts[st:st+40]
        la = ','.join(str(q[0]) for q in chunk)
        lo = ','.join(str(q[1]-360 if q[1] > 180 else q[1]) for q in chunk)
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={la}&longitude={lo}"
               "&hourly=wind_speed_10m,wind_direction_10m,pressure_msl&wind_speed_unit=ms&past_days=10&forecast_days=4")
        req = urllib.request.Request(url, headers={'User-Agent': 'cyclone-ri-warning/1.0'})
        res = json.load(urllib.request.urlopen(req, timeout=180))
        if isinstance(res, dict): res = [res]
        for k, loc in enumerate(res):
            idx = {t: ii for ii, t in enumerate(loc['hourly']['time'])}
            if not pruned:  # drop frames this feed does not cover instead of losing the whole file
                for w in [x for x in want if x not in idx]:
                    want.remove(w); U.pop(w, None); V.pop(w, None); P.pop(w, None)
                pruned = True
            spd = loc['hourly']['wind_speed_10m']; dr = loc['hourly']['wind_direction_10m']
            for w in want:
                ii = idx[w]; rad = math.radians(dr[ii])
                U[w][st+k] = round(-spd[ii]*math.sin(rad), 1); V[w][st+k] = round(-spd[ii]*math.cos(rad), 1)
            pres = loc['hourly'].get('pressure_msl') or []
            for k, gp in enumerate(range(st, st+len(chunk))):   # keep only the 5 deg subset
                gi, gj = divmod(gp, len(lons))
                if not pres or gi % PS or gj % PS: continue
                pi = (gi//PS)*PNX + (gj//PS)
                for w in want: P[w][pi] = round(pres[idx[w]])
        time.sleep(0.3)
    frames = [{'time': w, 'u': U[w], 'v': V[w], 'p': P[w]} for w in want]
    assert all(x is not None for f in frames for x in f['u'])
    wf = {'header': {'nx': len(lons), 'ny': len(lats), 'lo1': 0, 'la1': 87.5, 'dx': STEP, 'dy': STEP,
                     'source': 'Open-Meteo hourly 10 m wind (GFS/ICON blend), 3-hourly frames',
                     'p': {'nx': PNX, 'ny': PNY, 'lo1': 0, 'la1': 87.5, 'dx': STEP*PS, 'dy': STEP*PS,
                           'source': 'Open-Meteo MSLP, every 2nd grid point'}},
          'frames': frames}
    json.dump(wf, open(at('wind_frames.json'), 'w'), separators=(',', ':'))
    print('wind_frames.json written:', len(frames), 'frames')
except Exception as e:
    print('wind update skipped:', e)
