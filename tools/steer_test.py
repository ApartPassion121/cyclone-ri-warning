#!/usr/bin/env python3
"""Does the analysed wind field beat climatology at moving a storm 24 h on?

The page's pale-gold 24 h branch came from one vector per storm: half the
storm's own 6 h motion extrapolated to a day, half the typical 24 h displacement
of storms that passed through the cell and month it sits in (track_climatology*).
That climatology knows nothing about the flow the storm is actually in, so a
hand-placed or slow storm can be given a 1,000 km run.

This measures three ways to move a position 24 h on, against the 24 h position
IBTrACS itself reports later, for the storms in the live feed while
wind_frames.json covers them (Open-Meteo 10 m wind, 2.5 deg, 3-hourly):

  persistence  p + 4*(p - p6h)                       the storm keeps doing what it did
  clim blend   0.5*persistence + 0.5*cell climatology  what the page ships today
  steer blend  0.5*persistence + 0.5*wind steering     the analysed flow around it

Steering is the mean 10 m flow in an annulus 2.5-7.5 deg around the storm (the
core is skipped so its own circulation does not count as steering), integrated
for 24 h. It is a *surface* proxy for the deep-layer flow that really steers a
cyclone, which is the point of measuring rather than assuming.

    python3 tools/steer_test.py
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MS = {'W': 1e0}
KT = 1.94384          # m/s -> kt
EARTH = 111_000.0     # m per degree of latitude


def load(name):
    with open(os.path.join(ROOT, name)) as f:
        return json.load(f)


def frame_at(wind, t_ms):
    """The frame closest to t_ms, within 1.5 h, or None."""
    best, bd = None, 1.5 * 3600e3
    for fr in wind['frames']:
        d = abs(ms_of(fr['time']) - t_ms)
        if d < bd:
            bd, best = d, fr
    return best


def ms_of(s):
    y, mo, d = (int(x) for x in s[:10].split('-'))
    hh, mm = (int(x) for x in s[11:16].split(':'))
    # days since epoch, no datetime import needed for a fixed format
    return days_from_civil(y, mo, d) * 86400e3 + (hh * 3600 + mm * 60) * 1000.0


def days_from_civil(y, m, d):
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def sample_uv(wind, fr, lat, lon):
    h = wind['header']
    nx, ny = h['nx'], h['ny']
    fx = ((lon % 360) + 360) % 360 / h['dx']
    fy = max(0.0, min(ny - 1.001, (h['la1'] - lat) / h['dy']))
    x0 = int(fx) % nx
    x1 = (x0 + 1) % nx
    y0 = int(fy)
    y1 = min(y0 + 1, ny - 1)
    tx, ty = fx - int(fx), fy - y0
    u, v = fr['u'], fr['v']
    def bil(i00, i10, i01, i11):
        a = u[i00] + (u[i10] - u[i00]) * tx
        b = u[i01] + (u[i11] - u[i01]) * tx
        c = v[i00] + (v[i10] - v[i00]) * tx
        d = v[i01] + (v[i11] - v[i01]) * tx
        return a + (b - a) * ty, c + (d - c) * ty
    return bil(y0 * nx + x0, y0 * nx + x1, y1 * nx + x0, y1 * nx + x1)


def steering(wind, fr, lat, lon):
    """Mean 10 m flow (u, v in m/s) in a 2.5-7.5 deg annulus, or None."""
    us, vs = [], []
    for r in (2.5, 5.0, 7.5):
        for k in range(8):
            a = k * math.pi / 4
            la = lat + r * math.cos(a)
            if abs(la) > 87:
                continue
            lo = lon + r * math.sin(a) / max(0.25, math.cos(math.radians(lat)))
            u, v = sample_uv(wind, fr, la, lo)
            if u is None or v is None or not (math.isfinite(u) and math.isfinite(v)):
                continue
            us.append(u)
            vs.append(v)
    if len(us) < 6:
        return None
    return sum(us) / len(us), sum(vs) / len(vs)


def sample_p(wind, fr, lat, lon):
    h = wind['header']['p']
    P = fr.get('p')
    if not h or not P:
        return None
    nx, ny = h['nx'], h['ny']
    fx = ((lon % 360) + 360) % 360 / h['dx']
    fy = max(0.0, min(ny - 1.001, (h['la1'] - lat) / h['dy']))
    x0 = int(fx) % nx
    x1 = (x0 + 1) % nx
    y0 = int(fy)
    y1 = min(y0 + 1, ny - 1)
    tx, ty = fx - int(fx), fy - y0
    q = [P[y0 * nx + x0], P[y0 * nx + x1], P[y1 * nx + x0], P[y1 * nx + x1]]
    if any(v is None or not math.isfinite(v) for v in q):
        return None
    a = q[0] + (q[1] - q[0]) * tx
    b = q[2] + (q[3] - q[2]) * tx
    return a + (b - a) * ty


RHO = 1.15          # kg/m3, warm marine air
OMEGA = 7.292e-5


def steer_pressure(wind, fr, lat, lon):
    """Geostrophic flow from the analysed MSLP field, averaged 5-12 deg out.

    The 10 m wind close to a cyclone is mostly its own inflow, so the pressure
    field is sampled further out: a tropical cyclone tracks with the flow above
    the boundary layer, and the synoptic pressure pattern is the only proxy for
    that in this dataset.
    """
    us, vs = [], []
    f = 2 * OMEGA * math.sin(math.radians(lat))
    if abs(f) < 2e-5:
        return None
    for r in (5.0, 8.0, 12.0):
        for k in range(8):
            a = k * math.pi / 4
            la = lat + r * math.cos(a)
            lo = lon + r * math.sin(a) / max(0.25, math.cos(math.radians(lat)))
            if abs(la) > 87:
                continue
            d = 1.5
            p_n = sample_p(wind, fr, la + d, lo)
            p_s = sample_p(wind, fr, la - d, lo)
            p_e = sample_p(wind, fr, la, lo + d / max(0.25, math.cos(math.radians(la))))
            p_w = sample_p(wind, fr, la, lo - d / max(0.25, math.cos(math.radians(la))))
            if None in (p_n, p_s, p_e, p_w):
                continue
            dpdy = (p_n - p_s) * 100 / (2 * d * EARTH)       # Pa per m, north
            dpdx = (p_e - p_w) * 100 / (2 * d * EARTH * math.cos(math.radians(la)))
            u = -dpdy / (RHO * f)
            v = dpdx / (RHO * f)
            if abs(u) > 30 or abs(v) > 30:      # geostrophy fails near the equator
                continue
            us.append(u)
            vs.append(v)
    if len(us) < 6:
        return None
    return sum(us) / len(us), sum(vs) / len(vs)


def climb(lat, lon, dlat, dlon):
    cesc = max(0.25, math.cos(math.radians(lat)))
    return math.hypot(dlat * EARTH, dlon * cesc * EARTH * math.pi / 180) / 1000.0


def clim_for(track, cell, month):
    """The page's trackFor(): this cell+month (n>=5), nearest month, then the ring."""
    def ck(la, lo):
        return f"{la + 1.25:.2f},{lo + 1.25:.2f}"
    a, b = (float(x) for x in cell.split(','))
    e = track.get(f"{ck(a, b)},{month}")
    if e and e['n'] >= 5:
        return e
    for d2 in range(1, 12):
        for mm in (month - d2, month + d2):
            m2 = (mm - 1) % 12 + 1
            e2 = track.get(f"{ck(a, b)},{m2}")
            if e2 and e2['n'] >= 5:
                return e2
    n = dl = dn = 0.0
    cells = 0
    for dx in (-2.5, 0.0, 2.5):
        for dy in (-2.5, 0.0, 2.5):
            if not dx and not dy:
                continue
            e3 = track.get(f"{ck(a + dx, b + dy)},{month}")
            if e3 and e3['n']:
                n += e3['n']
                dl += e3['dLat'] * e3['n']
                dn += e3['dLon'] * e3['n']
                cells += 1
    if n >= 10:
        return {'dLat': dl / n, 'dLon': dn / n, 'n': n}
    return None


def main():
    wind = load('wind_frames.json')
    live = load('live_storms.json')
    tracks = {'east': load('track_climatology_east.json'),
              'wp': load('track_climatology_wp.json'),
              'shipped': load('track_climatology.json')}

    def track_for(lat, lon):
        # the page gates by box: east of 180..-70 W is the east fit, 110..180 E the west one
        if -180 <= lon <= -70 or lon >= 180:
            return tracks['east']
        if 110 <= lon <= 180:
            return tracks['wp']
        return tracks['shipped']

    rows = []
    for st in live['storms']:
        H = sorted(st.get('history') or [], key=lambda h: h['time_utc'])
        times = [ms_of(h['time_utc']) for h in H]
        for i in range(len(H)):
            t = times[i]
            # a 6 h step behind and a 24 h fix ahead, both within 1.5 h of nominal
            prev = next((j for j in range(i) if abs(times[j] - (t - 6 * 3600e3)) <= 1.5 * 3600e3), None)
            nxt = next((j for j in range(i + 1, len(H)) if abs(times[j] - (t + 24 * 3600e3)) <= 1.5 * 3600e3), None)
            if prev is None or nxt is None:
                continue
            fr = frame_at(wind, t)
            if not fr:
                continue
            p, q, a = H[i], H[prev], H[nxt]
            lat, lon = p['lat'], p['lon']
            dl, dn = a['lat'] - lat, ((a['lon'] - lon + 180) % 360) - 180
            act_dl, act_dn = dl, dn
            per_dl, per_dn = 4 * (lat - q['lat']), 4 * (((lon - q['lon'] + 180) % 360) - 180)
            st_uv = steering(wind, fr, lat, lon)
            if st_uv is None:
                continue
            su, sv = st_uv
            cesc = max(0.25, math.cos(math.radians(lat)))
            st_dl = sv * 86400 / EARTH                 # north component, deg
            st_dn = su * 86400 / (EARTH * cesc)        # east component, deg of lon
            p_uv = steer_pressure(wind, fr, lat, lon)
            p_dl = p_dn = None
            if p_uv:
                p_dl = p_uv[1] * 86400 / EARTH
                p_dn = p_uv[0] * 86400 / (EARTH * cesc)
            month = int(fr['time'][5:7])
            ck_lat = math.floor(lat / 2.5) * 2.5
            ck_lon = math.floor(lon / 2.5) * 2.5
            ce = clim_for(track_for(lat, lon), f'{ck_lat:.1f},{ck_lon:.1f}', month)
            row = {
                'storm': st['name'], 'time': p['time_utc'], 'lat': lat, 'lon': lon,
                'actual': (act_dl, act_dn),
                'persistence': (per_dl, per_dn),
                'steer': (st_dl, st_dn),
                'press': (p_dl, p_dn) if p_dl is not None else None,
                'clim': (ce['dLat'], ce['dLon']) if ce else None,
            }
            rows.append(row)

    seen = set()
    for r in rows:
        seen.add((r['storm'], r['time']))
    print(f"{len(rows)} cases over {len(seen)} storm-times, "
          f"{len({r['storm'] for r in rows})} storms, {min(r['time'] for r in rows)} .. {max(r['time'] for r in rows)}")

    def err(r, pred, act):
        return climb(r['lat'], r['lon'], pred[0] - act[0], pred[1] - act[1])

    methods = {
        'persistence': lambda r: r['persistence'],
        'clim blend (shipped)': lambda r: None if not r['clim'] else
            (0.5 * r['persistence'][0] + 0.5 * r['clim'][0], 0.5 * r['persistence'][1] + 0.5 * r['clim'][1]),
        'steer blend': lambda r: (0.5 * r['persistence'][0] + 0.5 * r['steer'][0],
                                  0.5 * r['persistence'][1] + 0.5 * r['steer'][1]),
        'steer only': lambda r: r['steer'],
        'press blend': lambda r: None if not r['press'] else
            (0.5 * r['persistence'][0] + 0.5 * r['press'][0], 0.5 * r['persistence'][1] + 0.5 * r['press'][1]),
        'press only': lambda r: r['press'],
        'press+steer blend': lambda r: None if not r['press'] else
            (0.5 * r['persistence'][0] + 0.25 * r['press'][0] + 0.25 * r['steer'][0],
             0.5 * r['persistence'][1] + 0.25 * r['press'][1] + 0.25 * r['steer'][1]),
    }
    print(f"\n{'method':24s} {'mean err':>9s} {'median':>8s} {'cases':>6s}")
    for name, fn in methods.items():
        es = sorted(err(r, fn(r), r['actual']) for r in rows if fn(r) is not None)
        if not es:
            print(f'{name:24s} {"-":>9s}')
            continue
        mean = sum(es) / len(es)
        med = es[len(es) // 2]
        print(f'{name:24s} {mean:9.0f} {med:8.0f} {len(es):6d}')
    print("\nsteering speed (kt) vs the storm's own 24 h speed (kt):")
    for r in rows[:8]:
        u, v = r['steer']
        kt_steer = math.hypot(u, v) * KT
        kt_own = climb(r['lat'], r['lon'], r['actual'][0], r['actual'][1]) / 1.852 / 24
        print(f"  {r['storm']:9s} {r['time']} steer {kt_steer:5.1f} kt   moved {kt_own:5.1f} kt"
              f"   steer dir {math.degrees(math.atan2(-u, -v)) % 360:5.0f}  moved dir"
              f" {math.degrees(math.atan2(-r['actual'][1], -r['actual'][0])) % 360:5.0f}")


if __name__ == '__main__':
    main()
