"""
RVSP Finder — Skyfield + Large MPC Catalog Edition
====================================================
Scans a large catalog of Trans-Neptunian Objects (TNOs) to find
Radial-Velocity Stationary Points (RVSPs): moments when the shadow
ground-track speed approaches zero, maximising effective integration time
for stellar occultation observations.

---------------------------------
    v_shadow = |v_perp_TNO − v_perp_Earth| / d_TNO   [km/s]

At an RVSP, v_shadow → 0 and t_eff = L / v_shadow is maximised.

Catalog strategy
-----------------
  1. Fetches the MPC "distant objects" JSON (~hundreds of TNOs / SDOs / Centaurs).
  2. Falls back to a built-in list of ~30 well-characterised objects.

Earth ephemeris : astropy built-in DE430 (no download required).
Timescales      : Skyfield (clean UTC/JD handling).
MPC data        : skyfield.data.mpc (minor-planet catalog utilities).

Dependencies
------------
    pip install skyfield astropy matplotlib numpy requests

Usage
-----
    python rvsp_skyfield.py                          # default 2025-2035
    python rvsp_skyfield.py --start 2026 --end 2033
    python rvsp_skyfield.py --threshold 0.5 --L 300
    python rvsp_skyfield.py --top 30                 # top-N on timeline
    python rvsp_skyfield.py --local                  # skip MPC, use built-in
    python rvsp_skyfield.py --min_dist 35 --max_dist 100
"""

import argparse
import gzip
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.ticker as mticker

import astropy.units as u
from astropy.time import Time as AstroTime
from astropy.coordinates import (
    get_body_barycentric_posvel,
    solar_system_ephemeris,
)
from skyfield.api import Loader

warnings.filterwarnings("ignore")
solar_system_ephemeris.set("builtin")   # astropy built-in DE430


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

AU_KM          = 1.495978707e8          # km per AU
DAY_S          = 86400.0               # s per day
LAMBDA_NM      = 550.0                 # reference wavelength [nm]
LAMBDA_M       = LAMBDA_NM * 1e-9
AU_M           = AU_KM * 1e3
GM_SUN_AU3_D2  = 2.959122082855911e-4  # AU³/day²


# ═══════════════════════════════════════════════════════════════════════════════
# BUILT-IN FALLBACK CATALOG  (~22 well-characterised TNOs)
# ═══════════════════════════════════════════════════════════════════════════════

BUILTIN_CATALOG = [
    dict(name="Eris",       a=67.78,  e=0.436, inc=44.04, Omega=35.87,  omega=151.4, M=204.0, epoch_jd=2459600.5, radius_km=1163.0, notes="Largest known dwarf planet."),
    dict(name="Makemake",   a=45.79,  e=0.162, inc=28.96, Omega=79.60,  omega=294.8, M=165.0, epoch_jd=2459600.5, radius_km=715.0,  notes="Large KBO; multi-chord occultation."),
    dict(name="Haumea",     a=43.13,  e=0.195, inc=28.21, Omega=122.1,  omega=239.5, M=218.3, epoch_jd=2459600.5, radius_km=798.0,  notes="Rapid rotator; ring system detected."),
    dict(name="Quaoar",     a=43.40,  e=0.034, inc=7.99,  Omega=188.9,  omega=155.0, M=328.0, epoch_jd=2459600.5, radius_km=555.0,  notes="Nearly circular; ring detections."),
    dict(name="Sedna",      a=506.0,  e=0.843, inc=11.93, Omega=144.5,  omega=311.2, M=358.1, epoch_jd=2459600.5, radius_km=497.5,  notes="Extreme TNO near perihelion."),
    dict(name="Orcus",      a=39.17,  e=0.227, inc=20.57, Omega=268.6,  omega=72.3,  M=178.2, epoch_jd=2459600.5, radius_km=459.0,  notes="Plutino; binary with Vanth."),
    dict(name="Salacia",    a=42.19,  e=0.109, inc=23.94, Omega=280.0,  omega=310.0, M=260.0, epoch_jd=2459600.5, radius_km=427.0,  notes="Binary (Actaea)."),
    dict(name="Varuna",     a=43.13,  e=0.051, inc=17.20, Omega=97.27,  omega=265.0, M=97.0,  epoch_jd=2459600.5, radius_km=339.0,  notes="Rapid rotator; elongated shape."),
    dict(name="Gonggong",   a=67.21,  e=0.500, inc=30.70, Omega=336.9,  omega=207.4, M=105.0, epoch_jd=2459600.5, radius_km=615.0,  notes="Highly eccentric; binary."),
    dict(name="Chaos",      a=45.85,  e=0.103, inc=12.02, Omega=58.50,  omega=57.5,  M=74.0,  epoch_jd=2459600.5, radius_km=300.0,  notes="Classical KBO."),
    dict(name="Arrokoth",   a=44.58,  e=0.042, inc=2.45,  Omega=158.7,  omega=174.7, M=316.5, epoch_jd=2459600.5, radius_km=9.0,   notes="New Horizons flyby target."),
    dict(name="2002 MS4",   a=41.93,  e=0.143, inc=17.69, Omega=215.9,  omega=212.2, M=132.5, epoch_jd=2459600.5, radius_km=385.0,  notes="Classical KBO."),
    dict(name="2002 AW197", a=47.1,   e=0.131, inc=24.4,  Omega=297.4,  omega=296.4, M=108.5, epoch_jd=2459600.5, radius_km=330.0,  notes="Classical KBO."),
    dict(name="Huya",       a=39.75,  e=0.282, inc=15.47, Omega=169.3,  omega=68.0,  M=222.0, epoch_jd=2459600.5, radius_km=214.0,  notes="Plutino; binary."),
    dict(name="Varda",      a=45.91,  e=0.141, inc=21.50, Omega=184.1,  omega=143.5, M=290.0, epoch_jd=2459600.5, radius_km=370.0,  notes="Binary TNO."),
    dict(name="2003 AZ84",  a=39.40,  e=0.175, inc=13.55, Omega=252.2,  omega=13.2,  M=199.0, epoch_jd=2459600.5, radius_km=340.0,  notes="Plutino; multi-chord occultation."),
    dict(name="Lempo",      a=39.79,  e=0.219, inc=8.41,  Omega=97.0,   omega=278.0, M=151.0, epoch_jd=2459600.5, radius_km=190.0,  notes="Triple system."),
    dict(name="2007 UK126", a=73.7,   e=0.490, inc=23.4,  Omega=131.0,  omega=345.0, M=15.0,  epoch_jd=2459600.5, radius_km=320.0,  notes="Scattered TNO."),
    dict(name="Dziewanna",  a=61.0,   e=0.288, inc=22.0,  Omega=184.7,  omega=323.0, M=325.0, epoch_jd=2459600.5, radius_km=294.0,  notes="Detached SDO."),
    dict(name="2013 FY27",  a=59.3,   e=0.392, inc=33.0,  Omega=199.0,  omega=191.0, M=326.0, epoch_jd=2459600.5, radius_km=370.0,  notes="Large SDO."),
    dict(name="2015 RR245", a=82.1,   e=0.590, inc=8.1,   Omega=65.4,   omega=105.0, M=8.5,   epoch_jd=2459600.5, radius_km=306.0,  notes="Large TNO; near perihelion."),
    dict(name="2018 VG18",  a=312.0,  e=0.832, inc=24.0,  Omega=95.0,   omega=270.0, M=0.3,   epoch_jd=2459600.5, radius_km=250.0,  notes="Farout — most distant known."),
]

# ═══════════════════════════════════════════════════════════════════════════════
# MPC CATALOG FETCHER  (distant objects JSON from Minor Planet Center)
# ═══════════════════════════════════════════════════════════════════════════════

## Orbits for TNOs, Centaurs and SDOs
MPC_DISTANT_URL  = "https://minorplanetcenter.net/Extended_Files/distant_extended.json.gz"

## Orbits for all asteroids in the MPC database (large fallback)
MPC_EXTENDED_URL = "https://www.minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz"

## Local cache path — avoids re-downloading on every run
MPC_CACHE_PATH   = Path("/tmp/mpc_distant_cache.json")


def fetch_mpc_catalog(min_a: float = 30.0, max_a: float = 1000.0,
                      min_radius_km: float = 80.0,
                      verbose: bool = True,
                      use_cache: bool = True) -> list:
    """
    Fetch TNOs from the MPC distant-objects JSON (gzipped).

    Field mapping (actual MPC Extended JSON keys, confirmed from live data):
        a        → semi-major axis [AU]
        e        → eccentricity
        i        → inclination [deg]
        Node     → longitude of ascending node [deg]
        Peri     → argument of perihelion [deg]
        M        → mean anomaly [deg]
        Epoch    → epoch [JD]
        H        → absolute magnitude
        Principal_desig → designation (when Name is absent)
        Name     → IAU name (may be empty)

    Applies semi-major axis and estimated-radius filters.
    Caches the raw JSON to /tmp so subsequent runs are instant.
    """
    # ── Try loading from local cache first ───────────────────────────────────
    records = None
    if use_cache and MPC_CACHE_PATH.exists():
        try:
            if verbose:
                print(f"  Loading MPC catalog from cache ({MPC_CACHE_PATH}) …")
            records = json.loads(MPC_CACHE_PATH.read_text())
            if verbose:
                print(f"  → {len(records):,} cached records.")
        except Exception:
            records = None  # cache corrupt, re-fetch

    # ── Fetch from MPC if not cached ─────────────────────────────────────────
    if records is None:
        for url in [MPC_DISTANT_URL, MPC_EXTENDED_URL]:
            try:
                label = "distant" if "distant" in url else "extended"
                if verbose:
                    print(f"  Fetching MPC {label} catalog from network …")
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                raw = gzip.decompress(resp.content)
                records = json.loads(raw)
                if verbose:
                    print(f"  → {len(records):,} raw records received.")
                # Save to cache
                MPC_CACHE_PATH.write_text(json.dumps(records))
                if verbose:
                    print(f"  → Cached to {MPC_CACHE_PATH}")
                break
            except Exception as ex:
                if verbose:
                    print(f"  ⚠  Fetch failed ({url}): {ex}")
                records = None

    if not records:
        raise RuntimeError("Could not fetch MPC catalog and no cache available.")

    # ── Parse records using the actual MPC Extended JSON field names ──────────
    objects = []
    skipped_a = skipped_e = skipped_r = 0
    for rec in records:
        try:
            # Orbital elements — MPC Extended JSON uses lowercase 'a', 'e', 'i'
            a     = float(rec.get("a")    or 0)
            e     = float(rec.get("e")    or 0)
            inc   = float(rec.get("i")    or 0)
            Omega = float(rec.get("Node") or 0)
            omega = float(rec.get("Peri") or 0)
            M_deg = float(rec.get("M")    or 0)
            # Epoch is already a JD float in this format
            epoch = float(rec.get("Epoch") or 2459600.5)
            H     = float(rec.get("H")    or 12)

            # Name: prefer proper name, fall back to designation
            name_raw = (rec.get("Name") or "").strip()
            desig    = (rec.get("Principal_desig") or
                        rec.get("Designation_and_Name") or
                        rec.get("Number") or "?").strip()
            name = name_raw if name_raw else desig

            # Filters
            if not (min_a <= a <= max_a):
                skipped_a += 1;  continue
            if not (0.0 <= e < 1.0):
                skipped_e += 1;  continue

            # Estimate radius from H magnitude (albedo ~0.09 typical for cold KBOs)
            albedo    = 0.09
            radius_km = 664.5 / np.sqrt(albedo) * 10 ** ((5.0 - H) / 5.0)
            if radius_km < min_radius_km:
                skipped_r += 1;  continue

            objects.append(dict(
                name=name[:35],
                a=a, e=e, inc=inc, Omega=Omega, omega=omega, M=M_deg,
                epoch_jd=epoch,
                radius_km=radius_km,
                notes=f"MPC (H={H:.1f}, est. r≈{radius_km:.0f} km)",
            ))
        except (TypeError, ValueError, KeyError):
            continue

    if verbose:
        print(f"  → {len(objects)} objects pass filters  "
              f"(a={min_a}–{max_a} AU, r≥{min_radius_km} km)  "
              f"[skipped: a={skipped_a}, e={skipped_e}, r={skipped_r}]")

    if not objects:
        raise RuntimeError(
            f"No objects survived filters (min_a={min_a}, max_a={max_a}, "
            f"min_radius={min_radius_km}). Try relaxing --min_radius."
        )
    return objects


# ═══════════════════════════════════════════════════════════════════════════════
# KEPLERIAN PROPAGATION  (two-body, heliocentric)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_kepler(M: np.ndarray, e: float, tol: float = 1e-12) -> np.ndarray:
    E = M.copy()
    for _ in range(100):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def heliocentric_posvel(tno: dict, t_jd: np.ndarray):
    """
    Two-body Keplerian heliocentric position [AU, (3,N)] and velocity
    [AU/day, (3,N)] for a TNO at times t_jd.
    """
    a     = float(tno["a"]);   e  = float(tno["e"])
    inc   = np.radians(float(tno["inc"]))
    Omega = np.radians(float(tno["Omega"]))
    omega = np.radians(float(tno["omega"]))
    M0    = np.radians(float(tno["M"]))
    t0    = float(tno["epoch_jd"])

    n  = np.sqrt(GM_SUN_AU3_D2 / a**3)
    M  = (M0 + n * (np.asarray(t_jd, float) - t0)) % (2 * np.pi)
    E  = solve_kepler(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    r  = a * (1.0 - e * np.cos(E))

    xo  = r * np.cos(nu);  yo  = r * np.sin(nu)
    h   = np.sqrt(GM_SUN_AU3_D2 * a * (1.0 - e**2))
    vxo = -GM_SUN_AU3_D2 / h * np.sin(nu)
    vyo =  GM_SUN_AU3_D2 / h * (e + np.cos(nu))

    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(inc),   np.sin(inc)
    co, so = np.cos(omega), np.sin(omega)

    Rx = np.array([
        [ cO*co - sO*so*ci, -cO*so - sO*co*ci,  sO*si],
        [ sO*co + cO*so*ci, -sO*so + cO*co*ci, -cO*si],
        [ so*si,             co*si,              ci   ],
    ])
    pos = Rx @ np.vstack([xo, yo, np.zeros_like(xo)])
    vel = Rx @ np.vstack([vxo, vyo, np.zeros_like(xo)])
    return pos, vel   # (3,N) AU, AU/day


# ═══════════════════════════════════════════════════════════════════════════════
# EARTH EPHEMERIS  via astropy built-in DE430
# ═══════════════════════════════════════════════════════════════════════════════

def earth_posvel(t_jd: np.ndarray):
    """
    Heliocentric Earth position [AU, (3,N)] and velocity [AU/day, (3,N)]
    using astropy's built-in DE430 (no internet required).
    """
    times = AstroTime(t_jd, format="jd", scale="tdb")
    pv    = get_body_barycentric_posvel("earth", times)

    e_pos = pv[0].xyz.to(u.AU).value          # (3,N)
    e_vel = pv[1].xyz.to(u.AU / u.day).value  # (3,N)

    # Sun barycentric (to convert bary→helio)
    sv    = get_body_barycentric_posvel("sun", times)
    s_pos = sv[0].xyz.to(u.AU).value
    s_vel = sv[1].xyz.to(u.AU / u.day).value

    return e_pos - s_pos, e_vel - s_vel


# ═══════════════════════════════════════════════════════════════════════════════
# RVSP PHYSICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_v_shadow(tno, t_jd, earth_pos, earth_vel):
    """
    v_shadow [km/s], geocentric distance [AU], RA [deg], Dec [deg] — all (N,).
    """
    tp, tv = heliocentric_posvel(tno, t_jd)   # (3,N) AU, AU/day

    geo = tp - earth_pos                        # geocentric TNO vector (3,N)
    dist_au = np.linalg.norm(geo, axis=0)      # (N,)
    los = geo / dist_au                        # unit LOS (3,N)

    ra_deg  = np.degrees(np.arctan2(los[1], los[0])) % 360.0
    dec_deg = np.degrees(np.arcsin(np.clip(los[2], -1, 1)))

    rel_vel = tv - earth_vel                   # (3,N) AU/day
    v_dot   = np.einsum("ij,ij->j", rel_vel, los)   # (N,)
    v_perp  = rel_vel - los * v_dot            # perpendicular component (3,N)
    v_perp_norm = np.linalg.norm(v_perp, axis=0)    # (N,)  AU/day

    v_shadow = (v_perp_norm / dist_au) * (AU_KM / DAY_S)  # km/s
    return v_shadow, dist_au, ra_deg, dec_deg


def fresnel_scale_km(dist_au: float) -> float:
    return np.sqrt(LAMBDA_M * dist_au * AU_M / 2.0) / 1e3


def check_viability(tno, v_min, dist_au, v_thresh=1.0):
    r_km   = float(tno.get("radius_km", 0))
    W      = 2 * r_km
    F      = fresnel_scale_km(dist_au)
    ratio  = W / F if F > 0 else 0.0
    ok_spd = v_min <= v_thresh
    ok_frn = ratio >= 5.0
    return dict(viable=(ok_spd and ok_frn),
                ratio_W_F=ratio, fresnel_km=F, shadow_width_km=W,
                ok_speed=ok_spd, ok_fresnel=ok_frn)


def jd_to_iso(jd: float, ts) -> str:
    return ts.tt_jd(float(jd)).utc_iso()[:10]


def find_rvsp(tno, t_jd, earth_pos, earth_vel, ts,
              v_threshold=1.0, L_km=500.0):
    v_shadow, dist_au, ra_deg, dec_deg = compute_v_shadow(
        tno, t_jd, earth_pos, earth_vel
    )

    idx_min  = int(np.argmin(v_shadow))
    v_min    = float(v_shadow[idx_min])
    d_best   = float(dist_au[idx_min])
    t_eff_best = L_km / v_min if v_min > 0 else 1e9
    iso_min  = jd_to_iso(t_jd[idx_min], ts)

    below = v_shadow < v_threshold
    windows = []
    if np.any(below):
        changes = np.diff(below.astype(int))
        starts  = list(np.where(changes == 1)[0] + 1)
        ends    = list(np.where(changes == -1)[0])
        if below[0]:  starts.insert(0, 0)
        if below[-1]: ends.append(len(below) - 1)
        for s, e_ in zip(starts, ends):
            pk = s + int(np.argmin(v_shadow[s:e_+1]))
            windows.append(dict(
                start=jd_to_iso(t_jd[s], ts),
                end=jd_to_iso(t_jd[e_], ts),
                peak=jd_to_iso(t_jd[pk], ts),
                v_min=float(v_shadow[pk]),
                t_eff=L_km / float(v_shadow[pk]) if v_shadow[pk] > 0 else 1e9,
                d_au=float(dist_au[pk]),
                ra=float(ra_deg[pk]),
            ))

    vis = check_viability(tno, v_min, d_best, v_thresh=v_threshold)

    return dict(
        tno=tno,
        v_shadow=v_shadow,
        dist_au=dist_au,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        v_min=v_min,
        idx_min=idx_min,
        d_best=d_best,
        ra_best=float(ra_deg[idx_min]),
        dec_best=float(dec_deg[idx_min]),
        iso_min=iso_min,
        t_eff_best=t_eff_best,
        windows=windows,
        viability=vis,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results, L_km, v_thresh, start, end):
    ranked = sorted(results, key=lambda r: r["v_min"])
    W = 105
    print("\n" + "╔" + "═"*(W-2) + "╗")
    print(f"║  RVSP FINDER (Skyfield + MPC Catalog){' '*(W-39)}║")
    print(f"║  Window: {start} → {end}  |  L={L_km:.0f} km  |  threshold={v_thresh:.2f} km/s{' '*(W-64)}║")
    print("╠" + "═"*(W-2) + "╣")
    hdr = f"  {'Name':<22} {'Best RVSP':<12} {'v_shadow':>9} {'t_eff [s]':>10} {'d [AU]':>8} {'RA°':>7} {'Dec°':>7} {'W/F':>8}  Viable"
    print(f"║{hdr:<{W-2}}║")
    print("╠" + "═"*(W-2) + "╣")
    for r in ranked:
        n   = r["tno"]["name"][:21]
        vis = r["viability"]
        flg = "✓ YES" if vis["viable"] else "✗ NO "
        row = (f"  {n:<22} {r['iso_min']:<12} {r['v_min']:>9.4f} "
               f"{r['t_eff_best']:>10.0f} {r['d_best']:>8.1f} "
               f"{r['ra_best']:>7.1f} {r['dec_best']:>7.1f} "
               f"{vis['ratio_W_F']:>8.2f}  {flg}")
        print(f"║{row:<{W-2}}║")
    print("╠" + "═"*(W-2) + "╣")
    note = "  v_shadow [km/s] · t_eff=L/v [s] · d geocentric [AU] · W/F=shadow_width/Fresnel (need ≥5)"
    print(f"║{note:<{W-2}}║")
    print("╚" + "═"*(W-2) + "╝")


def save_report(results, L_km, v_thresh, start, end, path):
    ranked = sorted(results, key=lambda r: r["v_min"])
    lines  = [
        "RVSP FINDER — Skyfield + MPC Catalog Edition",
        f"Window: {start} → {end}  |  L={L_km:.0f} km  |  threshold={v_thresh:.2f} km/s",
        "=" * 70,
    ]
    for r in ranked:
        tno = r["tno"]; vis = r["viability"]
        lines += [
            f"\n  {'─'*66}",
            f"  {tno['name']}",
            f"  {'─'*66}",
            f"  Best RVSP date  : {r['iso_min']}",
            f"  v_shadow (min)  : {r['v_min']:.5f} km/s",
            f"  t_eff           : {r['t_eff_best']:.1f} s  ({r['t_eff_best']/60:.1f} min)  [L={L_km:.0f} km]",
            f"  Geocentric dist : {r['d_best']:.3f} AU",
            f"  Sky position    : RA={r['ra_best']:.2f}°  Dec={r['dec_best']:.2f}°",
            f"  Shadow width    : {vis['shadow_width_km']:.0f} km (diameter)",
            f"  Fresnel scale   : {vis['fresnel_km']:.3f} km  at {r['d_best']:.1f} AU, λ={LAMBDA_NM:.0f} nm",
            f"  W / F_Fresnel   : {vis['ratio_W_F']:.2f}  {'✓' if vis['ok_fresnel'] else '✗'}  (need ≥ 5)",
            f"  Speed criterion : {'✓' if vis['ok_speed'] else '✗'}  v_shadow ≤ {v_thresh} km/s",
            f"  Overall viable  : {'✓ YES' if vis['viable'] else '✗ NO'}",
            f"  Notes           : {tno.get('notes','')}",
        ]
        if r["windows"]:
            lines.append(f"\n  RVSP Windows below {v_thresh:.1f} km/s:")
            lines.append(f"  {'Start':<14} {'End':<14} {'Peak':<14} {'v_min':>12} {'t_eff':>10} {'d[AU]':>8} {'RA°':>7}")
            for w in r["windows"]:
                lines.append(f"  {w['start']:<14} {w['end']:<14} {w['peak']:<14} "
                             f"{w['v_min']:>12.5f} {w['t_eff']:>10.1f} "
                             f"{w['d_au']:>8.2f} {w['ra']:>7.1f}")
    Path(path).write_text("\n".join(lines))
    print(f"  Report saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB PLOTS — dark theme
# ═══════════════════════════════════════════════════════════════════════════════

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
TEXT_COL = "#e6edf3"
ACCENT   = "#58a6ff"
HIGH     = "#f78166"
SUCCESS  = "#3fb950"
WARN     = "#d29922"
MID      = "#8b949e"
GRID_C   = "#21262d"


def obj_color(i, n):
    r, g, b, _ = plt.cm.turbo(i / max(n - 1, 1))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
        "axes.edgecolor": MID, "axes.labelcolor": TEXT_COL,
        "xtick.color": MID, "ytick.color": MID,
        "text.color": TEXT_COL, "grid.color": GRID_C,
        "legend.facecolor": PANEL_BG, "legend.edgecolor": MID,
    })


def jd_array_to_years(t_jd, ts):
    """Convert JD array to decimal years via Skyfield."""
    years = []
    for j in t_jd:
        dt = ts.tt_jd(float(j)).utc_datetime()
        years.append(dt.year + (dt.timetuple().tm_yday - 1) / 365.25)
    return np.array(years)


def panel_timeline(ax, results, t_jd, ts, v_thresh, L_km, top_n=15):
    years  = jd_array_to_years(t_jd, ts)
    ranked = sorted(results, key=lambda r: r["v_min"])[:top_n]

    ax.set_facecolor(PANEL_BG)
    for i, r in enumerate(ranked):
        v  = r["v_shadow"]
        te = np.clip(np.where(v > 0, L_km / v, 1e9), 1, 1e8)
        c  = obj_color(i, top_n)
        ax.plot(years, te, lw=1.3, color=c, alpha=0.85,
                label=f"{r['tno']['name'][:18]}  v={r['v_min']:.3f} km/s")
        ax.scatter(years[r["idx_min"]], te[r["idx_min"]],
                   s=70, c=c, marker="*", zorder=8)

    ax.set_yscale("log"); ax.set_ylim(10, 5e7)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("t_eff [s]  (log)", fontsize=9, color=ACCENT)
    ax.tick_params(axis="y", labelcolor=ACCENT)
    ax.set_title(f"Effective Integration Time  ★ = RVSP  (top {top_n} by v_shadow)",
                 fontsize=9, pad=5)
    ax.legend(loc="upper right", fontsize=6.0, framealpha=0.8, ncol=2)
    ax.grid(True, which="both", color=GRID_C, lw=0.35, alpha=0.6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def panel_bar(ax, results, L_km, top_n=30):
    ranked = sorted(results, key=lambda r: r["t_eff_best"], reverse=True)[:top_n]
    names  = [r["tno"]["name"][:20] for r in ranked]
    teffs  = [min(r["t_eff_best"], 5e6) for r in ranked]
    colors = [SUCCESS if r["viability"]["viable"] else WARN for r in ranked]

    ax.set_facecolor(PANEL_BG)
    y = np.arange(len(names))
    ax.barh(y, teffs, color=colors, alpha=0.8, height=0.72)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("Best t_eff [s]", fontsize=8)
    ax.set_title(f"Best Integration Times — Top {top_n}  (green=viable)", fontsize=8, pad=5)
    ax.axvline(3600,  color=ACCENT, lw=1.0, linestyle="--", alpha=0.6, label="1 hr")
    ax.axvline(86400, color=HIGH,   lw=1.0, linestyle=":", alpha=0.5, label="1 day")
    ax.legend(fontsize=7)
    ax.grid(True, axis="x", which="both", color=GRID_C, lw=0.35, alpha=0.5)


def panel_sky(ax, results, top_n=60):
    ranked  = sorted(results, key=lambda r: r["v_min"])
    viable  = [r for r in ranked if r["viability"]["viable"]]
    show    = (viable or ranked)[:top_n]

    ax.set_facecolor(PANEL_BG)
    ras   = [r["ra_best"]  for r in show]
    decs  = [r["dec_best"] for r in show]
    vmins = [r["v_min"]    for r in show]
    sizes = [max(15, min(250, 40/(v+0.01))) for v in vmins]
    sc = ax.scatter(ras, decs, c=vmins, s=sizes, cmap="plasma_r",
                    alpha=0.8, zorder=5, vmin=0,
                    vmax=min(1.0, max(vmins) if vmins else 1.0))
    cb = plt.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("v_shadow [km/s]", fontsize=7, color=TEXT_COL)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL, fontsize=6)
    for r in show[:20]:
        ax.annotate(r["tno"]["name"][:10],
                    xy=(r["ra_best"], r["dec_best"]),
                    fontsize=5.0, color=TEXT_COL, alpha=0.7,
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    ax.set_xlabel("RA [°]", fontsize=8); ax.set_ylabel("Dec [°]", fontsize=8)
    ax.set_title("Sky Positions at Best RVSP  (size ∝ t_eff)", fontsize=8, pad=5)
    ax.grid(True, color=GRID_C, lw=0.4, alpha=0.5)


def panel_vdist(ax, results):
    ax.set_facecolor(PANEL_BG)
    for r in results:
        col = SUCCESS if r["viability"]["viable"] else (WARN if r["v_min"] < 1.5 else MID)
        sz  = max(8, min(300, r["viability"]["shadow_width_km"] * 0.25))
        ax.scatter(r["d_best"], r["v_min"], c=col, s=sz, alpha=0.65, zorder=5)
        if r["v_min"] < 0.4:
            ax.annotate(r["tno"]["name"][:12],
                        xy=(r["d_best"], r["v_min"]),
                        fontsize=5.0, color=TEXT_COL, alpha=0.75,
                        xytext=(3, 2), textcoords="offset points")
    ax.axhline(1.0, color=WARN, lw=1.0, linestyle="--", alpha=0.6, label="1 km/s threshold")
    ax.set_xlabel("Geocentric distance [AU]", fontsize=8)
    ax.set_ylabel("v_shadow min [km/s]", fontsize=8)
    ax.set_title("v_shadow vs Distance  (size ∝ shadow width)", fontsize=8, pad=5)
    ax.legend(fontsize=7)
    ax.grid(True, color=GRID_C, lw=0.4, alpha=0.5)


def shadow_groundtrack(best_result, t_jd, earth_pos, earth_vel, ts,
                       window_hours=6.0, n_steps=400):
    """
    Compute the shadow ground-track on Earth's surface centred on the RVSP.

    The sub-TNO point (where the Earth→TNO line intersects Earth's surface)
    is tracked as a function of time via:
      - Geocentric unit vector to the TNO → equatorial RA/Dec
      - Earth's rotation (GST) → geographic longitude = RA − GST

    Returns dict with arrays: lats, lons, times_iso, v_shadow, dist_km, jd.
    """
    tno      = best_result["tno"]
    idx_rvsp = best_result["idx_min"]
    jd_rvsp  = float(t_jd[idx_rvsp])

    dt_days = window_hours / 24.0
    t_fine  = np.linspace(jd_rvsp - dt_days, jd_rvsp + dt_days, n_steps)

    times_ap = AstroTime(t_fine, format="jd", scale="tdb")
    pv_e = get_body_barycentric_posvel("earth", times_ap)
    pv_s = get_body_barycentric_posvel("sun",   times_ap)
    ep   = pv_e[0].xyz.to(u.AU).value - pv_s[0].xyz.to(u.AU).value
    ev   = pv_e[1].xyz.to(u.AU / u.day).value - pv_s[1].xyz.to(u.AU / u.day).value

    tp, tv  = heliocentric_posvel(tno, t_fine)
    geo     = tp - ep
    dist_au = np.linalg.norm(geo, axis=0)
    los     = geo / dist_au

    ra_los  = np.degrees(np.arctan2(los[1], los[0])) % 360.0
    dec_los = np.degrees(np.arcsin(np.clip(los[2], -1, 1)))

    # Greenwich Sidereal Time (mean, degrees)
    JD_J2000            = 2451545.0
    EARTH_ROT_DEG_PER_D = 360.9856235
    gst_deg = (t_fine - JD_J2000) * EARTH_ROT_DEG_PER_D % 360.0

    lons = (ra_los - gst_deg + 360.0) % 360.0
    lons[lons > 180] -= 360.0
    lats = dec_los

    rel_vel       = tv - ev
    v_dot         = np.einsum("ij,ij->j", rel_vel, los)
    v_perp        = rel_vel - los * v_dot
    v_shadow_fine = np.linalg.norm(v_perp, axis=0) / dist_au * (AU_KM / DAY_S)

    times_iso = [ts.tt_jd(float(j)).utc_iso()[:16] for j in t_fine]
    return dict(lats=lats, lons=lons, times_iso=times_iso,
                v_shadow=v_shadow_fine, dist_km=dist_au * AU_KM, jd=t_fine)


def panel_groundtrack(ax, best_result, trk, ts, window_hours):
    """World-map panel of the shadow ground-track for the best RVSP."""
    from matplotlib.collections import LineCollection as LC
    tno     = best_result["tno"]
    lats    = trk["lats"]
    lons    = trk["lons"]
    vshadow = trk["v_shadow"]
    n       = len(lats)
    idx_min = int(np.argmin(vshadow))

    ax.set_facecolor("#08131e")

    # Rough continent fills (no external geodata needed)
    land = [
        ([-170,-50,-50,-170,-170], [25,25,75,75,25]),
        ([-80,-35,-35,-80,-80],    [-55,-55,13,13,-55]),
        ([-5, 40, 40, -5, -5],     [35,35,72,72,35]),
        ([25, 60, 60, 25, 25],     [10,10,40,40,10]),
        ([60,150,150, 60, 60],     [5, 5,55,55, 5]),
        ([10, 55, 55, 10, 10],     [-35,-35,38,38,-35]),
        ([113,155,155,113,113],    [-40,-40,-5,-5,-40]),
    ]
    for lx, ly in land:
        ax.fill(lx, ly, color="#1a2e1a", alpha=0.55, zorder=1)
        ax.plot(lx, ly, color="#2a3e2a", lw=0.4, alpha=0.5, zorder=2)

    for lat0 in range(-90, 91, 30):
        ax.axhline(lat0, color="#0d2030", lw=0.6, alpha=0.8)
    for lon0 in range(-180, 181, 60):
        ax.axvline(lon0, color="#0d2030", lw=0.6, alpha=0.8)
    for lat0 in [-60,-30,0,30,60]:
        ax.text(-178, lat0+1, f"{lat0:+d}\u00b0", fontsize=5, color=MID)
    for lon0 in [-120,-60,0,60,120]:
        ax.text(lon0+1, -87, f"{lon0:+d}\u00b0", fontsize=5, color=MID)

    # Colour track by v_shadow — split at longitude wrap-arounds
    segments, colors_seg = [], []
    for i in range(n - 1):
        if abs(lons[i+1] - lons[i]) < 90:
            segments.append([(lons[i], lats[i]), (lons[i+1], lats[i+1])])
            colors_seg.append(0.5*(vshadow[i]+vshadow[i+1]))

    vlo = vshadow.min()
    vhi = max(vshadow.min() * 8, vshadow.min() + 0.05)
    lc = LC(segments, cmap="cool_r",
            norm=plt.Normalize(vlo, vhi),
            linewidth=3.0, zorder=6, alpha=0.9)
    lc.set_array(np.array(colors_seg))
    ax.add_collection(lc)
    cb = plt.colorbar(lc, ax=ax, pad=0.01, shrink=0.82, aspect=20)
    cb.set_label("v_shadow [km/s]", fontsize=7.5, color=TEXT_COL)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL, fontsize=6.5)

    ax.scatter(lons[idx_min], lats[idx_min], s=250, c=HIGH,
               marker="*", zorder=12, edgecolors="white", linewidths=0.7,
               label=f"\u2605 RVSP  {trk['times_iso'][idx_min]} UTC")

    # Direction arrows
    for frac in [0.25, 0.5, 0.75]:
        i = int(frac * (n - 3))
        if abs(lons[i+2] - lons[i]) < 90:
            ax.annotate("", xy=(lons[i+2], lats[i+2]), xytext=(lons[i], lats[i]),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT,
                                        lw=1.3, mutation_scale=12), zorder=9)

    # UTC labels
    step = max(1, n // 14)
    seen = set()
    for i in range(0, n, step):
        hh = trk["times_iso"][i][11:13]
        if hh not in seen:
            seen.add(hh)
            ax.annotate(trk["times_iso"][i][11:16], xy=(lons[i], lats[i]),
                        fontsize=5.5, color=TEXT_COL, alpha=0.85,
                        xytext=(3, 5), textcoords="offset points", zorder=10)

    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [\u00b0]", fontsize=8)
    ax.set_ylabel("Latitude [\u00b0]",  fontsize=8)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
    ax.set_title(
        f"Shadow Ground-Track \u2014 {tno['name']}  |  RVSP: {best_result['iso_min']}\n"
        f"\u00b1{window_hours:.0f} h window  |  "
        f"v_min = {best_result['v_min']:.4f} km/s  |  "
        f"shadow \u2205 = {best_result['viability']['shadow_width_km']:.0f} km",
        fontsize=8, pad=5,
    )


def make_plots(results, t_jd, earth_pos, earth_vel, ts, L_km, v_thresh, out_path,
               top_n=15, track_window_hours=6.0):
    dark_style()

    best = sorted(results, key=lambda r: r["v_min"])[0]

    print("  Computing shadow ground-track for best candidate … ", end="", flush=True)
    trk = shadow_groundtrack(best, t_jd, earth_pos, earth_vel, ts,
                              window_hours=track_window_hours)
    print("done.")

    fig = plt.figure(figsize=(26, 22), facecolor=DARK_BG)
    gs  = mgs.GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1.1],
                        hspace=0.44, left=0.05, right=0.97, top=0.94, bottom=0.04)

    ax_time = fig.add_subplot(gs[0])
    gs_mid  = mgs.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.32)
    ax_bar  = fig.add_subplot(gs_mid[0])
    ax_sky  = fig.add_subplot(gs_mid[1])
    ax_vd   = fig.add_subplot(gs_mid[2])
    ax_trk  = fig.add_subplot(gs[2])

    panel_timeline   (ax_time, results, t_jd, ts, v_thresh, L_km, top_n=top_n)
    panel_bar        (ax_bar,  results, L_km)
    panel_sky        (ax_sky,  results)
    panel_vdist      (ax_vd,   results)
    panel_groundtrack(ax_trk,  best, trk, ts, track_window_hours)

    n_viable = sum(1 for r in results if r["viability"]["viable"])
    fig.suptitle(
        f"RVSP Finder (Skyfield + MPC) — {len(results)} objects scanned, "
        f"{n_viable} viable  |  L={L_km:.0f} km  |  threshold={v_thresh:.2f} km/s",
        color=TEXT_COL, fontsize=12, fontweight="bold", y=0.995,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Figure saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="RVSP Finder (Skyfield + MPC) — large-catalog TNO RVSP scanner."
    )
    ap.add_argument("--start",      default="2025-01-01")
    ap.add_argument("--end",        default="2035-01-01")
    ap.add_argument("--step_days",  type=float, default=5.0)
    ap.add_argument("--threshold",  type=float, default=1.0,  help="v_shadow cut [km/s]")
    ap.add_argument("--L",          type=float, default=500.0, help="Array baseline [km]")
    ap.add_argument("--top",        type=int,   default=15,   help="Top N on timeline plot")
    ap.add_argument("--local",      action="store_true",      help="Skip MPC fetch")
    ap.add_argument("--min_dist",   type=float, default=30.0, help="Min a [AU] for MPC filter")
    ap.add_argument("--max_dist",   type=float, default=800.0,help="Max a [AU] for MPC filter")
    ap.add_argument("--min_radius", type=float, default=80.0, help="Min radius [km] for MPC filter")
    ap.add_argument("--out",        default="rvsp_skyfield.png")
    ap.add_argument("--report",     default="rvsp_skyfield_report.txt")
    ap.add_argument("--track-hours", type=float, default=6.0,  help="Ground-track window ±hours (default 6)")
    ap.add_argument("--no-cache",    action="store_true",      help="Ignore cached MPC data and re-fetch")
    args = ap.parse_args()

    print(f"\n{'═'*72}")
    print(f"  RVSP Finder  (Skyfield timescales + astropy DE430 + MPC catalog)")
    print(f"  Window : {args.start}  →  {args.end}")
    print(f"  Step   : {args.step_days:.0f} days")
    print(f"  L      : {args.L:.0f} km  |  v_threshold : {args.threshold:.2f} km/s")
    print(f"{'═'*72}\n")

    # ── Skyfield timescale (for date formatting only) ─────────────────────────
    load = Loader(".")
    ts   = load.timescale()

    # ── Time grid ─────────────────────────────────────────────────────────────
    def iso_to_jd(s):
        y, m, d = s.split("-")
        return ts.utc(int(y), int(m), int(d)).tt

    t_start = iso_to_jd(args.start)
    t_end   = iso_to_jd(args.end)
    t_jd    = np.arange(t_start, t_end, args.step_days)
    print(f"  Time grid: {len(t_jd)} steps over {t_end-t_start:.0f} days.\n")

    # ── Catalog ───────────────────────────────────────────────────────────────
    if args.local:
        catalog = list(BUILTIN_CATALOG)
        print(f"  Using built-in catalog: {len(catalog)} objects.\n")
    else:
        catalog = fetch_mpc_catalog(
            min_a=args.min_dist, max_a=args.max_dist,
            min_radius_km=args.min_radius, verbose=True,
            use_cache=not args.no_cache,
        )
    print()

    # ── Earth ephemeris (computed once, shared for all TNOs) ──────────────────
    print("  Computing Earth heliocentric state (astropy DE430) … ", end="", flush=True)
    earth_pos, earth_vel = earth_posvel(t_jd)
    print("done.\n")

    # ── Scan ──────────────────────────────────────────────────────────────────
    results = []
    n = len(catalog)
    for i, tno in enumerate(catalog):
        tag = f"[{i+1}/{n}]"
        print(f"  {tag:<8} {tno['name']:<25} … ", end="", flush=True)
        try:
            r = find_rvsp(tno, t_jd, earth_pos, earth_vel, ts,
                          v_threshold=args.threshold, L_km=args.L)
            results.append(r)
            flag = "✓" if r["viability"]["viable"] else " "
            print(f"{flag}  v={r['v_min']:.4f} km/s  t_eff={r['t_eff_best']:.0f} s  "
                  f"peak={r['iso_min']}")
        except Exception as ex:
            print(f"⚠ skipped: {ex}")

    if not results:
        print("  No results."); return

    # ── Report & plots ────────────────────────────────────────────────────────
    print()
    print_report(results, args.L, args.threshold, args.start, args.end)
    save_report (results, args.L, args.threshold, args.start, args.end, args.report)

    print("\n  Generating plots …")
    make_plots(results, t_jd, earth_pos, earth_vel, ts,
               args.L, args.threshold, args.out,
               top_n=args.top, track_window_hours=args.track_hours)


    n_viable = sum(1 for r in results if r["viability"]["viable"])
    print(f"  Outputs:")
    print(f"    PNG    → {args.out}")
    print(f"    Report → {args.report}")
    print(f"\n  Done.  ({len(results)} objects scanned, {n_viable} viable)\n")


if __name__ == "__main__":
    main()
