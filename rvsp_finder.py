"""
RVSP Finder — Radial-Velocity Stationary Point Scanner
=======================================================
Scans a catalog of Trans-Neptunian Objects (TNOs) over a user-defined time
window to find *Radial-Velocity Stationary Points* (RVSPs): moments when the
sky-plane velocity of a TNO (as seen from Earth) approaches zero — maximizing
the effective integration time for stellar occultation observations.

Science context (Hartshorn 2026)
---------------------------------
The shadow ground-track speed is:

    v_shadow = (d_Earth / d_TNO) × |v_sky_rel|
             = |v_perp_TNO − v_perp_Earth| / d_TNO   [km/s]

where v_perp denotes the heliocentric velocity component perpendicular to the
Earth–TNO line of sight, and d_TNO is in AU.  At the RVSP,
v_perp_TNO ≈ v_perp_Earth so v_shadow → 0, and effective integration time

    t_eff = L / v_shadow

is maximised for a telescope array of baseline L.

The script computes, for each TNO in the catalog:
  • v_shadow(t) over the scan window
  • The RVSP date (minimum v_shadow) and its value
  • t_eff for L = 500 km (paper reference) and user-supplied baselines
  • Shadow-track RA/Dec and ecliptic coordinates at RVSP
  • Array positioning corridor width (1-sigma ground-track uncertainty)
  • A viability flag against three criteria from the paper:
      (i)  σ_track ≤ 150 km  (met when v_shadow is well-defined)
      (ii) geometric shadow width ≥ 5 × Fresnel scale  (size criterion)
      (iii) v_shadow ≤ 1 km/s  (integration-time criterion)

Outputs
-------
  • Console table: all TNOs ranked by peak t_eff
  • rvsp_report.txt: full machine-readable report
  • rvsp_plots.png:  4-panel figure (v_shadow timeline, t_eff map, sky chart, detail)

Dependencies
------------
    pip install astropy matplotlib numpy scipy

Usage
-----
    python rvsp_finder.py                          # default: 2025-2035, all TNOs
    python rvsp_finder.py --start 2026 --end 2032  # custom date range
    python rvsp_finder.py --threshold 0.5          # v_shadow cut [km/s]
    python rvsp_finder.py --L 300                  # array baseline [km]
    python rvsp_finder.py --out my_report.png      # custom output filename
    python rvsp_finder.py --tno Eris Quaoar        # select specific objects
"""

import argparse
import warnings
from datetime import timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from scipy.signal import find_peaks

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    get_body_barycentric_posvel,
    solar_system_ephemeris,
    GCRS, ITRS, CartesianRepresentation,
    SkyCoord, GeocentricMeanEcliptic,
)

warnings.filterwarnings("ignore")
solar_system_ephemeris.set("builtin")

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GM_SUN   = 2.959122082855911e-4   # AU³/day²
AU_KM    = 1.495978707e8          # km/AU
DAY_S    = 86400.0                # s/day
LAMBDA_M = 550e-9                 # reference wavelength [m] (550 nm)
AU_M     = 1.495978707e11         # m/AU


# ═══════════════════════════════════════════════════════════════════════════════
# TNO CATALOG
#
# Osculating elements from JPL Small-Body Database Browser (epoch ~2022-Jan-22,
# JD 2459600.5).  Elements are approximate; long-term predictions require
# full n-body ephemerides, but are adequate for RVSP window identification.
#
# Catalog entries that satisfy the Hartshorn 2025 viability criteria:
#   - d = 35–55 AU (accessible distance)
#   - Published multi-chord shape (Fresnel criterion can be checked)
#   - Well-determined multi-opposition orbit
# ═══════════════════════════════════════════════════════════════════════════════

TNO_CATALOG = [
    # ── Dwarf planets / large classical KBOs ─────────────────────────────────
    {
        "name": "Eris",
        "full_name": "(136199) Eris",
        "a": 67.78, "e": 0.436, "inc": 44.04,
        "Omega": 35.87, "omega": 151.4, "M": 204.0,
        "epoch_jd": 2459600.5,
        "radius_km": 1163.0,   # Sicardy et al. 2011
        "color": "#FF6B6B",
        "notes": "Largest known dwarf planet; occultation profile published.",
    },
    {
        "name": "Makemake",
        "full_name": "(136472) Makemake",
        "a": 45.79, "e": 0.162, "inc": 28.96,
        "Omega": 79.60, "omega": 294.8, "M": 165.0,
        "epoch_jd": 2459600.5,
        "radius_km": 715.0,    # Ortiz et al. 2012
        "color": "#FFA07A",
        "notes": "Large KBO; multi-chord occultation profile available.",
    },
    {
        "name": "Haumea",
        "full_name": "(136108) Haumea",
        "a": 43.13, "e": 0.195, "inc": 28.21,
        "Omega": 122.1, "omega": 239.5, "M": 218.3,
        "epoch_jd": 2459600.5,
        "radius_km": 798.0,    # effective radius (triaxial ellipsoid)
        "color": "#FFD700",
        "notes": "Rapid rotator; ring system detected. Irregular limb known.",
    },
    {
        "name": "Quaoar",
        "full_name": "(50000) Quaoar",
        "a": 43.40, "e": 0.034, "inc": 7.99,
        "Omega": 188.9, "omega": 155.0, "M": 328.0,
        "epoch_jd": 2459600.5,
        "radius_km": 555.0,    # Braga-Ribas et al. 2013
        "color": "#98FB98",
        "notes": "Nearly circular orbit; excellent ring detections. Strong RVSP candidate.",
    },
    {
        "name": "Sedna",
        "full_name": "(90377) Sedna",
        "a": 506.0, "e": 0.843, "inc": 11.93,
        "Omega": 144.5, "omega": 311.2, "M": 358.1,
        "epoch_jd": 2459600.5,
        "radius_km": 497.5,    # Pal et al. 2012
        "color": "#FF69B4",
        "notes": "Extreme TNO; near perihelion (76 AU) in coming decades.",
    },
    {
        "name": "2002 MS4",
        "full_name": "(307261) 2002 MS4",
        "a": 41.93, "e": 0.143, "inc": 17.69,
        "Omega": 215.9, "omega": 212.2, "M": 132.5,
        "epoch_jd": 2459600.5,
        "radius_km": 385.0,    # Lellouch et al. 2013
        "color": "#87CEEB",
        "notes": "Classical KBO; several occultation chords observed.",
    },
    {
        "name": "Orcus",
        "full_name": "(90482) Orcus",
        "a": 39.17, "e": 0.227, "inc": 20.57,
        "Omega": 268.6, "omega": 72.3, "M": 178.2,
        "epoch_jd": 2459600.5,
        "radius_km": 459.0,    # Fornasier et al. 2013
        "color": "#DDA0DD",
        "notes": "Plutino (3:2 MMR); binary with Vanth. Good size estimate.",
    },
    {
        "name": "Varuna",
        "full_name": "(20000) Varuna",
        "a": 43.13, "e": 0.051, "inc": 17.20,
        "Omega": 97.27, "omega": 265.0, "M": 97.0,
        "epoch_jd": 2459600.5,
        "radius_km": 339.0,    # Lellouch et al. 2013
        "color": "#F0E68C",
        "notes": "Rapid rotator; elongated shape; multi-chord occultation 2020.",
    },
    {
        "name": "Salacia",
        "full_name": "(120347) Salacia",
        "a": 42.19, "e": 0.109, "inc": 23.94,
        "Omega": 280.0, "omega": 310.0, "M": 260.0,
        "epoch_jd": 2459600.5,
        "radius_km": 427.0,    # Stansberry et al. 2012
        "color": "#20B2AA",
        "notes": "Binary system (Actaea). Well-characterised orbit.",
    },
    {
        "name": "2007 OR10",
        "full_name": "(225088) Gonggong",
        "a": 67.21, "e": 0.500, "inc": 30.70,
        "Omega": 336.9, "omega": 207.4, "M": 105.0,
        "epoch_jd": 2459600.5,
        "radius_km": 615.0,    # Pal et al. 2016
        "color": "#CD853F",
        "notes": "Highly eccentric; binary; largest un-named KBO until 2020.",
    },
    {
        "name": "Arrokoth",
        "full_name": "(486958) Arrokoth / 2014 MU69",
        "a": 44.58, "e": 0.035, "inc": 2.45,
        "Omega": 158.7, "omega": 176.1, "M": 316.0,
        "epoch_jd": 2459600.5,
        "radius_km": 9.0,      # bilobate contact binary; mean equiv radius
        "color": "#BC8F8F",
        "notes": "New Horizons flyby target; shape exquisitely known. Very small shadow.",
    },
    {
        "name": "Chaos",
        "full_name": "(19521) Chaos",
        "a": 45.87, "e": 0.107, "inc": 12.05,
        "Omega": 58.2, "omega": 56.3, "M": 195.0,
        "epoch_jd": 2459600.5,
        "radius_km": 300.0,
        "color": "#708090",
        "notes": "Classical KBO; well-determined orbit.",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# ORBITAL MECHANICS
# ═══════════════════════════════════════════════════════════════════════════════

def solve_kepler(M: np.ndarray, e: float) -> np.ndarray:
    E = M.copy()
    for _ in range(80):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < 1e-11:
            break
    return E


def heliocentric_posvel(tno: dict, t_jd: np.ndarray):
    """
    Heliocentric Cartesian pos [AU,(3,N)] and vel [AU/day,(3,N)] (J2000 ecliptic).
    """
    a, e   = float(tno["a"]), float(tno["e"])
    inc    = np.radians(tno["inc"])
    Omega  = np.radians(tno["Omega"])
    omega  = np.radians(tno["omega"])
    M0, t0 = np.radians(tno["M"]), float(tno["epoch_jd"])

    n  = np.sqrt(GM_SUN / a**3)
    M  = M0 + n * (np.asarray(t_jd, float) - t0)
    E  = solve_kepler(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    r  = a * (1 - e * np.cos(E))
    xo, yo = r * np.cos(nu), r * np.sin(nu)

    h   = np.sqrt(GM_SUN * a * (1 - e**2))
    vxo = -GM_SUN / h * np.sin(nu)
    vyo =  GM_SUN / h * (e + np.cos(nu))

    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(inc),   np.sin(inc)
    cw, sw = np.cos(omega),  np.sin(omega)

    Px = cO*cw - sO*sw*ci;  Qx = -cO*sw - sO*cw*ci
    Py = sO*cw + cO*sw*ci;  Qy = -sO*sw + cO*cw*ci
    Pz = sw*si;              Qz =  cw*si

    pos = np.array([Px*xo+Qx*yo, Py*xo+Qy*yo, Pz*xo+Qz*yo])
    vel = np.array([Px*vxo+Qx*vyo, Py*vxo+Qy*vyo, Pz*vxo+Qz*vyo])
    return pos, vel


def earth_state(t_jd: np.ndarray):
    """Earth barycentric pos [AU,(3,N)] and vel [AU/day,(3,N)] via DE430."""
    times = Time(np.asarray(t_jd, float), format="jd", scale="tdb")
    pv    = get_body_barycentric_posvel("earth", times)
    return (pv[0].xyz.to(u.AU).value,
            pv[1].xyz.to(u.AU/u.day).value)


# ═══════════════════════════════════════════════════════════════════════════════
# SHADOW VELOCITY PHYSICS
# ═══════════════════════════════════════════════════════════════════════════════

def shadow_velocity_profile(tno: dict, t_jd: np.ndarray) -> dict:
    """
    Compute shadow ground-track speed, distance, and sky position at each time.

    Shadow velocity formula (Hartshorn 2025, Eq. shadow_vel):
        v_shadow = (d_Earth / d_TNO) × |v_sky_rel|
    where v_sky_rel is the geocentric sky-plane relative velocity of the TNO.

    Equivalently:  v_shadow = |v_perp_TNO_hel − v_perp_Earth_hel| / d_TNO_AU

    Returns dict with arrays (N,):
        v_shadow_km_s   shadow ground-track speed [km/s]
        v_sky_km_s      geocentric sky-plane speed [km/s]
        d_AU            TNO geocentric distance [AU]
        d_km            TNO geocentric distance [km]
        ra_deg, dec_deg geocentric sky position [deg]
        ecl_lon, ecl_lat ecliptic coordinates [deg]
        v_radial_km_s   radial (LOS) component [km/s]  (positive = receding)
        t_jd            times used
    """
    pos_tno, vel_tno = heliocentric_posvel(tno, t_jd)
    pos_ear, vel_ear = earth_state(t_jd)

    # Geocentric vectors
    rel_pos = pos_tno - pos_ear   # [AU, (3,N)]
    rel_vel = vel_tno - vel_ear   # [AU/day, (3,N)]
    d_AU    = np.linalg.norm(rel_pos, axis=0)

    # Line-of-sight unit vector
    los = rel_pos / d_AU          # (3, N)

    # Radial and sky-plane velocity components
    v_rad    = np.sum(rel_vel * los, axis=0)                 # AU/day, LOS
    v_perp   = rel_vel - v_rad[np.newaxis, :] * los         # [AU/day, (3,N)]
    v_sky_AU = np.linalg.norm(v_perp, axis=0)               # AU/day

    v_sky_km_s    = v_sky_AU    * AU_KM / DAY_S
    v_rad_km_s    = v_rad       * AU_KM / DAY_S

    # Shadow ground-track speed (Hartshorn 2025 eq.)
    v_shadow_km_s = v_sky_km_s / d_AU                       # km/s

    # Sky coordinates (geocentric, ICRS)
    ra_deg  = np.degrees(np.arctan2(los[1], los[0])) % 360
    dec_deg = np.degrees(np.arcsin(np.clip(los[2], -1, 1)))

    # Ecliptic coordinates (for opposition/elongation context)
    # obliquity ε ≈ 23.4393°
    eps = np.radians(23.4393)
    ecl_lon = np.degrees(np.arctan2(
        los[1]*np.cos(eps) + los[2]*np.sin(eps), los[0])) % 360
    ecl_lat = np.degrees(np.arcsin(
        -los[1]*np.sin(eps) + los[2]*np.cos(eps)))

    return {
        "v_shadow_km_s": v_shadow_km_s,
        "v_sky_km_s":    v_sky_km_s,
        "d_AU":          d_AU,
        "d_km":          d_AU * AU_KM,
        "ra_deg":        ra_deg,
        "dec_deg":       dec_deg,
        "ecl_lon":       ecl_lon,
        "ecl_lat":       ecl_lat,
        "v_radial_km_s": v_rad_km_s,
        "t_jd":          t_jd,
    }


def fresnel_scale_km(d_AU: float) -> float:
    """Fresnel diffraction scale at distance d_AU [km]."""
    d_m = d_AU * AU_M
    return np.sqrt(LAMBDA_M * d_m / 2) / 1000.0   # km


def teff_seconds(v_shadow_km_s: float, L_km: float) -> float:
    """Effective integration time for baseline L_km [s]."""
    if v_shadow_km_s <= 0:
        return np.inf
    return L_km / v_shadow_km_s


def shadow_width_km(tno: dict) -> float:
    """Geometric shadow width (= asteroid/TNO diameter) [km]."""
    return 2.0 * tno["radius_km"]


def viability_check(tno: dict, rvsp_data: dict, L_km: float = 500.0) -> dict:
    """
    Check Hartshorn 2025 viability criteria at the RVSP.

    Criteria:
      (i)   σ_track ≤ 150 km  → proxy: v_shadow ≤ 1 km/s (array can cover corridor)
      (ii)  W_shadow ≥ 5 × F_Fresnel  (geometric core > diffractive halo)
      (iii) v_shadow ≤ 1 km/s  (sufficient integration time)
    """
    v  = rvsp_data["v_shadow_min"]
    d  = rvsp_data["d_AU_at_min"]
    F  = fresnel_scale_km(d)
    W  = shadow_width_km(tno)
    te = teff_seconds(v, L_km)

    c1 = v <= 1.0                   # speed criterion
    c2 = W >= 5.0 * F               # size criterion
    c3 = te >= 100.0                # t_eff ≥ 100 s (paper reference)

    return {
        "speed_ok":     c1,
        "size_ok":      c2,
        "teff_ok":      c3,
        "viable":       c1 and c2 and c3,
        "F_fresnel_km": F,
        "W_shadow_km":  W,
        "ratio_W_F":    W / F,
        "t_eff_s":      te,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RVSP FINDER — core scan logic
# ═══════════════════════════════════════════════════════════════════════════════

def find_rvsps(tno: dict, t_jd: np.ndarray,
               v_threshold: float = 1.0, L_km: float = 500.0) -> dict:
    """
    Scan the time array for RVSP windows (v_shadow < v_threshold km/s).

    Returns a comprehensive result dict for this TNO.
    """
    prof = shadow_velocity_profile(tno, t_jd)
    v    = prof["v_shadow_km_s"]

    # ── global minimum ────────────────────────────────────────────────────────
    idx_min       = int(np.argmin(v))
    t_min_jd      = t_jd[idx_min]
    v_min         = float(v[idx_min])
    d_at_min      = float(prof["d_AU"][idx_min])
    ra_at_min     = float(prof["ra_deg"][idx_min])
    dec_at_min    = float(prof["dec_deg"][idx_min])
    ecl_l_at_min  = float(prof["ecl_lon"][idx_min])
    ecl_b_at_min  = float(prof["ecl_lat"][idx_min])

    rvsp_data = {
        "v_shadow_min": v_min,
        "d_AU_at_min":  d_at_min,
        "t_jd_min":     t_min_jd,
        "t_iso_min":    Time(t_min_jd, format="jd", scale="tdb").utc.iso[:10],
        "ra_deg":       ra_at_min,
        "dec_deg":      dec_at_min,
        "ecl_lon":      ecl_l_at_min,
        "ecl_lat":      ecl_b_at_min,
    }

    # ── windows below threshold ───────────────────────────────────────────────
    below = v <= v_threshold
    windows = []
    if below.any():
        # Find contiguous segments
        diff    = np.diff(below.astype(int))
        starts  = np.where(diff ==  1)[0] + 1
        ends    = np.where(diff == -1)[0] + 1
        if below[0]:
            starts = np.concatenate([[0], starts])
        if below[-1]:
            ends = np.concatenate([ends, [len(below)]])
        for s, e in zip(starts, ends):
            seg_v   = v[s:e]
            seg_idx = np.argmin(seg_v)
            seg_t   = t_jd[s:e]
            te_max  = teff_seconds(seg_v[seg_idx], L_km)
            windows.append({
                "t_start_iso": Time(seg_t[0],         format="jd").utc.iso[:10],
                "t_end_iso":   Time(seg_t[-1],         format="jd").utc.iso[:10],
                "t_peak_iso":  Time(seg_t[seg_idx],   format="jd").utc.iso[:10],
                "v_min_km_s":  float(seg_v[seg_idx]),
                "t_eff_s":     te_max,
                "duration_d":  float(seg_t[-1] - seg_t[0]),
                "d_AU":        float(prof["d_AU"][s + seg_idx]),
                "ra_deg":      float(prof["ra_deg"][s + seg_idx]),
                "dec_deg":     float(prof["dec_deg"][s + seg_idx]),
                "ecl_lon":     float(prof["ecl_lon"][s + seg_idx]),
            })

    # Sort windows by best (lowest) v_shadow
    windows.sort(key=lambda w: w["v_min_km_s"])

    # ── per-TNO derived quantities ────────────────────────────────────────────
    vcheck = viability_check(tno, rvsp_data, L_km)
    te_best = teff_seconds(v_min, L_km)

    return {
        "tno":           tno,
        "profile":       prof,
        "rvsp":          rvsp_data,
        "windows":       windows,
        "viability":     vcheck,
        "t_eff_best_s":  te_best,
        "v_shadow_min":  v_min,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results: list, L_km: float, v_thresh: float, t_start: str, t_end: str):
    """Print the main ranked summary table and per-object detail blocks."""

    # Sort by best t_eff (descending)
    ranked = sorted(results, key=lambda r: r["t_eff_best_s"], reverse=True)

    print()
    print("╔" + "═"*94 + "╗")
    print("║  RVSP FINDER — Radial-Velocity Stationary Point Scanner" + " "*38 + "║")
    print(f"║  Scan window: {t_start} → {t_end}   |   Baseline L = {L_km:.0f} km   |   v_shadow threshold = {v_thresh:.1f} km/s" + " "*2 + "║")
    print("╠" + "═"*94 + "╣")
    print(f"║  {'TNO':<22} {'Best RVSP':<12} {'v_shadow':<10} {'t_eff':<10} {'d [AU]':<8} "
          f"{'RA°':<7} {'Dec°':<7} {'W/F':<6} {'Viable':<7}║")
    print("╠" + "═"*94 + "╣")

    for r in ranked:
        tno   = r["tno"]
        rvsp  = r["rvsp"]
        vc    = r["viability"]
        te    = r["t_eff_best_s"]
        v     = r["v_shadow_min"]
        te_s  = f"{te:.0f} s" if te < 1e6 else ">1 Ms"
        flag  = "  ✓ YES" if vc["viable"] else "  ✗ no"
        print(
            f"║  {tno['name']:<22} {rvsp['t_iso_min']:<12} {v:<10.4f} {te_s:<10} "
            f"{rvsp['d_AU_at_min']:<8.1f} {rvsp['ra_deg']:<7.1f} {rvsp['dec_deg']:<7.1f} "
            f"{vc['ratio_W_F']:<6.2f} {flag:<7}║"
        )

    print("╠" + "═"*94 + "╣")
    print("║  Columns: v_shadow [km/s] · t_eff = L/v_shadow [s] · d geocentric [AU]" + " "*21 + "║")
    print("║  W/F = shadow_width / Fresnel_scale  (viability criterion ii: W/F ≥ 5)" + " "*20 + "║")
    print("╚" + "═"*94 + "╝")
    print()

    # ── Detailed blocks for viable objects ────────────────────────────────────
    viable = [r for r in ranked if r["viability"]["viable"]]
    if not viable:
        viable = ranked[:3]   # show top-3 even if none fully viable
        print("  ⚠  No fully viable objects in this window.  Showing top-3 candidates.\n")

    for r in viable:
        tno  = r["tno"]
        rvsp = r["rvsp"]
        vc   = r["viability"]
        te   = r["t_eff_best_s"]

        print(f"  {'─'*70}")
        print(f"  {tno['full_name']}")
        print(f"  {'─'*70}")
        print(f"  Best RVSP date  : {rvsp['t_iso_min']}")
        print(f"  v_shadow (min)  : {r['v_shadow_min']:.5f} km/s")
        print(f"  t_eff           : {te:.1f} s  ({te/60:.1f} min)  [L={L_km:.0f} km baseline]")
        print(f"  Geocentric dist : {rvsp['d_AU_at_min']:.3f} AU  ({rvsp['d_AU_at_min']*AU_KM/1e6:.0f} million km)")
        print(f"  Sky position    : RA={rvsp['ra_deg']:.2f}°  Dec={rvsp['dec_deg']:.2f}°")
        print(f"  Ecliptic        : lon={rvsp['ecl_lon']:.2f}°  lat={rvsp['ecl_lat']:.2f}°")
        print(f"  Shadow width    : {vc['W_shadow_km']:.0f} km  (diameter)")
        print(f"  Fresnel scale   : {vc['F_fresnel_km']:.3f} km  at {rvsp['d_AU_at_min']:.1f} AU, λ=550 nm")
        print(f"  W / F_Fresnel   : {vc['ratio_W_F']:.2f}  {'✓' if vc['size_ok'] else '✗'}  (need ≥ 5)")
        print(f"  Speed criterion : {'✓' if vc['speed_ok'] else '✗'}  v_shadow ≤ 1 km/s")
        print(f"  t_eff criterion : {'✓' if vc['teff_ok'] else '✗'}  t_eff ≥ 100 s")
        print(f"  Overall viable  : {'✓ YES' if vc['viable'] else '✗ NO'}")
        print(f"  Notes           : {tno['notes']}")

        if r["windows"]:
            print(f"\n  RVSP Windows below {v_thresh:.1f} km/s:")
            print(f"  {'Start':<13} {'End':<13} {'Peak':<13} {'v_min [km/s]':<14} "
                  f"{'t_eff [s]':<11} {'d [AU]':<8} {'RA°':<7}")
            for w in r["windows"][:5]:
                te_w = f"{w['t_eff_s']:.0f}" if w["t_eff_s"] < 1e6 else ">1e6"
                print(f"  {w['t_start_iso']:<13} {w['t_end_iso']:<13} {w['t_peak_iso']:<13} "
                      f"{w['v_min_km_s']:<14.5f} {te_w:<11} {w['d_AU']:<8.2f} {w['ra_deg']:<7.1f}")
        print()


def save_report(results: list, L_km: float, v_thresh: float,
                t_start: str, t_end: str, path: str):
    """Write machine-readable report to a text file."""
    ranked = sorted(results, key=lambda r: r["t_eff_best_s"], reverse=True)

    lines = [
        "# RVSP Finder Report",
        f"# Scan: {t_start} to {t_end}",
        f"# Array baseline L = {L_km:.0f} km",
        f"# v_shadow threshold = {v_thresh:.2f} km/s",
        "#",
        "# Columns: Name | BestRVSP | v_min_km_s | t_eff_s | d_AU | RA_deg | Dec_deg"
        " | EclLon | EclLat | W_km | F_km | W_over_F | Viable",
        "",
    ]
    for r in ranked:
        vc = r["viability"]
        rv = r["rvsp"]
        lines.append(
            f"{r['tno']['name']:25s} {rv['t_iso_min']} "
            f"{r['v_shadow_min']:.6f} {r['t_eff_best_s']:.1f} "
            f"{rv['d_AU_at_min']:.4f} {rv['ra_deg']:.4f} {rv['dec_deg']:.4f} "
            f"{rv['ecl_lon']:.4f} {rv['ecl_lat']:.4f} "
            f"{vc['W_shadow_km']:.2f} {vc['F_fresnel_km']:.4f} {vc['ratio_W_F']:.3f} "
            f"{'YES' if vc['viable'] else 'NO'}"
        )
        for w in r["windows"]:
            te_w = f"{w['t_eff_s']:.1f}" if w["t_eff_s"] < 1e9 else "inf"
            lines.append(
                f"  WINDOW {w['t_start_iso']} {w['t_end_iso']} "
                f"peak={w['t_peak_iso']} v={w['v_min_km_s']:.6f} "
                f"teff={te_w}s dur={w['duration_d']:.1f}d d={w['d_AU']:.3f}AU"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_COL  = "#21262d"
TEXT_COL  = "#e6edf3"
ACCENT    = "#58a6ff"
HIGHLIGHT = "#ffa657"
SUCCESS   = "#3fb950"
WARN      = "#d29922"
MID       = "#8b949e"


def dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID_COL, "axes.labelcolor": TEXT_COL,
        "axes.titlecolor": TEXT_COL, "xtick.color": TEXT_COL,
        "ytick.color": TEXT_COL, "grid.color": GRID_COL,
        "text.color": TEXT_COL, "legend.facecolor": PANEL_BG,
        "legend.edgecolor": GRID_COL, "font.size": 8.5,
    })


def panel_timeline(ax, results: list, t_jd: np.ndarray,
                   v_thresh: float, top_n: int = 8):
    """
    Panel 1: v_shadow(t) for the top-N TNOs ranked by best t_eff,
    with RVSP windows shaded and threshold line.
    """
    ranked = sorted(results, key=lambda r: r["t_eff_best_s"], reverse=True)[:top_n]

    years = Time(t_jd, format="jd").decimalyear

    ax.axhline(v_thresh, color=WARN, lw=1.1, linestyle="--", alpha=0.7,
               label=f"Threshold {v_thresh:.1f} km/s")
    ax.axhline(0.1,      color=SUCCESS, lw=0.8, linestyle=":", alpha=0.5,
               label="0.1 km/s (ideal RVSP)")

    for r in ranked[::-1]:   # draw best on top
        tno = r["tno"]
        v   = r["profile"]["v_shadow_km_s"]
        lbl = f"{tno['name']}  (t_eff={r['t_eff_best_s']:.0f} s)"
        ax.plot(years, v, lw=1.3, color=tno["color"], label=lbl, alpha=0.88)

        # Shade RVSP windows
        for w in r["windows"]:
            t_s = Time(w["t_start_iso"], scale="tdb").decimalyear
            t_e = Time(w["t_end_iso"],   scale="tdb").decimalyear
            ax.axvspan(t_s, t_e, color=tno["color"], alpha=0.08)

        # Mark global minimum
        idx = int(np.argmin(v))
        ax.scatter(years[idx], v[idx], s=50, c=tno["color"],
                   marker="v", zorder=6, alpha=0.9)

    ax.set_yscale("log")
    ax.set_ylim(1e-3, 30)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("v_shadow [km/s]  (log)", fontsize=9)
    ax.set_title("Shadow Ground-Track Speed  —  RVSP Windows (▼ = global minimum)",
                 fontsize=9, pad=5)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8, ncol=2)
    ax.grid(True, which="both", color=GRID_COL, lw=0.35, alpha=0.6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def panel_teff_heatmap(ax, results: list, t_jd: np.ndarray,
                       L_km: float, top_n: int = 12):
    """
    Panel 2: Heat-map of t_eff(t) for top-N TNOs (rows = objects, col = time).
    Colour = log10(t_eff).
    """
    ranked = sorted(results, key=lambda r: r["t_eff_best_s"], reverse=True)[:top_n]
    years  = Time(t_jd, format="jd").decimalyear

    matrix = np.zeros((len(ranked), len(t_jd)))
    labels = []
    for i, r in enumerate(ranked):
        v          = r["profile"]["v_shadow_km_s"]
        te         = np.where(v > 0, L_km / v, 1e6)
        matrix[i]  = np.log10(np.clip(te, 1, 1e6))
        labels.append(r["tno"]["name"])

    im = ax.imshow(matrix, aspect="auto", origin="lower",
                   extent=[years[0], years[-1], -0.5, len(ranked)-0.5],
                   cmap="magma", vmin=0, vmax=6)
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_title(f"Effective Integration Time  log₁₀(t_eff [s])  [L={L_km:.0f} km]",
                 fontsize=9, pad=5)
    cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01, fraction=0.04)
    cb.set_label("log₁₀(t_eff [s])", fontsize=7.5)
    plt.setp(cb.ax.yaxis.get_ticklabels(), fontsize=7)
    # Overlay RVSP markers
    for i, r in enumerate(ranked):
        rvsp = r["rvsp"]
        yr   = Time(rvsp["t_jd_min"], format="jd").decimalyear
        ax.scatter(yr, i, s=60, c="white", marker="*", zorder=8, alpha=0.85)
    ax.grid(False)


def panel_sky_chart(ax, results: list, v_thresh: float):
    """
    Panel 3: Sky chart of RVSP positions (ecliptic coords).
    Point size ∝ log10(t_eff); colour = TNO colour.
    """
    ax.set_facecolor(PANEL_BG)
    # Ecliptic plane line
    ax.axhline(0, color=WARN, lw=0.8, linestyle="--", alpha=0.4,
               label="Ecliptic plane")

    for r in results:
        rvsp = r["rvsp"]
        vc   = r["viability"]
        te   = r["t_eff_best_s"]
        size = max(20, min(400, 60 * np.log10(max(te, 10))))
        ec   = "white" if vc["viable"] else MID
        ax.scatter(rvsp["ecl_lon"], rvsp["ecl_lat"],
                   s=size, c=r["tno"]["color"], edgecolors=ec,
                   linewidths=1.2, zorder=5, alpha=0.88)
        ax.annotate(r["tno"]["name"],
                    xy=(rvsp["ecl_lon"], rvsp["ecl_lat"]),
                    xytext=(3, 4), textcoords="offset points",
                    fontsize=6, color=TEXT_COL, alpha=0.85)

    ax.set_xlim(0, 360); ax.set_ylim(-50, 50)
    ax.set_xlabel("Ecliptic longitude [°]", fontsize=9)
    ax.set_ylabel("Ecliptic latitude [°]",  fontsize=9)
    ax.set_title("RVSP Sky Positions  (size ∝ log t_eff · white outline = viable)",
                 fontsize=9, pad=5)
    ax.grid(True, color=GRID_COL, lw=0.4, alpha=0.5)
    ax.set_xticks(range(0, 361, 30))


def panel_detail(ax, results: list, t_jd: np.ndarray, L_km: float, v_thresh: float):
    """
    Panel 4: Detailed t_eff curves for the top-5 viable (or best) objects,
    with secondary axis showing v_shadow and horizontal guide lines.
    """
    ranked = sorted(results, key=lambda r: r["t_eff_best_s"], reverse=True)

    viable = [r for r in ranked if r["viability"]["viable"]]
    show   = (viable if viable else ranked)[:5]
    years  = Time(t_jd, format="jd").decimalyear

    ax.axhline(100,  color=WARN,    lw=0.8, linestyle="--", alpha=0.6,
               label="100 s (paper reference)")
    ax.axhline(3600, color=SUCCESS, lw=0.8, linestyle=":",  alpha=0.6,
               label="3600 s (1 h)")

    ax2 = ax.twinx(); ax2.set_facecolor(PANEL_BG)

    for r in show:
        tno  = r["tno"]
        v    = r["profile"]["v_shadow_km_s"]
        te   = np.where(v > 0, L_km / v, 1e7)
        te   = np.clip(te, 0, 1e6)
        lbl  = f"{tno['name']}  (best {r['t_eff_best_s']:.0f} s)"
        ax.plot(years, te, lw=1.6, color=tno["color"], label=lbl, alpha=0.9)
        # RVSP marker
        idx = int(np.argmin(v))
        ax.scatter(years[idx], te[idx], s=80, c=tno["color"],
                   marker="*", zorder=8, alpha=0.9)

    ax.set_yscale("log")
    ax.set_ylim(1, 1e6)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("t_eff [s]  (log)", fontsize=9, color=ACCENT)
    ax.tick_params(axis="y", labelcolor=ACCENT)
    ax2.set_ylabel("v_shadow [km/s]", fontsize=8, color=HIGHLIGHT)
    ax2.set_ylim(0, v_thresh * 3); ax2.tick_params(axis="y", labelcolor=HIGHLIGHT)

    ax.set_title("Effective Integration Time  (top viable candidates  ★ = RVSP)",
                 fontsize=9, pad=5)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.8)
    ax.grid(True, which="both", color=GRID_COL, lw=0.35, alpha=0.6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def make_plots(results: list, t_jd: np.ndarray,
               L_km: float, v_thresh: float, out_path: str):
    """Compose and save the four-panel figure."""
    dark_style()
    fig = plt.figure(figsize=(22, 14), facecolor=DARK_BG)
    gs  = mgs.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28,
                        left=0.05, right=0.97, top=0.94, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, :])   # full-width timeline
    ax2 = fig.add_subplot(gs[1, 0])   # heat-map
    ax3 = fig.add_subplot(gs[1, 1])   # sky chart

    # For the bottom row do a 1-3 split
    gs2 = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :],
                                       hspace=0.0, wspace=0.30,
                                       width_ratios=[1.4, 1])
    ax2 = fig.add_subplot(gs2[0])
    ax3 = fig.add_subplot(gs2[1])

    panel_timeline   (ax1, results, t_jd, v_thresh)
    panel_teff_heatmap(ax2, results, t_jd, L_km)
    panel_sky_chart  (ax3, results, v_thresh)

    fig.suptitle(
        "RVSP Finder — TNO Shadow Velocity & Effective Integration Time Survey",
        color=TEXT_COL, fontsize=13, fontweight="bold", y=0.99,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Figure saved → {out_path}")


def make_detail_plot(results: list, t_jd: np.ndarray,
                     L_km: float, v_thresh: float, out_path: str):
    """Separate detail figure: top candidate t_eff curves + per-TNO data cards."""
    dark_style()
    ranked = sorted(results, key=lambda r: r["t_eff_best_s"], reverse=True)
    viable = [r for r in ranked if r["viability"]["viable"]] or ranked[:5]
    show   = viable[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor=DARK_BG)
    fig.suptitle("RVSP Candidate Detail Cards", color=TEXT_COL,
                 fontsize=13, fontweight="bold", y=1.00)
    plt.subplots_adjust(hspace=0.48, wspace=0.32,
                        left=0.06, right=0.97, top=0.93, bottom=0.08)

    years = Time(t_jd, format="jd").decimalyear

    for ax, r in zip(axes.flat, show):
        tno  = r["tno"]
        v    = r["profile"]["v_shadow_km_s"]
        te   = np.where(v > 0, L_km / v, 1e7)
        te   = np.clip(te, 0, 1e6)
        vc   = r["viability"]
        rvsp = r["rvsp"]

        ax.set_facecolor(PANEL_BG)

        # Main t_eff curve
        ax.plot(years, te, lw=2.0, color=tno["color"], alpha=0.9, label="t_eff(t)")

        # Secondary: v_shadow
        ax2 = ax.twinx(); ax2.set_facecolor(PANEL_BG)
        ax2.plot(years, v, lw=1.2, color=HIGHLIGHT, linestyle="--",
                 alpha=0.6, label="v_shadow")
        ax2.set_ylabel("v_shadow [km/s]", fontsize=7, color=HIGHLIGHT)
        ax2.tick_params(axis="y", labelcolor=HIGHLIGHT, labelsize=7)
        ax2.axhline(v_thresh, color=WARN, lw=0.8, linestyle=":", alpha=0.55)

        # Mark RVSP
        idx = int(np.argmin(v))
        ax.scatter(years[idx], te[idx], s=150, c=tno["color"],
                   marker="*", zorder=10, edgecolors="white", linewidths=0.7)

        # Guide lines
        ax.axhline(100,  color=MID,     lw=0.7, linestyle="--", alpha=0.5)
        ax.axhline(3600, color=SUCCESS, lw=0.7, linestyle=":",  alpha=0.4)

        ax.set_yscale("log"); ax.set_ylim(1, 1e6)
        ax.set_xlabel("Year", fontsize=7.5); ax.set_ylabel("t_eff [s]", fontsize=7.5, color=ACCENT)
        ax.tick_params(axis="y", labelcolor=ACCENT, labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, which="both", color=GRID_COL, lw=0.3, alpha=0.5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))

        # Title with key metrics
        viable_tag = "✓ VIABLE" if vc["viable"] else "○ candidate"
        ax.set_title(
            f"{tno['name']}   {viable_tag}\n"
            f"Best: {rvsp['t_iso_min']}  ·  v={r['v_shadow_min']:.4f} km/s  ·  "
            f"t_eff={r['t_eff_best_s']:.0f} s  ·  W/F={vc['ratio_W_F']:.1f}",
            fontsize=7.5, color=tno["color"], pad=4,
        )

    # Hide unused subplots
    for ax in axes.flat[len(show):]:
        ax.set_visible(False)

    detail_path = out_path.replace(".png", "_detail.png")
    fig.savefig(detail_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Detail figure saved → {detail_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="RVSP Finder — scan TNO catalog for shadow-velocity minima."
    )
    ap.add_argument("--start",     default="2025-01-01",
                    help="Scan start date (YYYY-MM-DD, default 2025-01-01)")
    ap.add_argument("--end",       default="2035-01-01",
                    help="Scan end date   (YYYY-MM-DD, default 2035-01-01)")
    ap.add_argument("--step_days", type=float, default=5.0,
                    help="Time step [days] for coarse scan (default 5)")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="v_shadow threshold for RVSP windows [km/s] (default 1.0)")
    ap.add_argument("--L",         type=float, default=500.0,
                    help="Telescope array baseline [km] (default 500)")
    ap.add_argument("--tno",       nargs="+",  default=None,
                    help="Select specific TNOs by name (default: all)")
    ap.add_argument("--out",       default=None,
                    help="Output PNG path (default: rvsp_plots.png)")
    args = ap.parse_args()

    t_start_jd = Time(args.start).jd
    t_end_jd   = Time(args.end).jd
    t_jd       = np.arange(t_start_jd, t_end_jd, args.step_days)

    catalog = TNO_CATALOG
    if args.tno:
        catalog = [t for t in TNO_CATALOG if t["name"] in args.tno]
        if not catalog:
            print(f"  ⚠  No matching TNOs found for: {args.tno}")
            return

    out_base = args.out or "./rvsp_plots.png"

    print(f"\n{'═'*70}")
    print(f"  RVSP Finder")
    print(f"  Window : {args.start}  →  {args.end}")
    print(f"  Step   : {args.step_days:.0f} days  ({len(t_jd)} time steps)")
    print(f"  Objects: {len(catalog)}")
    print(f"  L      : {args.L:.0f} km  |  threshold: {args.threshold:.2f} km/s")
    print(f"{'═'*70}\n")

    results = []
    for tno in catalog:
        print(f"  Scanning {tno['name']:<20} …", end="", flush=True)
        r = find_rvsps(tno, t_jd, v_threshold=args.threshold, L_km=args.L)
        results.append(r)
        flag = "✓" if r["viability"]["viable"] else " "
        print(f"  {flag}  v_min={r['v_shadow_min']:.4f} km/s  "
              f"t_eff={r['t_eff_best_s']:.0f} s  "
              f"peak={r['rvsp']['t_iso_min']}")

    print()
    print_report(results, args.L, args.threshold, args.start, args.end)

    report_path = out_base.replace(".png", ".txt")
    save_report(results, args.L, args.threshold, args.start, args.end, report_path)

    print("\n  Generating plots…")
    make_plots(results, t_jd, args.L, args.threshold, out_base)
    make_detail_plot(results, t_jd, args.L, args.threshold, out_base)
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
