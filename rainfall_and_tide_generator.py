# rainfall_and_tide_generator.py
from __future__ import annotations
from typing import Tuple, Optional
import re
import numpy as np
import pandas as pd

from scipy.signal import find_peaks

__all__ = [
    "pf_df",
    "convert_units",
    "generate_rainfall",
    "moon_tide_ranges",
    "generate_tide_curve",
    "find_tide_extrema",
    "align_rainfall_to_tide",
    "fetch_greenstream_dataframe",
    "build_timestep_and_resample_15min",
    "get_tide_real_or_synthetic",
    "get_aligned_rainfall",
]

pf_df = pd.DataFrame({
    "Duration_Minutes": [120, 180, 360, 720, 1440],
    "1":   [1.68, 1.80, 2.18, 2.57, 2.94],
    "2":   [2.02, 2.17, 2.62, 3.08, 3.58],
    "5":   [2.49, 2.69, 3.25, 3.85, 4.62],
    "10":  [2.98, 3.24, 3.91, 4.66, 5.51],
    "25":  [3.58, 3.93, 4.77, 5.73, 6.82],
    "50":  [4.13, 4.57, 5.58, 6.75, 7.96],
    "100": [4.67, 5.23, 6.41, 7.82, 9.20],
})

def convert_units(value_in_inches: float, unit: str) -> float:
    """
    Convert rainfall depth from inches -> inches (U.S.) or centimeters (Metric).
    """
    if unit == "U.S. Customary":
        return value_in_inches
    elif unit == "Metric (SI)":
        return value_in_inches * 2.54
    else:
        raise ValueError("Unsupported unit. Use 'U.S. Customary' or 'Metric (SI)'.")

def feet_to_meters(series_or_array):
    return series_or_array * 0.3048

# Your Type III cumulative dimensionless curve (0..1). Assumed sampled at equal
# time-fraction steps from 0 to 1. Replace with your own list if needed.
SCS_TYPE_III_CUM = np.array([
    0.0000, 0.0050, 0.0110, 0.0150, 0.0200, 0.0232, 0.0308, 0.0367, 0.0430, 0.0497,
    0.0568, 0.0642, 0.0720, 0.0806, 0.0905, 0.1016, 0.1140, 0.1284, 0.1458, 0.1659,
    0.1899, 0.2165, 0.2500, 0.2980, 0.5000, 0.7020, 0.7500, 0.7835, 0.8110, 0.8341,
    0.8542, 0.8716, 0.8860, 0.8984, 0.9095, 0.9194, 0.9280, 0.9358, 0.9432, 0.9503,
    0.9570, 0.9634, 0.9694, 0.9752, 0.9808, 0.9860, 0.9900, 0.9956, 1.0000
], dtype=float)

def generate_rainfall(total_inches: float,
                      duration_minutes: int,
                      curve: np.ndarray = SCS_TYPE_III_CUM) -> np.ndarray:
    """
    SCS Type III hyetograph generator.
    Returns INCREMENTAL 15-min depths that sum to `total_inches` over `duration_minutes`.
    `duration_minutes` must be divisible by 15.
    `curve` is the dimensionless cumulative P/Ptotal vs. time-fraction (0..1), monotone 0->1.
    """
    if duration_minutes % 15 != 0:
        raise ValueError("duration_minutes must be divisible by 15.")
    if curve.ndim != 1 or len(curve) < 2:
        raise ValueError("curve must be a 1D array with at least 2 points.")
    if not (np.isclose(curve[0], 0.0) and np.isclose(curve[-1], 1.0)):
        raise ValueError("curve must start at 0.0 and end at 1.0 (dimensionless cumulative).")
    if np.any(np.diff(curve) < -1e-12):
        raise ValueError("curve must be nondecreasing.")

    intervals = duration_minutes // 15

    # Time fractions at the provided curve points (assumed uniform from 0..1)
    x_src = np.linspace(0.0, 1.0, len(curve))

    # We need cumulative at BIN EDGES (0..1 with intervals+1 points)
    edges = np.linspace(0.0, 1.0, intervals + 1)

    # Interpolate cumulative curve at bin edges
    cum_edges = np.interp(edges, x_src, curve)

    # Incremental (per 15-min) dimensionless depths
    inc_dimless = np.diff(cum_edges)
    # Numerical safety: clip tiny negatives to zero
    inc_dimless = np.clip(inc_dimless, 0.0, None)

    # Scale to total_inches
    y = total_inches * inc_dimless

    # Force exact sum by putting any tiny rounding error into the last bin
    err = total_inches - y.sum()
    if abs(err) > 1e-10:
        y[-1] += err

    return y  # inches per 15-min interval

moon_tide_ranges = {
    "🌓 First Quarter: Neap": (-0.8, 1.2),   # feet
    "🌕 Full Moon: Spring":   (-1.2, 1.8),   # feet
    "🌗 Last Quarter: Neap":  (-0.9, 1.3),   # feet
    "🌑 New Moon: Spring":    (-1.1, 1.7),   # feet
}

def generate_tide_curve(moon_phase: str, unit: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic tide over 48h at 15-min resolution.
    Returns (minutes_15, tide_15) where tide units are ft (U.S.) or m (Metric).
    """
    if moon_phase not in moon_tide_ranges:
        raise ValueError(f"Unknown moon phase: {moon_phase}")

    tide_min_ft, tide_max_ft = moon_tide_ranges[moon_phase]

    # Build high-res (1-min) sine-ish tide across 48h (2880 min), then downsample to 15-min.
    minutes_full = np.arange(0, 2880, 1)
    tide_full_ft = ((np.sin(2 * np.pi * minutes_full / 720 - np.pi / 2) + 1) / 2) \
                   * (tide_max_ft - tide_min_ft) + tide_min_ft

    minutes_15 = np.arange(0, 2880, 15)
    tide_15_ft = tide_full_ft[minutes_15]

    if unit == "Metric (SI)":
        tide_15 = feet_to_meters(tide_15_ft)  # meters
    else:
        tide_15 = tide_15_ft  # feet

    return minutes_15, tide_15

def find_tide_extrema(
    tide_curve_15min: np.ndarray,
    distance_bins: int = 40,      # ~10 h / 15 min ≈ 40 bins; avoids double-detecting
    prominence: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return indices of tide highs (peaks) and lows (troughs) on a 15-min series.
    Tune 'distance_bins' and 'prominence' to control detection strictness.
    """
    peaks, _   = find_peaks(tide_curve_15min, distance=distance_bins, prominence=prominence)
    troughs, _ = find_peaks(-tide_curve_15min, distance=distance_bins, prominence=prominence)
    return peaks, troughs

# ============================================================
# 6) Align rainfall to a tide curve (length-agnostic)
# ============================================================
def align_rainfall_to_tide(total_inches: float,
                           duration_minutes: int,
                           tide_curve_15min: np.ndarray,
                           align: str = "peak",
                           method: str = "SCS_TypeIII",
                           target_index: Optional[int] = None,
                           prominence: Optional[float] = None  # kept for API compatibility; unused here
                           ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build an SCS Type III hyetograph (incremental, 15-min bins) and align it to a tide series.
    - total_inches: storm depth [in]
    - duration_minutes: storm duration (multiple of 15)
    - tide_curve_15min: array of tide elevations at 15-min steps (length N)
    - align: "peak" or "low" (align to nearest peak/low near the series mid), or use target_index
    - method: only "SCS_TypeIII" is supported (kept to avoid breaking callers)
    - target_index: if set, center the storm at this index (overrides 'align')
    Returns (minutes_15, rain_15, center_index_used)
    """
    if method != "SCS_TypeIII":
        raise ValueError("Only 'SCS_TypeIII' is supported. Remove 'Normal' and 'Randomized' uses.")

    if duration_minutes % 15 != 0:
        raise ValueError("duration_minutes must be divisible by 15.")
    n = int(len(tide_curve_15min))
    if n == 0:
        raise ValueError("Empty tide curve.")
    intervals = duration_minutes // 15
    if intervals > n:
        raise ValueError(f"Storm of {intervals} bins does not fit into tide series of length {n}.")

    # 1) Build incremental 15-min rainfall profile using SCS Type III.
    rain_profile = generate_rainfall(total_inches, duration_minutes)

    # 2) Choose center index on the tide series.
    if target_index is not None:
        center_index = int(target_index)
        if not (0 <= center_index < n):
            raise ValueError("target_index out of range.")
    else:
        # Try project-provided extrema finder if it exists; otherwise use the fallback above.
        try:
            # if your module defines find_tide_extrema(peaks, troughs), use it; otherwise the fallback:
            from rainfall_and_tide_generator import find_tide_extrema  # noqa
            peaks, troughs = find_tide_extrema(tide_curve_15min, prominence=prominence)
            candidates = peaks if align == "peak" else troughs
            candidates = np.asarray(candidates, dtype=int)
        except Exception:
            candidates = generate_rainfall(tide_curve_15min, "peak" if align == "peak" else "trough")

        if candidates.size == 0:
            center_index = n // 2
        else:
            mid = n // 2
            center_index = int(candidates[np.argmin(np.abs(candidates - mid))])

    # 3) Place the storm fully within bounds (shift if needed so no depth is lost).
    start = center_index - (intervals // 2)
    start = max(0, min(start, n - intervals))  # clamp so [start, start+intervals) fits inside [0, n)
    end = start + intervals

    rain = np.zeros(n, dtype=float)
    rain[start:end] = rain_profile  # fits exactly by construction

    minutes_15 = np.arange(n, dtype=int) * 15
    return minutes_15, rain, center_index

GREENSTREAM_URL = "https://dashboard.greenstream.cloud/detail?id=SITE#d935fec2-7a0b-4df0-986c-76f25d773070"
WATER_COL_LIVE = "Water Level NAVD88 (ft)"  # column name in CSV/XLSX

# Toolbar geometry from your scans (pixels). Adjust if your viewport differs.
X_FILTER   = 1565   # rightmost icon (Filter)
X_DOWNLOAD = 1470   # Download icon (has title='Download')
TOL        = 40
TOP_Y_MAX  = 100
ICON_MIN_W, ICON_MAX_W = 24, 56

# --- Internal: Playwright helpers (kept inside function scope to avoid hard dependency when not used) ---
def _find_icon_by_x(page, x_target, tol, top_y_max, min_w, max_w):
    nodes = page.locator("div, span, svg")
    n = nodes.count()
    best = None
    best_dx = float("inf")
    for i in range(n):
        h = nodes.nth(i).element_handle()
        if not h:
            continue
        box = h.bounding_box()
        if not box:
            continue
        if (box["y"] < top_y_max and min_w <= box["width"] <= max_w
                and min_w <= box["height"] <= max_w):
            dx = abs(box["x"] - x_target)
            if dx < best_dx:
                best_dx, best = dx, h
    if best is None or best_dx > tol:
        raise RuntimeError(f"Icon not found near x≈{x_target} (Δx={best_dx:.1f}). Adjust TOL/viewport.")
    return best

def _click_handle_by_center(page, handle):
    box = handle.bounding_box()
    if not box:
        raise RuntimeError("No bounding box for element.")
    try:
        handle.click(timeout=1500)
    except Exception:
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        page.mouse.click(cx, cy)

def fetch_greenstream_dataframe() -> pd.DataFrame:
    """
    Opens the Greenstream dashboard, does:
      Filter -> Last 2 Days -> OK -> Download,
    and returns the file as a pandas DataFrame (no saving to disk).
    """
    # Lazy import so your module imports even without Playwright installed
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = browser.new_context(accept_downloads=True, viewport={"width": 1600, "height": 900})
        page = context.new_page()
        page.goto(GREENSTREAM_URL, wait_until="domcontentloaded")

        # Filter (right-most icon)
        filter_handle = _find_icon_by_x(page, X_FILTER, TOL, TOP_Y_MAX, ICON_MIN_W, ICON_MAX_W)
        _click_handle_by_center(page, filter_handle)

        # Drawer: Last 2 Days -> OK
        drawer = page.locator("div.drawer")
        drawer.wait_for(state="visible", timeout=5000)

        try:
            drawer.get_by_text("Last 2 Days", exact=True).click(timeout=2000)
        except PWTimeout:
            drawer.locator("div.radioButton").nth(1).click()  # 0=Today, 1=Last 2 Days

        # OK button (robust)
        ok_clicked = False
        for loc in [
            drawer.get_by_role("button", name=re.compile(r"^OK$", re.I)),
            drawer.locator("button:has-text('OK')"),
            drawer.locator("[role=button]:has-text('OK')"),
            drawer.locator("[class*=Button]:has-text('OK')"),
            drawer.get_by_text(re.compile(r"^\s*OK\s*$", re.I)),
        ]:
            try:
                if loc.count():
                    el = loc.first
                    el.scroll_into_view_if_needed()
                    el.click(timeout=1500)
                    ok_clicked = True
                    break
            except Exception:
                pass
        if not ok_clicked:
            bb = drawer.bounding_box()
            if bb:
                page.mouse.click(bb["x"] + bb["width"] - 60, bb["y"] + bb["height"] - 40)

        # Settle
        try:
            drawer.wait_for(state="hidden", timeout=4000)
        except PWTimeout:
            pass
        try:
            page.wait_for_load_state("networkidle", timeout=6000)
        except PWTimeout:
            pass

        # Download → read directly
        dl_el = page.locator("[title='Download']").first
        if dl_el.count():
            with page.expect_download(timeout=20000) as dl_info:
                try:
                    dl_el.click(timeout=1500)
                except Exception:
                    _click_handle_by_center(page, dl_el.element_handle())
            dl = dl_info.value
        else:
            # geometry fallback
            download_handle = _find_icon_by_x(page, X_DOWNLOAD, TOL, TOP_Y_MAX, ICON_MIN_W, ICON_MAX_W)
            with page.expect_download(timeout=20000) as dl_info:
                _click_handle_by_center(page, download_handle)
            dl = dl_info.value

        tmp_path = dl.path()
        suggested = (dl.suggested_filename or "").lower()

        # CSV vs Excel
        try:
            if suggested.endswith((".xlsx", ".xls")):
                df = pd.read_excel(tmp_path)
            else:
                df = pd.read_csv(tmp_path)
        except Exception:
            # try both if extension lies
            try:
                df = pd.read_csv(tmp_path)
            except Exception:
                df = pd.read_excel(tmp_path)

        browser.close()
        return df

def build_timestep_and_resample_15min(df_raw: pd.DataFrame,
                                      water_col: str = WATER_COL_LIVE,
                                      unit: str = "U.S. Customary",
                                      start_ts: Optional[pd.Timestamp] = None,
                                      navd88_to_sea_level_offset_ft: float = 0.0
                                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input: df_raw from Greenstream (~>=480 rows @ 6-min).
    Returns exactly 48h @ 15-min (192 bins), no padding/interpolation.
    """

    # Find the water level column
    if water_col not in df_raw.columns:
        matches = [c for c in df_raw.columns if "water" in c.lower() and "level" in c.lower()]
        if not matches:
            raise ValueError(f"Water level column not found. Columns={list(df_raw.columns)}")
        water_col = matches[0]

    tide_df = df_raw[[water_col]].copy()
    n = len(tide_df)
    if n == 0:
        raise ValueError("Empty tide DataFrame (live).")

    # We only want the last 48h of 6-min data: 48*60/6 = 480 rows.
    # If more than 480 are present, keep the most recent 480. If fewer, fail fast.
    REQUIRED_6MIN = 48 * 60 // 6  # 480
    if n < REQUIRED_6MIN:
        raise ValueError(f"Live dataset too short ({n} rows). Expected at least {REQUIRED_6MIN} rows for 48h.")
    if n > REQUIRED_6MIN:
        tide_df = tide_df.tail(REQUIRED_6MIN)
        n = REQUIRED_6MIN

    # Build a strict 6-min timeline ending "now" (or aligned to provided start_ts)
    if start_ts is None:
        end6 = pd.Timestamp.now().floor("6min")
    else:
        # If caller provides a start_ts (meaning the first of the n samples),
        # compute the corresponding end timestamp consistently:
        end6 = (pd.Timestamp(start_ts).floor("6min") + pd.Timedelta(minutes=6*(n-1)))
    idx6 = pd.date_range(end=end6, periods=n, freq="6min")
    tide_df = tide_df.set_index(idx6)

    # Unit conversion (live data are in feet)
    vals = tide_df[water_col].astype(float)
    if unit == "Metric (SI)":
        vals = feet_to_meters(vals)
        offset = navd88_to_sea_level_offset_ft * 0.3048   # ft -> m
    else:
        offset = navd88_to_sea_level_offset_ft             # ft
    if offset != 0.0:
        vals = vals - offset
    tide_df[water_col] = vals

    # Downsample to 15-min means, then keep exactly the last 48h (192 bins)
    tide_15_series = tide_df.resample("15min").mean(numeric_only=True)[water_col]
    REQUIRED_15MIN = 48 * 60 // 15  # 192
    if len(tide_15_series) < REQUIRED_15MIN:
        # If Greenstream gave exactly 480×6-min rows, this won’t happen; guard anyway.
        raise ValueError(f"After resampling, got {len(tide_15_series)}×15-min bins; expected {REQUIRED_15MIN}.")
    tide_15_series = tide_15_series.tail(REQUIRED_15MIN)

    # Return fixed 0..48h minutes and values (no padding; pure slice+resample)
    minutes_15 = np.arange(0, 48*60, 15, dtype=int)  # 0,15,...,287*15
    tide_15 = tide_15_series.to_numpy()

    return minutes_15, tide_15

def get_tide_real_or_synthetic(moon_phase: str,
                               unit: str,
                               start_ts: Optional[pd.Timestamp] = None,
                               navd88_to_sea_level_offset_ft: float = 1.36
                               ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Try live tide (Greenstream). If it fails, return synthetic tide.
    Returns:
      minutes_15 (np.ndarray),
      tide_15    (np.ndarray) in selected 'unit' (ft or m), already shifted to MSL,
      used_live  (bool)
    """
    try:
        df_live = fetch_greenstream_dataframe()
        m15, tide_15 = build_timestep_and_resample_15min(
            df_raw=df_live,
            water_col=WATER_COL_LIVE,
            unit=unit,
            start_ts=start_ts,
            navd88_to_sea_level_offset_ft=navd88_to_sea_level_offset_ft  # << MSL shift here
        )
        return m15, tide_15, True
    except Exception:
        # Synthetic fallback (already outputs selected unit; no geodetic datum shift)
        m15, tide_15 = generate_tide_curve(moon_phase, unit)
        return m15, tide_15, False


# ============================================================
# 10) High-level helper: get tide (live or synthetic) and aligned rainfall
# ============================================================
def get_aligned_rainfall(
    total_inches: float,
    duration_minutes: int,
    moon_phase: str,
    unit: str,
    align: str = "peak",
    method: str = "Normal",
    start_ts: Optional[pd.Timestamp] = None,
    prominence: Optional[float] = None,
    navd88_to_sea_level_offset_ft: float = 1.36       # << add this
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, int]:
    """
    Returns (minutes_15, tide_15, rain_15, used_live, center_idx).
    """
    m15, tide_15, used_live = get_tide_real_or_synthetic(
        moon_phase, unit, start_ts, navd88_to_sea_level_offset_ft  # << forward it
    )
    # ...rest unchanged...

    # Choose a target index near the series midpoint based on peaks/lows.
    peaks, troughs = find_tide_extrema(tide_15, prominence=prominence)
    if align == "peak":
        cand = peaks
    else:
        cand = troughs

    if cand.size == 0:
        target_idx = len(tide_15) // 2
    else:
        target_idx = cand[np.argmin(np.abs(cand - len(tide_15)//2))]

    _, rain_15, center_idx = align_rainfall_to_tide(
        total_inches=total_inches,
        duration_minutes=duration_minutes,
        tide_curve_15min=tide_15,
        align=align,
        method=method,
        target_index=target_idx,
        prominence=prominence
    )
    return m15, tide_15, rain_15, used_live, center_idx
