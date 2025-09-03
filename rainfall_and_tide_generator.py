# rainfall_and_tide_generator.py
from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from scipy.signal import find_peaks

# --------------------------
# Public Exports (what your app imports)
# --------------------------
__all__ = [
    "pf_df",
    "convert_units",
    "generate_rainfall",
    "moon_tide_ranges",
    "generate_tide_curve",
    "align_rainfall_to_tide",
    "fetch_greenstream_dataframe",
    "build_timestep_and_resample_15min",
    "get_tide_real_or_synthetic",
]

# ============================================================
# 1) PF Table (Depths in inches for given durations/return periods)
# ============================================================
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

# ============================================================
# 2) Unit conversions
# ============================================================
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

# ============================================================
# 3) Synthetic rainfall generator (15-min resolution)
# ============================================================
def generate_rainfall(total_inches: float,
                      duration_minutes: int,
                      method: str = "Normal") -> np.ndarray:
    """
    Returns a 1D array with 15-min bins summing to 'total_inches' across duration_minutes.
    duration_minutes must be a multiple of 15.
    """
    if duration_minutes % 15 != 0:
        raise ValueError("duration_minutes must be divisible by 15.")
    intervals = duration_minutes // 15

    if method == "Normal":
        x = np.linspace(-3, 3, intervals)
        y = np.exp(-0.5 * x**2)
        y /= y.sum()
    elif method == "Randomized":
        rng = np.random.default_rng()
        y = rng.random(intervals)
        y /= y.sum()
    else:
        raise ValueError("Invalid rainfall distribution method.")

    return total_inches * y  # inches by design

# ============================================================
# 4) Tide definitions + synthetic tide generator (15-min)
# ============================================================
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

# ============================================================
# 5) Align rainfall to a 15-min tide curve
# ============================================================
def align_rainfall_to_tide(total_inches: float,
                           duration_minutes: int,
                           tide_curve_15min: np.ndarray,
                           align: str = "peak",
                           method: str = "Normal") -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns a generated rainfall hyetograph to a 15-min tide curve.
    - total_inches: storm depth (inches)
    - duration_minutes: storm duration (multiple of 15)
    - tide_curve_15min: 192-length tide values over 48h at 15-min
    - align: "peak" (center rain at a high tide) or "low" (center at a low tide)
    Returns (minutes_15, rain_15) both length 192.
    """
    if duration_minutes % 15 != 0:
        raise ValueError("duration_minutes must be divisible by 15.")
    intervals = duration_minutes // 15

    if method == "Normal":
        x = np.linspace(-3, 3, intervals)
        rain_profile = np.exp(-0.5 * x**2)
        rain_profile /= rain_profile.sum()
    elif method == "Randomized":
        rng = np.random.default_rng()
        rain_profile = rng.random(intervals)
        rain_profile /= rain_profile.sum()
    else:
        raise ValueError("Unsupported method for alignment.")

    rain_profile *= total_inches

    search_curve = tide_curve_15min if align == "peak" else -tide_curve_15min
    peaks, _ = find_peaks(search_curve)
    if peaks.size == 0:
        raise ValueError("Could not detect a tide peak/dip for alignment.")
    center_index = peaks[1] if len(peaks) > 1 else peaks[0]  # pick a central-ish peak

    full_rain = np.zeros(192, dtype=float)  # 48h at 15-min
    start = center_index - intervals // 2
    for i in range(intervals):
        idx = start + i
        if 0 <= idx < 192:
            full_rain[idx] = rain_profile[i]

    minutes_15 = np.arange(0, 2880, 15)
    return minutes_15, full_rain

# ============================================================
# 6) Real-time Greenstream fetch (48h @ 6-min) -> DataFrame
#     Then resample to 15-min with unit handling
# ============================================================

# --- You may tune these constants if the site moves icons around ---
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
        context = browser.new_context(accept_downloads=True, viewport={"width": 1920, "height": 1080})
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

# ============================================================
# 7) Build 6-min timeline and resample to 15-min (unit aware)
# ============================================================
def build_timestep_and_resample_15min(df_raw: pd.DataFrame,
                                      water_col: str = WATER_COL_LIVE,
                                      unit: str = "U.S. Customary",
                                      start_ts: Optional[pd.Timestamp] = None
                                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input: df_raw from Greenstream (≈480 rows @ 6-minute interval).
    - Creates a 6-min datetime index across len(df)
    - Converts ft->m if unit == 'Metric (SI)'
    - Resamples to 15-min mean
    Returns: (minutes_15, tide_15) where tide_15 is in ft or m per 'unit'.
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

    # Anchor 6-min timeline. If you know the real start time, pass start_ts.
    if start_ts is None:
        # Snap to the quarter-hour so resample bins align nicely
        start_ts = (pd.Timestamp.now().floor("15T") - pd.Timedelta(minutes=6*(n-1)))
    tide_df["TimeStep"] = pd.date_range(start=start_ts, periods=n, freq="6T")
    tide_df = tide_df.set_index("TimeStep")

    # Unit conversion (live data are in feet)
    vals = tide_df[water_col].astype(float)
    if unit == "Metric (SI)":
        vals = feet_to_meters(vals)
    tide_df[water_col] = vals

    # Resample to 15-minute bins
    tide_15 = tide_df.resample("15T").mean(numeric_only=True)[water_col].to_numpy()

    # Minutes array (trim to actual length if some points missing)
    minutes_15 = np.arange(0, 2880, 15)[: len(tide_15)]
    return minutes_15, tide_15

# ============================================================
# 8) Orchestrator: prefer real-time, fallback to synthetic
# ============================================================
def get_tide_real_or_synthetic(moon_phase: str,
                               unit: str,
                               start_ts: Optional[pd.Timestamp] = None
                               ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Try live tide (Greenstream). If it fails, return synthetic tide.
    Returns:
      minutes_15 (np.ndarray),
      tide_15    (np.ndarray) in selected 'unit' (ft or m),
      used_live  (bool)
    """
    try:
        df_live = fetch_greenstream_dataframe()
        m15, tide_15 = build_timestep_and_resample_15min(
            df_raw=df_live, water_col=WATER_COL_LIVE, unit=unit, start_ts=start_ts
        )
        return m15, tide_15, True
    except Exception:
        # Synthetic fallback (already outputs selected unit)
        m15, tide_15 = generate_tide_curve(moon_phase, unit)
        return m15, tide_15, False
