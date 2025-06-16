import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# === Manually Created PF Table ===
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

# === Unit Conversion ===
def convert_units(value_in_inches, unit):
    if unit == "U.S. Customary":
        return value_in_inches
    elif unit == "Metric (SI)":
        return value_in_inches * 2.54
    else:
        raise ValueError("Unsupported unit.")

# === Generate Rainfall ===
def generate_rainfall(total_inches, duration_minutes, method="Normal"):
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
    return total_inches * y

# === Moon Phase Tide Ranges (in ft) ===
moon_tide_ranges = {
    "🌓 First Quarter: Neap": (-0.8, 1.2),
    "🌕 Full Moon: Spring": (-1.2, 1.8),
    "🌗 Last Quarter: Neap": (-0.9, 1.3),
    "🌑 New Moon: Spring": (-1.1, 1.7),
    
}

# === Generate Tide Curve (15-min intervals) ===
def generate_tide_curve(moon_phase, unit):
    tide_min, tide_max = moon_tide_ranges[moon_phase]
    minutes_full = np.arange(0, 2880, 1)  # 48 hours at 1-min resolution
    tide_full = ((np.sin(2 * np.pi * minutes_full / 720 - np.pi / 2) + 1) / 2) * (tide_max - tide_min) + tide_min
    if unit == "Metric (SI)":
        tide_full = tide_full * 0.3048
    minutes_15 = np.arange(0, 2880, 15)
    tide_15 = tide_full[minutes_15]
    return minutes_15, tide_15


# === Align Rainfall to Tide (15-min resolution) ===
def align_rainfall_to_tide(total_inches, duration_minutes, tide_curve, align="peak", method="Normal"):
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
        raise ValueError("Unsupported method for alignment")

    rain_profile *= total_inches

    search_curve = tide_curve if align == "peak" else -tide_curve
    peaks, _ = find_peaks(search_curve)
    if not peaks.any():
        raise ValueError("Could not detect tide peak or dip.")
    center_index = peaks[1]  # pick first peak (can be changed)

    full_rain = np.zeros(192)  # 2880 / 15 = 192 intervals
    start = center_index - intervals // 2
    for i in range(intervals):
        idx = start + i
        if 0 <= idx < 192:
            full_rain[idx] = rain_profile[i]

    rain_minutes = np.arange(0, 2880, 15)
    return rain_minutes, full_rain
