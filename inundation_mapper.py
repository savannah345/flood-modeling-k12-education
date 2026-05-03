"""
inundation_mapper.py

Consumes flooding time series obtained during Simulation iteration,
not from SWMM OUT file.

Required input:
- flooding_df: DataFrame with columns ["datetime", node1, node2, ...]
- storm_start: datetime
- storm_dur_hours: float

We compute inundation from:
1 hr before storm -> end of storm + 2 hr.
"""
from datetime import datetime, timedelta
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt


def extract_flooding_window(flood_df, storm_start, storm_dur_hours):
    """
    Clip full flooding time series to:
    storm_start -1 hr  → storm_start + storm_dur + 2 hr
    """
    clip_start = storm_start - timedelta(hours=1)
    clip_end   = storm_start + timedelta(hours=storm_dur_hours + 2)

    df = flood_df[(flood_df["datetime"] >= clip_start) &
                  (flood_df["datetime"] <= clip_end)].copy()

    return df


def integrate_flood_volume(df, report_step_minutes=5):
    """Convert cfs to ft³."""
    dt = report_step_minutes * 60.0
    node_cols = [c for c in df.columns if c != "datetime"]

    volumes = {}
    for nid in node_cols:
        s = df[nid].fillna(0).astype(float)
        volumes[nid] = float(s.sum() * dt)

    return volumes


def clean_dem(dem, nodata):
    mask_bad = (dem > 1e20) | (dem < -1e20)
    if np.any(mask_bad):
        dem = dem.astype(float)
        dem[mask_bad] = np.nan

    mask_large = dem > 1000
    if np.any(mask_large):
        dem[mask_large] = np.nan

    if np.isnan(dem).any():
        m = np.nanmin(dem)
        dem = np.where(np.isnan(dem), m, dem)

    return dem


def simulate_inundation(dem_path, nodes_path, volumes, out_png):
    """Very simplified inundation just for demonstration."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        dem = clean_dem(dem, src.nodata)
        transform = src.transform
        crs = src.crs

    gdf = gpd.read_file(nodes_path).to_crs(crs)

    depth = np.zeros_like(dem, float)

    for node, vol in volumes.items():
        row = gdf[gdf["NAME"] == node]
        if row.empty:
            continue
        pt = row.iloc[0].geometry
        c, r = ~transform * (pt.x, pt.y)
        r, c = int(r), int(c)

        # very simplistic: put all water in that 1 cell
        depth[r,c] += vol / (transform.a * transform.a)

    plt.figure(figsize=(6,6))
    plt.imshow(depth, cmap="Blues", vmin=0, vmax=np.nanpercentile(depth,95))
    plt.axis("off")
    plt.savefig(out_png, dpi=150)
    plt.close()

    return depth