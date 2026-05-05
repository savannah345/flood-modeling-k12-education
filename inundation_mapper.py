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

def simulate_inundation_timestep(dem, transform, gdf_nodes, volumes_dict):
    """
    Compute inundation depth for ONE timestep.
    volumes_dict: {node: volume_ft3}
    Returns a list of {
       'lon': float,
       'lat': float,
       'depth': float
    }
    """
    depth = np.zeros_like(dem, float)
    cell_area = transform.a * transform.a

    for node, vol in volumes_dict.items():
        row = gdf_nodes[gdf_nodes["NAME"] == node]
        if row.empty:
            continue
        pt = row.iloc[0].geometry
        c, r = ~transform * (pt.x, pt.y)
        r, c = int(r), int(c)

        if 0 <= r < depth.shape[0] and 0 <= c < depth.shape[1]:
            depth[r, c] += vol / cell_area

    points = []
    H, W = depth.shape
    for r in range(H):
        for c in range(W):
            d = depth[r, c]
            if d <= 0:
                continue
            x, y = transform * (c + 0.5, r + 0.5)
            points.append({"lon": x, "lat": y, "depth": float(d)})

    return points