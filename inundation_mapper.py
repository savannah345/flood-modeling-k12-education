import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
import streamlit as st
from rasterio.warp import transform as rio_transform


# ================================================================
# DEM CLEANING
# ================================================================
def clean_dem(dem: np.ndarray) -> np.ndarray:
    dem2 = dem.astype(float)
    bad = (dem2 > 1e20) | (dem2 < -1e20)
    dem2[bad] = np.nan

    if np.isnan(dem2).any():
        mn = np.nanmin(dem2)
        dem2 = np.where(np.isnan(dem2), mn, dem2)

    return dem2


# ================================================================
# ArcGIS FlowDirection (powers of two) → D8 index (0..7)
# ================================================================
ARC_TO_IDX = {
    64: 0,   # N
    128: 1,  # NE
    1: 2,    # E
    2: 3,    # SE
    4: 4,    # S
    8: 5,    # SW
    16: 6,   # W
    32: 7    # NW
}

# D8 OFFSETS (index 0..7)
# N, NE, E, SE, S, SW, W, NW
D8_OFFSETS = [
    (-1, 0),   # 0 N
    (-1, 1),   # 1 NE
    (0, 1),    # 2 E
    (1, 1),    # 3 SE
    (1, 0),    # 4 S
    (1, -1),   # 5 SW
    (0, -1),   # 6 W
    (-1, -1)   # 7 NW
]


# ================================================================
# STATIC LOADER (DEM, D8, nodes)
# ================================================================
def load_static_inundation_data(dem_path, nodes_path, flowdir_path, unit_system):
    """
    Loads DEM, flowdir (D8), node locations and caches in session_state.
    """

    if "inundation_static" in st.session_state:
        return st.session_state["inundation_static"]

    # --- Load DEM ---
    with rasterio.open(dem_path) as src:
        raw_dem = src.read(1)
        dem = clean_dem(raw_dem)
        transform = src.transform
        crs = src.crs

        # pixel size (assume square)
        cell_dx = transform.a
        cell_size = float(cell_dx)

    # --- Unit conversion ---
    if unit_system == "U.S. Customary":
        dem_use = dem * 3.28084
        cell_area = (cell_size * 3.28084) ** 2
        cell_size = cell_size * 3.28084
    else:
        dem_use = dem
        cell_area = cell_size * cell_size

    # --- Load FlowDirection raster (ArcGIS format) ---
    with rasterio.open(flowdir_path) as fd:
        flow_raw = fd.read(1).astype(int)

    # Map ArcGIS values → model indices
    flowdir = np.full(flow_raw.shape, -1, dtype=int)
    for val, idx in ARC_TO_IDX.items():
        flowdir[flow_raw == val] = idx

    # --- Load nodes ---
    gdf_nodes = gpd.read_file(nodes_path)
    node_locs = {}
    for _, row in gdf_nodes.iterrows():
        x, y = row.geometry.x, row.geometry.y
        r, c = rasterio.transform.rowcol(transform, x, y)
        node_locs[row["NAME"]] = (int(r), int(c), float(x), float(y))

    # Cache all
    st.session_state["inundation_static"] = {
        "dem": dem_use,
        "flowdir": flowdir,
        "transform": transform,
        "crs": crs,
        "cell_area": cell_area,
        "cell_size": cell_size,
        "node_locs": node_locs,
    }
    return st.session_state["inundation_static"]


# ================================================================
# ADAPTIVE SLOPE-BASED FLOW FRACTION
# ================================================================
def compute_flow_fraction(dz, dist):
    """
    Adaptive fraction based on local slope.
    slope = max(dz/dist, 0)
    fraction = clipped linear scale: 0.05 to 1.0
    """
    if dz <= 0:
        return 0.0

    slope = dz / dist
    frac = slope * 10.0
    return float(max(0.05, min(1.0, frac)))


# ================================================================
# ROUTING FOR ONE NODE’S CUMULATIVE VOLUME
# ================================================================
def route_volume_from_node(dem, flowdir, r0, c0, volume_ft3, cell_area, cell_size):
    """
    Routes cumulative volume from (r0,c0) across D8 grid.
    Returns a 2D array of depths contributed by this node.
    """

    H, W = dem.shape
    depth = np.zeros((H, W), dtype=float)

    if not (0 <= r0 < H and 0 <= c0 < W):
        return depth

    initial_depth = volume_ft3 / cell_area
    depth[r0, c0] = initial_depth

    dist_card = cell_size
    dist_diag = cell_size * np.sqrt(2)

    # iterative relaxation
    for _ in range(500):  # large cap; stops early when stable
        moved = False
        new_depth = depth.copy()

        for r in range(H):
            for c in range(W):
                h = depth[r, c]
                if h <= 1e-12:
                    continue

                d8 = flowdir[r, c]
                if d8 < 0 or d8 > 7:
                    continue

                dr, dc = D8_OFFSETS[d8]
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    continue

                # compute slope based on DEM + depth (fills depressions)
                dz = (dem[r, c] + depth[r, c]) - (dem[rr, cc] + depth[rr, cc])
                dist = dist_diag if (dr != 0 and dc != 0) else dist_card

                frac = compute_flow_fraction(dz, dist)
                if frac <= 0:
                    continue

                move_amt = h * frac
                if move_amt <= 0:
                    continue

                new_depth[r, c] -= move_amt
                new_depth[rr, cc] += move_amt
                moved = True

        depth = new_depth
        if not moved:
            break

    return depth


# ================================================================
# MAIN ENTRY: compute_peak_inundation
# ================================================================
def compute_peak_inundation(
    flooding_df: pd.DataFrame,
    dem_path: str,
    nodes_shp_path: str,
    unit_system: str,
    flowdir_path="map_files/D8_flowdir.tif"
):
    """
    flooding_df must have columns:
        datetime, Node1, Node2, ...
    Returns DataFrame: lon, lat, depth
    """

    static = load_static_inundation_data(
        dem_path=dem_path,
        nodes_path=nodes_shp_path,
        flowdir_path=flowdir_path,
        unit_system=unit_system
    )

    dem = static["dem"]
    flowdir = static["flowdir"]
    transform = static["transform"]
    crs = static["crs"]
    cell_area = static["cell_area"]
    cell_size = static["cell_size"]
    node_locs = static["node_locs"]

    H, W = dem.shape

    # Determine dt
    if len(flooding_df) > 1:
        dt = (
            flooding_df["datetime"].iloc[1] -
            flooding_df["datetime"].iloc[0]
        ).total_seconds()
    else:
        dt = 0.0

    # Cumulative storm volume per node
    node_cols = [c for c in flooding_df.columns if c != "datetime"]
    volumes = {}
    for col in node_cols:
        arr = flooding_df[col].fillna(0).to_numpy()
        total_vol = float((arr * dt).sum())
        if total_vol > 0:
            volumes[col] = total_vol

    # Routing accumulation
    peak_grid = np.zeros((H, W), dtype=float)

    for node, vol_ft3 in volumes.items():
        if node not in node_locs:
            continue

        r0, c0, _, _ = node_locs[node]
        routed = route_volume_from_node(
            dem=dem, flowdir=flowdir,
            r0=r0, c0=c0,
            volume_ft3=vol_ft3,
            cell_area=cell_area,
            cell_size=cell_size
        )
        peak_grid = np.maximum(peak_grid, routed)

    # Convert to lon/lat/depth list
    pts = []
    for r in range(H):
        for c in range(W):
            d = peak_grid[r, c]
            if d <= 1e-12:
                continue

            x, y = transform * (c + 0.5, r + 0.5)
            lon, lat = rio_transform(crs, "EPSG:4326", [x], [y])
            pts.append({
                "lon": float(lon[0]),
                "lat": float(lat[0]),
                "depth": float(d)
            })

    return pd.DataFrame(pts)