import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd


import streamlit as st

def load_static_inundation_data(dem_path, flowdir_path, nodes_path, unit_system):
    if "inundation_static" in st.session_state:
        return st.session_state["inundation_static"]

    # --- Load DEM ---
    with rasterio.open(dem_path) as src:
        dem = clean_dem(src.read(1))
        transform = src.transform
        crs = src.crs

    # --- Unit convert DEM ONCE ---
    if unit_system == "U.S. Customary":
        dem_use = dem * 3.28084
        cell_area = (1.0 * 3.28084)**2   # ft²
    else:
        dem_use = dem
        cell_area = 1.0                  # m²

    # --- Load ArcGIS D8 Raster ---
    with rasterio.open(flowdir_path) as src_fd:
        d8_raw = src_fd.read(1).astype(int)

    # Convert ArcGIS flowdir (powers of 2) → 0..7 model indices
    flowdir = np.full_like(d8_raw, -1, dtype=np.int32)
    for arc_val, idx in ARC_TO_IDX.items():
        flowdir[d8_raw == arc_val] = idx

    # --- Load nodes and convert to raster locations ---
    gdf_nodes = gpd.read_file(nodes_path).to_crs(crs)

    node_locs = {}
    for _, row in gdf_nodes.iterrows():
        x, y = row.geometry.x, row.geometry.y
        col, r = ~transform * (x, y)
        node_locs[row["NAME"]] = (int(r), int(col), x, y)

    # --- Store in session cache ---
    st.session_state["inundation_static"] = {
        "dem": dem_use,
        "flowdir": flowdir,
        "transform": transform,
        "cell_area": cell_area,
        "node_locs": node_locs
    }
    return st.session_state["inundation_static"]

# --------------------------------------------------------------------------------------
# DEM CLEANING
# --------------------------------------------------------------------------------------

def clean_dem(dem: np.ndarray) -> np.ndarray:
    """
    Replace impossible DEM values with min DEM elevation.
    DEM provided in meters.
    """
    dem2 = dem.astype(float)
    mask = (dem2 > 1e20) | (dem2 < -1e20)
    dem2[mask] = np.nan

    if np.isnan(dem2).any():
        mn = np.nanmin(dem2)
        dem2 = np.where(np.isnan(dem2), mn, dem2)

    return dem2

# Row/col neighbor offsets in D8 order
D8_OFFSETS = [
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1)   # NW
]

# Mapping from ArcGIS Flow Direction (powers of 2) → model D8 index 0..7
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

# --------------------------------------------------------------------------------------
# ROUTE A SINGLE NODE'S MAX DEPTH OVER DEM
# --------------------------------------------------------------------------------------

def route_node_volume(max_depth: float,
                      node_r: int,
                      node_c: int,
                      dem: np.ndarray,
                      flowdir: np.ndarray) -> np.ndarray:
    """
    Spread a node's max depth through D8 downhill routing.
    This is a simplified steady-state downhill spread.

    Returns: 2D array of cell depths (same units as max_depth).
    """
    H, W = dem.shape
    depth = np.zeros((H, W), dtype=float)

    # Start with all water at the node cell
    depth[node_r, node_c] = max_depth

    # Perform repeated downhill spreading (not time-based, steady routing)
    for _ in range(200):  # iteration count for stability
        moved = False
        new_depth = depth.copy()

        for r in range(H):
            for c in range(W):
                if depth[r, c] <= 0:
                    continue

                d = flowdir[r, c]
                if d < 0:
                    continue  # pit or no downhill neighbor

                dr, dc = D8_OFFSETS[d]
                rr, cc = r + dr, c + dc

                # Move fraction of water downhill
                frac = 0.5
                delta = depth[r, c] * frac
                if delta > 0:
                    new_depth[r, c] -= delta
                    new_depth[rr, cc] += delta
                    moved = True

        depth = new_depth
        if not moved:
            break

    return depth


# --------------------------------------------------------------------------------------
# MAIN PEAK INUNDATION ROUTINE
# --------------------------------------------------------------------------------------

def compute_peak_inundation(
    flooding_df: pd.DataFrame,
    dem_path: str,
    nodes_shp_path: str,
    unit_system: str,
    flowdir_path="map_files/D8_flowdir.tif"
):
    """
    Computes peak inundation depth using:
      - precomputed ArcGIS D8 flow directions,
      - cached DEM,
      - cached raster transform,
      - cached node locations.

    flooding_df: timeseries of node flooding (cfs)
    Returns: DataFrame of flooded cell depths with lon, lat, depth
    """

    # ---- Load cached DEM + D8 + node locations ----
    static = load_static_inundation_data(
        dem_path=dem_path,
        flowdir_path=flowdir_path,
        nodes_path=nodes_shp_path,
        unit_system=unit_system
    )

    dem_use   = static["dem"]
    flowdir   = static["flowdir"]
    transform = static["transform"]
    cell_area = static["cell_area"]
    node_locs = static["node_locs"]

    H, W = dem_use.shape

    # ---- Determine timestep dt ----
    if len(flooding_df) < 2:
        raise ValueError("Flooding DF must have >=2 rows to detect timestep.")

    dt = (flooding_df["datetime"].iloc[1] -
          flooding_df["datetime"].iloc[0]).total_seconds()

    # ---- Compute max flood volume per node ----
    node_cols = [c for c in flooding_df.columns if c != "datetime"]
    max_vol = {}

    for col in node_cols:
        cfs_vals = flooding_df[col].fillna(0).to_numpy()
        vol_ft3 = cfs_vals * dt                 # ft³ per timestep
        max_vol[col] = float(np.max(vol_ft3))   # peak ft³

    # ---- Routing: accumulate depth across grid ----
    peak_grid = np.zeros((H, W), dtype=float)

    for node, vol_ft3 in max_vol.items():
        if vol_ft3 <= 0:
            continue
        if node not in node_locs:
            continue

        r0, c0, x0, y0 = node_locs[node]

        if not (0 <= r0 < H and 0 <= c0 < W):
            continue

        # Convert volume → depth at starting cell
        max_depth = vol_ft3 / cell_area    # in ft or m

        # Initialize temporary grid for routing
        depth = np.zeros((H, W), dtype=float)
        depth[r0, c0] = max_depth

        # Downhill spreading (iterative steady state)
        for _ in range(200):
            moved = False
            nd = depth.copy()

            for r in range(H):
                for c in range(W):
                    h = depth[r, c]
                    if h <= 0:
                        continue

                    d = flowdir[r, c]
                    if d < 0:
                        continue

                    dr, dc = D8_OFFSETS[d]
                    rr = r + dr
                    cc = c + dc

                    if 0 <= rr < H and 0 <= cc < W:
                        frac = 0.5
                        amt = h * frac
                        if amt > 0:
                            nd[r, c] -= amt
                            nd[rr, cc] += amt
                            moved = True

            depth = nd
            if not moved:
                break

        peak_grid = np.maximum(peak_grid, depth)

    # ---- Convert routed depths to point features ----
    pts = []
    for r in range(H):
        for c in range(W):
            d = peak_grid[r, c]
            if d <= 0:
                continue

            lon, lat = transform * (c + 0.5, r + 0.5)
            pts.append({"lon": lon, "lat": lat, "depth": float(d)})

    return pd.DataFrame(pts)