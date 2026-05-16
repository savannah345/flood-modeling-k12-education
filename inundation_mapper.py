
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform as rio_transform

from numba import njit
from concurrent.futures import ProcessPoolExecutor

# ESRI Codes:
# 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE

ESRI_D8_TO_IDX = {
    1:   2,   # E
    2:   3,   # SE
    4:   4,   # S
    8:   5,   # SW
    16:  6,   # W
    32:  7,   # NW
    64:  0,   # N
    128: 1    # NE
}

# Offsets (row, col)
D8_OFFSETS = np.array([
    (-1,  0),   # N
    (-1,  1),   # NE
    ( 0,  1),   # E
    ( 1,  1),   # SE
    ( 1,  0),   # S
    ( 1, -1),   # SW
    ( 0, -1),   # W
    (-1, -1),   # NW
], dtype=np.int64)


# ------------------------------------------------
# DEM Cleaning
# ------------------------------------------------
def clean_dem(dem: np.ndarray) -> np.ndarray:
    """Replace insane values with NaN. Assumes DEM in feet."""
    dem2 = dem.astype(float)
    bad = (dem2 > 1e20) | (dem2 < -1e20)
    dem2[bad] = np.nan
    return dem2


# ------------------------------------------------
# Numba-accelerated routing for a single node
# ------------------------------------------------
@njit
def route_volume_from_node_numba(dem, flowdir, start_r, start_c, volume_ft3, cell_area_ft2,
                                 d8_offsets, d8_map_keys, d8_map_vals):
    """
    Routes a volume (ft³) from a single node across the DEM using D8 directions.
    Returns a depth raster (ft).
    """
    nrows, ncols = dem.shape
    depth = np.zeros((nrows, ncols))

    # Convert Python dict to arrays for Numba:
    # d8_map_keys = array of valid D8 codes
    # d8_map_vals = array of corresponding D8 indices (0..7)
    # We'll linear search because map small (8 entries)
    stack_r = [start_r]
    stack_c = [start_c]
    stack_v = [volume_ft3]

    visited = set()

    while len(stack_r) > 0:
        r = stack_r.pop()
        c = stack_c.pop()
        vol = stack_v.pop()

        key = r * ncols + c
        if key in visited:
            continue
        visited.add(key)

        # Deposit water
        depth[r, c] += vol / cell_area_ft2

        fd = flowdir[r, c]

        # lookup D8 idx
        idx = -1
        for i in range(len(d8_map_keys)):
            if fd == d8_map_keys[i]:
                idx = d8_map_vals[i]
                break
        if idx == -1:
            # invalid or no-flow direction cell
            continue

        dr = d8_offsets[idx, 0]
        dc = d8_offsets[idx, 1]
        r2 = r + dr
        c2 = c + dc

        if (r2 < 0) or (r2 >= nrows) or (c2 < 0) or (c2 >= ncols):
            continue

        # only route downhill
        dz = dem[r, c] - dem[r2, c2]
        if dz <= 0:
            continue

        stack_r.append(r2)
        stack_c.append(c2)
        stack_v.append(vol)

    return depth


# Wrapper for multiprocessing
def _route_wrapper(args):
    """
    Unpacks arguments and runs Numba-routing for use with multiprocessing.
    """
    (dem, flowdir, r0, c0, vol, cell_area_ft2, d8_offsets,
     d8_map_keys, d8_map_vals) = args

    return route_volume_from_node_numba(
        dem, flowdir, r0, c0, vol, cell_area_ft2,
        d8_offsets, d8_map_keys, d8_map_vals
    )


# ------------------------------------------------
# MAIN Inundation Function
# ------------------------------------------------
def compute_peak_inundation(
    flooding_df: pd.DataFrame,
    dem_path: str,
    nodes_shp_path: str,
    flowdir_path: str
):
    """
    flooding_df: DataFrame with datetime + one column per node ID.
    All values must be **volume in ft³**, not ft³/s.
    (Your streamlit scenario builder will handle dt * flow conversion.)

    Returns: DataFrame containing lon, lat, depth_ft.
    """

    # ------------------------------------------------
    # Load DEM (feet)
    # ------------------------------------------------
    with rasterio.open(dem_path) as src_dem:
        dem = src_dem.read(1)
        transform = src_dem.transform
        dem = clean_dem(dem)

        # Cell size in feet
        cell_size_x = transform.a
        cell_size_y = -transform.e
        cell_area_ft2 = cell_size_x * cell_size_y

    # ------------------------------------------------
    # Load FlowDir
    # ------------------------------------------------
    with rasterio.open(flowdir_path) as src_fd:
        flowdir = src_fd.read(1).astype(np.int32)

    # ------------------------------------------------
    # Load Nodes (feet)
    # ------------------------------------------------
    gdf_nodes = gpd.read_file(nodes_shp_path)
    gdf_nodes = gdf_nodes.to_crs("EPSG:2284")  # VA South (ft)

    node_locs = {}
    for _, row in gdf_nodes.iterrows():
        node_id = row["NAME"] if "NAME" in row else row["nodeid"]
        x_ft = row.geometry.x
        y_ft = row.geometry.y

        col = int((x_ft - transform.c) / cell_size_x)
        row_pix = int((transform.f - y_ft) / cell_size_y)

        if 0 <= row_pix < dem.shape[0] and 0 <= col < dem.shape[1]:
            node_locs[node_id] = (row_pix, col)

    # ------------------------------------------------
    # Determine peak volumes per node (ft³)
    # ------------------------------------------------
    flood_cols = [c for c in flooding_df.columns if c != "datetime"]
    peak = flooding_df[flood_cols].max(axis=0)  # already in ft³ (your streamlit code will ensure this)

    # ------------------------------------------------
    # Prepare parallel tasks
    # ------------------------------------------------
    d8_keys = np.array(list(ESRI_D8_TO_IDX.keys()), dtype=np.int32)
    d8_vals = np.array(list(ESRI_D8_TO_IDX.values()), dtype=np.int32)
    d8_offsets = D8_OFFSETS.copy()

    tasks = []
    for node_id, vol in peak.items():
        if node_id not in node_locs:
            continue
        r0, c0 = node_locs[node_id]
        tasks.append(
            (dem, flowdir, r0, c0, vol, cell_area_ft2, d8_offsets, d8_keys, d8_vals)
        )

    # ------------------------------------------------
    # Parallel Routing
    # ------------------------------------------------
    depth_layers = []

    if len(tasks) == 1:
        # no reason for multiprocessing
        depth_layers = [_route_wrapper(tasks[0])]
    else:
        with ProcessPoolExecutor() as exe:
            for result in exe.map(_route_wrapper, tasks):
                depth_layers.append(result)

    # Sum all routed depths
    total_depth = np.sum(depth_layers, axis=0)

    # ------------------------------------------------
    # Convert non-zero depth cells into lon/lat
    # ------------------------------------------------
    rows, cols = np.where(total_depth > 0)

    xs = transform.c + cols * cell_size_x
    ys = transform.f - rows * cell_size_y

    lons, lats = rio_transform(
        "EPSG:2284",
        "EPSG:4326",
        xs.tolist(),
        ys.tolist()
    )

    df = pd.DataFrame({
        "lon": lons,
        "lat": lats,
        "depth_ft": total_depth[rows, cols]
    })

    return df