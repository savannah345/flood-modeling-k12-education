# app.py — optimized with side-by-side runoff maps + flooded Y/N nodes

# --- stdlib
import os, io, re, glob, sys, shutil, tempfile, subprocess
from datetime import datetime, timedelta
from typing import Dict, Tuple, List

# --- third-party
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import geopandas as gpd
import pydeck as pdk
import matplotlib as mpl
from matplotlib import colors as mcolors
from pyswmm import Simulation, Nodes
from streamlit import components

# --- project
from auth_supabase import create_user, authenticate_user, reset_password
from rainfall_and_tide_generator import (
    pf_df,
    moon_tide_ranges,
    get_tide_real_or_synthetic,
    get_aligned_rainfall,
)

st.set_page_config(page_title="CoastWise", layout="centered")

@st.cache_resource(show_spinner=False)
def ensure_playwright_browsers():
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
ensure_playwright_browsers()

MSL_OFFSET_NAVD88_FT = 1.36  # NAVD88 -> MSL

@st.cache_resource(show_spinner=False)
def load_ws(path="Subcatchments.shp") -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = (gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326))
    gdf = gdf[gdf.geometry.notnull() & gdf.is_valid].copy()   # <—
    return gdf

@st.cache_resource(show_spinner=False)
def load_nodes(path="Nodes.shp") -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = (gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326))
    gdf = gdf[gdf.geometry.notnull() & gdf.is_valid].copy()   # <—
    return gdf

@st.cache_resource(show_spinner=False)
def load_pipes(path: str):
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.geom_type.isin(["LineString","MultiLineString"])]
    return gdf

@st.cache_data(show_spinner=False)
def load_raster_cells(path="raster_cells_per_sub.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)
    def extract_number(name):
        if not isinstance(name, str): return float('inf')
        m = re.search(r"_(\d+)", name)
        return int(m.group(1)) if m else float('inf')
    return df.sort_values(by="NAME", key=lambda x: x.map(extract_number)).reset_index(drop=True)

def tide_to_feet_for_swmm(tide_curve_ui: np.ndarray, ui_unit: str) -> np.ndarray:
    arr = np.asarray(tide_curve_ui, dtype=float)
    return arr if ui_unit == "U.S. Customary" else (arr / 0.3048)

def format_timeseries(name: str, minutes: np.ndarray, values: np.ndarray, start_datetime: str) -> List[str]:
    start_dt = datetime.strptime(start_datetime, "%m/%d/%Y %H:%M")
    out = []
    for m, v in zip(minutes, values):
        ts = (start_dt + timedelta(minutes=int(m))).strftime("%m/%d/%Y %H:%M")
        out.append(f"{name} {ts} {float(v):.5f}")
    return out

def return_period_labels(duration_min: int, unit_ui: str) -> Dict[str, str]:
    row = pf_df[pf_df["Duration_Minutes"] == duration_min]
    if row.empty: return {}
    row = row.iloc[0]
    label = "inches" if unit_ui == "U.S. Customary" else "centimeters"
    factor = 1 if unit_ui == "U.S. Customary" else 2.54
    return {
        col: f"{col}-year storm ({100//int(col)}% annual): {row[col]*factor:.2f} {label}"
        for col in pf_df.columns[1:]
    }

def make_color(values, vmin, vmax, a=0.9):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = mpl.colormaps.get_cmap("Blues")
    out = []
    for v in values:
        if pd.isna(v):
            out.append([200,200,200,int(0.4*255)])
        else:
            r,g,b,_ = cmap(norm(float(v)))
            out.append([int(r*255), int(g*255), int(b*255), int(a*255)])
    return out

def prep_total_runoff_gdf(rpt_df: pd.DataFrame, unit_ui: str, ws_gdf: gpd.GeoDataFrame):
    """
    Join per-subcatchment Total Runoff to polygons.

    Expects rpt_df with:
      - Subcatchment
      - Total_in  (depth units from report; your RPT shows inches)

    Back-compat:
      - If Total_in is missing but Impervious/Pervious exist, uses their sum.
    Returns: (GeoDataFrame_with_Total_R, display_unit_str)
    """
    g = ws_gdf.copy()

    # Empty -> return NaNs with unit label
    if rpt_df is None or rpt_df.empty:
        g["Total_R"] = np.nan
        return g, ("in" if unit_ui == "U.S. Customary" else "cm")

    # Choose a source column for total depth
    if "Total_in" in rpt_df.columns:
        total_in = rpt_df["Total_in"].astype(float)
    elif {"Impervious","Pervious"} <= set(rpt_df.columns):
        total_in = (rpt_df["Impervious"].astype(float).fillna(0.0) +
                    rpt_df["Pervious"].astype(float).fillna(0.0))
    else:
        # nothing usable
        g["Total_R"] = np.nan
        return g, ("in" if unit_ui == "U.S. Customary" else "cm")

    # Normalize names on both sides for a robust join
    def _norm(n):
        n = "" if n is None else str(n)
        n = n.strip().lower()
        m = re.fullmatch(r"sub[_\s-]*0*(\d+)", n)  # Sub_01 -> sub_1 ; Sub-1 -> sub_1
        return f"sub_{m.group(1)}" if m else re.sub(r"\s+", "", n)

    left = rpt_df.assign(NAME_JOIN=rpt_df["Subcatchment"].astype(str).map(_norm))[
        ["NAME_JOIN"]
    ].copy()
    left["Total_in"] = total_in.values

    g["NAME_JOIN"] = g["NAME"].astype(str).map(_norm)
    merged = g.merge(left, on="NAME_JOIN", how="left").drop(columns=["NAME_JOIN"])

    # Convert to display units for the UI
    if unit_ui == "U.S. Customary":
        merged["Total_R"] = merged["Total_in"]  # inches
        unit = "in"
    else:
        merged["Total_R"] = merged["Total_in"] * 2.54  # -> cm
        unit = "cm"

    return merged, unit

def extract_total_runoff(rpt_file: str):
    """
    Robustly parse 'Subcatchment Runoff Summary' from an SWMM .rpt file.
    Returns DataFrame: Subcatchment, Total_in  (Total Runoff depth)
    Strategy: find the section, skip header lines, then for each data row
    read the first token as the name and take the 7th numeric value.
    """
    import os, re
    import pandas as pd

    if not os.path.exists(rpt_file):
        return pd.DataFrame(columns=["Subcatchment","Total_in"])

    with open(rpt_file, "r", errors="ignore") as f:
        lines = f.readlines()

    # 1) locate the section
    sec_i = next((i for i, l in enumerate(lines) if "Subcatchment Runoff Summary" in l), None)
    if sec_i is None:
        return pd.DataFrame(columns=["Subcatchment","Total_in"])

    def is_dash(s: str) -> bool:
        s = s.strip()
        return len(s) > 0 and set(s) <= set("- ")

    # 2) advance to the line that actually starts with "Subcatchment"
    i = sec_i + 1
    while i < len(lines) and "Subcatchment" not in lines[i]:
        i += 1
    if i >= len(lines):
        return pd.DataFrame(columns=["Subcatchment","Total_in"])

    # 3) skip the underline dashes after that units line
    i += 1
    while i < len(lines) and not is_dash(lines[i]):
        i += 1
    if i >= len(lines):
        return pd.DataFrame(columns=["Subcatchment","Total_in"])
    i += 1  # move to first data row

    # 4) parse data rows
    rows = []
    float_re = re.compile(r"[-+]?\d+(?:\.\d+)?")
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        if not raw.strip() or is_dash(raw):
            break
        # stop at next section just in case
        if raw.lstrip().startswith("*") or "Node Depth Summary" in raw or "Analysis Options" in raw:
            break

        # name = first contiguous non-space token
        parts = raw.strip().split()
        name = parts[0] if parts else None

        # grab all numeric tokens on the line
        nums = [float(m.group(0)) for m in float_re.finditer(raw)]
        # Expect at least 10 numbers; Total Runoff (in) is the 7th numeric (0-based idx 6)
        if name and len(nums) >= 7:
            total_in = nums[6]
            rows.append({"Subcatchment": name, "Total_in": total_in})

        i += 1

    return pd.DataFrame(rows, columns=["Subcatchment","Total_in"])


# ---- Node overlay helpers (flooded YES/NO) ----
def node_layer_from_shp(node_shp_path: str, node_vol_dict_post5h: Dict, name_field_hint: str = "NAME"):
    try:
        nodes_gdf = load_nodes(node_shp_path)
        # choose join field
        name_field = name_field_hint if name_field_hint in nodes_gdf.columns else None
        if name_field is None:
            for c in ["NAME","Name","node_id","NODEID","NodeID","id","ID"]:
                if c in nodes_gdf.columns:
                    name_field = c
                    break
        if name_field is None:
            for c in nodes_gdf.columns:
                if c.lower() != "geometry":
                    name_field = c; break
        if name_field is None:
            return None

        nodes_gdf["_node_id"] = nodes_gdf[name_field].astype(str).str.strip()
        vol_map = {str(k).strip(): float(v) for k, v in (node_vol_dict_post5h or {}).items()}
        nodes_gdf["_cuft_post5h"] = nodes_gdf["_node_id"].map(lambda nid: vol_map.get(nid, 0.0))
        nodes_gdf["_flooded"] = nodes_gdf["_cuft_post5h"] > 0.0
        nodes_gdf["_color_rgba"] = nodes_gdf["_flooded"].map(lambda f: [255,0,0,255] if f else [0,0,0,255])

        return pdk.Layer(
            "GeoJsonLayer",
            data=nodes_gdf.__geo_interface__,
            pickable=True,
            stroked=False,
            filled=True,
            get_fill_color="properties._color_rgba",
            get_point_radius=6,
            pointRadiusMinPixels=6,
        )
    except Exception:
        return None

def render_side_by_side_total_runoff_maps(
    left_df_in_inches: pd.DataFrame,  left_title: str,  left_nodes_post5h_dict: Dict,
    right_df_in_inches: pd.DataFrame, right_title: str, right_nodes_post5h_dict: Dict,
    unit_ui: str, ws_shp_path: str, node_shp_path: str, pipe_shp_path: str,
    node_name_field_hint: str = "NAME"
):
    # --- Prep watershed + runoff geodata ---
    ws_gdf = load_ws(ws_shp_path)
    gdf_left,  runoff_unit_L = prep_total_runoff_gdf(left_df_in_inches,  unit_ui, ws_gdf)
    gdf_right, runoff_unit_R = prep_total_runoff_gdf(right_df_in_inches, unit_ui, ws_gdf)
    runoff_unit = runoff_unit_L  # both should match

    # Shared color scale across both sides
    vals = pd.concat([gdf_left["Total_R"], gdf_right["Total_R"]], ignore_index=True)
    if len(vals) == 0 or not np.isfinite(np.nanmin(vals)) or not np.isfinite(np.nanmax(vals)):
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1e-6

    # Fill colors + labels
    gdf_left["_fill_total"]  = make_color(gdf_left["Total_R"],  vmin, vmax)
    gdf_right["_fill_total"] = make_color(gdf_right["Total_R"], vmin, vmax)
    for g in (gdf_left, gdf_right):
        g["_label"] = g["NAME"]

    # View state from combined centroid
    centroid = pd.concat([gdf_left.geometry, gdf_right.geometry]).union_all().centroid
    view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=14.25)

    # --- Pipes layer (single instance, reused on both maps) ---
    pipe_layer = None
    try:
        pipes = gpd.read_file(pipe_shp_path)
        pipes = pipes[pipes.geometry.notnull() & pipes.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        if not pipes.empty:
            if pipes.crs != ws_gdf.crs:
                pipes = pipes.to_crs(ws_gdf.crs)
            if "NAME" in pipes.columns:
                pipes["_pipe_label"] = pipes["NAME"]
            elif "Conduit" in pipes.columns:
                pipes["_pipe_label"] = pipes["Conduit"]
            else:
                pipes["_pipe_label"] = ""
            pipe_layer = pdk.Layer(
                "GeoJsonLayer",
                data=pipes.__geo_interface__,
                pickable=True, filled=False, stroked=True,
                get_line_color=[80, 80, 80, 255],
                get_line_width=2,
                line_width_min_pixels=2,
                tooltip={"text": "{_pipe_label}"}
            )
    except Exception as _:
        pipe_layer = None  # Fail quietly; map still renders

    # --- Helpers for polygon + labels ---
    def _poly_and_label_layers(gdf: "gpd.GeoDataFrame"):
        poly_layer = pdk.Layer(
            "GeoJsonLayer",
            data=gdf.__geo_interface__,
            pickable=True, stroked=True, filled=True,
            get_fill_color="properties._fill_total",
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=1,
        )
        reps = gdf.geometry.representative_point()
        labels = pd.DataFrame({"lon": reps.x, "lat": reps.y, "text": gdf["_label"]})
        text_layer = pdk.Layer(
            "TextLayer",
            data=labels,
            get_position='[lon, lat]',
            get_text="text",
            get_size=12,
            get_color=[0, 0, 0],
            get_alignment_baseline="'center'"
        )
        return poly_layer, text_layer

    left_poly,  left_text  = _poly_and_label_layers(gdf_left)
    right_poly, right_text = _poly_and_label_layers(gdf_right)

    # Nodes (topmost data layer)
    left_nodes_layer  = node_layer_from_shp(node_shp_path,  left_nodes_post5h_dict,  node_name_field_hint)
    right_nodes_layer = node_layer_from_shp(node_shp_path, right_nodes_post5h_dict, node_name_field_hint)

    # --- Render side-by-side ---
    c1, c2 = st.columns(2, gap="medium")
    tooltip_html = "<b>{NAME}</b><br/>Total runoff: {Total_R} " + runoff_unit

    with c1:
        st.markdown(f"**{left_title}**")
        layers_left = [left_poly]
        if pipe_layer is not None: layers_left.append(pipe_layer)       # subcatchments < pipes
        if left_nodes_layer is not None: layers_left.append(left_nodes_layer)  # < nodes
        layers_left.append(left_text)  # labels on top
        st.pydeck_chart(
            pdk.Deck(
                layers=layers_left,
                initial_view_state=view_state,
                map_provider="carto",
                map_style="light",
                tooltip={"html": tooltip_html, "style": {"backgroundColor": "white", "color": "black"}},
            ),
            use_container_width=True
        )

    with c2:
        st.markdown(f"**{right_title}**")
        layers_right = [right_poly]
        if pipe_layer is not None: layers_right.append(pipe_layer)
        if right_nodes_layer is not None: layers_right.append(right_nodes_layer)
        layers_right.append(right_text)
        st.pydeck_chart(
            pdk.Deck(
                layers=layers_right,
                initial_view_state=view_state,
                map_provider="carto",
                map_style="light",
                tooltip={"html": tooltip_html,
                        "style": {"backgroundColor": "white", "color": "black",
                            "fontFamily": "Inter, Arial, Helvetica, sans-serif", "fontSize": "12px"}},
            ),
            use_container_width=True
        )

    # --- Legend (shared scale) ---
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = mpl.colormaps.get_cmap("Blues")
    c0  = [int(v * 255) for v in cmap(norm(vmin))[:3]]
    c1b = [int(v * 255) for v in cmap(norm(vmax))[:3]]
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin-top:6px;">
          <div style="min-width:260px; max-width:640px; width:60%;">
            <div style="text-align:center; font-size:13px;"><b>Runoff Legend ({runoff_unit})</b></div>
            <div style="display:flex; align-items:center; gap:10px;">
              <span>{vmin:.2f}</span>
              <div style="flex:1; height:12px;
                  background:linear-gradient(to right,
                  rgb({c0[0]},{c0[1]},{c0[2]}),
                  rgb({c1b[0]},{c1b[1]},{c1b[2]}));
                  border:1px solid #888;"></div>
              <span>{vmax:.2f}</span>
            </div>
            <div style="color:#555; font-size:12px; text-align:center; margin-top:6px;">
              Same scale for both maps
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def _workspace_dir_from_session() -> str:
    paths = [
        st.session_state.get("WS_SHP_PATH", "Subcatchments.shp"),
        st.session_state.get("NODE_SHP_PATH", "Nodes.shp"),
        st.session_state.get("PIPE_SHP_PATH", "Conduits.shp"),
    ]
    for p in paths:
        if isinstance(p, str):
            d = os.path.dirname(os.path.abspath(p)) or os.getcwd()
            # if the shp exists or the directory looks real, accept it
            if os.path.isdir(d) and (os.path.exists(p) or any(fn.lower().endswith(".shp") for fn in os.listdir(d))):
                return d
    return os.getcwd()


def _safe_copy(src: str, dst_dir: str, new_name: str | None = None) -> str | None:
    try:
        if not (src and os.path.exists(src) and os.path.isdir(dst_dir)):
            return None
        base = new_name if new_name else os.path.basename(src)
        root, ext = os.path.splitext(base)
        dst = os.path.join(dst_dir, base)
        i = 1
        while os.path.exists(dst):
            dst = os.path.join(dst_dir, f"{root}__{i}{ext}")
            i += 1
        shutil.copy2(src, dst)
        return dst
    except Exception:
        return None

def run_swmm_scenario(
    scenario_name: str,
    rain_lines: List[str],
    tide_lines: List[str],
    lid_lines: List[str],
    gate_flag: str,
    report_interval=timedelta(minutes=5),
    template_path="swmm_project.inp",
    warmup_hours=5,
) -> Tuple[List[float], List[str], str]:
    temp_dir = st.session_state.temp_dir
    inp_path = os.path.join(temp_dir, f"{scenario_name}.inp")
    rpt_path = os.path.join(temp_dir, f"{scenario_name}.rpt")
    out_path = os.path.join(temp_dir, f"{scenario_name}.out")

    with open(template_path, "r") as f:
        text = f.read()
    text = (text
            .replace("$RAINFALL_TIMESERIES$", "\n".join(rain_lines))
            .replace("$TIDE_TIMESERIES$", "\n".join(tide_lines))
            .replace("$LID_USAGE$", "\n".join(lid_lines))
            .replace("$TIDE_GATE_CONTROL$", gate_flag))
    with open(inp_path, "w") as f:
        f.write(text)

    cumulative_flooding_acft, timestamps = [], []
    cum_cuft_all = 0.0
    cum_cuft_post5h = 0.0
    node_cuft_post5h = {}

    step_s = int(report_interval.total_seconds())
    warmup_s = int(warmup_hours * 3600)

    with Simulation(inp_path) as sim:
        sim.step_advance(step_s)
        nodes = list(Nodes(sim))
        t0 = None
        for _ in sim:
            now = sim.current_time
            if t0 is None:
                t0 = now
            elapsed_s = int((now - t0).total_seconds())

            total_cfs = 0.0
            post5h = elapsed_s >= warmup_s
            for n in nodes:
                q = float(n.flooding)  # cfs
                total_cfs += q
                if post5h and q > 0.0:
                    nid = getattr(n, "nodeid", None) or getattr(n, "id", None) or str(n)
                    vol = q * step_s  # ft^3
                    node_cuft_post5h[nid] = node_cuft_post5h.get(nid, 0.0) + vol
                    cum_cuft_post5h += vol

            cum_cuft_all += total_cfs * step_s
            cumulative_flooding_acft.append(cum_cuft_all / 43560.0)
            timestamps.append(now.strftime("%m-%d %H:%M"))

    # move SWMM’s default-named outputs into our scenario-named paths if present
    for src, dst in [("updated_model.rpt", rpt_path), ("updated_model.out", out_path)]:
        p = os.path.join(temp_dir, src)
        if os.path.exists(p):
            shutil.move(p, dst)
    try:
        ws_dir = _workspace_dir_from_session()
        # Make filenames clear and user-specific
        # e.g., user_123_baseline_nogate_current.rpt
        _safe_copy(rpt_path, ws_dir, new_name=f"{scenario_name}.rpt")
        _safe_copy(out_path, ws_dir, new_name=f"{scenario_name}.out")
        _safe_copy(inp_path, ws_dir, new_name=f"{scenario_name}.inp")
    except Exception:
        pass  # don't block the app if workspace copy fails

    st.session_state[f"{scenario_name}_total_flood"] = (cumulative_flooding_acft[-1] if cumulative_flooding_acft else 0.0)
    st.session_state[f"{scenario_name}_post5h_total_flood"] = (cum_cuft_post5h / 43560.0) if cum_cuft_post5h > 0 else 0.0
    st.session_state[f"{scenario_name}_node_flood_post5h_cuft"] = node_cuft_post5h
    return cumulative_flooding_acft, timestamps, rpt_path


def _build_baseline_map_html(df_swmm_local: pd.DataFrame, unit_ui: str, ws_shp_path: str) -> str:

    # Build geodataframe with Total_R joined and unit label returned
    ws_gdf_local = load_ws(ws_shp_path)
    gdf, unit_r = prep_total_runoff_gdf(df_swmm_local, unit_ui, ws_gdf_local)

    # Color scale from Total_R (robust to NaNs and constant arrays)
    vals = gdf["Total_R"].to_numpy()
    vmin_raw = np.nanmin(vals)
    vmax_raw = np.nanmax(vals)
    vmin = float(vmin_raw) if np.isfinite(vmin_raw) else 0.0
    vmax = float(vmax_raw) if np.isfinite(vmax_raw) else 1.0
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1e-6

    gdf["_fill"] = make_color(gdf["Total_R"], vmin, vmax)
    gdf["_label"] = gdf["NAME"]

    # Map view centered on overall centroid
    centroid = gdf.geometry.unary_union.centroid
    view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=14.25)

    # Polygons
    poly_layer = pdk.Layer(
        "GeoJsonLayer",
        data=gdf.__geo_interface__,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="properties._fill",
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=1,
    )

    # Labels at representative points
    reps = gdf.geometry.representative_point()
    labels_df = pd.DataFrame({"lon": reps.x, "lat": reps.y, "text": gdf["_label"]})
    text_layer = pdk.Layer(
        "TextLayer",
        data=labels_df,
        get_position='[lon, lat]',
        get_text="text",
        get_size=12,
        get_color=[0, 0, 0],
        get_alignment_baseline="'center'",
    )

    tooltip = {
        "html": "<b>{NAME}</b><br/>Total runoff: {Total_R} " + unit_r,
        "style": {"backgroundColor": "white", "color": "black",
                "fontFamily": "Inter, Arial, Helvetica, sans-serif", "fontSize": "12px"},
    }

    deck_obj = pdk.Deck(
        layers=[poly_layer, text_layer],   # NO PIPES here
        initial_view_state=view_state,
        map_provider="carto",
        map_style="light",
        tooltip=tooltip,
    )
    return deck_obj.to_html(as_string=True)


def generate_lid_usage_lines(lid_config: Dict[str, Dict[str, int]], excel_df: pd.DataFrame) -> List[str]:

    lines: List[str] = []
    tpl = (
        "{sub:<15}{proc:<16}{num:>7}{area:>8}{width:>7}{initsat:>8}"
        "{fromimp:>8}{toperv:>8}{rptfile:>24}{drainto:>16}{fromperv:>9}"
    )

    # Quick lookup to avoid repeated filtering
    df = excel_df.set_index("NAME", drop=False)

    for sub, cfg in lid_config.items():
        if sub not in df.index:
            continue

        row = df.loc[sub]
        imperv = float(row.get("Impervious_ft2", 0.0) or 0.0)
        perv   = float(row.get("Pervious_ft2",   0.0) or 0.0)

        # Rain barrels
        rb = int(cfg.get("rain_barrels", 0) or 0)
        if rb > 0:
            # % of impervious receiving RB drainage; 300 ft² per barrel
            pct_imp = (rb * 300.0) / (imperv if imperv > 0 else 1e-9) * 100.0
            lines.append(
                tpl.format(
                    sub=sub,
                    proc="rain_barrel",
                    num=rb,
                    area=f"{2.58:.2f}",   # nominal plan area per barrel (ft²); keep from your prior spec
                    width=0,
                    initsat=0,
                    fromimp=f"{pct_imp:.2f}",
                    toperv=1,
                    rptfile="*",
                    drainto="*",
                    fromperv=0,
                )
            )

        # Rain gardens
        rg = int(cfg.get("rain_gardens", 0) or 0)
        if rg > 0:
            # % of pervious contributing to RG; 500 ft² per garden as contribution basis
            pct_perv = (rg * 500.0) / (perv if perv > 0 else 1e-9) * 100.0
            lines.append(
                tpl.format(
                    sub=sub,
                    proc="rain_garden",
                    num=rg,
                    area=f"{100:.0f}",    # plan area per garden (ft²)
                    width=0,
                    initsat=0,
                    fromimp=0,
                    toperv=1,
                    rptfile="*",
                    drainto="*",
                    fromperv=f"{pct_perv:.2f}",
                )
            )

    return lines


def ensure_temp_dir():
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()

def delete_user_files(user_id):
    patterns = [f"user_{user_id}_*.inp", f"user_{user_id}_*.rpt", f"user_{user_id}_*.out"]
    for pattern in patterns:
        for file in glob.glob(pattern):
            try: os.remove(file)
            except Exception: pass

def login_ui():
    st.title("🌊 CoastWise Login")
    tab1, tab2, tab3 = st.tabs(["🔐 Login", "🆕 Sign Up", "🔁 Reset Password"])
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            user_id = authenticate_user(email, password)
            if user_id:
                st.session_state["user_id"] = user_id
                st.session_state["email"] = email
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid email or password.")
    with tab2:
        email = st.text_input("New Email", key="signup_email")
        password = st.text_input("New Password", type="password", key="signup_pass")
        if st.button("Create Account"):
            if create_user(email, password):
                st.success("Account created! You can now log in.")
            else:
                st.error("Email already in use.")
    with tab3:
        reset_email = st.text_input("Your Email", key="reset_email")
        new_password = st.text_input("New Password", type="password", key="reset_pass")
        if st.button("Reset Password"):
            if reset_password(reset_email, new_password):
                st.success("Password updated. You can now log in.")
            else:
                st.error("Failed to reset password. Check your email.")

def app_ui():
    """Main Streamlit UI — all scenario controls, runs, maps, export, and logout."""

    # ---------- Setup / shared state ----------
    st.success(f"Logged in as: {st.session_state.get('email', 'user')}")
    ensure_temp_dir()

    user_id = st.session_state.get("user_id", "guest")
    prefix = f"user_{user_id}_"

    # Defaults / resources
    simulation_date = "05/31/2025 12:00"
    template_inp    = "swmm_project.inp"
    WS_SHP_PATH     = st.session_state.get("WS_SHP_PATH", "Subcatchments.shp")
    NODE_SHP_PATH   = st.session_state.get("NODE_SHP_PATH", "Nodes.shp")
    PIPE_SHP_PATH   = st.session_state.get("PIPE_SHP_PATH", "Conduits.shp")

    st.title("CoastWise: Watershed Design Toolkit (SWMM)")

    # ---------- Persisted config ----------
    if "cfg" not in st.session_state:
        st.session_state.cfg = {
            "unit": "U.S. Customary",
            "moon_phase": list(moon_tide_ranges.keys())[0],
            "duration_minutes": int(pf_df["Duration_Minutes"].iloc[0]),
            "return_period": "1",
            "align_mode": "peak",   # "peak" or "low"
            "settings_ready": False,
        }
    cfg = st.session_state.cfg

    # ---------- Settings form ----------
    with st.form("scenario_settings"):
        unit = st.selectbox(
            "Preferred Units",
            ["U.S. Customary", "Metric (SI)"],
            index=0 if cfg["unit"] == "U.S. Customary" else 1
        )
        moon_phase = st.selectbox(
            "Synthetic Tide (fallback)",
            list(moon_tide_ranges.keys()),
            index=(list(moon_tide_ranges.keys()).index(cfg["moon_phase"])
                   if cfg["moon_phase"] in moon_tide_ranges else 0)
        )
        duration_minutes = st.selectbox(
            "Storm Duration",
            options=pf_df["Duration_Minutes"],
            index=int(np.where(pf_df["Duration_Minutes"].values == cfg["duration_minutes"])[0][0])
                  if cfg["duration_minutes"] in pf_df["Duration_Minutes"].values else 0,
            format_func=lambda x: f"{int(x)//60} hr"
        )
        # Return period depends on duration & unit
        rp_opts = return_period_labels(int(duration_minutes), unit)
        rp_keys = list(rp_opts.keys())
        rp_idx  = rp_keys.index(cfg["return_period"]) if cfg["return_period"] in rp_keys else 0
        rp_label = st.selectbox("Return Year", list(rp_opts.values()), index=rp_idx)
        return_period = [k for k, v in rp_opts.items() if v == rp_label][0]

        align_choice = st.radio(
            "Tide Alignment",
            ["Peak aligned with High Tide", "Peak aligned with Low Tide"],
            index=0 if cfg["align_mode"] == "peak" else 1
        )
        submitted = st.form_submit_button("Apply Settings")

    if submitted:
        st.session_state.cfg = {
            "unit": unit,
            "moon_phase": moon_phase,
            "duration_minutes": int(duration_minutes),
            "return_period": str(return_period),
            "align_mode": ("peak" if "High" in align_choice else "low"),
            "settings_ready": True,
        }
        cfg = st.session_state.cfg
        st.success("Settings applied.")

    if not cfg.get("settings_ready", False):
        st.info("Apply settings to generate rainfall and tide series.")
        # Still show Logout even if settings not applied
        if st.button("🚪 Logout"):
            try: shutil.rmtree(st.session_state.temp_dir)
            except Exception: pass
            st.session_state.clear()
            st.success("Logged out and cleaned up all files.")
            st.experimental_rerun()
        return

    # ---------- Build rainfall + tide ----------
    align_mode       = cfg["align_mode"]
    unit             = cfg["unit"]
    moon_phase       = cfg["moon_phase"]
    duration_minutes = int(cfg["duration_minutes"])
    return_period    = cfg["return_period"]

    total_inches = float(
        pf_df.loc[pf_df["Duration_Minutes"] == duration_minutes, return_period].values[0]
    )

    try:
        minutes_15, tide_curve_ui, rain_curve_in, used_live, _ = get_aligned_rainfall(
            total_inches=total_inches,
            duration_minutes=duration_minutes,
            moon_phase=moon_phase,
            unit=unit,
            align=align_mode,
            method="Normal",
            start_ts=None,
            prominence=None,
            navd88_to_sea_level_offset_ft=MSL_OFFSET_NAVD88_FT
        )
        tide_source = "live" if used_live else "synthetic"
    except Exception:
        minutes_15, tide_curve_ui, _used_live = get_tide_real_or_synthetic(
            moon_phase, unit, start_ts=None, navd88_to_sea_level_offset_ft=MSL_OFFSET_NAVD88_FT
        )
        from rainfall_and_tide_generator import align_rainfall_to_tide
        _m, rain_curve_in, _ = align_rainfall_to_tide(
            total_inches=total_inches,
            duration_minutes=duration_minutes,
            tide_curve_15min=tide_curve_ui,
            align=align_mode,
            method="Normal",
            target_index=None,
            prominence=None
        )
        tide_source = "synthetic"

    # Display conversions
    if unit == "U.S. Customary":
        display_rain_curve = rain_curve_in         # inches
        display_tide_curve = tide_curve_ui         # ft
        rain_disp_unit = "inches"; tide_disp_unit = "ft"
    else:
        display_rain_curve = rain_curve_in * 2.54  # cm
        display_tide_curve = tide_curve_ui         # meters (UI already meters for SI)
        rain_disp_unit = "centimeters"; tide_disp_unit = "meters"

    # Persist series + labels
    st.session_state.update({
        "rain_minutes": minutes_15,
        "tide_minutes": minutes_15,
        "display_rain_curve_current": display_rain_curve,
        "display_rain_curve_future": display_rain_curve * 1.2,
        "display_tide_curve": display_tide_curve,
        "rain_sim_curve_current_in": rain_curve_in,
        "rain_sim_curve_future_in": rain_curve_in * 1.2,
        "rain_disp_unit": rain_disp_unit,
        "tide_disp_unit": tide_disp_unit,
        "unit_ui": unit,
        "tide_source": tide_source,
        "moon_phase": moon_phase,
        "align_mode": align_mode,
        f"{prefix}simulation_date": simulation_date,
        f"{prefix}template_inp": template_inp,
    })

    # ---------- Build SWMM-ready TIMESERIES lines ----------
    tide_lines = format_timeseries(
        "tide",
        minutes_15,
        tide_to_feet_for_swmm(st.session_state["display_tide_curve"], unit),
        simulation_date
    )
    def _rain_lines_pair(sim_minutes, rain_curve_in, sim_start_str):
        cur = format_timeseries("rain_gage_timeseries", sim_minutes, rain_curve_in, sim_start_str)
        fut = format_timeseries("rain_gage_timeseries", sim_minutes, (np.array(rain_curve_in) * 1.2), sim_start_str)
        return cur, fut
    rain_lines_cur, rain_lines_fut = _rain_lines_pair(minutes_15, st.session_state["rain_sim_curve_current_in"], simulation_date)

    st.session_state.update({
        "tide_lines": tide_lines,
        "rain_lines_cur": rain_lines_cur,
        "rain_lines_fut": rain_lines_fut,
        f"{prefix}tide_lines": tide_lines,
        f"{prefix}rain_lines_cur": rain_lines_cur,
        f"{prefix}rain_lines_fut": rain_lines_fut,
    })

    # ---------- Charts ----------
    time_hours = np.array(minutes_15, dtype=float) / 60.0
    # Rainfall
    st.subheader("Rainfall Distribution")
    rain_df = pd.DataFrame({
        "Hour": time_hours,
        "Current": st.session_state["display_rain_curve_current"],
        "Future (+20%)": st.session_state["display_rain_curve_future"],
    }).set_index("Hour")
    st.line_chart(rain_df, height=220, use_container_width=True)
    st.caption(f"Rainfall units: {rain_disp_unit}. X-axis = hours since start (0–48, 15-min steps).")

    # Tide
    st.subheader("Tide Profile")
    tide_df = pd.DataFrame({
        "Hour": time_hours,
        "Tide": st.session_state["display_tide_curve"],
    }).set_index("Hour")
    st.line_chart(tide_df, height=220, use_container_width=True)
    st.caption(("Source: Real-time tide (last 48 h)" if tide_source == "live"
            else f"Source: Synthetic tide ({moon_phase})") + ". X-axis = hours since start.")

    # ---------- Run Baseline Scenario ----------
    st.markdown("---")
    if st.button("Run Baseline Scenario"):
        try:
            lid_lines = [";"]  # none
            # current, with/without tide gate
            fill_nogate_cur, ts, rpt1 = run_swmm_scenario(
                f"{prefix}baseline_nogate_current", rain_lines_cur, tide_lines, lid_lines, "NO",
                template_path=template_inp
            )
            fill_gate_cur,  _,  rpt2 = run_swmm_scenario(
                f"{prefix}baseline_gate_current",   rain_lines_cur, tide_lines, lid_lines, "YES",
                template_path=template_inp
            )
            # +20%, with/without tide gate
            fill_nogate_fut,_,  rpt3 = run_swmm_scenario(
                f"{prefix}baseline_nogate_future",  rain_lines_fut, tide_lines, lid_lines, "NO",
                template_path=template_inp
            )
            fill_gate_fut,  _,  rpt4 = run_swmm_scenario(
                f"{prefix}baseline_gate_future",    rain_lines_fut, tide_lines, lid_lines, "YES",
                template_path=template_inp
            )
            st.session_state.update({
                f"{prefix}baseline_timestamps": ts,
                f"{prefix}baseline_fill_current": fill_nogate_cur,
                f"{prefix}baseline_gate_fill_current": fill_gate_cur,
                f"{prefix}baseline_fill_future": fill_nogate_fut,
                f"{prefix}baseline_gate_fill_future": fill_gate_fut,
                f"{prefix}df_base_nogate_current": extract_total_runoff(rpt1),
                f"{prefix}df_base_gate_current":   extract_total_runoff(rpt2),
                f"{prefix}df_base_nogate_future":  extract_total_runoff(rpt3),
                f"{prefix}df_base_gate_future":    extract_total_runoff(rpt4),
            })
            df_swmm_now = st.session_state[f"{prefix}df_base_nogate_current"]
            if len(df_swmm_now) == 0:
                st.error("No rows parsed from 'Subcatchment Runoff Summary' in the .rpt")
            else:
                st.success("Baseline scenarios complete.")
                html_str = _build_baseline_map_html(
                    df_swmm_local=df_swmm_now,
                    unit_ui=st.session_state["unit_ui"],
                    ws_shp_path=WS_SHP_PATH
                )
                st.session_state[f"{prefix}baseline_map_html"] = html_str
                st.session_state[f"{prefix}show_baseline_map"] = True
        except Exception as e:
            st.error(f"Baseline simulation failed: {e}")

    REQUIRED_RASTER_COLS = {
        "NAME", "Max_RG_DEM_Considered", "MaxNumber_RB",
        "Impervious_ft2", "Pervious_ft2"
    }

    def _load_raster_df() -> pd.DataFrame:
        """
        Replace this stub with your real loader.
        Options:
        - return load_raster_cells()           # if you already have this helper
        - return pd.read_csv("raster.csv")     # if it lives in CSV
        - return pd.read_excel("raster.xlsx")  # if it lives in Excel
        """
        # Example using an existing helper:
        return load_raster_cells()

    # cache in session so reruns are cheap
    if "raster_df" not in st.session_state:
        try:
            st.session_state["raster_df"] = _load_raster_df()
        except Exception as e:
            st.error(f"Failed to load raster/subcatchment table: {e}")
            st.stop()

    raster_df: pd.DataFrame = st.session_state["raster_df"]

    # sanity check columns
    missing = REQUIRED_RASTER_COLS - set(raster_df.columns)
    if missing:
        st.error(f"raster_df is missing required columns: {sorted(missing)}")
        st.stop()

    # ---------- Baseline Map + LID selection UI ----------
    if st.session_state.get(f"{prefix}show_baseline_map") and st.session_state.get(f"{prefix}baseline_map_html"):
        st.subheader("Watershed Baseline Runoff Map")
        components.v1.html(st.session_state[f"{prefix}baseline_map_html"], height=620, scrolling=False)
        st.subheader("Add LID Features")
        if f"{prefix}user_lid_config" not in st.session_state:
            st.session_state[f"{prefix}user_lid_config"] = {}
        available_subs = raster_df["NAME"].tolist()
        selected_subs = st.multiselect("Select subcatchments", options=available_subs)

        if selected_subs:
            for _, row in raster_df[raster_df["NAME"].isin(selected_subs)].iterrows():
                sub = row["NAME"]
                rg_max = int(row["Max_RG_DEM_Considered"])
                rb_max = int(row["MaxNumber_RB"])
                c1, c2, c3 = st.columns([2,2,2])
                with c1: st.write(f"**{sub}**")
                with c2: rg_val = st.number_input(
                    f"Rain Gardens ({rg_max} max) — {sub}", 0, rg_max, 0, step=5, key=f"rg_{sub}"
                )
                with c3: rb_val = st.number_input(
                    f"Rain Barrels ({rb_max} max) — {sub}", 0, rb_max, 0, step=5, key=f"rb_{sub}"
                )
                st.session_state[f"{prefix}user_lid_config"][sub] = {
                    "rain_gardens": rg_val, "rain_barrels": rb_val
                }

    # ---------- Run Custom LID ----------
    if st.button("Run Custom LID Scenario"):
        lid_cfg = st.session_state.get(f"{prefix}user_lid_config", {})
        if not lid_cfg or all((v.get("rain_gardens",0)==0 and v.get("rain_barrels",0)==0) for v in lid_cfg.values()):
            st.warning("No LIDs selected.")
        else:
            try:
                lid_lines = generate_lid_usage_lines(lid_cfg, raster_df)
                fill_lid_cur, ts, rpt1 = run_swmm_scenario(
                    f"{prefix}lid_nogate_current", st.session_state["rain_lines_cur"],
                    st.session_state["tide_lines"], lid_lines, "NO",
                    template_path=template_inp
                )
                fill_lid_gate_cur,_, rpt2 = run_swmm_scenario(
                    f"{prefix}lid_gate_current",   st.session_state["rain_lines_cur"],
                    st.session_state["tide_lines"], lid_lines, "YES",
                    template_path=template_inp
                )
                fill_lid_fut,_,      rpt3 = run_swmm_scenario(
                    f"{prefix}lid_nogate_future",  st.session_state["rain_lines_fut"],
                    st.session_state["tide_lines"], lid_lines, "NO",
                    template_path=template_inp
                )
                fill_lid_gate_fut,_, rpt4 = run_swmm_scenario(
                    f"{prefix}lid_gate_future",    st.session_state["rain_lines_fut"],
                    st.session_state["tide_lines"], lid_lines, "YES",
                    template_path=template_inp
                )
                st.session_state.update({
                    f"{prefix}lid_timestamps": ts,
                    f"{prefix}lid_fill_current": fill_lid_cur,
                    f"{prefix}lid_gate_fill_current": fill_lid_gate_cur,
                    f"{prefix}lid_fill_future": fill_lid_fut,
                    f"{prefix}lid_gate_fill_future": fill_lid_gate_fut,
                    f"{prefix}df_lid_nogate_current": extract_total_runoff(rpt1),
                    f"{prefix}df_lid_gate_current":   extract_total_runoff(rpt2),
                    f"{prefix}df_lid_nogate_future":  extract_total_runoff(rpt3),
                    f"{prefix}df_lid_gate_future":    extract_total_runoff(rpt4),
                })
                st.success("Custom LID scenarios complete.")
            except Exception as e:
                st.error(f"LID simulation failed: {e}")

    # ---------- Run Max LID ----------
    if st.button("Run Max LID Scenario"):
        lid_cfg = {row["NAME"]: {"rain_gardens": int(row["Max_RG_DEM_Considered"]),
                                 "rain_barrels": int(row["MaxNumber_RB"])}
                   for _, row in raster_df.iterrows()}
        try:
            lid_lines = generate_lid_usage_lines(lid_cfg, raster_df)
            fill_max_cur, ts, rpt1 = run_swmm_scenario(
                f"{prefix}lid_max_nogate_current", st.session_state["rain_lines_cur"],
                st.session_state["tide_lines"], lid_lines, "NO", template_path=template_inp
            )
            fill_max_gate_cur,_, rpt2 = run_swmm_scenario(
                f"{prefix}lid_max_gate_current",   st.session_state["rain_lines_cur"],
                st.session_state["tide_lines"], lid_lines, "YES", template_path=template_inp
            )
            fill_max_fut,_,      rpt3 = run_swmm_scenario(
                f"{prefix}lid_max_nogate_future",  st.session_state["rain_lines_fut"],
                st.session_state["tide_lines"], lid_lines, "NO", template_path=template_inp
            )
            fill_max_gate_fut,_, rpt4 = run_swmm_scenario(
                f"{prefix}lid_max_gate_future",    st.session_state["rain_lines_fut"],
                st.session_state["tide_lines"], lid_lines, "YES", template_path=template_inp
            )
            st.session_state.update({
                f"{prefix}lid_max_timestamps": ts,
                f"{prefix}lid_max_fill_current": fill_max_cur,
                f"{prefix}lid_max_gate_fill_current": fill_max_gate_cur,
                f"{prefix}lid_max_fill_future": fill_max_fut,
                f"{prefix}lid_max_gate_fill_future": fill_max_gate_fut,
                f"{prefix}df_lid_max_nogate_current": extract_total_runoff(rpt1),
                f"{prefix}df_lid_max_gate_current":   extract_total_runoff(rpt2),
                f"{prefix}df_lid_max_nogate_future":  extract_total_runoff(rpt3),
                f"{prefix}df_lid_max_gate_future":    extract_total_runoff(rpt4),
            })
            st.success("Max LID scenarios complete.")
        except Exception as e:
            st.error(f"Max LID simulation failed: {e}")

    # ---------- Comparison Maps ----------
    st.subheader("Scenario Comparison Maps (Total Runoff + Flooded Nodes)")
    left_df_key   = f"{prefix}df_lid_nogate_future"     # LID (+20%)
    right_df_key  = f"{prefix}df_base_nogate_future"    # Baseline (+20%)
    if left_df_key in st.session_state and right_df_key in st.session_state:
        render_side_by_side_total_runoff_maps(
            left_df_in_inches  = st.session_state[left_df_key],
            left_title         = "LID — +20% Rainfall",
            left_nodes_post5h_dict  = st.session_state.get(f"{prefix}lid_nogate_future_node_flood_post5h_cuft", {}),
            right_df_in_inches = st.session_state[right_df_key],
            right_title        = "Baseline — +20% Rainfall",
            right_nodes_post5h_dict = st.session_state.get(f"{prefix}baseline_nogate_future_node_flood_post5h_cuft", {}),
            unit_ui=st.session_state["unit_ui"],
            ws_shp_path=WS_SHP_PATH,
            pipe_shp_path=PIPE_SHP_PATH,
            node_shp_path=NODE_SHP_PATH,
            node_name_field_hint="NAME",
        )
    else:
        st.info("Run both scenarios: LID (+20%) and Baseline (+20%) to view the comparison maps.")

    # ---------- Summary table + Excel export ----------
    temp_dir = st.session_state.temp_dir
    rpt_scenarios = {
        "Baseline (No Tide Gate) – Current": os.path.join(temp_dir, f"{prefix}baseline_nogate_current.rpt"),
        "Baseline + Tide Gate – Current":    os.path.join(temp_dir, f"{prefix}baseline_gate_current.rpt"),
        "Baseline (No Tide Gate) – +20%":    os.path.join(temp_dir, f"{prefix}baseline_nogate_future.rpt"),
        "Baseline + Tide Gate – +20%":       os.path.join(temp_dir, f"{prefix}baseline_gate_future.rpt"),
        "LID (No Tide Gate) – Current":      os.path.join(temp_dir, f"{prefix}lid_nogate_current.rpt"),
        "LID + Tide Gate – Current":         os.path.join(temp_dir, f"{prefix}lid_gate_current.rpt"),
        "LID (No Tide Gate) – +20%":         os.path.join(temp_dir, f"{prefix}lid_nogate_future.rpt"),
        "LID + Tide Gate – +20%":            os.path.join(temp_dir, f"{prefix}lid_gate_future.rpt"),
        "Max LID (No Tide Gate) – Current":  os.path.join(temp_dir, f"{prefix}lid_max_nogate_current.rpt"),
        "Max LID + Tide Gate – Current":     os.path.join(temp_dir, f"{prefix}lid_max_gate_current.rpt"),
        "Max LID (No Tide Gate) – +20%":     os.path.join(temp_dir, f"{prefix}lid_max_nogate_future.rpt"),
        "Max LID + Tide Gate – +20%":        os.path.join(temp_dir, f"{prefix}lid_max_gate_future.rpt"),
    }

    def _extract_summary_table() -> pd.DataFrame:
        friendly_map = {
            "Baseline (No Tide Gate) – Current":  "baseline_nogate_current",
            "Baseline + Tide Gate – Current":     "baseline_gate_current",
            "Baseline (No Tide Gate) – +20%":     "baseline_nogate_future",
            "Baseline + Tide Gate – +20%":        "baseline_gate_future",
            "LID (No Tide Gate) – Current":       "lid_nogate_current",
            "LID + Tide Gate – Current":          "lid_gate_current",
            "LID (No Tide Gate) – +20%":          "lid_nogate_future",
            "LID + Tide Gate – +20%":             "lid_gate_future",
            "Max LID (No Tide Gate) – Current":   "lid_max_nogate_current",
            "Max LID + Tide Gate – Current":      "lid_max_gate_current",
            "Max LID (No Tide Gate) – +20%":      "lid_max_nogate_future",
            "Max LID + Tide Gate – +20%":         "lid_max_gate_future",
        }
        rows = []
        for disp, path in rpt_scenarios.items():
            if not os.path.exists(path): continue
            key = friendly_map[disp]
            flood_acft = st.session_state.get(f"{prefix}{key}_post5h_total_flood",
                                              st.session_state.get(f"{prefix}{key}_total_flood", 0.0))
            rows.append({"Scenario": disp, "Flooding (ac-ft)": flood_acft})
        return pd.DataFrame(rows).set_index("Scenario")

    if st.button("Show Water Balance Summary Table"):
        df_balance = _extract_summary_table()
        if df_balance.empty:
            st.info("Run scenarios first.")
        else:
            convert_to_m3 = (st.session_state["unit_ui"] == "Metric (SI)")
            ACF_TO_FT3 = 43560.0
            FT3_TO_M3 = 0.0283168

            def to_disp_ft3(x):
                if x is None: return 0.0
                return float(x) * ACF_TO_FT3

            def maybe_m3(v_ft3):
                return v_ft3 * FT3_TO_M3 if convert_to_m3 else v_ft3

            df_conv = pd.DataFrame(index=df_balance.index)
            df_conv["Flooded Volume"] = (
                df_balance["Flooding (ac-ft)"].apply(to_disp_ft3).apply(maybe_m3)
            ).round(0).astype(int)

            st.subheader(f"Summary ({'m³' if convert_to_m3 else 'ft³'})")
            st.dataframe(df_conv)

            # Build Excel payload
            sim_start = datetime.strptime(simulation_date, "%m/%d/%Y %H:%M")
            rain_minutes = st.session_state.get("rain_minutes", [])
            tide_minutes = st.session_state.get("tide_minutes", [])
            rain_disp_unit = st.session_state.get("rain_disp_unit", "inches")
            tide_disp_unit = st.session_state.get("tide_disp_unit", "ft")
            rain_ts = st.session_state.get("display_rain_curve_current", [])
            rain_ts_f = st.session_state.get("display_rain_curve_future", [])
            tide_ts = st.session_state.get("display_tide_curve", [])

            if len(rain_ts) > 0:
                r_t = [(sim_start + timedelta(minutes=int(m))).strftime("%m/%d/%Y %H:%M")
                       for m in rain_minutes[:len(rain_ts)]]
                df_rain = pd.DataFrame({
                    "Timestamp": r_t,
                    f"Rainfall – Current ({rain_disp_unit})": rain_ts[:len(r_t)],
                    f"Rainfall – +20% ({rain_disp_unit})":    rain_ts_f[:len(r_t)]
                })
            else:
                df_rain = pd.DataFrame(columns=[
                    "Timestamp",
                    f"Rainfall – Current ({rain_disp_unit})",
                    f"Rainfall – +20% ({rain_disp_unit})"
                ])

            if len(tide_ts) > 0:
                t_t = [(sim_start + timedelta(minutes=int(m))).strftime("%m/%d/%Y %H:%M")
                       for m in tide_minutes[:len(tide_ts)]]
                df_tide = pd.DataFrame({"Timestamp": t_t, f"Tide ({tide_disp_unit})": tide_ts[:len(t_t)]})
            else:
                df_tide = pd.DataFrame(columns=["Timestamp", f"Tide ({tide_disp_unit})"])

            lid_cfg = st.session_state.get(f"{prefix}user_lid_config", {})
            if lid_cfg:
                rows = [{"Subcatchment": sub,
                         "Selected Rain Gardens": cfg.get("rain_gardens", 0),
                         "Selected Rain Barrels":  cfg.get("rain_barrels", 0)}
                        for sub, cfg in lid_cfg.items()]
                df_user_lid = pd.DataFrame(rows)
            else:
                df_user_lid = pd.DataFrame(columns=["Subcatchment","Selected Rain Gardens","Selected Rain Barrels"])

            excel_output = io.BytesIO()
            with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
                scenario_summary = pd.DataFrame([{
                    "Storm Duration (hr)": duration_minutes // 60,
                    "Return Period (yr)": return_period,
                    "Tide": ("Real-time" if st.session_state["tide_source"]=="live" else st.session_state["moon_phase"]),
                    "Tide Alignment": "High Tide Peak" if st.session_state["align_mode"] == "peak" else "Low Tide Dip",
                    "Units": st.session_state["unit_ui"]
                }])
                scenario_summary.to_excel(writer, sheet_name="Scenario Settings", index=False)
                df_rain.to_excel(writer, sheet_name="Rainfall Event", index=False)
                df_tide.to_excel(writer, sheet_name="Tide Event", index=False)
                df_user_lid.to_excel(writer, sheet_name="User LID Selections", index=False)
                df_balance.reset_index().rename(columns={"index":"Scenario"}).to_excel(
                    writer, sheet_name="Scenario Summary", index=False
                )

            st.download_button(
                label="Generate & Download Scenario Results (Excel)",
                data=excel_output.getvalue(),
                file_name="CoastWise_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    # ---------- Logout ----------
    st.markdown("---")
    if st.button("🚪 Logout"):
        try: shutil.rmtree(st.session_state.temp_dir)
        except Exception: pass
        st.session_state.clear()
        st.success("Logged out and cleaned up all files.")
        st.experimental_rerun()

if "user_id" not in st.session_state:
    login_ui()
else:
    app_ui()
