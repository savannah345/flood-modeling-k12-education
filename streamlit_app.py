import os, io, re, glob, sys, shutil, tempfile, subprocess
from datetime import datetime, timedelta
from typing import Dict, Tuple, List


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


from auth_supabase import create_user, authenticate_user, reset_password
from rainfall_and_tide_generator import (
    pf_df,
    moon_tide_ranges,
    get_tide_real_or_synthetic,
    get_aligned_rainfall,
)

st.set_page_config(page_title="CoastWise", layout="centered")


def _read_text_keep(path: str) -> str:
    """Read file contents and keep artifacts in temp for later use."""
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def ensure_temp_dir():
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()

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

MSL_OFFSET_NAVD88_FT = 0
WS_SHP_PATH  = st.session_state.get("WS_SHP_PATH", "map_files/Subcatchments.shp")
NODE_SHP_PATH= st.session_state.get("NODE_SHP_PATH", "map_files/Nodes.shp")
PIPE_SHP_PATH= st.session_state.get("PIPE_SHP_PATH", "map_files/Conduits.shp")


@st.cache_resource(show_spinner=False)
def load_ws(path="map_files/Subcatchments.shp") -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = (gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326))
    gdf = gdf[gdf.geometry.notnull() & gdf.is_valid].copy()   
    return gdf

@st.cache_resource(show_spinner=False)
def load_nodes(path="map_files/Nodes.shp") -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = (gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326))
    gdf = gdf[gdf.geometry.notnull() & gdf.is_valid].copy()   
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


def _ensure_scenario_loaded(name: str, prefix: str):
    """Guarantee a scenario exists in the in-memory store by rehydrating from DF or RPT."""
    store = _sc_store(prefix)
    if name in store:
        return

    # Map scenario -> DF session key
    df_keys = {
        "baseline_nogate_current": f"{prefix}df_base_nogate_current",
        "baseline_gate_current":   f"{prefix}df_base_gate_current",
        "baseline_nogate_future":  f"{prefix}df_base_nogate_future",
        "baseline_gate_future":    f"{prefix}df_base_gate_future",
        "lid_nogate_current":      f"{prefix}df_lid_nogate_current",
        "lid_gate_current":        f"{prefix}df_lid_gate_current",
        "lid_nogate_future":       f"{prefix}df_lid_nogate_future",
        "lid_gate_future":         f"{prefix}df_lid_gate_future",
        "lid_max_nogate_current":  f"{prefix}df_lid_max_nogate_current",
        "lid_max_gate_current":    f"{prefix}df_lid_max_gate_current",
        "lid_max_nogate_future":   f"{prefix}df_lid_max_nogate_future",
        "lid_max_gate_future":     f"{prefix}df_lid_max_gate_future",
    }

    # 1) Try DF in session_state
    df = st.session_state.get(df_keys.get(name, ""), None)
    nodes = st.session_state.get(f"{prefix}{name}_node_flood_event_cuft", {}) or {}

    if isinstance(df, pd.DataFrame) and not df.empty:
        remember_scenario(name, df, nodes, prefix)
        return

    # 2) Fall back to RPT text in memory
    rpts = st.session_state.get("rpts", {})
    rpt_text = rpts.get(f"{prefix}{name}", "")
    if rpt_text:
        df2 = extract_total_runoff_from_text(rpt_text)
        if isinstance(df2, pd.DataFrame) and not df2.empty:
            remember_scenario(name, df2, nodes, prefix)


def prep_total_runoff_gdf(rpt_df: pd.DataFrame, unit_ui: str, ws_gdf: gpd.GeoDataFrame):

    g = ws_gdf.copy()
    if rpt_df is None or rpt_df.empty:
        g["Total_R"] = np.nan
        return g, ("in" if unit_ui == "U.S. Customary" else "cm")

    if "Total_in" in rpt_df.columns:
        total_in = rpt_df["Total_in"].astype(float)
    elif {"Impervious","Pervious"} <= set(rpt_df.columns):
        total_in = (rpt_df["Impervious"].astype(float).fillna(0.0) +
                    rpt_df["Pervious"].astype(float).fillna(0.0))
    else:
        g["Total_R"] = np.nan
        return g, ("in" if unit_ui == "U.S. Customary" else "cm")

    def _norm(n):
        n = "" if n is None else str(n)
        n = n.strip().lower()
        m = re.fullmatch(r"sub[_\s-]*0*(\d+)", n) 
        return f"sub_{m.group(1)}" if m else re.sub(r"\s+", "", n)

    left = rpt_df.assign(NAME_JOIN=rpt_df["Subcatchment"].astype(str).map(_norm))[
        ["NAME_JOIN"]
    ].copy()
    left["Total_in"] = total_in.values

    g["NAME_JOIN"] = g["NAME"].astype(str).map(_norm)
    merged = g.merge(left, on="NAME_JOIN", how="left").drop(columns=["NAME_JOIN"])

    if unit_ui == "U.S. Customary":
        merged["Total_R"] = merged["Total_in"]  
        unit = "in"
    else:
        merged["Total_R"] = merged["Total_in"] * 2.54  
        unit = "cm"

    return merged, unit

def parse_node_flooding_summary_from_text(txt: str) -> pd.DataFrame:
    lines = txt.splitlines(True)
    if not lines: return pd.DataFrame(columns=["Node","Gallons"])
    def _is_dashes(s: str) -> bool:
        s = s.strip(); return bool(s) and set(s) <= {"-"}
    hdr = next((i for i,l in enumerate(lines) if l.strip().lower().startswith("node flooding summary")), None)
    if hdr is None: return pd.DataFrame(columns=["Node","Gallons"])
    i = hdr + 1; dash = 0
    while i < len(lines) and dash < 2:
        if _is_dashes(lines[i]): dash += 1
        i += 1
    if dash < 2 or i >= len(lines): return pd.DataFrame(columns=["Node","Gallons"])
    out = []
    num_re = re.compile(r"[-+]?\d+(?:\.\d+)?")
    while i < len(lines):
        raw = lines[i].rstrip("\n"); i += 1
        if not raw.strip() or _is_dashes(raw) or raw.lstrip().startswith("*"): break
        parts = raw.split()
        if not parts: continue
        node = parts[0]
        nums = [float(m.group(0)) for m in num_re.finditer(raw)]
        if len(nums) < 2: continue
        mgal = nums[-2]
        gallons = mgal * 1_000_000.0
        out.append({"Node": node, "Gallons": gallons})
    return pd.DataFrame(out, columns=["Node","Gallons"])

_GAL_TO_FT3 = 7.48051948     
_GAL_TO_M3  = 0.003785411784 

def summarize_node_flooding_in_window(
    df_nodes: pd.DataFrame,
    to_metric: bool
) -> tuple[float, dict[str, float]]:
    """
    Returns:
      total_volume (m³ if to_metric else ft³),
      per_node_ft3 (dict) — always ft³ for mapping/flagging
    """
    if df_nodes is None or df_nodes.empty:
        return 0.0, {}

    sub = df_nodes.copy()
    if "Gallons" not in sub.columns and "TotalFlood_Mgal" in sub.columns:
        sub["Gallons"] = pd.to_numeric(sub["TotalFlood_Mgal"], errors="coerce").fillna(0.0) * 1_000_000.0
    else:
        sub["Gallons"] = pd.to_numeric(sub["Gallons"], errors="coerce").fillna(0.0)

    per_node_ft3 = {r["Node"]: float(r["Gallons"]) / _GAL_TO_FT3 for _, r in sub.iterrows()}

    total_gal = float(sub["Gallons"].sum())
    if to_metric:
        total = total_gal * _GAL_TO_M3     
    else:
        total = total_gal / _GAL_TO_FT3    

    return total, per_node_ft3



def _global_runoff_range_across(
    dfs: List[pd.DataFrame], unit_ui: str, ws_path: str
) -> Tuple[float,float]:
    vals = []
    ws_gdf = load_ws(ws_path)
    for df in dfs:
        gdf, _unit = prep_total_runoff_gdf(df, unit_ui, ws_gdf)
        if "Total_R" in gdf:
            vals.append(gdf["Total_R"].to_numpy(dtype=float))
    if not vals:
        return (0.0, 1.0)
    arr = np.concatenate(vals)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, 1.0)
    vmin, vmax = float(finite.min()), float(finite.max())
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1e-6
    return (vmin, vmax)

def extract_total_runoff_from_text(txt: str) -> pd.DataFrame:
    lines = txt.splitlines(True)
    if not lines: return pd.DataFrame(columns=["Subcatchment","Total_in"])
    def is_dash(s: str) -> bool:
        s = s.strip(); return len(s) > 0 and set(s) <= set("- ")
    sec_i = next((i for i,l in enumerate(lines) if "Subcatchment Runoff Summary" in l), None)
    if sec_i is None: return pd.DataFrame(columns=["Subcatchment","Total_in"])
    i = sec_i + 1
    while i < len(lines) and "Subcatchment" not in lines[i]: i += 1
    if i >= len(lines): return pd.DataFrame(columns=["Subcatchment","Total_in"])
    i += 1
    while i < len(lines) and not is_dash(lines[i]): i += 1
    if i >= len(lines): return pd.DataFrame(columns=["Subcatchment","Total_in"])
    i += 1
    rows = []
    float_re = re.compile(r"[-+]?\d+(?:\.\d+)?")
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        if not raw.strip() or is_dash(raw): break
        if raw.lstrip().startswith("*") or "Node Depth Summary" in raw or "Analysis Options" in raw: break
        parts = raw.strip().split()
        name = parts[0] if parts else None
        nums = [float(m.group(0)) for m in float_re.finditer(raw)]
        if name and len(nums) >= 7:
            total_in = nums[7]
            rows.append({"Subcatchment": name, "Total_in": total_in})
        i += 1
    return pd.DataFrame(rows, columns=["Subcatchment","Total_in"])

def node_layer_from_shp(node_shp_path: str, node_vol_dict: Dict, name_field_hint: str = "NAME"):
    try:
        nodes_gdf = load_nodes(node_shp_path)
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
        vol_map = {str(k).strip(): float(v) for k, v in (node_vol_dict or {}).items()}
        nodes_gdf["_cuft_event"] = nodes_gdf["_node_id"].map(lambda nid: vol_map.get(nid, 0.0))
        nodes_gdf["_flooded"] = nodes_gdf["_cuft_event"] > 0.0
        nodes_gdf["_color_rgba"] = nodes_gdf["_flooded"].map(lambda f: [255,0,0,255] if f else [0,0,0,255])

        return pdk.Layer(
            "GeoJsonLayer",
            data=nodes_gdf.__geo_interface__,
            pickable=True,
            stroked=False,
            filled=True,
            get_fill_color="properties._color_rgba",
            get_point_radius=4,
            pointRadiusMinPixels=4,
        )
    except Exception:
        return None


def render_total_runoff_map_single(
    df_in_inches,
    title,
    nodes_post5h_dict,
    unit_ui,
    ws_shp_path,
    pipe_shp_path,
    node_shp_path,
    node_name_field_hint="NAME",
    legend_range: Tuple[float,float] | None = None,
    widget_key: str | None = None,
    show_title: bool = True,   # <-- NEW
):
    ws_gdf = load_ws(ws_shp_path)
    gdf, runoff_unit = prep_total_runoff_gdf(df_in_inches, unit_ui, ws_gdf)

    if legend_range is not None:
        vmin, vmax = legend_range
    else:
        vals = gdf["Total_R"].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(finite.min()), float(finite.max())
            if abs(vmax - vmin) < 1e-9:
                vmax = vmin + 1e-6

    gdf["_fill_total"] = make_color(gdf["Total_R"], vmin, vmax)
    gdf["_label"] = gdf["NAME"]

    centroid = gdf.geometry.union_all().centroid
    view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=13.75)

    poly_layer = pdk.Layer(
        "GeoJsonLayer",
        data=gdf.__geo_interface__,
        pickable=True, stroked=True, filled=True,
        get_fill_color="properties._fill_total",
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=1,
    )
    pipe_layer = None
    try:
        pipes = gpd.read_file(pipe_shp_path)
        pipes = pipes[pipes.geometry.notnull() & pipes.geometry.geom_type.isin(["LineString","MultiLineString"])]
        if not pipes.empty:
            if pipes.crs != ws_gdf.crs:
                pipes = pipes.to_crs(ws_gdf.crs)
            pipes["_pipe_label"] = pipes["NAME"] if "NAME" in pipes.columns else ""
            pipe_layer = pdk.Layer(
                "GeoJsonLayer",
                data=pipes.__geo_interface__,
                pickable=True, filled=False, stroked=True,
                get_line_color=[80,80,80,255], get_line_width=2, line_width_min_pixels=2,
                tooltip={"text": "{_pipe_label}"},
            )
    except Exception:
        pass

    nodes_layer = node_layer_from_shp(node_shp_path, nodes_post5h_dict, node_name_field_hint)

    reps = gdf.geometry.representative_point()
    labels = pd.DataFrame({"lon": reps.x, "lat": reps.y, "text": gdf["_label"]})
    text_layer = pdk.Layer("TextLayer", data=labels,
                           get_position='[lon, lat]', get_text="text",
                           get_size=10, get_color=[0,0,0], get_alignment_baseline="'center'")

    layers = [poly_layer]
    if pipe_layer is not None: layers.append(pipe_layer)
    if nodes_layer is not None: layers.append(nodes_layer)
    layers.append(text_layer)

    if show_title:
        st.markdown(f"**{title}**")
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_provider="carto",
            map_style="light",
            tooltip={"html": f"<b>{{NAME}}</b><br/>Total runoff: {{Total_R}} {runoff_unit}",
                    "style": {"backgroundColor":"white","color":"black"}},
        ),
        use_container_width=True,
        height=250,
        key=(widget_key or f"cmpmap_{title.replace(' ', '_')}")
    )

    return runoff_unit, (vmin, vmax)

from typing import Dict, Tuple, Set

def load_caps_from_df(df: pd.DataFrame) -> Tuple[Dict[str,int], Dict[str,int]]:
    if "NAME" not in df.columns or "Max_RG_DEM_Considered" not in df.columns or "MaxNumber_RB" not in df.columns:
        raise ValueError("raster_df must have NAME, Max_RG_DEM_Considered, MaxNumber_RB")
    name = df["NAME"].astype(str)
    caps_RG = dict(zip(name, pd.to_numeric(df["Max_RG_DEM_Considered"], errors="coerce").fillna(0).astype(int)))
    caps_RB = dict(zip(name, pd.to_numeric(df["MaxNumber_RB"], errors="coerce").fillna(0).astype(int)))
    return caps_RG, caps_RB

def realize_percent_uptake(pRG_percent: float, pRB_percent: float,
                           caps_RG: Dict[str,int], caps_RB: Dict[str,int]) -> Dict[str, Dict[str,int]]:
    pRG = max(0.0, min(100.0, float(pRG_percent)))
    pRB = max(0.0, min(100.0, float(pRB_percent)))
    out = {}
    all_names = set(caps_RG) | set(caps_RB)
    for s in all_names:
        rg = int(np.floor((pRG/100.0) * (caps_RG.get(s, 0) or 0)))
        rb = int(np.floor((pRB/100.0) * (caps_RB.get(s, 0) or 0)))
        out[s] = {"rain_gardens": rg, "rain_barrels": rb}
    return out

def compute_percent_pair_for_budget(B: float, cRG: float, cRB: float,
                                    CapRG_total: int, CapRB_total: int,
                                    pin_type: str, pin_value_percent: float) -> Tuple[float, float]:
    pin_type = pin_type.upper().strip()
    pin_value = max(0.0, min(100.0, float(pin_value_percent)))
    if pin_type == "RG":
        pRG = pin_value
        pRB_star = (B - cRG * (pRG/100.0) * CapRG_total) / (cRB * CapRB_total) if CapRB_total > 0 else 0.0
        pRB = max(0.0, min(100.0, pRB_star * 100.0))
        return round(pRG, 1), round(pRB, 1)
    elif pin_type == "RB":
        pRB = pin_value
        pRG_star = (B - cRB * (pRB/100.0) * CapRB_total) / (cRG * CapRG_total) if CapRG_total > 0 else 0.0
        pRG = max(0.0, min(100.0, pRG_star * 100.0))
        return round(pRG, 1), round(pRB, 1)
    else:
        raise ValueError("pin_type must be 'RG' or 'RB'")

def restrict_plan_to_group(plan: Dict[str, Dict[str,int]], group_names: Set[str]) -> Dict[str, Dict[str,int]]:
    group = set(group_names or set())
    return {s: v for s, v in plan.items() if s in group}

def summarize_plan(plan: Dict[str, Dict[str,int]], cRG: float, cRB: float) -> dict:
    total_rg = sum(v["rain_gardens"] for v in plan.values())
    total_rb = sum(v["rain_barrels"] for v in plan.values())
    spent = cRG * total_rg + cRB * total_rb
    treated_ft2 = 400.0 * total_rg + 300.0 * total_rb
    return {"rg": int(total_rg), "rb": int(total_rb), "spent": float(spent), "treated_ft2": float(treated_ft2)}

# --- LID placement map (logarithmic green shading; zero-LID transparent; Deck-level tooltip) ---
def render_focus_placement_map_log(plan: Dict[str, Dict[str,int]],
                                   title: str,
                                   ws_shp_path: str,
                                   widget_key: str):
    # Load subcatchments
    gdf = load_ws(ws_shp_path).copy()

    # Attach counts to each subcatchment
    gdf["_RG"] = gdf["NAME"].map(lambda s: plan.get(s, {}).get("rain_gardens", 0))
    gdf["_RB"] = gdf["NAME"].map(lambda s: plan.get(s, {}).get("rain_barrels", 0))
    gdf["_TOTAL_LID"] = gdf["_RG"] + gdf["_RB"]

    # Log-scale green fill; zero-LID is transparent
    vmax = max(1, int(gdf["_TOTAL_LID"].max()))
    def fill_color(n):
        if n <= 0:
            return [0, 0, 0, 0]  # transparent
        alpha = int(80 + 175 * (np.log1p(n) / np.log1p(vmax)))  # 80..255 (log scale)
        return [34, 139, 34, alpha]  # ForestGreen
    gdf["_fill"] = gdf["_TOTAL_LID"].map(fill_color)

    # Labels for map (keep simple; details in tooltip)
    reps = gdf.geometry.representative_point()
    labels = pd.DataFrame({
        "lon": reps.x, "lat": reps.y,
        "text": gdf["NAME"].astype(str)
    })
    text_layer = pdk.Layer(
        "TextLayer",
        data=labels,
        get_position='[lon, lat]',
        get_text="text",
        get_size=10,
        get_color=[0,0,0],
        get_alignment_baseline="'center'"
    )

    # GeoJson layer (no tooltip here; Streamlit reads it from Deck)
    poly_layer = pdk.Layer(
        "GeoJsonLayer",
        data=gdf.__geo_interface__,
        pickable=True,
        autoHighlight=True,
        highlightColor=[0, 0, 0, 160],
        stroked=True,
        filled=True,
        get_fill_color="properties._fill",
        get_line_color=[0,0,0,255],  # black outlines
        line_width_min_pixels=1,
    )

    # View
    centroid = gdf.geometry.union_all().centroid
    view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=13.45)

    deck_tooltip = {
        "html": (
            "<b>{NAME}</b><br/>"
            "RG (rain gardens): {_RG}<br/>"
            "RB (rain barrels): {_RB}<br/>"
            "Total LIDs: {_TOTAL_LID}"
        ),
        "style": {"backgroundColor": "white", "color": "black"}
    }

    # Render
    st.markdown(f"**{title}**")
    st.pydeck_chart(
        pdk.Deck(
            layers=[poly_layer, text_layer],
            initial_view_state=view_state,
            map_provider="carto",
            map_style="light",
            tooltip=deck_tooltip,  # <-- move tooltip here
        ),
        use_container_width=True,
        height=260,
        key=widget_key
    )

def _legend_html(unit: str, vmin: float, vmax: float) -> str:
    cmap = mpl.colormaps.get_cmap("Blues")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    c0 = [int(v*255) for v in cmap(norm(vmin))[:3]]
    c1 = [int(v*255) for v in cmap(norm(vmax))[:3]]
    return f"""
    <div style="display:flex; justify-content:center; margin-top:6px;">
      <div style="min-width:260px; max-width:640px; width:60%;">
        <div style="text-align:center; font-size:13px;"><b>Runoff Legend ({unit})</b></div>
        <div style="display:flex; align-items:center; gap:10px;">
          <span>{vmin:.2f}</span>
          <div style="flex:1; height:12px;
              background:linear-gradient(to right,
              rgb({c0[0]},{c0[1]},{c0[2]}),
              rgb({c1[0]},{c1[1]},{c1[2]}));
              border:1px solid #888;"></div>
          <span>{vmax:.2f}</span>
        </div>
        <div style="color:#555; font-size:12px; text-align:center; margin-top:6px;">
          Same scale for all 8 comparison.
        </div>
      </div>
    </div>
    """


def rain_window_with_buffer(sim_start_str: str,
                            minutes_15: List[int] | np.ndarray,
                            rain_curve_in: List[float] | np.ndarray,
                            buffer_hours: float = 2.0,
                            eps: float = 1e-9) -> Tuple[datetime, datetime] | None:
    """Return (t_start, t_end) covering all times where rain>eps, plus +buffer_hours."""
    if minutes_15 is None or rain_curve_in is None:
        return None
    if len(minutes_15) == 0 or len(rain_curve_in) == 0:
        return None

    t0 = datetime.strptime(sim_start_str, "%m/%d/%Y %H:%M")
    wet_idx = [i for i, v in enumerate(rain_curve_in) if (float(v) if v is not None else 0.0) > eps]
    if not wet_idx:
        return None

    i0, i1 = wet_idx[0], wet_idx[-1]
    t_start = t0 + timedelta(minutes=int(minutes_15[i0]))
    t_end   = t0 + timedelta(minutes=int(minutes_15[i1])) + timedelta(hours=float(buffer_hours))
    return (t_start, t_end)

def run_swmm_scenario(
    scenario_name: str,
    rain_lines: List[str],
    tide_lines: List[str],
    lid_lines: List[str],
    gate_flag: str,
    report_interval=timedelta(minutes=5),
    template_path="SWMM_Project.inp",
    event_window_mode: str = "rain+2h",
    event_buffer_hours: float = 2.0,
    rain_minutes: List[int] | None = None,
    rain_curve_in: List[float] | None = None,
    sim_start_str: str | None = None,
) -> dict:
    """Run SWMM; keep RPT only in session_state; return parsed artifacts."""
    temp_dir  = st.session_state.temp_dir
    inp_path  = os.path.join(temp_dir, f"{scenario_name}.inp")
    rpt_path  = os.path.join(temp_dir, f"{scenario_name}.rpt")   
    out_path  = os.path.join(temp_dir, f"{scenario_name}.out")   

    with open(template_path, "r") as f:
        text = f.read()
    text = (
        text.replace("$RAINFALL_TIMESERIES$", "\n".join(rain_lines))
            .replace("$TIDE_TIMESERIES$", "\n".join(tide_lines))
            .replace("$LID_USAGE$", "\n".join(lid_lines))
            .replace("$TIDE_GATE_CONTROL$", gate_flag)
    )
    with open(inp_path, "w") as f:
        f.write(text)

    step_s = int(report_interval.total_seconds())

    with Simulation(inp_path, rpt_path, out_path) as sim:
        sim.step_advance(step_s)
        for _ in sim:
            pass

    rpt_text = _read_text_keep(rpt_path)


    df_nf = parse_node_flooding_summary_from_text(rpt_text)
    to_metric = (st.session_state.get("unit_ui") == "Metric (SI)")
    total_event_vol, node_cuft_event = summarize_node_flooding_in_window(df_nf, to_metric=to_metric)
    df_runoff = extract_total_runoff_from_text(rpt_text)
    df_ir     = extract_infiltration_and_runoff_from_text(rpt_text)


    st.session_state.setdefault("rpts", {})[scenario_name] = rpt_text
    st.session_state[f"{scenario_name}_event_total_flood"] = total_event_vol
    st.session_state[f"{scenario_name}_node_flood_event_cuft"] = node_cuft_event

    return {
        "df_runoff": df_runoff,
        "node_cuft_event": node_cuft_event,
        "df_continuity": df_ir,
        "total_event_vol": total_event_vol,
    }

def extract_infiltration_and_runoff_from_text(txt: str) -> pd.DataFrame:
    lines = txt.splitlines(True)
    if not lines: return pd.DataFrame(columns=["label","volume_acft"])
    hdr_i = next((i for i,l in enumerate(lines) if "Runoff Quantity Continuity" in l), None)
    if hdr_i is None: return pd.DataFrame(columns=["label","volume_acft"])
    i = hdr_i + 1
    def _decor(s):
        s = s.strip(); return not s or all(ch in "*- " for ch in s)
    while i < len(lines) and _decor(lines[i]): i += 1
    targets = {"Infiltration Loss": "infiltration_loss", "Surface Runoff": "surface_runoff"}
    rows = []
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        if not raw.strip() or raw.lstrip().startswith("*"): break
        for disp, key in targets.items():
            if disp.lower() in raw.lower():
                nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", raw)]
                vol = nums[0] if nums else float("nan")
                rows.append({"label": key, "volume_acft": vol})
                break
        i += 1
    return pd.DataFrame(rows, columns=["label","volume_acft"])

def _ensure_baselines_visible(prefix: str):
    """If the scenario store lacks the 4 baseline entries, rebuild them from saved DFs / RPT text."""
    store = _sc_store(prefix)
    needed = {
        "baseline_nogate_current": f"{prefix}df_base_nogate_current",
        "baseline_gate_current":   f"{prefix}df_base_gate_current",
        "baseline_nogate_future":  f"{prefix}df_base_nogate_future",
        "baseline_gate_future":    f"{prefix}df_base_gate_future",
    }
    rpts = st.session_state.get("rpts", {})
    changed = False

    for name, df_key in needed.items():
        if name in store:
            continue
        df = st.session_state.get(df_key, None)

        if (df is None) or (isinstance(df, pd.DataFrame) and df.empty):
            rpt_text = rpts.get(f"{prefix}{name}", "")
            if rpt_text:
                df = extract_total_runoff_from_text(rpt_text)

        if isinstance(df, pd.DataFrame) and not df.empty:
            node_key = f"{prefix}{name}_node_flood_event_cuft"
            nodes = st.session_state.get(node_key, {}) or {}
            remember_scenario(name, df, nodes, prefix)
            changed = True

    return changed

def _rehydrate_baseline_into_store(prefix: str):
    store = _sc_store(prefix)
    df_keys = {
        "baseline_nogate_current": f"{prefix}df_base_nogate_current",
        "baseline_gate_current":   f"{prefix}df_base_gate_current",
        "baseline_nogate_future":  f"{prefix}df_base_nogate_future",
        "baseline_gate_future":    f"{prefix}df_base_gate_future",
    }

    rpts = st.session_state.get("rpts", {})

    for name, df_key in df_keys.items():
        if name in store:
            continue

        df = st.session_state.get(df_key, None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            node_key = f"{prefix}{name}_node_flood_event_cuft"
            nodes = st.session_state.get(node_key, {}) or {}
            remember_scenario(name, df, nodes, prefix)
            continue

        rpt_text = rpts.get(f"{prefix}{name}", "")
        if rpt_text:
            df = extract_total_runoff_from_text(rpt_text)
            node_key = f"{prefix}{name}_node_flood_event_cuft"
            nodes = st.session_state.get(node_key, {}) or {}
            if not df.empty:
                remember_scenario(name, df, nodes, prefix)

def _build_baseline_map_html(df_swmm_local: pd.DataFrame, unit_ui: str, ws_shp_path: str) -> str:
    ws_gdf_local = load_ws(ws_shp_path)
    gdf, unit_r = prep_total_runoff_gdf(df_swmm_local, unit_ui, ws_gdf_local)

    vals = gdf["Total_R"].to_numpy()
    vmin_raw, vmax_raw = np.nanmin(vals), np.nanmax(vals)
    vmin = float(vmin_raw) if np.isfinite(vmin_raw) else 0.0
    vmax = float(vmax_raw) if np.isfinite(vmax_raw) else 1.0
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1e-6

    gdf["_fill"] = make_color(gdf["Total_R"], vmin, vmax)
    gdf["_label"] = gdf["NAME"]
    gdf["Total_R_disp"] = gdf["Total_R"].apply(lambda v: f"{float(v):,.2f}")

    centroid = gdf.geometry.union_all().centroid
    view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=14.5)

    poly_layer = pdk.Layer(
        "GeoJsonLayer", data=gdf.__geo_interface__,
        pickable=True, stroked=True, filled=True,
        get_fill_color="properties._fill",
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=1,
    )

    reps = gdf.geometry.representative_point()
    labels_df = pd.DataFrame({"lon": reps.x, "lat": reps.y, "text": gdf["_label"]})
    text_layer = pdk.Layer(
        "TextLayer", data=labels_df,
        get_position='[lon, lat]', get_text="text",
        get_size=14, get_color=[0, 0, 0], get_alignment_baseline="'center'"
    )

    tooltip = {
        "html": (
            "<div style='font-size:16px; line-height:1.5;'>"
            "<div style='font-weight:600;'>{NAME}</div>"
            f"<div>Runoff: {{Total_R_disp}} {unit_r}</div>"
            "</div>"
        ),
        "style": {
            "backgroundColor": "white",
            "color": "black",
            "fontFamily": "Inter, Arial, Helvetica, sans-serif",
            "fontSize": "16px",
            "padding": "8px 10px",
            "borderRadius": "8px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.20)",
        },
    }

    deck_obj = pdk.Deck(
        layers=[poly_layer, text_layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style="light",
        tooltip=tooltip,
    )
    map_html = deck_obj.to_html(as_string=True)

    map_html = map_html.replace(
        "</head>",
        "<style>.deck-tooltip{font-size:16px !important; line-height:1.5 !important;"
        "padding:8px 10px !important; border-radius:8px !important;}</style></head>"
    )

    # Legend (optional: bump text size a bit)
    cmap = mpl.colormaps.get_cmap("Blues")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    c0 = [int(v * 255) for v in cmap(norm(vmin))[:3]]
    c1 = [int(v * 255) for v in cmap(norm(vmax))[:3]]

    legend_html = f"""
    <div style="display:flex; justify-content:center; margin-top:6px;">
      <div style="min-width:260px; max-width:640px; width:60%;">
        <div style="text-align:center; font-size:14px;"><b>Runoff Legend ({unit_r})</b></div>
        <div style="display:flex; align-items:center; gap:10px;">
          <span>{vmin:.2f}</span>
          <div style="flex:1; height:12px;
              background:linear-gradient(to right,
              rgb({c0[0]},{c0[1]},{c0[2]}),
              rgb({c1[0]},{c1[1]},{c1[2]}));
              border:1px solid #888;"></div>
          <span>{vmax:.2f}</span>
        </div>
      </div>
    </div>
    """
    return map_html, legend_html

def generate_lid_usage_lines(lid_config: Dict[str, Dict[str, int]], excel_df: pd.DataFrame) -> List[str]:

    lines: List[str] = []
    tpl = (
        "{sub:<15}{proc:<16}{num:>7}{area:>8}{width:>7}{initsat:>8}"
        "{fromimp:>8}{toperv:>8}{rptfile:>24}{drainto:>16}{fromperv:>9}"
    )

    df = excel_df.set_index("NAME", drop=False)

    for sub, cfg in lid_config.items():
        if sub not in df.index:
            continue

        row = df.loc[sub]
        imperv = float(row.get("Impervious_ft2", 0.0) or 0.0)
        perv   = float(row.get("Pervious_ft2",   0.0) or 0.0)


        rb = int(cfg.get("rain_barrels", 0) or 0)
        if rb > 0:
            pct_imp = (rb * 300.0) / (imperv if imperv > 0 else 1e-9) * 100.0
            lines.append(
                tpl.format(
                    sub=sub,
                    proc="rain_barrel",
                    num=rb,
                    area=f"{2.58:.2f}",  
                    width=0,
                    initsat=0,
                    fromimp=f"{pct_imp:.2f}",
                    toperv=1,
                    rptfile="*",
                    drainto="*",
                    fromperv=0,
                )
            )


        rg = int(cfg.get("rain_gardens", 0) or 0)
        if rg > 0:
           
            pct_perv = (rg * 400.0) / (perv if perv > 0 else 1e-9) * 100.0
            lines.append(
                tpl.format(
                    sub=sub,
                    proc="rain_garden",
                    num=rg,
                    area=f"{100:.0f}",  
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

def _sc_store(prefix: str) -> dict:
    key = f"{prefix}scenarios"
    if key not in st.session_state:
        st.session_state[key] = {}
    return st.session_state[key]

def remember_scenario(name: str, df_runoff: pd.DataFrame, node_dict: dict, prefix: str = ""):
    store = _sc_store(prefix)

    store[name] = {
        "df": df_runoff.copy(deep=True) if df_runoff is not None else None,
        "nodes": dict(node_dict) if node_dict is not None else {},
    }

def recall_scenario(name: str, prefix: str = ""):
    return _sc_store(prefix).get(name)

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

                st.session_state["scenario_prefix"] = f"user_{user_id}_"
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

    st.success(f"Logged in as: {st.session_state.get('email', 'user')}")
    ensure_temp_dir()
    prefix = st.session_state.get("scenario_prefix", "")

    st.session_state.setdefault("rpts", {})

    _rehydrate_baseline_into_store(prefix)
    _ensure_baselines_visible(prefix)

    simulation_date = "05/31/2025 12:00"
    template_inp    = "SWMM_Project.inp"
    WS_SHP_PATH     = st.session_state.get("WS_SHP_PATH", "map_files/Subcatchments.shp")
    NODE_SHP_PATH   = st.session_state.get("NODE_SHP_PATH", "map_files/Nodes.shp")
    PIPE_SHP_PATH   = st.session_state.get("PIPE_SHP_PATH", "map_files/Conduits.shp")

    future_mult = 1.2

    st.title("CoastWise")
    st.markdown('<a href="https://github.com/savannah345/flood-modeling-k12-education/blob/main/CoastWise_Tutorial.docx" target="_blank">Tutorial</a>', unsafe_allow_html=True)

    # --- Fixed subcatchment lists (must match shapefile NAME like "Sub_20") ---
    UPSTREAM_LIST = {f"Sub_{n}" for n in [20,22,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]}
    DOWNSTREAM_LIST = {f"Sub_{n}" for n in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23]}
    HIGHRUNOFF_LIST = {f"Sub_{n}" for n in [3,4,5,6,7,10,18,19,21,22,27,28,29,30,31,32,33,35,37,38]}
    
    st.markdown("""
    <style>
    /* Make all help (?) tooltips larger and easier to read */
    [data-testid="stTooltipContent"] {
        font-size: 10.5rem !important;
        line-height: 1.55 !important;
        max-width: 360px !important;
        white-space: normal !important;  /* allow wrapping */
    }

    /* Fallback for older Streamlit builds using BaseWeb tooltip container */
    div[data-baseweb="tooltip"] {
        font-size: 10rem !important;
        line-height: 1.55 !important;
        max-width: 360px !important;
    }

    /* Optional: slightly increase tooltip padding */
    div[data-baseweb="tooltip"] .content,
    [data-testid="stTooltipContent"] {
        padding: 0.5rem 0.75rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


    if "cfg" not in st.session_state:
        st.session_state.cfg = {
            "unit": "U.S. Customary",
            "moon_phase": list(moon_tide_ranges.keys())[0],
            "duration_minutes": int(pf_df["Duration_Minutes"].iloc[0]),
            "return_period": "1",
            "align_mode": "peak",  
            "settings_ready": False,
        }
        st.session_state.pop(f"{prefix}grid_ready", None)
    cfg = st.session_state.cfg
    st.success(
        "Use the controls below to configure your storm scenario. "
        "These settings determine the rainfall event, tide conditions, and alignment "
        "that will be used to generate boundary inputs for the SWMM simulation."
    )

    with st.form("scenario_settings"):
        # ---------------- Units ----------------
        unit = st.radio(
            "Preferred Units",
            ["U.S. Customary", "Metric (SI)"],
            index=(0 if cfg["unit"] == "U.S. Customary" else 1),
            horizontal=True,
            help=(
                "Choose how rainfall and tide inputs should be displayed. "
                "U.S. Customary = rainfall in inches, tides in feet. "
                "Metric (SI) = rainfall in centimeters, tides in meters."
            )
        )

        # ---------------- Tide Source ----------------
        moon_phase_keys = list(moon_tide_ranges.keys())
        moon_phase = st.radio(
            "Synthetic Tide (fallback)",
            moon_phase_keys,
            index=(moon_phase_keys.index(cfg["moon_phase"]) if cfg["moon_phase"] in moon_phase_keys else 0),
            horizontal=True,
            help=(
                "If real-time tide data from the Norfolk gauge is unavailable, "
                "CoastWise generates a fallback tide curve. The moon phase determines "
                "the approximate high/low tide range used to build the synthetic series."
            )
        )

        # ---------------- Storm Duration Slider ----------------
        duration_options = sorted(
            map(int, pd.Series(pf_df['Duration_Minutes']).dropna().unique().tolist())
        )
        duration_default = cfg["duration_minutes"] if cfg["duration_minutes"] in duration_options else duration_options[0]

        duration_minutes = st.select_slider(
            "Storm Duration",
            options=duration_options,
            value=duration_default,
            format_func=lambda x: f"{int(x)//60} hr",
            help=(
                "Controls the total rainfall event length used to scale the NOAA Atlas 14 depth. "
                "Shorter storms (2–3 hr) generally produce more intense peaks, while longer durations "
                "spread the rain over a wider window."
            )
        )

        d_low, d_high = st.columns([1, 3])
        with d_low:
            st.caption("Short storm")
        with d_high:
            st.caption("<div style='text-align:right;'>Long storm</div>", unsafe_allow_html=True)


        # ---------------- Return Period Slider ----------------
        rp_cols = [c for c in pf_df.columns if c != "Duration_Minutes" and str(c).isdigit()]
        rp_years = sorted(map(int, rp_cols)) if rp_cols else [1]
        rp_default = cfg.get("return_period")
        rp_default = str(rp_default) if rp_default in map(str, rp_years) else str(rp_years[0])

        return_period = st.select_slider(
            "Return Period (years)",
            options=list(map(str, rp_years)),
            value=str(rp_default),
            format_func=lambda k: f"{k}-year",
            help=(
                "Selects the rainfall intensity based on NOAA Atlas 14. "
                "Higher return periods represent rarer, more severe storms. "
                "For example, a 10-year storm is more intense than a 2-year storm."
            )
        )
        r_low, r_high = st.columns([1, 3])
        with r_low:
            st.caption("Less intense")
        with r_high:
            st.caption("<div style='text-align:right;'>More intense</div>", unsafe_allow_html=True)

        rain_variant_choice = st.radio(
            "Rainfall Scenario",
            ["Current", "Future (+20%)"],
            index=(0 if cfg.get("rain_variant", "current") == "current" else 1),
            horizontal=True,
            help="Choose whether to run simulations using current rainfall or a fixed +20% future rainfall."
        )

        # ---------------- Tide Alignment ----------------
        align_choice = st.radio(
            "Tide Alignment",
            ["Peak aligned with High Tide", "Peak aligned with Low Tide"],
            index=(0 if cfg["align_mode"] == "peak" else 1),
            horizontal=True,
            help=(
                "Controls how the rainfall peak is aligned with the tidal cycle. "
                "High-tide alignment typically creates more adverse compound flooding. "
                "Low-tide alignment represents a more favorable drainage condition."
            )
        )

        submitted = st.form_submit_button("Apply Settings")

    if submitted:
        st.session_state.cfg = {
            "unit": unit,
            "moon_phase": moon_phase,
            "duration_minutes": int(duration_minutes),
            "return_period": str(return_period),  # keep as string; downstream expects str
             "align_mode": ("peak" if "High" in align_choice else "low"),
            "settings_ready": True,
            "rain_variant": ("future" if "Future" in rain_variant_choice else "current")
            }
        st.session_state["rain_variant"] = st.session_state.cfg["rain_variant"]
        cfg = st.session_state.cfg
        st.success("Settings applied.")

    if not cfg.get("settings_ready", False):
        st.info("Apply settings to generate rainfall and tide series.")
        if st.button("🚪 Logout"):
            try:
                shutil.rmtree(st.session_state.temp_dir)
            except Exception:
                pass
            st.session_state.clear()
            st.success("Logged out and cleaned up all files.")
            st.experimental_rerun()
        st.stop()  # cleanly halt Streamlit

        # Safe to read cfg from here onward
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
            method="SCS_TypeIII",
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
            method="SCS_TypeIII",
            target_index=None,
            prominence=None
        )
        tide_source = "synthetic"


    if unit == "U.S. Customary":
        display_rain_curve = rain_curve_in        
        display_tide_curve = tide_curve_ui        
        rain_disp_unit = "inches"; tide_disp_unit = "ft"
    else:
        display_rain_curve = rain_curve_in * 2.54  
        display_tide_curve = tide_curve_ui         
        rain_disp_unit = "centimeters"; tide_disp_unit = "meters"

    st.session_state.update({
        "rain_minutes": minutes_15,
        "tide_minutes": minutes_15,
        "display_rain_curve_current": display_rain_curve,
        "display_rain_curve_future": display_rain_curve * future_mult,
        "display_tide_curve": display_tide_curve,
        "rain_sim_curve_current_in": rain_curve_in,
        "rain_sim_curve_future_in": rain_curve_in * future_mult,
        "rain_disp_unit": rain_disp_unit,
        "tide_disp_unit": tide_disp_unit,
        "unit_ui": unit,
        "tide_source": tide_source,
        "moon_phase": moon_phase,
        "align_mode": align_mode,
        f"{prefix}simulation_date": simulation_date,
        f"{prefix}template_inp": template_inp,
    })

    tide_lines = format_timeseries(
        "tide",
        minutes_15,
        tide_to_feet_for_swmm(st.session_state["display_tide_curve"], unit),
        simulation_date
    )
    def _rain_lines_pair(sim_minutes, rain_curve_in, sim_start_str, future_mult=future_mult):
        cur = format_timeseries("rain_gage_timeseries", sim_minutes, rain_curve_in, sim_start_str)
        fut = format_timeseries("rain_gage_timeseries", sim_minutes, (np.array(rain_curve_in) * future_mult), sim_start_str)
        return cur, fut

    rain_lines_cur, rain_lines_fut = _rain_lines_pair(minutes_15, st.session_state["rain_sim_curve_current_in"], simulation_date, future_mult=future_mult)

    st.session_state.update({
        "tide_lines": tide_lines,
        "rain_lines_cur": rain_lines_cur,
        "rain_lines_fut": rain_lines_fut,
        f"{prefix}tide_lines": tide_lines,
        f"{prefix}rain_lines_cur": rain_lines_cur,
        f"{prefix}rain_lines_fut": rain_lines_fut,
    })


    time_hours = np.array(minutes_15, dtype=float) / 60.0

    # Build tidy dataframe for the chart
    df_rt = pd.DataFrame({
        "Hour": time_hours,
        "Rain_Current": st.session_state["display_rain_curve_current"],
        "Rain_Future":  st.session_state["display_rain_curve_future"],  
        "Tide":         st.session_state["display_tide_curve"],
    })


    rain_variant = cfg.get("rain_variant", "current")
    run_current = (rain_variant == "current")
    run_future  = (rain_variant == "future")

    # Optional quick metrics row (keep or remove as you prefer)
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.metric(f"Total Current Rainfall ({rain_disp_unit})", f"{np.nansum(df_rt['Rain_Current']):.2f}")
    with c_m2:
        st.metric(f"Total Future Rainfall (+20%) ({rain_disp_unit})", f"{np.nansum(df_rt['Rain_Future']):.2f}")
    with c_m3:
        st.metric(f"Tide Range ({tide_disp_unit})", f"{(np.nanmax(df_rt['Tide'])-np.nanmin(df_rt['Tide'])):.2f}")

    # ----- Altair combined chart (left y-axis = rainfall, right y-axis = tide) -----
    import altair as alt
    alt.data_transformers.disable_max_rows()

    base = alt.Chart(df_rt).encode(
        x=alt.X("Hour:Q", title="Hour", scale=alt.Scale(nice=False))
    )

    # Rainfall (Current) — darker area, left axis
    rain_current = base.mark_area(color="steelblue", opacity=0.50).encode(
        y=alt.Y("Rain_Current:Q",
                title=f"Rainfall ({rain_disp_unit})",
                axis=alt.Axis(titleColor="steelblue")),
        tooltip=[
            alt.Tooltip("Hour:Q", format=".1f"),
            alt.Tooltip("Rain_Current:Q", title=f"Current ({rain_disp_unit})", format=".3f"),
        ],
    )

    # Rainfall (Future +20%) — lighter area, left axis
    rain_future = base.mark_area(color="#f74f4f", opacity=0.20).encode(
        y=alt.Y("Rain_Future:Q",
                axis=alt.Axis(title=None, labels=False, ticks=False)),  # share left axis visually
        tooltip=[
            alt.Tooltip("Hour:Q", format=".1f"),
            alt.Tooltip("Rain_Future:Q", title=f"Future (+20%) ({rain_disp_unit})", format=".3f"),
        ],
    )

    # Tide — line on right axis
    tide_line = base.mark_line(color="#2e9144", strokeWidth=3).encode(
        y=alt.Y("Tide:Q",
                title=f"Tide ({tide_disp_unit})",
                axis=alt.Axis(titleColor="#2e9144", orient="right")),
        tooltip=[
            alt.Tooltip("Hour:Q", format=".1f"),
            alt.Tooltip("Tide:Q", title=f"Tide ({tide_disp_unit})", format=".2f"),
        ],
    )

    combo_chart = alt.layer(tide_line, rain_future, rain_current).resolve_scale(
        y="independent"  # let rainfall and tide use separate axes/scales
    ).properties(
        title="Rainfall (Current & +20%) and Tide Combined",
        height=380,
    ).configure_title(
        fontSize=18
    ).configure_axis(
        labelFontSize=13,
        titleFontSize=14,
        grid=False  # remove gridlines
    ).interactive()  # zoom + pan

    st.altair_chart(combo_chart, use_container_width=True)

    # Clear, short explanation under the chart (optional)
    st.caption(
        f"Rainfall shown as areas (left axis: {rain_disp_unit}); Tide shown as line (right axis: {tide_disp_unit}). "
        "Zoom/pan to inspect alignment and peaks."
    )

    if st.button("Run Baseline Scenario" , key=f"{prefix}btn_run_baseline"):
        try:
            lid_lines = [";"] 
            if run_current:  

                info1 = run_swmm_scenario(
                    f"{prefix}baseline_nogate_current", rain_lines_cur, tide_lines, lid_lines, "NO",
                    template_path=template_inp,
                    event_window_mode="rain+2h", event_buffer_hours=2.0,
                    rain_minutes=minutes_15,
                    rain_curve_in=st.session_state["rain_sim_curve_current_in"],
                    sim_start_str=simulation_date,
                )
                info2 = run_swmm_scenario(
                    f"{prefix}baseline_gate_current",   rain_lines_cur, tide_lines, lid_lines, "YES",
                    template_path=template_inp,
                    event_window_mode="rain+2h", event_buffer_hours=2.0,
                    rain_minutes=minutes_15,
                    rain_curve_in=st.session_state["rain_sim_curve_current_in"],
                    sim_start_str=simulation_date,
                )
                st.session_state.update({
                    f"{prefix}df_base_nogate_current": info1["df_runoff"],
                    f"{prefix}df_base_gate_current":   info2["df_runoff"]
                })

                remember_scenario("baseline_nogate_current", info1["df_runoff"], info1["node_cuft_event"], prefix)
                remember_scenario("baseline_gate_current",   info2["df_runoff"], info2["node_cuft_event"], prefix)

            if run_future: 
                info3 = run_swmm_scenario(
                    f"{prefix}baseline_nogate_future",  rain_lines_fut, tide_lines, lid_lines, "NO",
                    template_path=template_inp,
                    event_window_mode="rain+2h", event_buffer_hours=2.0,
                    rain_minutes=minutes_15,
                    rain_curve_in=st.session_state["rain_sim_curve_future_in"],
                    sim_start_str=simulation_date,
                )
                info4 = run_swmm_scenario(
                    f"{prefix}baseline_gate_future",    rain_lines_fut, tide_lines, lid_lines, "YES",
                    template_path=template_inp,
                    event_window_mode="rain+2h", event_buffer_hours=2.0,
                    rain_minutes=minutes_15,
                    rain_curve_in=st.session_state["rain_sim_curve_future_in"],
                    sim_start_str=simulation_date,
                )

                st.session_state.update({
                    f"{prefix}df_base_nogate_future":  info3["df_runoff"],
                    f"{prefix}df_base_gate_future":    info4["df_runoff"]
                })
 
                remember_scenario("baseline_nogate_future",  info3["df_runoff"], info3["node_cuft_event"], prefix)
                remember_scenario("baseline_gate_future",    info4["df_runoff"], info4["node_cuft_event"], prefix)

            # Replace your df_swmm_now block with:
            df_swmm_now = None
            if run_current:
                df_swmm_now = st.session_state.get(f"{prefix}df_base_nogate_current")
            elif run_future:
                df_swmm_now = st.session_state.get(f"{prefix}df_base_nogate_future")

            if df_swmm_now is None or len(df_swmm_now) == 0:
                st.error("No rows parsed from 'Subcatchment Runoff Summary' in the .rpt")
            else:
                st.success("Baseline scenarios complete.")
                st.session_state[f"{prefix}ran_baseline"] = True
                map_html, legend_html = _build_baseline_map_html(
                    df_swmm_local=df_swmm_now,
                    unit_ui=st.session_state["unit_ui"],
                    ws_shp_path=WS_SHP_PATH
                )
                st.session_state[f"{prefix}baseline_map_html"]    = map_html
                st.session_state[f"{prefix}baseline_legend_html"] = legend_html
                st.session_state[f"{prefix}grid_ready"] = True
        except Exception as e:
            st.error(f"Baseline simulation failed: {e}")

    if f"{prefix}baseline_map_html" in st.session_state:
        st.subheader("Runoff and Land Cover Context")

        st.markdown("""
        Higher runoff values typically occur in areas with more **impervious surfaces** such as 
        roads, parking lots, rooftops, and other paved or compacted areas. These surfaces prevent 
        rainfall from soaking into the soil, increasing the amount of direct surface runoff. This 
        is why developed or urbanized areas often show darker (higher) runoff values. 
        The land cover map to the right helps show where these impervious areas are located.
        """)

        # --- two columns side-by-side ---
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("### Modeled Runoff")
            components.v1.html(st.session_state[f"{prefix}baseline_map_html"], height=500, scrolling=False)

        with c2:
            st.markdown("### Land Use / Land Cover")
            st.image("lulc_image.png", use_container_width=True)

    REQUIRED_RASTER_COLS = {
        "NAME", "Max_RG_DEM_Considered", "MaxNumber_RB",
        "Impervious_ft2", "Pervious_ft2"
    }

    def _load_raster_df() -> pd.DataFrame:

        return load_raster_cells()


    if "raster_df" not in st.session_state:
        try:
            st.session_state["raster_df"] = _load_raster_df()
        except Exception as e:
            st.error(f"Failed to load raster/subcatchment table: {e}")
            st.stop()

    raster_df: pd.DataFrame = st.session_state["raster_df"]


    missing = REQUIRED_RASTER_COLS - set(raster_df.columns)
    if missing:
        st.error(f"raster_df is missing required columns: {sorted(missing)}")
        st.stop()

    st.subheader("Plan LIDs")

    # Explain the two paths *before* any controls
    st.info(
        "• **Path A – Percent Uptake (whole watershed):** Choose RG% and RB%. CoastWise applies these to each "
        "subcatchment’s feasible maximum RG and RB, highlighting "
        "collective action across the watershed.\n"
    )

    st.info(

        "• **Path B – Known Budget → Feasible Uptake:** Enter a budget and unit costs. Pin RG% or RB%. CoastWise solves "
        "the other percent so the plan fits the budget as closely as integer caps allow, then reports counts, spend, and treated area.\n\n"
    )

    path_choice = st.selectbox(
        "Select a planning path:",
        ["— Select a planning path —", "Path A — Percent Uptake", "Path B — Budget to Percent"],
        index=0
    )

    if path_choice == "— Select a planning path —":
        st.stop()  

    # Load per-subcatchment maxima from raster_df
    caps_RG, caps_RB = load_caps_from_df(raster_df)
    CapRG_total = int(sum(caps_RG.values()))
    CapRB_total = int(sum(caps_RB.values()))



    # Build plan_base from the selected path
    plan_base = None

    if "Path A" in path_choice:
        st.markdown("### Path A — Percent Uptake (whole watershed)")

        # Unit costs (shared)
        c1, c2 = st.columns(2)
        with c1:
            unit_cost_rg = st.number_input("Rain Garden cost ($/unit)", min_value=0.0, value=500.0, step=25.0)
        with c2:
            unit_cost_rb = st.number_input("Rain Barrel cost ($/unit)", min_value=0.0, value=150.0, step=5.0)

        a1, a2 = st.columns(2)
        with a1:
            pct_rg = st.slider("RG uptake (%)", 0, 100, 20, step=1,
                            help="Applied uniformly to each subcatchment’s RG cap")
        with a2:
            pct_rb = st.slider("RB uptake (%)", 0, 100, 30, step=1,
                            help="Applied uniformly to each subcatchment’s RB cap")

        plan_base = realize_percent_uptake(pct_rg, pct_rb, caps_RG, caps_RB)
        summary = summarize_plan(plan_base, unit_cost_rg, unit_cost_rb)
        st.success(
            f"RG={summary['rg']} | RB={summary['rb']}               "
            f"   Estimated Cost: ${summary['spent']:,.0f}"
        )

    elif "Path B" in path_choice:
        st.markdown("### Path B — Known Budget → Feasible Uptake")

        st.info(
            "Path B starts from your **budget** and **unit costs**. You pick **one percent to pin** (RG or RB), "
            "which CoastWise applies across **every subcatchment’s maximum** for that LID type. Then CoastWise solves "
            "the **other percent** so the **total cost fits your budget**.\n\n"
            "**Example (Pin RG = 30%)**: CoastWise applies 30% of each subcatchment’s RG max, totals that cost, then finds the RB% "
            "that best uses the remaining budget.\n"
        )

        # Unit costs (shared)
        c1, c2 = st.columns(2)
        with c1:
            unit_cost_rg = st.number_input("Rain Garden cost ($/unit)", min_value=0.0, value=500.0, step=25.0)
        with c2:
            unit_cost_rb = st.number_input("Rain Barrel cost ($/unit)", min_value=0.0, value=150.0, step=5.0)

        # ---- Budget + pinned percent controls ----
        b1, b2, b3 = st.columns([1,1,2])
        with b1:
            budget_total = st.number_input(
                "Total budget ($)",
                min_value=0.0, value=500_000.0, step=1_000.0,
                help="Target total spend for this plan."
            )
        with b2:
            pin_type_choice = st.radio(
                "Pin which percent?",
                ["RG", "RB"], horizontal=True,
                help="Apply this percent uniformly across all subcatchments for the selected LID type."
            )
        with b3:
            pin_value = st.slider(
                f"Pinned {pin_type_choice} uptake (%)",
                0, 100, 30, step=1,
                help=f"Percent of each subcatchment’s maximum {pin_type_choice} capacity."
            )

        # ---- Solve the other percent to match the budget (as closely as caps/integers allow) ----
        pRG_solved, pRB_solved = compute_percent_pair_for_budget(
            B=budget_total, cRG=unit_cost_rg, cRB=unit_cost_rb,
            CapRG_total=CapRG_total, CapRB_total=CapRB_total,
            pin_type=pin_type_choice, pin_value_percent=pin_value
        )

        # ---- Realize plan as discrete counts per subcatchment ----
        plan_base = realize_percent_uptake(pRG_solved, pRB_solved, caps_RG, caps_RB)

        # Per-type counts and costs
        total_rg = sum(v["rain_gardens"] for v in plan_base.values())
        total_rb = sum(v["rain_barrels"] for v in plan_base.values())
        est_cost_rg = unit_cost_rg * total_rg
        est_cost_rb = unit_cost_rb * total_rb
        est_cost_total = est_cost_rg + est_cost_rb

        # Treated area (using your constants: 400 ft² per RG; 300 ft² per RB)
        treated_ft2 = 400.0 * total_rg + 300.0 * total_rb

        # ---- Row 1: Budget + per-type estimated costs ----
        s1c2, s1c3, s1c4 = st.columns([1,1,1])
        with s1c2:
            st.metric("Estimated RG Cost", f"${est_cost_rg:,.0f}")
        with s1c3:
            st.metric("Estimated RB Cost", f"${est_cost_rb:,.0f}")
        with s1c4:
            st.metric("Estimated Total Cost", f"${est_cost_total:,.0f}")

        # ---- Row 2: Pinned + Solved percents (plus optional counts/treated area) ----
        s2c1, s2c2, s2c3 = st.columns([1,1,1])
        with s2c1:
            st.metric(f"Pinned {pin_type_choice} (%)",
                    f"{pin_value:.1f}%")
        with s2c2:
            # Show the counterpart label properly
            solved_label = "RB (%)" if pin_type_choice == "RG" else "RG (%)"
            solved_value = pRB_solved if pin_type_choice == "RG" else pRG_solved
            st.metric(f"Solved {solved_label}", f"{solved_value:.1f}%")
        with s2c3:
            st.metric("Counts (RG | RB)", f"{int(total_rg)} | {int(total_rb)}")

    if plan_base is not None:
        plan_all      = plan_base
        plan_upstream = restrict_plan_to_group(plan_base, UPSTREAM_LIST)
        plan_downstr  = restrict_plan_to_group(plan_base, DOWNSTREAM_LIST)
        plan_highro   = restrict_plan_to_group(plan_base, HIGHRUNOFF_LIST)

        st.markdown("### LID Placement — Compare Different Focus Areas")
        row1 = st.columns(2, gap="large")
        with row1[0]:
            render_focus_placement_map_log(
                plan_all,
                "All subcatchments",
                WS_SHP_PATH,
                "fa_place_all"
            )
        with row1[1]:
            render_focus_placement_map_log(
                plan_upstream,
                "Upstream",
                WS_SHP_PATH,
                "fa_place_up"
            )

        row2 = st.columns(2, gap="large")
        with row2[0]:
            render_focus_placement_map_log(
                plan_downstr,
                "Downstream/outlet",
                WS_SHP_PATH,
                "fa_place_dn"
            )
        with row2[1]:
            render_focus_placement_map_log(
                plan_highro,
                "Highest runoff",
                WS_SHP_PATH,
                "fa_place_hi"
            )

    if plan_base is not None:
        st.markdown("### Run Focus Area Scenarios")

        def run_focus_pair(prefix_key: str, plan: Dict[str, Dict[str,int]],
                        rain_lines, tide_lines, template_inp, minutes_15, sim_date, rain_curve):
            lid_lines = generate_lid_usage_lines(plan, raster_df)  # your existing generator
            info_off = run_swmm_scenario(f"{prefix}{prefix_key}_nogate", rain_lines, tide_lines, lid_lines, "NO",
                                        template_path=template_inp, event_window_mode="rain+2h", event_buffer_hours=2.0,
                                        rain_minutes=minutes_15, rain_curve_in=rain_curve, sim_start_str=sim_date)
            info_on  = run_swmm_scenario(f"{prefix}{prefix_key}_gate",   rain_lines, tide_lines, lid_lines, "YES",
                                        template_path=template_inp, event_window_mode="rain+2h", event_buffer_hours=2.0,
                                        rain_minutes=minutes_15, rain_curve_in=rain_curve, sim_start_str=sim_date)
            remember_scenario(f"{prefix_key}_nogate", info_off["df_runoff"], info_off["node_cuft_event"], prefix)
            remember_scenario(f"{prefix_key}_gate",   info_on["df_runoff"],  info_on["node_cuft_event"],  prefix)
            return info_off, info_on

        rain_variant = st.session_state.get("rain_variant", "current")
        use_cur = (rain_variant == "current")
        rain_lines = st.session_state["rain_lines_cur"] if use_cur else st.session_state["rain_lines_fut"]
        rain_curve = st.session_state["rain_sim_curve_current_in"] if use_cur else st.session_state["rain_sim_curve_future_in"]

        tag = "current" if use_cur else "future"
        c_run1, c_run2 = st.columns(2)
        with c_run1:
            if st.button("Run ALL subcatchments (Gate OFF/ON)"):
                run_focus_pair(f"focus_all_{tag}",      plan_all,      rain_lines, tide_lines, template_inp, minutes_15, simulation_date, rain_curve)
                st.success("All subcatchments scenarios complete.")
            if st.button("Run UPSTREAM (Gate OFF/ON)"):
                run_focus_pair(f"focus_upstream_{tag}", plan_upstream, rain_lines, tide_lines, template_inp, minutes_15, simulation_date, rain_curve)
                st.success("Upstream scenarios complete.")
        with c_run2:
            if st.button("Run DOWNSTREAM/OUTLET (Gate OFF/ON)"):
                run_focus_pair(f"focus_downstream_{tag}", plan_downstr,  rain_lines, tide_lines, template_inp, minutes_15, simulation_date, rain_curve)
                st.success("Downstream/outlet scenarios complete.")
            if st.button("Run HIGHEST RUNOFF (Gate OFF/ON)"):
                run_focus_pair(f"focus_highrunoff_{tag}", plan_highro,   rain_lines, tide_lines, template_inp, minutes_15, simulation_date, rain_curve)

    st.subheader("Focus Area Comparison Maps")

    set_tag = "current" if st.session_state.get("rain_variant", "current") == "current" else "future"
    FOCUS_ROWS = [
        ("All subcatchments", f"focus_all_{set_tag}_nogate",      f"focus_all_{set_tag}_gate"),
        ("Upstream",          f"focus_upstream_{set_tag}_nogate", f"focus_upstream_{set_tag}_gate"),
        ("Downstream/outlet", f"focus_downstream_{set_tag}_nogate", f"focus_downstream_{set_tag}_gate"),
        ("Highest runoff",    f"focus_highrunoff_{set_tag}_nogate", f"focus_highrunoff_{set_tag}_gate"),
    ]

    recs = [recall_scenario(name, prefix) for _, n_off, n_on in FOCUS_ROWS for name in (n_off, n_on)]
    dfs_available = [r["df"] for r in recs if r and isinstance(r.get("df"), pd.DataFrame) and not r["df"].empty]
    legend_range = _global_runoff_range_across(dfs_available, st.session_state["unit_ui"], WS_SHP_PATH) if dfs_available else (0.0, 1.0)

    for (label, name_off, name_on) in FOCUS_ROWS:
        st.markdown(f"#### {label}")
        cols = st.columns(2, gap="medium")

        with cols[0]:
            rec_off = recall_scenario(name_off, prefix)
            if not rec_off:
                st.info("Not run yet.")
            else:
                render_total_runoff_map_single(
                    df_in_inches=rec_off["df"], title=f"{label} – NO gate",
                    nodes_post5h_dict=rec_off.get("nodes", {}) or {},
                    unit_ui=st.session_state["unit_ui"],
                    ws_shp_path=WS_SHP_PATH, pipe_shp_path=PIPE_SHP_PATH, node_shp_path=NODE_SHP_PATH,
                    node_name_field_hint="NAME", legend_range=legend_range,
                    widget_key=f"focus_cmp_{label}_{set_tag}_nogate", show_title=False
                )

        with cols[1]:
            rec_on = recall_scenario(name_on, prefix)
            if not rec_on:
                st.info("Not run yet.")
            else:
                render_total_runoff_map_single(
                    df_in_inches=rec_on["df"], title=f"{label} – gate ON",
                    nodes_post5h_dict=rec_on.get("nodes", {}) or {},
                    unit_ui=st.session_state["unit_ui"],
                    ws_shp_path=WS_SHP_PATH, pipe_shp_path=PIPE_SHP_PATH, node_shp_path=NODE_SHP_PATH,
                    node_name_field_hint="NAME", legend_range=legend_range,
                    widget_key=f"focus_cmp_{label}_{set_tag}_gate", show_title=False
                )

    st.markdown(_legend_html("in" if st.session_state["unit_ui"] == "U.S. Customary" else "cm",
                            legend_range[0], legend_range[1]), unsafe_allow_html=True)

    # --- Summary grouped bar chart across the four focus areas ---
    def build_focus_summary_df(plans_dict, unit_cost_rg, unit_cost_rb):
        rows = []
        for label, plan in plans_dict.items():
            s = summarize_plan(plan, unit_cost_rg, unit_cost_rb)
            rows.append({
                "Focus Area": label,
                "RG_count": s["rg"],
                "RB_count": s["rb"],
                "TotalCost_K": s["spent"] / 1_000.0,      # $ thousands
                "Treated_kft2": s["treated_ft2"] / 1_000.0  # thousand ft²
            })
        return pd.DataFrame(rows)

    if plan_base is not None:
        plans_dict = {
            "All":       plan_all,
            "Upstream":  plan_upstream,
            "Downstream":plan_downstr,
            "High runoff": plan_highro,
        }
        df_sum = build_focus_summary_df(plans_dict, unit_cost_rg, unit_cost_rb)

        # long format for Altair
        df_long_counts = df_sum.melt(id_vars=["Focus Area"], value_vars=["RG_count","RB_count"],
                                    var_name="Metric", value_name="Value")
        df_long_other  = df_sum.melt(id_vars=["Focus Area"], value_vars=["TotalCost_K","Treated_kft2"],
                                    var_name="Metric", value_name="Value")
        df_long = pd.concat([df_long_counts, df_long_other], ignore_index=True)

        # Nice labels
        metric_labels = {
            "RG_count": "RG count",
            "RB_count": "RB count",
            "TotalCost_K": "Total cost ($K)",
            "Treated_kft2": "Treated area (kft²)"
        }
        df_long["Metric"] = df_long["Metric"].map(metric_labels)

        # Color palette (metrics legend)
        metric_colors = {"RG count": "#228B22", "RB count": "#1f77b4", "Total cost ($K)": "#ff7f0e", "Treated area (kft²)": "#9467bd"}

        st.markdown("### Summary — Counts, Cost, and Treated Area by Focus Area")
        try:
            import altair as alt
            chart = (
                alt.Chart(df_long)
                .mark_bar()
                .encode(
                    x=alt.X("Focus Area:N", title=None),
                    y=alt.Y("Value:Q", title="Value (counts · $K · kft²)", axis=alt.Axis(grid=True)),
                    color=alt.Color("Metric:N", scale=alt.Scale(domain=list(metric_colors.keys()), range=list(metric_colors.values())),
                                    legend=alt.Legend(title="Metrics")),
                    column=alt.Column("Focus Area:N", title=None, header=alt.Header(labelFontSize=14)),  # optional small multiples
                    tooltip=[
                        alt.Tooltip("Focus Area:N"),
                        alt.Tooltip("Metric:N"),
                        alt.Tooltip("Value:Q", format=",.2f")
                    ],
                )
                .properties(height=280)
            )
            # Alternative: grouped bars without small multiples (comment line above and use a single x with order)
            chart = chart.configure_axis(labelFontSize=12, titleFontSize=13).configure_legend(titleFontSize=13, labelFontSize=12)
            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Summary chart failed: {e}")



    def _gather_scenario_volumes() -> tuple[pd.DataFrame, str]:
        ACF_TO_FT3 = 43560.0
        FT3_TO_M3  = 0.0283168
        to_m3 = (st.session_state.get("unit_ui") == "Metric (SI)")
        unit_label = "m³" if to_m3 else "ft³"

        friendly_map = {
            # --- Baseline entries (keep) ---
            "Baseline (No Tide Gate) – Current":  "baseline_nogate_current",
            "Baseline + Tide Gate – Current":     "baseline_gate_current",
            "Baseline (No Tide Gate) – +20%":     "baseline_nogate_future",
            "Baseline + Tide Gate – +20%":        "baseline_gate_future",

            # --- NEW: pattern scenarios — Current rainfall ---
            "Even Pattern (No Tide Gate) – Current":       "pattern_even_current_nogate",
            "Even Pattern + Tide Gate – Current":          "pattern_even_current_gate",
            "Upstream Pattern (No Tide Gate) – Current":   "pattern_upstream_current_nogate",
            "Upstream Pattern + Tide Gate – Current":      "pattern_upstream_current_gate",
            "Downstream Pattern (No Tide Gate) – Current": "pattern_downstream_current_nogate",
            "Downstream Pattern + Tide Gate – Current":    "pattern_downstream_current_gate",
            "High‑Runoff Pattern (No Tide Gate) – Current": "pattern_highrunoff_current_nogate",
            "High‑Runoff Pattern + Tide Gate – Current":    "pattern_highrunoff_current_gate",

            # --- NEW: pattern scenarios — +20% rainfall ---
            "Even Pattern (No Tide Gate) – +20%":       "pattern_even_future_nogate",
            "Even Pattern + Tide Gate – +20%":          "pattern_even_future_gate",
            "Upstream Pattern (No Tide Gate) – +20%":   "pattern_upstream_future_nogate",
            "Upstream Pattern + Tide Gate – +20%":      "pattern_upstream_future_gate",
            "Downstream Pattern (No Tide Gate) – +20%": "pattern_downstream_future_nogate",
            "Downstream Pattern + Tide Gate – +20%":    "pattern_downstream_future_gate",
            "High‑Runoff Pattern (No Tide Gate) – +20%": "pattern_highrunoff_future_nogate",
            "High‑Runoff Pattern + Tide Gate – +20%":    "pattern_highrunoff_future_gate",
        }

        rows = []
        rpts = st.session_state.get("rpts", {})
        prefix = st.session_state["scenario_prefix"]

        def acft_to_display(v_acft: float) -> float:
            v_ft3 = v_acft * ACF_TO_FT3
            return v_ft3 * FT3_TO_M3 if to_m3 else v_ft3

        for disp, canon in friendly_map.items():
            key = f"{prefix}{canon}"

            flood_display = float(st.session_state.get(f"{key}_event_total_flood", 0.0))

            rpt_text = rpts.get(key, "")
            df_ir = extract_infiltration_and_runoff_from_text(rpt_text) if rpt_text else pd.DataFrame(columns=["label","volume_acft"])
            infil_acft  = float(df_ir.loc[df_ir["label"]=="infiltration_loss", "volume_acft"].squeeze() or 0.0)
            runoff_acft = float(df_ir.loc[df_ir["label"]=="surface_runoff",   "volume_acft"].squeeze() or 0.0)
            rows.append({
                "Scenario": disp,
                "Flooding":       flood_display,
                "Infiltration":   acft_to_display(infil_acft),
                "Surface Runoff": acft_to_display(runoff_acft),
            })

        df = pd.DataFrame(rows).set_index("Scenario").round(0).astype(int)
        return df, unit_label


    if st.button("Watershed Volumes: Flooding / Infiltration / Surface Runoff" , key=f"{prefix}btn_volumes"):
        df_vol, unit_lbl = _gather_scenario_volumes()
        if df_vol.empty:
            st.info("Run scenarios first.")
        else:
            st.subheader(f"Scenario Volumes ({unit_lbl})")

            def scenario_group(name: str) -> str:
                prefix = name.split("–", 1)[0].strip()
                prefix_to_group = {
                    "Baseline (No Tide Gate)":  "Baseline (No Gate)",
                    "Baseline + Tide Gate":     "Baseline (With Gate)",
                    "LID (No Tide Gate)":       "LID (No Gate)",
                    "LID + Tide Gate":          "LID (With Gate)",
                    "Max LID (No Tide Gate)":   "Max LID (No Gate)",
                    "Max LID + Tide Gate":      "Max LID (With Gate)",
                }
                return prefix_to_group.get(prefix, "Other")

            group_domain = [
                "Baseline (No Gate)", "Baseline (With Gate)",
                "LID (No Gate)", "LID (With Gate)",
                "Max LID (No Gate)", "Max LID (With Gate)"
            ]
            group_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

            def bar_chart(df_vals: pd.DataFrame, value_col: str, title_text: str | None):
                df = df_vals.reset_index().rename(columns={"index": "Scenario", value_col: unit_lbl})
                df["Group"] = df["Scenario"].map(scenario_group)
                order = df.sort_values(unit_lbl, ascending=False)["Scenario"].tolist()
                height_px = max(44 * len(order), 400)
                chart = (
                    alt.Chart(df)
                    .mark_bar(size=25)
                    .encode(
                        x=alt.X(f"{unit_lbl}:Q", sort='-x', title=unit_lbl),
                        y=alt.Y("Scenario:N", sort=order, title=None,
                                axis=alt.Axis(labelFontSize=18, titleFontSize=22, labelLimit=1000)),
                        color=alt.Color("Group:N",
                                        scale=alt.Scale(domain=group_domain, range=group_colors),
                                        legend=alt.Legend(title="Scenario Group")),
                        tooltip=["Scenario", f"{unit_lbl}:Q", "Group:N"]
                    )
                    .properties(height=height_px)
                    .configure_axis(labelFontSize=18, titleFontSize=22)
                    .configure_view(strokeWidth=0)
                    .configure_axis(labelColor="black", titleColor="black")
                    .configure_title(color="black", fontSize=20, fontWeight="bold")
                    .configure_legend(labelColor="black", titleColor="black", titleFontSize=16, labelFontSize=14)
                    .configure_view(strokeWidth=0)
                )
                if isinstance(title_text, str) and title_text != "":
                    chart = chart.properties(title=title_text)
                st.altair_chart(chart, use_container_width=True, key=f"{prefix}_vol_flood_chart")

            st.markdown("**Flooding**")
            s_flood  = df_vol["Flooding"].sort_values(ascending=False)
            bar_chart(s_flood.to_frame(), "Flooding", None)


            st.markdown("**Infiltration**")
            gate_mask = df_vol.index.str.contains(r"\+ Tide Gate")
            s_infil_gate = df_vol.loc[gate_mask, "Infiltration"].sort_values(ascending=False)

            df_i = s_infil_gate.to_frame().reset_index().rename(columns={"index":"Scenario","Infiltration":unit_lbl})
            order_i = df_i.sort_values(unit_lbl, ascending=False)["Scenario"].tolist()
            height_i = max(44 * len(order_i), 400)

            chart_i = (
                alt.Chart(df_i)
                .mark_bar(size=25)
                .encode(
                    x=alt.X(f"{unit_lbl}:Q", sort='-x', title=unit_lbl),
                    y=alt.Y("Scenario:N", sort=order_i, title=None,
                            axis=alt.Axis(labelFontSize=18, titleFontSize=22, labelLimit=1000)),

                    color=alt.Color("Scenario:N", legend=None)
                )
                .properties(height=height_i)
                .configure_axis(labelFontSize=18, titleFontSize=22)
                .configure_view(strokeWidth=0)
                .configure_axis(labelColor="black", titleColor="black")
                .configure_title(color="black", fontSize=20, fontWeight="bold")
                .configure_legend(labelColor="black", titleColor="black", titleFontSize=16, labelFontSize=14)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart_i, use_container_width=True, key=f"{prefix}_vol_infil_chart")

            st.markdown("**Surface Runoff**")
            s_runoff_gate = df_vol.loc[gate_mask, "Surface Runoff"].sort_values(ascending=False)

            df_r = s_runoff_gate.to_frame().reset_index().rename(columns={"index":"Scenario","Surface Runoff":unit_lbl})
            order_r = df_r.sort_values(unit_lbl, ascending=False)["Scenario"].tolist()
            height_r = max(44 * len(order_r), 400)

            chart_r = (
                alt.Chart(df_r)
                .mark_bar(size=25)
                .encode(
                    x=alt.X(f"{unit_lbl}:Q", sort='-x', title=unit_lbl),
                    y=alt.Y("Scenario:N", sort=order_r, title=None,
                            axis=alt.Axis(labelFontSize=18, titleFontSize=22, labelLimit=1000)),

                    color=alt.Color("Scenario:N", legend=None)
                )
                .properties(height=height_r)
                .configure_axis(labelFontSize=18, titleFontSize=22)
                .configure_view(strokeWidth=0)
                .configure_axis(labelColor="black", titleColor="black")
                .configure_title(color="black", fontSize=20, fontWeight="bold")
                .configure_legend(labelColor="black", titleColor="black", titleFontSize=16, labelFontSize=14)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart_r, use_container_width=True, key=f"{prefix}_vol_runoff_chart")


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
                df_rain.to_excel(writer, sheet_name="Rainfall Event", index=False)

                if len(tide_ts) > 0:
                    t_t = [(sim_start + timedelta(minutes=int(m))).strftime("%m/%d/%Y %H:%M")
                        for m in tide_minutes[:len(tide_ts)]]
                    df_tide = pd.DataFrame({"Timestamp": t_t, f"Tide ({tide_disp_unit})": tide_ts[:len(t_t)]})
                else:
                    df_tide = pd.DataFrame(columns=["Timestamp", f"Tide ({tide_disp_unit})"])
                df_tide.to_excel(writer, sheet_name="Tide Event", index=False)

                lid_cfg = st.session_state.get(f"{prefix}user_lid_config", {})
                if lid_cfg:
                    rows = [{"Subcatchment": sub,
                            "Selected Rain Gardens": cfg.get("rain_gardens", 0),
                            "Selected Rain Barrels":  cfg.get("rain_barrels", 0)}
                            for sub, cfg in lid_cfg.items()]
                    df_user_lid = pd.DataFrame(rows)
                else:
                    df_user_lid = pd.DataFrame(columns=["Subcatchment","Selected Rain Gardens","Selected Rain Barrels"])
                df_user_lid.to_excel(writer, sheet_name="User LID Selections", index=False)


                out = df_vol.reset_index()
                out.columns = ["Scenario",
                            f"Flooding ({unit_lbl})",
                            f"Infiltration ({unit_lbl})",
                            f"Surface Runoff ({unit_lbl})"]
                out.to_excel(writer, sheet_name="Scenario Volumes", index=False)

            st.download_button(
                label=f"Download Results",
                data=excel_output.getvalue(),
                file_name="CoastWise_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

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
