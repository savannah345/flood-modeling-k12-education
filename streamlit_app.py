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

@contextmanager
def pushd(path):
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)

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
            get_point_radius=6,
            pointRadiusMinPixels=6,
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
    view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=13.5)

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
                           get_size=12, get_color=[0,0,0], get_alignment_baseline="'center'")

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
          Same scale for all 12 maps
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

    centroid = gdf.geometry.union_all().centroid
    view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=14.75)

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
        get_size=12, get_color=[0, 0, 0], get_alignment_baseline="'center'"
    )

    tooltip = {
        "html": "<b>{NAME}</b><br/>Total runoff: {Total_R} " + unit_r,
        "style": {"backgroundColor": "white", "color": "black",
                  "fontFamily": "Inter, Arial, Helvetica, sans-serif", "fontSize": "12px"},
    }

    deck_obj = pdk.Deck(
        layers=[poly_layer, text_layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style="light",
        tooltip=tooltip,
    )
    map_html = deck_obj.to_html(as_string=True)

    cmap = mpl.colormaps.get_cmap("Blues")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    c0 = [int(v * 255) for v in cmap(norm(vmin))[:3]]
    c1 = [int(v * 255) for v in cmap(norm(vmax))[:3]]
    legend_html = f"""
    <div style="display:flex; justify-content:center; margin-top:6px;">
      <div style="min-width:260px; max-width:640px; width:60%;">
        <div style="text-align:center; font-size:13px;"><b>Runoff Legend ({unit_r})</b></div>
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

    st.title("CoastWise: Watershed Design Toolkit")
    st.markdown('<a href="https://github.com/savannah345/flood-modeling-k12-education/blob/main/CoastWise_Tutorial.docx" target="_blank">Tutorial</a>', unsafe_allow_html=True)

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
    st.success("Settings applied.")

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
        if st.button("🚪 Logout"):
            try: shutil.rmtree(st.session_state.temp_dir)
            except Exception: pass
            st.session_state.clear()
            st.success("Logged out and cleaned up all files.")
            st.experimental_rerun()
        return

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


    time_hours = np.array(minutes_15, dtype=float) / 60.0
    st.subheader("Rainfall Distribution")
    rain_df = pd.DataFrame({
        "Hour": time_hours,
        "Current": st.session_state["display_rain_curve_current"],
        "Future (+20%)": st.session_state["display_rain_curve_future"],
    }).set_index("Hour")
    st.line_chart(rain_df, height=320, use_container_width=True)
    st.markdown(f"Rainfall units: {rain_disp_unit}")

    st.subheader("Tide Profile")
    tide_df = pd.DataFrame({
        "Hour": time_hours,
        "Tide": st.session_state["display_tide_curve"],
    }).set_index("Hour")
    st.line_chart(tide_df, height=220, use_container_width=True)
    st.markdown(
        f"Source: Real-time tide (last 48 h) {tide_disp_unit}"
        if tide_source == "live"
        else f"Source: Synthetic tide    Units: {tide_disp_unit}    Phase: ({moon_phase})"
    )

    if st.button("Run Baseline Scenario" , key=f"{prefix}btn_run_baseline"):
        try:
            lid_lines = [";"]  

            info1 = run_swmm_scenario(
                f"{prefix}baseline_nogate_current", rain_lines_cur, tide_lines, [";"], "NO",
                template_path=template_inp,
                event_window_mode="rain+2h", event_buffer_hours=2.0,
                rain_minutes=minutes_15,
                rain_curve_in=st.session_state["rain_sim_curve_current_in"],
                sim_start_str=simulation_date,
            )
            info2 = run_swmm_scenario(
                f"{prefix}baseline_gate_current",   rain_lines_cur, tide_lines, [";"], "YES",
                template_path=template_inp,
                event_window_mode="rain+2h", event_buffer_hours=2.0,
                rain_minutes=minutes_15,
                rain_curve_in=st.session_state["rain_sim_curve_current_in"],
                sim_start_str=simulation_date,
            )
            info3 = run_swmm_scenario(
                f"{prefix}baseline_nogate_future",  rain_lines_fut, tide_lines, [";"], "NO",
                template_path=template_inp,
                event_window_mode="rain+2h", event_buffer_hours=2.0,
                rain_minutes=minutes_15,
                rain_curve_in=st.session_state["rain_sim_curve_future_in"],
                sim_start_str=simulation_date,
            )
            info4 = run_swmm_scenario(
                f"{prefix}baseline_gate_future",    rain_lines_fut, tide_lines, [";"], "YES",
                template_path=template_inp,
                event_window_mode="rain+2h", event_buffer_hours=2.0,
                rain_minutes=minutes_15,
                rain_curve_in=st.session_state["rain_sim_curve_future_in"],
                sim_start_str=simulation_date,
            )

            st.session_state.update({
                f"{prefix}df_base_nogate_current": info1["df_runoff"],
                f"{prefix}df_base_gate_current":   info2["df_runoff"],
                f"{prefix}df_base_nogate_future":  info3["df_runoff"],
                f"{prefix}df_base_gate_future":    info4["df_runoff"],
            })

            remember_scenario("baseline_nogate_current", info1["df_runoff"], info1["node_cuft_event"], prefix)
            remember_scenario("baseline_gate_current",   info2["df_runoff"], info2["node_cuft_event"], prefix)
            remember_scenario("baseline_nogate_future",  info3["df_runoff"], info3["node_cuft_event"], prefix)
            remember_scenario("baseline_gate_future",    info4["df_runoff"], info4["node_cuft_event"], prefix)

            df_swmm_now = st.session_state[f"{prefix}df_base_nogate_current"]
            if len(df_swmm_now) == 0:
                st.error("No rows parsed from 'Subcatchment Runoff Summary' in the .rpt")
            else:
                st.success("Baseline scenarios complete.")
                st.session_state[f"{prefix}ran_baseline"] = True
                map_html, legend_html = _build_baseline_map_html(
                    df_swmm_local=df_swmm_now,
                    unit_ui=st.session_state["unit_ui"],
                    ws_shp_path=WS_SHP_PATH
                )
                st.session_state[f"{prefix}baseline_map_html"] = map_html
                st.session_state[f"{prefix}baseline_legend_html"] = legend_html
                st.session_state[f"{prefix}grid_ready"] = True
        except Exception as e:
            st.error(f"Baseline simulation failed: {e}")

    if f"{prefix}baseline_map_html" in st.session_state:
        st.subheader("Watershed Baseline Runoff Map")
        components.v1.html(st.session_state[f"{prefix}baseline_map_html"], height=600, scrolling=False)
        st.markdown(st.session_state[f"{prefix}baseline_legend_html"], unsafe_allow_html=True)

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

    st.subheader("Add LID Features (by % of community application)")
    if f"{prefix}user_lid_config" not in st.session_state:
        st.session_state[f"{prefix}user_lid_config"] = {}

    c1, c2, c3, c4 = st.columns([2,2,2,2])
    with c1:
        pct_rg = st.slider("Rain Gardens uptake (%)", 0, 100, 10, step=1,
                        help="Applied to each subcatchment’s RG max")
    with c2:
        pct_rb = st.slider("Rain Barrels uptake (%)", 0, 100, 10, step=1,
                        help="Applied to each subcatchment’s RB max")
    with c3:
        unit_cost_rg = st.number_input("Rain Garden ($/ea)", min_value=0.0, value=500.0, step=100.0)
    with c4:
        unit_cost_rb = st.number_input("Rain Barrel ($/ea)", min_value=0.0, value=150.0, step=10.0)

    df0 = raster_df[["NAME", "Max_RG_DEM_Considered", "MaxNumber_RB"]].copy()
    df0["Max_RG_DEM_Considered"] = df0["Max_RG_DEM_Considered"].astype(int)
    df0["MaxNumber_RB"]          = df0["MaxNumber_RB"].astype(int)
    df0["Rain Gardens"] = np.rint(df0["Max_RG_DEM_Considered"] * (pct_rg / 100.0)).astype(int)
    df0["Rain Barrels"] = np.rint(df0["MaxNumber_RB"]          * (pct_rb / 100.0)).astype(int)
    df0["Rain Gardens"] = df0["Rain Gardens"].clip(lower=0, upper=df0["Max_RG_DEM_Considered"])
    df0["Rain Barrels"] = df0["Rain Barrels"].clip(lower=0, upper=df0["MaxNumber_RB"])
    df0["Cost ($)"]      = (df0["Rain Gardens"] * unit_cost_rg) + (df0["Rain Barrels"] * unit_cost_rb)
    df0["Cost Display"]  = df0["Cost ($)"].apply(lambda x: f"${x:,.0f}")

    table_init = (
        df0.rename(columns={"Max_RG_DEM_Considered": "RG Max", "MaxNumber_RB": "RB Max"})
        [["NAME", "RG Max", "RB Max", "Rain Gardens", "Rain Barrels", "Cost Display", "Cost ($)"]]
    )

 
    table_init = table_init.sort_values("Cost ($)", ascending=True).reset_index(drop=True)

    st.markdown("Edit Rain Gardens and Rain Barrels numbers after % application if you want to change any subcatchments individually. Do not exceed the RG Max or RB Max, respectfully.")
    edited_display = st.data_editor(
        table_init.drop(columns=["Cost ($)"]), 
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "NAME": st.column_config.TextColumn("Subcatchment", disabled=True),
            "RG Max": st.column_config.NumberColumn("RG Max", disabled=True, format="%d"),
            "RB Max": st.column_config.NumberColumn("RB Max", disabled=True, format="%d"),
            "Rain Gardens": st.column_config.NumberColumn(min_value=0, step=1, format="%d"),
            "Rain Barrels": st.column_config.NumberColumn(min_value=0, step=1, format="%d"),
            "Cost Display": st.column_config.TextColumn("Cost ($)", disabled=True),
        },
        key=f"{prefix}lid_percent_editor",
    )

    edited = edited_display.copy()
    edited["RG Max"] = edited["RG Max"].astype(int)
    edited["RB Max"] = edited["RB Max"].astype(int)

    rg_before = edited["Rain Gardens"].copy()
    rb_before = edited["Rain Barrels"].copy()

    edited["Rain Gardens"] = (
        pd.to_numeric(edited["Rain Gardens"], errors="coerce").fillna(0)
        .round().astype(int).clip(lower=0, upper=edited["RG Max"])
    )
    edited["Rain Barrels"] = (
        pd.to_numeric(edited["Rain Barrels"], errors="coerce").fillna(0)
        .round().astype(int).clip(lower=0, upper=edited["RB Max"])
    )
    clipped = ((rg_before != edited["Rain Gardens"]) | (rb_before != edited["Rain Barrels"])).sum()
    if clipped > 0:
        st.warning(f"{clipped} row(s) exceeded the max and were clipped.")

    edited["Cost ($)"]     = (edited["Rain Gardens"] * unit_cost_rg) + (edited["Rain Barrels"] * unit_cost_rb)
    edited["Cost Display"] = edited["Cost ($)"].apply(lambda x: f"${x:,.0f}")

    edited = edited[["NAME", "RG Max", "RB Max", "Rain Gardens", "Rain Barrels", "Cost Display", "Cost ($)"]]
    edited = edited.sort_values("Cost ($)", ascending=True).reset_index(drop=True)

    c_budget1, c_budget2, c_budget3 = st.columns([2,2,2])
    with c_budget1:
        budget_total = st.number_input("Total budget ($)", min_value=0.0, value=500000.0, step=1000.0)
    with c_budget2:
        total_cost = float(edited["Cost ($)"].sum())
        st.metric("Total cost (current plan)", f"${total_cost:,.0f}")
    with c_budget3:
        remaining = max(budget_total - total_cost, 0.0)
        st.metric("Remaining", f"${remaining:,.0f}")

    if budget_total > 0:
        pct = total_cost / budget_total
        pct_clamped = min(pct, 1.0)
        if pct <= 1.0:
            prog_text = f"{pct*100:.1f}% of ${budget_total:,.0f} budget"
        else:
            prog_text = f"100%+  (over by ${total_cost - budget_total:,.0f})"
        st.progress(pct_clamped, text=prog_text)

    if budget_total > 0 and total_cost > budget_total:
        st.error(f"Over budget by ${total_cost - budget_total:,.0f}. Reduce counts or increase the budget.")

    try:
        import altair as alt
        chart_df = edited.sort_values("Cost ($)", ascending=False)  
        if not chart_df.empty:
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("Cost ($):Q", title="Cost ($)"),
                    y=alt.Y("NAME:N", sort='-x', title="Subcatchment"),
                    tooltip=[
                        alt.Tooltip("NAME:N", title="Subcatchment"),
                        alt.Tooltip("Cost ($):Q", title="Cost ($)", format="$.0f"),
                        alt.Tooltip("Rain Gardens:Q", title="Rain Gardens"),
                        alt.Tooltip("Rain Barrels:Q", title="Rain Barrels"),
                        alt.Tooltip("RG Max:Q", title="RG Max"),
                        alt.Tooltip("RB Max:Q", title="RB Max"),
                    ],
                )
                .configure_axisY(labelFontSize=15, labelOverlap=False, labelSeparation=8)
                .properties(height=max(320, 20 * len(chart_df)))
            )
            st.altair_chart(chart, use_container_width=True, key=f"{prefix}_vol_flood_chart")
    except Exception as e:
        st.error(f"Chart rendering failed: {e}")


    with st.container(key=f"{prefix}sec_apply"):
        if st.button("Apply table to LID selections", key=f"{prefix}btn_apply_lid"):
            st.session_state[f"{prefix}user_lid_config"] = {
                row["NAME"]: {
                    "rain_gardens": int(row["Rain Gardens"]),
                    "rain_barrels": int(row["Rain Barrels"]),
                }
                for _, row in edited.iterrows()
                if int(row["Rain Gardens"]) > 0 or int(row["Rain Barrels"]) > 0
            }
            st.success("Applied. Use **Run Custom LID Scenario** to simulate.")
            # Keep baseline scenarios present after this button rerun
            _rehydrate_baseline_into_store(prefix)
            _ensure_baselines_visible(prefix)
            for nm in ["baseline_nogate_current","baseline_gate_current",
                    "baseline_nogate_future","baseline_gate_future"]:
                _ensure_scenario_loaded(nm, prefix)
            # harmless, but keeps other logic happy if it ever checks this
            st.session_state[f"{prefix}grid_ready"] = True

    with st.container(key=f"{prefix}sec_custom"):
        if st.button("Run Custom LID Scenario", key=f"{prefix}btn_run_custom"):
            existing_store = _sc_store(prefix).copy()
            lid_cfg = st.session_state.get(f"{prefix}user_lid_config", {})
            if not lid_cfg or all((v.get("rain_gardens",0)==0 and v.get("rain_barrels",0)==0) for v in lid_cfg.values()):
                st.warning("No LIDs selected.")
            else:
                try:
                    lid_lines = generate_lid_usage_lines(lid_cfg, raster_df)
                    info1 = run_swmm_scenario(
                        f"{prefix}lid_nogate_current",
                        st.session_state["rain_lines_cur"],
                        st.session_state["tide_lines"],
                        lid_lines,
                        "NO",
                        template_path=template_inp,
                        event_window_mode="rain+2h",
                        event_buffer_hours=2.0,
                        rain_minutes=minutes_15,
                        rain_curve_in=st.session_state["rain_sim_curve_current_in"],
                        sim_start_str=simulation_date,
                    )
                    info2 = run_swmm_scenario(
                        f"{prefix}lid_gate_current",
                        st.session_state["rain_lines_cur"],
                        st.session_state["tide_lines"],
                        lid_lines,
                        "YES",
                        template_path=template_inp,
                        event_window_mode="rain+2h",
                        event_buffer_hours=2.0,
                        rain_minutes=minutes_15,
                        rain_curve_in=st.session_state["rain_sim_curve_current_in"],
                        sim_start_str=simulation_date,
                    )
                    info3 = run_swmm_scenario(
                        f"{prefix}lid_nogate_future",
                        st.session_state["rain_lines_fut"],
                        st.session_state["tide_lines"],
                        lid_lines,
                        "NO",
                        template_path=template_inp,
                        event_window_mode="rain+2h",
                        event_buffer_hours=2.0,
                        rain_minutes=minutes_15,
                        rain_curve_in=st.session_state["rain_sim_curve_future_in"],
                        sim_start_str=simulation_date,
                    )
                    info4 = run_swmm_scenario(
                        f"{prefix}lid_gate_future",
                        st.session_state["rain_lines_fut"],
                        st.session_state["tide_lines"],
                        lid_lines,
                        "YES",
                        template_path=template_inp,
                        event_window_mode="rain+2h",
                        event_buffer_hours=2.0,
                        rain_minutes=minutes_15,
                        rain_curve_in=st.session_state["rain_sim_curve_future_in"],
                        sim_start_str=simulation_date,
                    )

                    st.session_state.update({
                        f"{prefix}df_lid_nogate_current": info1["df_runoff"],
                        f"{prefix}df_lid_gate_current":   info2["df_runoff"],
                        f"{prefix}df_lid_nogate_future":  info3["df_runoff"],
                        f"{prefix}df_lid_gate_future":    info4["df_runoff"],
                    })

                    remember_scenario("lid_nogate_current", info1["df_runoff"], info1["node_cuft_event"], prefix)
                    remember_scenario("lid_gate_current",   info2["df_runoff"], info2["node_cuft_event"], prefix)
                    remember_scenario("lid_nogate_future",  info3["df_runoff"], info3["node_cuft_event"], prefix)
                    remember_scenario("lid_gate_future",    info4["df_runoff"], info4["node_cuft_event"], prefix)

                    st.success("Custom LID scenarios complete.")
                    st.session_state[f"{prefix}ran_custom"] = True

                    store = _sc_store(prefix)
                    for k, v in existing_store.items():
                        store.setdefault(k, v)

                    _rehydrate_baseline_into_store(prefix)
                    _ensure_baselines_visible(prefix)
                    st.session_state[f"{prefix}grid_ready"] = True
                except Exception as e:
                    st.error(f"LID simulation failed: {e}")

    with st.container(key=f"{prefix}sec_max"):
        if st.button("Run Max LID Scenario", key=f"{prefix}btn_run_max"): 
            existing_store = _sc_store(prefix).copy()
            lid_cfg = {row["NAME"]: {"rain_gardens": int(row["Max_RG_DEM_Considered"]),
                                    "rain_barrels": int(row["MaxNumber_RB"])}
                    for _, row in raster_df.iterrows()}
            try:
                lid_lines = generate_lid_usage_lines(lid_cfg, raster_df)

                info1 = run_swmm_scenario(
                    f"{prefix}lid_max_nogate_current",
                    st.session_state["rain_lines_cur"],
                    st.session_state["tide_lines"],
                    lid_lines,
                    "NO",
                    template_path=template_inp,
                    event_window_mode="rain+2h",
                    event_buffer_hours=2.0,
                    rain_minutes=minutes_15,
                    rain_curve_in=st.session_state["rain_sim_curve_current_in"],
                    sim_start_str=simulation_date,
                )
                info2 = run_swmm_scenario(
                    f"{prefix}lid_max_gate_current",
                    st.session_state["rain_lines_cur"],
                    st.session_state["tide_lines"],
                    lid_lines,
                    "YES",
                    template_path=template_inp,
                    event_window_mode="rain+2h",
                    event_buffer_hours=2.0,
                    rain_minutes=minutes_15,
                    rain_curve_in=st.session_state["rain_sim_curve_current_in"],
                    sim_start_str=simulation_date,
                )
                info3 = run_swmm_scenario(
                    f"{prefix}lid_max_nogate_future",
                    st.session_state["rain_lines_fut"],
                    st.session_state["tide_lines"],
                    lid_lines,
                    "NO",
                    template_path=template_inp,
                    event_window_mode="rain+2h",
                    event_buffer_hours=2.0,
                    rain_minutes=minutes_15,
                    rain_curve_in=st.session_state["rain_sim_curve_future_in"],
                    sim_start_str=simulation_date,
                )
                info4 = run_swmm_scenario(
                    f"{prefix}lid_max_gate_future",
                    st.session_state["rain_lines_fut"],
                    st.session_state["tide_lines"],
                    lid_lines,
                    "YES",
                    template_path=template_inp,
                    event_window_mode="rain+2h",
                    event_buffer_hours=2.0,
                    rain_minutes=minutes_15,
                    rain_curve_in=st.session_state["rain_sim_curve_future_in"],
                    sim_start_str=simulation_date,
                )


                st.session_state.update({
                    f"{prefix}df_lid_max_nogate_current": info1["df_runoff"],
                    f"{prefix}df_lid_max_gate_current":   info2["df_runoff"],
                    f"{prefix}df_lid_max_nogate_future":  info3["df_runoff"],
                    f"{prefix}df_lid_max_gate_future":    info4["df_runoff"],
                })

                existing_store = _sc_store(prefix).copy()
                remember_scenario("lid_max_nogate_current", info1["df_runoff"], info1["node_cuft_event"], prefix)
                remember_scenario("lid_max_gate_current",   info2["df_runoff"], info2["node_cuft_event"], prefix)
                remember_scenario("lid_max_nogate_future",  info3["df_runoff"], info3["node_cuft_event"], prefix)
                remember_scenario("lid_max_gate_future",    info4["df_runoff"], info4["node_cuft_event"], prefix)

                store = _sc_store(prefix)
                for k, v in existing_store.items():
                    store.setdefault(k, v)

                st.success("Max LID scenarios complete.")
                st.session_state[f"{prefix}ran_max"] = True
                _rehydrate_baseline_into_store(prefix)
                _ensure_baselines_visible(prefix)
                st.session_state[f"{prefix}grid_ready"] = True
            except Exception as e:
                st.error(f"Max LID simulation failed: {e}")

    ran_baseline = bool(st.session_state.get(f"{prefix}ran_baseline"))
    ran_custom   = bool(st.session_state.get(f"{prefix}ran_custom"))
    ran_max      = bool(st.session_state.get(f"{prefix}ran_max"))

    have_all_three = ran_baseline and ran_custom and ran_max
    if have_all_three:
        st.session_state[f"{prefix}show_comparison"] = True

    show_comparison = bool(st.session_state.get(f"{prefix}show_comparison", False))

    st.subheader("Scenario Comparison Maps")

    ran_baseline = bool(st.session_state.get(f"{prefix}ran_baseline"))
    ran_custom   = bool(st.session_state.get(f"{prefix}ran_custom"))
    ran_max      = bool(st.session_state.get(f"{prefix}ran_max"))
    have_all_three = ran_baseline and ran_custom and ran_max
    if have_all_three:
        st.session_state[f"{prefix}show_comparison"] = True

    show_comparison = bool(st.session_state.get(f"{prefix}show_comparison", False))
    if not show_comparison:
        st.info("Run Baseline, Custom LID, and Max LID to display comparison maps.")
    else:
        # Keep all scenarios in memory
        _rehydrate_baseline_into_store(prefix)
        _ensure_baselines_visible(prefix)

        # Two 6-map sets (CURRENT vs +20%)
        CURRENT_SCENS = [
            "baseline_nogate_current", "baseline_gate_current",
            "lid_nogate_current",      "lid_gate_current",
            "lid_max_nogate_current",  "lid_max_gate_current",
        ]
        CURRENT_TITLES = [
            "Baseline-No Gate-Current",
            "Baseline-Gate-Current",
            "Custom LID-No Gate-Current",
            "Custom LID-Gate-Current",
            "Max LID-No Gate-Current",
            "Max LID-Gate-Current",
        ]

        FUTURE_SCENS = [
            "baseline_nogate_future", "baseline_gate_future",
            "lid_nogate_future",      "lid_gate_future",
            "lid_max_nogate_future",  "lid_max_gate_future",
        ]
        FUTURE_TITLES = [
            "Baseline-No Gate-+20%",
            "Baseline-Gate-+20%",
            "Custom LID-No Gate-+20%",
            "Custom LID-Gate-+20%",
            "Max LID-No Gate-+20%",
            "Max LID-Gate-+20%",
        ]

        # Ensure all needed scenarios are loaded so we can compute a shared legend range
        for nm in CURRENT_SCENS + FUTURE_SCENS:
            _ensure_scenario_loaded(nm, prefix)

        # Build DF list for global legend range (across all 12 so scales stay consistent)
        sc_records_all = [recall_scenario(nm, prefix) for nm in (CURRENT_SCENS + FUTURE_SCENS)]
        dfs_available  = [rec["df"] for rec in sc_records_all if rec is not None]
        global_range = (
            _global_runoff_range_across(dfs_available, st.session_state["unit_ui"], WS_SHP_PATH)
            if dfs_available else (0.0, 1.0)
        )

        # --- UI: two buttons to switch set ---
        if f"{prefix}cmp_set" not in st.session_state:
            st.session_state[f"{prefix}cmp_set"] = "current"

        cbtn1, cbtn2 = st.columns([1,1])
        with cbtn1:
            if st.button("Show CURRENT rainfall maps", key=f"{prefix}btn_cmp_current"):
                st.session_state[f"{prefix}cmp_set"] = "current"
        with cbtn2:
            if st.button("Show +20% rainfall maps", key=f"{prefix}btn_cmp_future"):
                st.session_state[f"{prefix}cmp_set"] = "future"

        # Pick which set to render (6 maps max)
        if st.session_state[f"{prefix}cmp_set"] == "future":
            SCENARIO_NAMES = FUTURE_SCENS
            TITLES         = FUTURE_TITLES
        else:
            SCENARIO_NAMES = CURRENT_SCENS
            TITLES         = CURRENT_TITLES

        # Recollect only the selected set
        sc_records = [recall_scenario(nm, prefix) for nm in SCENARIO_NAMES]

        unit_lbl_for_legend = None
        any_rendered = False

        # Stable 2x3 grid (6 canvases total)
        tiles = []
        for _row in range(2):
            cols = st.columns(3, gap="medium")
            tiles.extend([cols[0].container(), cols[1].container(), cols[2].container()])

        # Fill the 6 tiles
        for i, tile in enumerate(tiles):
            title = TITLES[i]
            name  = SCENARIO_NAMES[i]
            rec   = sc_records[i]

            with tile:
                st.markdown(f"**{title}**")
                body = st.empty()
                if rec is None:
                    body.info("Not run yet.")
                else:
                    df_i    = rec["df"]
                    nodes_i = rec.get("nodes", {}) or {}
                    with body:
                        unit_lbl_for_legend, _ = render_total_runoff_map_single(
                            df_in_inches=df_i,
                            title=title,
                            nodes_post5h_dict=nodes_i,
                            unit_ui=st.session_state["unit_ui"],
                            ws_shp_path=WS_SHP_PATH,
                            pipe_shp_path=PIPE_SHP_PATH,
                            node_shp_path=NODE_SHP_PATH,
                            node_name_field_hint="NAME",
                            legend_range=global_range,              # SAME SCALE across both pages
                            widget_key=f"cmpmap_{st.session_state[f'{prefix}cmp_set']}_{i}_{name}",
                            show_title=False,
                        )
                    any_rendered = True

        if any_rendered and unit_lbl_for_legend is not None:
            st.markdown(_legend_html(unit_lbl_for_legend, global_range[0], global_range[1]), unsafe_allow_html=True)

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

    def _gather_scenario_volumes() -> tuple[pd.DataFrame, str]:
        ACF_TO_FT3 = 43560.0
        FT3_TO_M3  = 0.0283168
        to_m3 = (st.session_state.get("unit_ui") == "Metric (SI)")
        unit_label = "m³" if to_m3 else "ft³"

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
                                axis=alt.Axis(labelFontSize=14, titleFontSize=20, labelLimit=1000)),
                        color=alt.Color("Group:N",
                                        scale=alt.Scale(domain=group_domain, range=group_colors),
                                        legend=alt.Legend(title="Scenario Group")),
                        tooltip=["Scenario", f"{unit_lbl}:Q", "Group:N"]
                    )
                    .properties(height=height_px)
                    .configure_axis(labelFontSize=14, titleFontSize=20)
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
                            axis=alt.Axis(labelFontSize=14, titleFontSize=20, labelLimit=1000)),

                    color=alt.Color("Scenario:N", legend=None)
                )
                .properties(height=height_i)
                .configure_axis(labelFontSize=14, titleFontSize=16)
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
                            axis=alt.Axis(labelFontSize=14, titleFontSize=16, labelLimit=1000)),

                    color=alt.Color("Scenario:N", legend=None)
                )
                .properties(height=height_r)
                .configure_axis(labelFontSize=14, titleFontSize=16)
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
