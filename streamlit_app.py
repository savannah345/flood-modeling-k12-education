import os, re, sys, shutil, tempfile, subprocess
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


# Your existing theme dictionary
alt_theme = {
    "config": {
        "title": {"fontSize": 20, "color": "black"},
        "axis": {
            "labelFontSize": 18,
            "titleFontSize": 20,
            "labelColor": "black",
            "titleColor": "black"
        },
        "legend": {
            "labelFontSize": 18,
            "titleFontSize": 20,
            "labelColor": "black",
            "titleColor": "black"
        }
    }
}

@alt.theme.register("bigger_black", enable= True)
def bigger_black_theme():
    return alt.theme.ThemeConfig(alt_theme)


st.markdown("""
<style>
/* PyDeck TextLayer labels */
.deck-tooltip, .mapboxgl-popup-content {
    color: black !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

/* TextLayer labels (subcatchment names) */
text {
    fill: black !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* PyDeck canvas default font */
canvas {
    color: black !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

def to_ui_depth(ft, ui_unit):
    if ui_unit == "Metric (SI)":
        return ft * 0.3048
    return ft

def to_ui_volume(ft3, ui_unit):
    if ui_unit == "Metric (SI)":
        return ft3 * 0.0283168
    return ft3

def to_ui_rain(inches, ui_unit):
    if ui_unit == "Metric (SI)":
        return inches * 2.54
    return inches

def to_ui_tide(feet, ui_unit):
    if ui_unit == "Metric (SI)":
        return feet * 0.3048
    return feet

def make_scenario_key(prefix: str, subset: str, gate_flag: str, rain_variant: str) -> str:
    prefix = prefix.strip("_") + "_"

    subset = subset.lower()
    gate = "gate" if gate_flag.upper() == "YES" else "nogate"
    rain = "future" if rain_variant.lower() == "future" else "current"

    return f"{prefix}{subset}_{gate}_{rain}"

def make_display_label(subset: str, gate_flag: str) -> str:
    subset_map = {
        "baseline": "Baseline",
        "all": "All Subcatchments",
        "upstream": "Upstream",
        "downstream": "Downstream",
        "highrunoff": "High-Runoff",
    }

    gate_text = "TG & Pump ON" if gate_flag.upper() == "YES" else "TG & Pump OFF"

    return f"{subset_map.get(subset, subset)} – {gate_text}"

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

WS_SHP_PATH  = st.session_state.get("WS_SHP_PATH", "map_files/Subcatchment.shp")
PIPES_SHP_PATH= st.session_state.get("PIPES_SHP_PATH", "map_files/Conduits.shp")
NODES_SHP_PATH= st.session_state.get("NODES_SHP_PATH", "map_files/Nodes.shp")
DEM_PATH = st.session_state.get("DEM_PATH", "map_files/DEM.tif")
FLOWDIR_PATH = st.session_state.get("FLOWDIR_PATH", "map_files/flow_direction.tif")

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
    return pd.read_excel(path)



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

future_mult = 1.2
    
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


def plan_cost(plan: dict, cRG: float, cRB: float) -> float:
    return cRG * sum(v["rain_gardens"] for v in plan.values()) + cRB * sum(v["rain_barrels"] for v in plan.values())

def summarize_plan(plan: Dict[str, Dict[str,int]], cRG: float, cRB: float) -> dict:
    total_rg = sum(v["rain_gardens"] for v in plan.values())
    total_rb = sum(v["rain_barrels"] for v in plan.values())
    spent = cRG * total_rg + cRB * total_rb
    treated_ft2 = 400.0 * total_rg + 300.0 * total_rb
    return {"rg": int(total_rg), "rb": int(total_rb), "spent": float(spent), "treated_ft2": float(treated_ft2)}

def enforce_rg_limit(plan, RG_limit_all, caps_RG):
    """
    Ensures no scenario uses more rain gardens than the 'All' scenario.
    If RG exceeds limit, remove RG from lowest-capacity subs first.
    """
    current_rg = sum(v["rain_gardens"] for v in plan.values())
    if current_rg <= RG_limit_all:
        return plan

    remove_needed = current_rg - RG_limit_all

    # remove from lowest RG-capacity subs first to avoid breaking constraints later
    subs_sorted = sorted(plan.keys(), key=lambda s: caps_RG.get(s, 0))

    for s in subs_sorted:
        if remove_needed <= 0:
            break

        rg_here = plan[s]["rain_gardens"]
        if rg_here > 0:
            rm = min(rg_here, remove_needed)
            plan[s]["rain_gardens"] -= rm
            remove_needed -= rm

    return plan


def fill_with_rb_to_budget(plan, caps_RB, cRG, cRB, target_budget):
    """
    After RG counts are capped, fill remaining budget using RB until target_budget is reached.
    """
    spend = cRG * sum(v["rain_gardens"] for v in plan.values()) + \
            cRB * sum(v["rain_barrels"]  for v in plan.values())

    if spend >= target_budget:
        return plan

    # add RB in subs with the most RB capacity
    subs_sorted = sorted(plan.keys(), key=lambda s: caps_RB.get(s, 0), reverse=True)

    for s in subs_sorted:
        max_rb = caps_RB.get(s, 0)
        while plan[s]["rain_barrels"] < max_rb and spend + cRB <= target_budget:
            plan[s]["rain_barrels"] += 1
            spend += cRB
            if spend >= target_budget:
                break

    return plan

def render_focus_placement_map(plan, title, ws_shp_path, widget_key):
    gdf = load_ws(ws_shp_path).copy()

    # Attach counts
    gdf["_RG"] = gdf["NAME"].map(lambda s: plan.get(s, {}).get("rain_gardens", 0))
    gdf["_RB"] = gdf["NAME"].map(lambda s: plan.get(s, {}).get("rain_barrels", 0))
    gdf["_TOTAL_LID"] = gdf["_RG"] + gdf["_RB"]

    FIXED_COLOR = [34, 139, 34, 160]   

    def fixed_color(n):
        if n <= 0:
            return [0, 0, 0, 0] 
        return FIXED_COLOR

    gdf["_fill"] = gdf["_TOTAL_LID"].map(fixed_color)

    # Labels
    reps = gdf.geometry.representative_point()
    labels = pd.DataFrame({
        "lon": reps.x,
        "lat": reps.y,
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

    # GeoJSON layer
    poly_layer = pdk.Layer(
        "GeoJsonLayer",
        data=gdf.__geo_interface__,
        pickable=True,
        autoHighlight=True,
        highlightColor=[0, 0, 0, 160],
        stroked=True,
        filled=True,
        get_fill_color="properties._fill",
        get_line_color=[0,0,0,255],
        line_width_min_pixels=1,
    )

    centroid = gdf.geometry.union_all().centroid
    view_state = pdk.ViewState(
        latitude=centroid.y,
        longitude=centroid.x,
        zoom=13.45
    )

    tooltip = {
        "html": (
            "<b>{NAME}</b><br/>"
            "RG (rain gardens): {_RG}<br/>"
            "RB (rain barrels): {_RB}<br/>"
            "Total LIDs: {_TOTAL_LID}"
        ),
        "style": {"backgroundColor": "white", "color": "black"}
    }

    st.markdown(f"**{title}**")
    st.pydeck_chart(
        pdk.Deck(
            layers=[poly_layer, text_layer],
            initial_view_state=view_state,
            map_provider="carto",
            map_style="light",
            tooltip=tooltip,
        ),
        use_container_width=True,
        height=260,
        key=widget_key
    )

def build_fixed_sim_windows(storm_duration_minutes):
    sim_start = datetime(2025, 5, 1, 6, 0)
    storm_duration_hr = storm_duration_minutes / 60
    sim_duration_hr = storm_duration_hr + 3

    sim_end = sim_start + timedelta(hours=sim_duration_hr)
    report_start = datetime(2025, 5, 1, 8, 0)

    return sim_start, sim_end, report_start

PUMP_BLOCK = "Pump_1  J14  PumpOutfall  PumpCurve1  ON"

PUMP_CURVE_BLOCK = (
    "PumpCurve1 Pump3 0 0\n"
    "PumpCurve1 2 8\n"
    "PumpCurve1 5 20\n"
    "PumpCurve1 10 20\n"
)

PUMP_RULES = (
    "RULE Turn_Pump_On\n"
    "IF NODE outfall Head >= NODE J14 Head - 0.05\n"
    "AND NODE J14 Depth > 0.1\n"
    "THEN PUMP Pump_1 STATUS = ON\n\n"
    "RULE Turn_Pump_Off\n"
    "IF NODE outfall Head < NODE J14 Head - 0.15\n"
    "THEN PUMP Pump_1 STATUS = OFF"
)

def extract_node_flooding_from_rpt(txt: str, preview: bool = False) -> pd.DataFrame:
    """
    Robustly parse the Node Flooding Summary from an RPT file.
    Includes optional preview of the first 5 rows for debugging table alignment.
    """

    cols = ["node", "hours_flooded", "max_rate_cfs", "total_flood_ft3", "max_depth_ft"]

    if not txt:
        df = pd.DataFrame(columns=cols)
        if preview:
            print("Node Flooding Summary not found – empty RPT")
        return df

    lines = txt.splitlines(True)

    # 1. Locate the section start
    sec_i = next(
        (i for i, line in enumerate(lines) if "Node Flooding Summary" in line),
        None
    )
    if sec_i is None:
        df = pd.DataFrame(columns=cols)
        if preview:
            print("Node Flooding Summary section not found.")
        return df

    # 2. Skip to the header
    i = sec_i + 1
    while i < len(lines) and "Node" not in lines[i]:
        i += 1
    if i >= len(lines):
        df = pd.DataFrame(columns=cols)
        if preview:
            print("Node Flooding Summary header not found.")
        return df

    # 3. Skip header underline (dashes)
    i += 1
    while i < len(lines) and "-" in lines[i]:
        i += 1

    rows = []
    float_re = re.compile(r"[-+]?\d+(?:\.\d+)?")

    # 4. Parse rows of the table
    while i < len(lines):
        raw = lines[i].rstrip("\n")

        # Stop conditions (same style as runoff parser)
        if not raw.strip():
            break
        if raw.lstrip().startswith("*"):
            break
        if "Subcatchment" in raw or "Analysis Options" in raw or "Node Depth Summary" in raw:
            break

        parts = raw.split()
        nums = [float(m.group(0)) for m in float_re.finditer(raw)]

        # SWMM flooding summary format basically ends with two key numbers:
        # [... Hours, MaxRate, ... , TotalFlood(MG), MaxDepth(ft)]
        if len(parts) >= 7 and len(nums) >= 5:
            node = parts[0]
            hours = nums[0]
            max_rate = nums[1]
            total_mg = nums[-2]     # second-to-last numeric
            max_depth = nums[-1]    # last numeric

            total_ft3 = total_mg * 1_000_000 * 0.133681  # MG → ft³

            rows.append({
                "node": node,
                "hours_flooded": hours,
                "max_rate_cfs": max_rate,
                "total_flood_ft3": total_ft3,
                "max_depth_ft": max_depth
            })

        i += 1

    df = pd.DataFrame(rows, columns=cols)

    return df

def run_swmm_senario(
    scenario_name: str,
    rain_lines: List[str],
    tide_lines: List[str],
    lid_lines: List[str],
    gate_flag: str,
    duration_minutes: int,
    template_path: str = "SWMM_Project.inp",
):
    """
    Clean SWMM runner using only RPT parsing.
    Assumes rainfall + tide lines already clipped and
    time series rebuilt starting at 06:00.
    """
    output_dir = tempfile.mkdtemp(prefix=f"{scenario_name}_")

    inp_path = os.path.join(output_dir, f"{scenario_name}.inp")
    rpt_path = os.path.join(output_dir, f"{scenario_name}.rpt")
    out_path = os.path.join(output_dir, f"{scenario_name}.out")

    # Build fixed simulation window
    sim_start_dt, sim_end_dt, report_start_dt = build_fixed_sim_windows(duration_minutes)

    sim_start_date = sim_start_dt.strftime("%m/%d/%Y")
    sim_start_time = sim_start_dt.strftime("%H:%M:%S")
    sim_end_date   = sim_end_dt.strftime("%m/%d/%Y")
    sim_end_time   = sim_end_dt.strftime("%H:%M:%S")

    report_start_date = report_start_dt.strftime("%m/%d/%Y")
    report_start_time = report_start_dt.strftime("%H:%M:%S")

    # Store metadata
    st.session_state[f"{scenario_name}_sim_start"] = sim_start_dt
    st.session_state[f"{scenario_name}_sim_end"] = sim_end_dt
    st.session_state[f"{scenario_name}_report_start"] = report_start_dt
    st.session_state[f"{scenario_name}_report_end"] = sim_end_dt

    # Pump logic
    if gate_flag == "YES":
        pump_block = PUMP_BLOCK
        pump_curve_block = PUMP_CURVE_BLOCK
        pump_rule_block = PUMP_RULES
    else:
        pump_block = ""
        pump_curve_block = ""
        pump_rule_block = ""

    with open(template_path, "r") as f:
        text = f.read()

    text = (
        text.replace("$RAINFALL_TIMESERIES$", "\n".join(rain_lines))
            .replace("$TIDE_TIMESERIES$", "\n".join(tide_lines))
            .replace("$LID_USAGE$", "\n".join(lid_lines))
            .replace("$TIDE_GATE_CONTROL$", gate_flag)
            .replace("$SIM_START_DATE$", sim_start_date)
            .replace("$SIM_START_TIME$", sim_start_time)
            .replace("$REPORT_START_DATE$", report_start_date)
            .replace("$REPORT_START_TIME$", report_start_time)
            .replace("$SIM_END_DATE$", sim_end_date)
            .replace("$SIM_END_TIME$", sim_end_time)
            .replace("$PUMP_BLOCK$", pump_block)
            .replace("$PUMP_CURVE_BLOCK$", pump_curve_block)
            .replace("$PUMP_RULES$", pump_rule_block)
    )

    with open(inp_path, "w") as f:
        f.write(text)

    with Simulation(inp_path, rpt_path, out_path) as sim:
        sim.execute()

    rpt_text = _read_text_keep(rpt_path)

    df_node_flooding = extract_node_flooding_from_rpt(rpt_text)
    df_runoff        = extract_total_runoff_from_text(rpt_text)
    df_ir            = extract_infiltration_and_runoff_from_text(rpt_text)

    st.session_state[f"{scenario_name}_node_flooding_summary"] = df_node_flooding

    total_flood_ft3 = df_node_flooding["total_flood_ft3"].sum()
    st.session_state[f"{scenario_name}_storm_total_flood"] = total_flood_ft3

    infiltration_ft3 = 0.0
    if df_ir is not None and not df_ir.empty:
        infil_acft = df_ir.loc[df_ir["label"] == "infiltration_loss", "volume_acft"].sum()
        infiltration_ft3 = float(infil_acft) * 43560.0

    st.session_state[f"{scenario_name}_storm_infiltration"] = infiltration_ft3

    total_sim_minutes = duration_minutes + 180
    st.session_state[f"{scenario_name}_storm_dur_hours"] = total_sim_minutes / 60.0

    return {
        "df_runoff": df_runoff,
        "df_continuity": df_ir,
        "df_flooding": df_node_flooding
    }

def flooding_summary_ui():
    if not st.session_state.get("scenarios_finished", False):
        return

    ui_unit = st.session_state.get("unit_ui", "U.S. Customary")

    labels = st.session_state.get("scenario_display_labels", {})
    flood_data = st.session_state.get("scenario_flood_volume", {})
    infil_data = st.session_state.get("scenario_infiltration", {})

    rows = []
    for scen_key, label in labels.items():
        flood_ft3 = float(flood_data.get(scen_key, 0.0))
        infil_ft3 = float(infil_data.get(scen_key, 0.0))

        rows.append({
            "ScenarioKey": scen_key,
            "Scenario": label,     # readable label
            "Flooding_ft3": to_ui_volume(flood_ft3, ui_unit),
            "Infiltration_ft3": to_ui_volume(infil_ft3, ui_unit)
        })

    df = pd.DataFrame(rows)

    if df.empty:
        st.warning("No scenarios found.")
        return

    df_sorted = df.sort_values("Flooding_ft3", ascending=True)

    min_val = float(df_sorted["Flooding_ft3"].min())
    max_val = float(df_sorted["Flooding_ft3"].max())

    display_unit = "ft³" if ui_unit == "U.S. Customary" else "m³"

    chart = (
        alt.Chart(df_sorted)
        .mark_bar()
        .encode(
            x=alt.X("Flooding_ft3:Q", title=f"Total Flood Volume ({display_unit})", scale=alt.Scale(domain=[min_val, max_val])),
            y=alt.Y("Scenario:N", sort=alt.SortField(field="Flooding_ft3", order="ascending"), axis=None),
            color=alt.Color("Scenario:N", legend=None),
            tooltip=[
                alt.Tooltip("Scenario:N", title="Scenario"),
                alt.Tooltip("Flooding_ft3:Q", title="Flooding", format=",.1f")
            ]
        )
    )

    labels = (
        alt.Chart(df_sorted)
        .mark_text(
            align="left", baseline="middle", dx=10, fontSize =15
        )
        .encode(
            x="Flooding_ft3:Q",
            y=alt.Y("Scenario:N", sort=alt.SortField("Flooding_ft3", "ascending")),
            text="Scenario"
        )
    )

    st.markdown("### Flood Summary")
    final_chart = (chart + labels).properties(
        padding={"bottom": 40}
    )

    st.altair_chart(final_chart, use_container_width=True)


def scenario_comparison_map_ui():
    st.subheader("Compare Two Scenarios")

    st.markdown("What this shows: circle color = which scenario has more flooding")
    st.markdown("Red: Scenario A has more flooding")
    st.markdown("Blue: Scenario B has more")
    st.markdown("Gray: no meaningful change")
    st.markdown("Circle size = magnitude of difference")

    labels = st.session_state.get("scenario_display_labels", {})
    if not labels:
        st.info("Run scenarios first.")
        return

    opts = [(label, key) for key, label in labels.items()]

    col1, col2 = st.columns(2)
    with col1:
        scenA = st.selectbox("Scenario A", opts, format_func=lambda x: x[0])
    with col2:
        scenB = st.selectbox("Scenario B", opts, format_func=lambda x: x[0])

    labelA, keyA = scenA
    labelB, keyB = scenB

    if not st.button("Compare Flooding"):
        return

    dfA = st.session_state.get(f"{keyA}_node_flooding_summary")
    dfB = st.session_state.get(f"{keyB}_node_flooding_summary")

    if dfA is None or dfB is None:
        st.error("Missing flooding data for selected scenarios.")
        return

    nodes_gdf = load_nodes(NODES_SHP_PATH)
    nodes_gdf["lon"] = nodes_gdf.geometry.x
    nodes_gdf["lat"] = nodes_gdf.geometry.y

    # ---- MERGE A AND B ----
    merged = (
        nodes_gdf
        .merge(dfA[["node", "max_depth_ft"]].rename(columns={"max_depth_ft": "A_depth"}),
               left_on="NAME", right_on="node", how="left")
        .merge(dfB[["node", "max_depth_ft"]].rename(columns={"max_depth_ft": "B_depth"}),
               left_on="NAME", right_on="node", how="left")
    )

    merged["A_depth"] = merged["A_depth"].fillna(0)
    merged["B_depth"] = merged["B_depth"].fillna(0)
    merged["difference"] = (merged["A_depth"] - merged["B_depth"]).round(2)

    # ---- COLOR RULES ----
    def color(diff):
        if diff > 0:
            return [220, 50, 40, 200]    # Scenario A worse
        if diff < 0:
            return [40, 80, 220, 200]    # Scenario B worse
        return [150, 150, 150, 150]      # No change

    merged["color"] = merged["difference"].apply(color)

    merged["size"] = 40 + 80 * (
        merged["difference"].abs() /
        max(merged["difference"].abs().max(), 1e-6)
    )

    # ---- SUBCATCHMENTS SHAPEFILE ----
    ws_gdf_local = load_ws(WS_SHP_PATH)
    centroid = ws_gdf_local.geometry.union_all().centroid
    view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=14.5)

    poly_layer = pdk.Layer(
        "GeoJsonLayer", data=ws_gdf_local.__geo_interface__,
        pickable=True, stroked=True, filled=True,
        get_fill_color=[255, 255, 255, 0],   
        get_line_color=[0, 0, 0, 255],      
        line_width_min_pixels=1,
    )


    # ---- NODES LAYER ----
    nodes_layer = pdk.Layer(
        "ScatterplotLayer",
        data=merged,
        get_position='[lon, lat]',
        get_fill_color='color',
        get_radius='size',
        pickable=True,
    )

    # ---- MUCH IMPROVED TOOLTIP ----
    tooltip = {
        "html": (
            "<b>Node:</b> {NAME}<br>"
            f"<b>{labelA}:</b> {{A_depth}} ft<br>"
            f"<b>{labelB}:</b> {{B_depth}} ft<br>"
            "<b>Difference:</b> {difference} ft"
        ),
        "style": {
            "backgroundColor": "white",
            "color": "black",
            "fontSize": "14px",
            "padding": "10px"
        }
    }

    view_state = pdk.ViewState(
        latitude=merged["lat"].mean(),
        longitude=merged["lon"].mean(),
        zoom=14,
    )

    deck = pdk.Deck(
        layers=[poly_layer, nodes_layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip=tooltip,
    )

    st.pydeck_chart(deck)


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

    # Legend
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

def clip_rain_and_tide(rain, tide, duration_minutes):
    # find first rain
    rain = np.array(rain)
    tide = np.array(tide)

    idx0 = np.argmax(rain > 0)
    storm_steps = duration_minutes // 15
    idx_end = idx0 + storm_steps - 1

    pre_steps = 2 * 4     # 2 hours = 8 steps
    post_steps = 1 * 4    # 1 hour = 4 steps

    start = idx0 - pre_steps
    end = idx_end + post_steps

    return rain[start:end+1], tide[start:end+1]

def build_rain_timeseries(sim_start_dt, values):
    lines = []
    for i, v in enumerate(values):
        ts = sim_start_dt + timedelta(minutes=15*i)
        lines.append(f"rain_gage_timeseries {ts:%m/%d/%Y %H:%M} {v:.5f}")
    return lines

def build_tide_timeseries(sim_start_dt, values):
    lines = []
    for i, v in enumerate(values):
        ts = sim_start_dt + timedelta(minutes=15*i)
        lines.append(f"tide {ts:%m/%d/%Y %H:%M} {float(v):.5f}")
    return lines

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

    st.session_state.setdefault("scenario_display_labels", {})
    st.session_state.setdefault("scenario_flood_ts", {})
    st.session_state.setdefault("scenario_flood_volume", {})
    st.session_state.setdefault("scenario_infiltration", {})
    st.session_state.setdefault("scenario_labels_saved", {})
    st.session_state.setdefault("scenario_flood_saved", {})
    st.session_state.setdefault("scenario_infil_saved", {})
    st.session_state.setdefault("display_summary", False)
    st.session_state.setdefault("scenario_runoff", {})

        # Initialize storage
    if "flood_summary_chart" not in st.session_state:
        st.session_state["flood_summary_chart"] = None

    st.success(f"Logged in as: {st.session_state.get('email', 'user')}")
    ensure_temp_dir()
    prefix = st.session_state.get("scenario_prefix", "")

    st.session_state.setdefault("rpts", {})

    template_inp    = "SWMM_Project.inp"
    WS_SHP_PATH     = st.session_state.get("WS_SHP_PATH", "map_files/Subcatchment.shp")

    future_mult = 1.2

    st.title("CoastWise")
    
    # --- Fixed subcatchment lists (must match shapefile NAME like "Sub_20") ---
    UPSTREAM_LIST = {f"Sub_{n}" for n in [20,22,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]}
    DOWNSTREAM_LIST = {f"Sub_{n}" for n in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23]}
    HIGHRUNOFF_LIST = {f"Sub_{n}" for n in [3,4,5,6,7,10,18,19,21,22,27,28,29,30,31,32,33,35,37,38]}
    
    st.markdown("""
    <style>
    [data-testid="stTooltipContent"] {
        font-size: 12.5rem !important;
        line-height: 1.55 !important;
        max-width: 360px !important;
        white-space: normal !important;  
    }

    div[data-baseweb="tooltip"] {
        font-size: 12.5rem !important;
        line-height: 1.55 !important;
        max-width: 360px !important;
    }

    div[data-baseweb="tooltip"] .content,
    [data-testid="stTooltipContent"] {
        padding: 0.5rem 0.75rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    cfg = st.session_state.get("cfg", {})

    # Ensure all required keys exist, even if cfg was created in an older version
    cfg.setdefault("ui_unit", "U.S. Customary")
    cfg.setdefault("moon_phase", list(moon_tide_ranges.keys())[0])
    cfg.setdefault("duration_minutes", int(pf_df["Duration_Minutes"].iloc[0]))
    cfg.setdefault("return_period", "1")
    cfg.setdefault("align_mode", "peak")
    cfg.setdefault("settings_ready", False)

    # Save back
    st.session_state.cfg = cfg
    st.session_state["ui_unit"] = cfg["ui_unit"]

    st.success(
        "Use the controls below to configure your storm scenario. "
        "These settings determine the rainfall event, tide conditions, and alignment "
        "that will be used to generate inputs for the simulation."
    )

    with st.form("scenario_settings"):
        # ---------------- Units ----------------
        unit = st.radio(
            "Preferred Units",
            ["U.S. Customary", "Metric (SI)"],
            index=(0 if cfg["ui_unit"] == "U.S. Customary" else 1),
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
        duration_options = [120, 240, 360, 480, 600, 720]  # 2, 4, 6, 8, 10, 12 hours

        # Reset default duration if old value is incompatible
        if cfg["duration_minutes"] not in duration_options:
            duration_default = 120
        else:
            duration_default = cfg["duration_minutes"]

        duration_minutes = st.select_slider(
            "Storm Duration",
            options=duration_options,
            value=duration_default,
            format_func=lambda x: f"{int(x)//60} hr",
            help=(
                "Controls the total rainfall event length."
            )
        )

        d_low, d_high = st.columns([1, 3])
        with d_low:
            st.caption("Shorter storm")
        with d_high:
            st.caption("<div style='text-align:right;'>Longer storm</div>", unsafe_allow_html=True)


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
        r_low, r_high = st.columns([1, 2])
        with r_low:
            st.caption("Less intense and more common")
        with r_high:
            st.caption("<div style='text-align:right;'>More intense and less common</div>", unsafe_allow_html=True)

        rain_variant_choice = st.radio(
            "Rainfall Scenario",
            ["Current", "Future (+20%)"],
            index=(0 if cfg.get("rain_variant", "current") == "current" else 1),
            horizontal=True,
            help="Choose whether to run simulations using current rainfall (NOAA Atlas 14 Values) or a +20% future rainfall that Norfolk, VA utilizes in its design standards for new builds in preparation for the future climate conditions of the area."
        )

        # ---------------- Tide Alignment ----------------
        align_choice = st.radio(
            "Tide Alignment",
            ["Rainfall aligned with High Tide", "Rainfall aligned with Low Tide"],
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
            "ui_unit": unit,
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
        st.stop() 

    align_mode       = cfg["align_mode"]
    unit             = cfg["ui_unit"]
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



    ui_unit = st.session_state.get("ui_unit", "U.S. Customary")

    display_rain_curve = to_ui_rain(rain_curve_in, ui_unit)
    display_tide_curve = to_ui_tide(tide_curve_ui, ui_unit)

    rain_disp_unit = "in" if ui_unit == "U.S. Customary" else "cm"
    tide_disp_unit = "ft" if ui_unit == "U.S. Customary" else "m"

    # flooding / infiltration always computed in ft³
    display_flood_unit = "ft³" if ui_unit == "U.S. Customary" else "m³"

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
        f"{prefix}template_inp": template_inp,
    })

    # ------------------------------------------------------------
    # BUILD RAINFALL + TIDE FOR SWMM (must run before buttons)
    # ------------------------------------------------------------

    # --- Clip rain + tide using the aligned 48hr curves ---
    rain_clip, tide_clip = clip_rain_and_tide(rain_curve_in, tide_curve_ui, duration_minutes)

    # --- Build new timestamps ---
    sim_start_dt, sim_end_dt, report_start_dt = build_fixed_sim_windows(duration_minutes)

    rain_lines_cur = build_rain_timeseries(sim_start_dt, rain_clip)
    tide_lines = build_tide_timeseries(sim_start_dt, tide_clip)
    rain_lines_fut = build_rain_timeseries(sim_start_dt, rain_clip * 1.2)


    # SAVE
    st.session_state["rain_lines_cur"] = rain_lines_cur
    st.session_state["rain_lines_fut"] = rain_lines_fut
    st.session_state["tide_lines"] = tide_lines

    st.session_state[f"{prefix}rain_lines_cur"] = rain_lines_cur
    st.session_state[f"{prefix}rain_lines_fut"] = rain_lines_fut
    st.session_state[f"{prefix}tide_lines"] = tide_lines


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

    import altair as alt
    alt.data_transformers.disable_max_rows()

    # Determine the hour at which rainfall is aligned (the rainfall peak)
    align_idx = int(np.argmax(df_rt["Rain_Current"]))
    align_hour = float(df_rt["Hour"].iloc[align_idx])

    # ------------------- Vertical Alignment Line -------------------
    align_line = alt.Chart(df_rt).mark_rule(
        stroke="red",
        strokeDash=[6,6],
        strokeWidth=2
    ).encode(
        x="Hour:Q"
    ).transform_filter(
        alt.datum.Hour == align_hour
    )

    # ------------------- Rainfall Chart -------------------
    rain_base = alt.Chart(df_rt).encode(
        x=alt.X("Hour:Q", title="Hour")
    )

    rain_current = rain_base.mark_area(color="steelblue", opacity=0.5).encode(
        y=alt.Y("Rain_Current:Q", title=f"Rainfall ({rain_disp_unit})")
    )

    rain_future = rain_base.mark_area(color="#f74f4f", opacity=0.2).encode(
        y="Rain_Future:Q"
    )

    rain_chart = alt.layer(rain_current, rain_future, align_line).properties(
        title="Rainfall Event",
        height=300
    ).configure_title(fontSize=18).interactive()

    st.altair_chart(rain_chart, use_container_width=True)

    # ------------------- Tide Chart -------------------
    tide_base = alt.Chart(df_rt).encode(
        x=alt.X("Hour:Q", title="Hour")
    )

    tide_line = tide_base.mark_line(color="#2e9144", strokeWidth=3).encode(
        y=alt.Y("Tide:Q", title=f"Tide ({tide_disp_unit})")
    )

    tide_chart = alt.layer(tide_line, align_line).properties(
        title="Tide Event",
        height=300
    ).configure_title(fontSize=18).interactive()

    st.altair_chart(tide_chart, use_container_width=True)


    if st.button("Run Baseline Scenario", key=f"{prefix}_btn_run_baseline"):

        try:
            lid_lines = [";"]  # no LIDs in baseline
            rain_variant = st.session_state.get("rain_variant", "current")

            # canonical scenario name
            baseline_key = make_scenario_key(prefix, "baseline", "NO", rain_variant)

            # choose rain series
            rain_lines_use = rain_lines_cur if rain_variant == "current" else rain_lines_fut

            # run SWMM
            info = run_swmm_senario(
                baseline_key,
                rain_lines_use,
                tide_lines,
                lid_lines,
                "NO",  # gate off
                duration_minutes,
                template_path=template_inp
            )

            # store baseline results under canonical keys
            st.session_state["scenario_runoff"][baseline_key] = info["df_runoff"]
            st.session_state["scenario_flood_volume"][baseline_key] = st.session_state.get(f"{baseline_key}_storm_total_flood", 0.0)
            st.session_state["scenario_infiltration"][baseline_key] = st.session_state.get(f"{baseline_key}_storm_infiltration", 0.0)

            # label for UI only
            st.session_state["scenario_display_labels"][baseline_key] = make_display_label("baseline", "NO")

            # build baseline runoff map
            map_html, legend_html = _build_baseline_map_html(
                df_swmm_local=info["df_runoff"],
                unit_ui=st.session_state["unit_ui"],
                ws_shp_path=WS_SHP_PATH
            )

            # save visuals
            st.session_state[f"{baseline_key}_map_html"] = map_html
            st.session_state[f"{baseline_key}_legend_html"] = legend_html
            st.session_state[f"{baseline_key}_ready"] = True

            st.success("Baseline scenario complete.")

        except Exception as e:
            st.error(f"Baseline simulation failed: {e}")

    baseline_key = make_scenario_key(
        prefix,
        "baseline",
        "NO",
        st.session_state.get("rain_variant", "current")
    )

    if st.session_state.get(f"{baseline_key}_ready", False):

        st.markdown("### Baseline Runoff and Land Cover")

        c1, c2 = st.columns([1, 1])

        # left side = runoff map
        with c1:
            st.markdown("#### Modeled Runoff")
            components.v1.html(
                st.session_state[f"{baseline_key}_map_html"],
                height=500,
                scrolling=False
            )
            st.markdown(
                st.session_state[f"{baseline_key}_legend_html"],
                unsafe_allow_html=True
            )


        with c2:
            st.markdown("#### Land Use / Land Cover")

            # container to match the 500px height of the runoff map
            image_container = st.container()
            with image_container:
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
        "**Percent Uptake (whole watershed):** Choose the percent rain garden and rain barrel uptake across the whole watershed, highlighting collective action across the watershed."
    )

    path_choice = st.selectbox(
        "Select a planning path:",
        ["Percent Uptake"],
        index=0
    )

    # Load per-subcatchment maxima from raster_df
    caps_RG, caps_RB = load_caps_from_df(raster_df)


    if "Percent Uptake" in path_choice:
        st.markdown("### Percent Uptake (whole watershed)")

        # Unit costs
        c1, c2 = st.columns(2)
        with c1:
            unit_cost_rg = st.number_input(
                "Rain Garden cost ($/unit)", min_value=0.0, value=500.0, step=25.0
            )
        with c2:
            unit_cost_rb = st.number_input(
                "Rain Barrel cost ($/unit)", min_value=0.0, value=150.0, step=5.0
            )

        # Percent sliders
        a1, a2 = st.columns(2)
        with a1:
            pct_rg = st.slider("RG uptake (%)", 0, 100, 20)
        with a2:
            pct_rb = st.slider("RB uptake (%)", 0, 100, 30)

        # RAW PLAN (this is the ONLY source of truth)
        plan_base = realize_percent_uptake(pct_rg, pct_rb, caps_RG, caps_RB)
        plan_all_raw = plan_base.copy()           # ← LOCKED FOREVER
        cost_all_raw = plan_cost(plan_all_raw, unit_cost_rg, unit_cost_rb)
        RG_limit_all = sum(v["rain_gardens"] for v in plan_all_raw.values())

        # Display raw Path A summary
        summary = summarize_plan(plan_all_raw, unit_cost_rg, unit_cost_rb)
        st.success(
            f"RG={summary['rg']} | RB={summary['rb']}   Estimated Cost: ${summary['spent']:,.0f}"
        )

        # This is the target cost for all other scenarios
        target_cost = cost_all_raw

        def build_focus_plan(group_set):

            # 1. Compute max RG capacity of this group
            max_RG_group = sum(caps_RG.get(s, 0) for s in group_set)

            # 2. Determine how many RG we attempt to place
            RG_for_group = min(RG_limit_all, max_RG_group)

            # 3. Allocate RG proportionally to each subcatchment’s capacity
            plan_group = {}
            for s in group_set:
                cap_s = caps_RG.get(s, 0)
                if max_RG_group > 0:
                    share = cap_s / max_RG_group
                    plan_group[s] = {
                        "rain_gardens": int(round(RG_for_group * share)),
                        "rain_barrels": 0
                    }
                else:
                    plan_group[s] = {"rain_gardens": 0, "rain_barrels": 0}

            # 4. Fill remaining budget using RB
            plan_group = fill_with_rb_to_budget(
                plan_group, caps_RB, unit_cost_rg, unit_cost_rb, target_cost
            )

            return plan_group

        plan_all = plan_all_raw

        plan_upstream  = build_focus_plan(UPSTREAM_LIST)
        plan_downstr   = build_focus_plan(DOWNSTREAM_LIST)
        plan_highro    = build_focus_plan(HIGHRUNOFF_LIST)


        with st.expander("Why CoastWise Shows Four Different Spatial Layouts"):
            st.markdown("""
        CoastWise creates four different LID layouts to help you understand how placement patterns influence outcomes such as runoff and flooding.

        Each layout uses the **same total investment**, but places the LIDs in different priority areas:

        1. **All Subcatchments** – spread across the whole watershed  
        2. **Upstream** – focused in the upper part of the watershed   
        3. **Downstream** – focused in the lower part of the watershed near the outlet  
        4. **High-Runoff Areas** – focused on the top 20 subcatchments that generated the most runoff (i.e., baseline simulation, darker subcatchments)  

        The purpose is to compare whether spreading LIDs across the watershed or concentrating them in specific areas has a larger benefit for managing stormwater and reducing flooding.

        By holding the total cost constant, CoastWise allows you to see how the same investment performs under different placement strategies.
        """)
            
        with st.expander("How CoastWise Places Rain Gardens and Rain Barrels"):
            st.markdown("""
        CoastWise places Rain Gardens and Rain Barrels based on what each part of the watershed can realistically support.

        Some areas have good soils and open space, so they can hold more rain gardens. Other areas have shallow groundwater or limited pervious land, so they can only take a few rain gardens but can still use rain barrels.

        When you select a percentage of Rain Gardens and Rain Barrels for the watershed:

        - The percentage is applied to each subcatchment's **own capacity** based on land use.
        - Areas that cannot support as many rain gardens are filled with **rain barrels instead**.
        - This keeps the overall investment the same across the watershed.
        - For more detail on how RG and RB placement works, see the [supporting document](https://docs.google.com/document/d/1I3sWiiGf6CqeSLmHuhr8rE60eiyLlCsPaQ_FMWZ4Ckc/edit?usp=sharing).
        """)
        
        row1 = st.columns(2, gap="small")
        with row1[0]:
            render_focus_placement_map(
                plan_all,
                "All subcatchments",
                WS_SHP_PATH,
                "fa_place_all"
            )
        with row1[1]:
            render_focus_placement_map(
                plan_upstream,
                "Upstream",
                WS_SHP_PATH,
                "fa_place_up"
            )

        row2 = st.columns(2, gap="small")
        with row2[0]:
            render_focus_placement_map(
                plan_downstr,
                "Downstream/outlet",
                WS_SHP_PATH,
                "fa_place_dn"
            )
        with row2[1]:
            render_focus_placement_map(
                plan_highro,
                "Highest runoff",
                WS_SHP_PATH,
                "fa_place_hi"
            )

    RG_STORAGE_FT3 = 107.27
    RB_STORAGE_FT3 = 7.34

    def build_focus_summary_df(plans_dict: dict, unit_cost_rg: float, unit_cost_rb: float) -> pd.DataFrame:
        """
        Build one row per focus area with:
        - RG_count, RB_count
        - TotalCost_K  (in $ thousands)
        - Storage_ft3  (RG*140.6 + RB*7.35)
        """
        rows = []
        for label, plan in plans_dict.items():
            # Counts
            rg_cnt = int(sum(v.get("rain_gardens", 0) for v in plan.values()))
            rb_cnt = int(sum(v.get("rain_barrels", 0)  for v in plan.values()))
            # Cost
            total_cost = unit_cost_rg * rg_cnt + unit_cost_rb * rb_cnt
            # Storage (ft³)
            storage_ft3 = rg_cnt * RG_STORAGE_FT3 + rb_cnt * RB_STORAGE_FT3 

            rows.append({
                "Focus Area": label,
                "RG_count": rg_cnt,
                "RB_count": rb_cnt,
                "TotalCost_K": total_cost / 1_000.0,   # dollars → $K for display
                "Storage_ft3": storage_ft3
            })
        return pd.DataFrame(rows)


    if plan_base is not None:
        # Prepare inputs
        plans_dict = {
            "All":        plan_all,
            "Upstream":   plan_upstream,
            "Downstream": plan_downstr,
            "High runoff": plan_highro,
        }
        df_sum = build_focus_summary_df(plans_dict, unit_cost_rg, unit_cost_rb)  
        # Apply unit conversion to storage volumes for UI
        ui_unit = st.session_state.get("unit_ui", "U.S. Customary")

        df_sum["Storage_ft3"] = df_sum["Storage_ft3"].apply(lambda v: to_ui_volume(v, ui_unit))
        storage_unit = "ft³" if ui_unit == "U.S. Customary" else "m³"

        # Focus area ordering: sort by Total Cost ($K) descending
        focus_order = df_sum.sort_values("TotalCost_K", ascending=False)["Focus Area"].tolist()

        # Focus area colors (keep your palette)
        focus_colors = {
            "All":        "#046d64",  # dark teal
            "Upstream":   "#1316AC",  # indigo
            "Downstream": "#3e95d3",  # steel blue
            "High runoff":"#9555d1",  # purple
        }

        # Helper: robust axis domain (always start at zero, pad by 10%)
        def _domain_zero_to_max(values: pd.Series, pad_ratio: float = 0.10) -> list[float]:
            arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if arr.size == 0:
                return [0.0, 1.0]
            mx = float(np.max(arr))
            if not np.isfinite(mx) or mx <= 0:
                return [0.0, 1.0]
            return [0.0, mx * (1.0 + pad_ratio)]

        # Build one metric chart with shared styling & labels
        def _metric_chart(df: pd.DataFrame,
                        value_col: str,
                        title_text: str,
                        color_map: dict,
                        order: list[str],
                        height_px: int = 280):
            import altair as alt
            try:
                alt.data_transformers.disable_max_rows()
            except Exception:
                pass

            # Prepare data
            df_plot = df[["Focus Area", value_col]].copy()
            df_plot.rename(columns={value_col: "Value"}, inplace=True)
            df_plot["Color"] = df_plot["Focus Area"].map(color_map).fillna("#777777")

            x_dom = _domain_zero_to_max(df_plot["Value"], pad_ratio=0.10)

            base = alt.Chart(df_plot)

            bars = (
                base.mark_bar()
                .encode(
                    y=alt.Y("Focus Area:N", sort=order, title=None,
                            axis=alt.Axis(labelFontSize=16)),
                    x=alt.X("Value:Q",
                            title=title_text,
                            scale=alt.Scale(domain=x_dom, nice=False, zero=True),
                            axis=alt.Axis(format=",.0f", titleFontSize=16, labelFontSize=16)),
                    color=alt.Color("Color:N", scale=None, legend=None),
                    tooltip=[
                        alt.Tooltip("Focus Area:N"),
                        alt.Tooltip("Value:Q", title=title_text, format=",.0f")
                    ]
                )
                .properties(height=height_px)
            )

            chart = (bars).configure_view(strokeWidth=0)
            return chart

        st.markdown("### Summary")

        with st.expander("Understanding Costs, LID Mix, and Storage"):
            st.markdown("""
        CoastWise may show different numbers of rain gardens and rain barrels across the four layouts even when the total cost is the same. This is because each part of the watershed has different limits on how many rain gardens it can support.

        When an area reaches its rain garden limit, rain barrels are added instead to keep the total investment equal.

        Although the cost stays the same, the storage can differ because each LID type holds a different amount of water:

        - Each rain barrel provides about **55 gallons** or 0.2 cubic meters.
        - Each rain garden provides about **1,050 gallons** or 4 cubic meters.

        A layout with fewer rain gardens and more rain barrels will have **less total storage**, even with the same budget. 
        """)

        # --- Row 1: RG count | RB count ---
        row1 = st.columns(2, gap="large")
        with row1[0]:
            st.markdown("**RG count**")
            ch_rg = _metric_chart(df_sum, "RG_count", "Count", focus_colors, focus_order)
            st.altair_chart(ch_rg, use_container_width=True)
        with row1[1]:
            st.markdown("**RB count**")
            ch_rb = _metric_chart(df_sum, "RB_count", "Count", focus_colors, focus_order)
            st.altair_chart(ch_rb, use_container_width=True)

        # --- Row 2: Total cost ($K) | Storage (yd³) ---
        row2 = st.columns(2, gap="large")
        with row2[0]:
            st.markdown("**Total cost ($K)**")
            ch_cost = _metric_chart(df_sum, "TotalCost_K", "Total cost ($K)", focus_colors, focus_order)
            st.altair_chart(ch_cost, use_container_width=True)
        with row2[1]:
            st.markdown(f"**Storage ({storage_unit})**")
            ch_storage = _metric_chart(df_sum, "Storage_ft3", f"Storage ({storage_unit})", focus_colors, focus_order)
            st.altair_chart(ch_storage, use_container_width=True)

        if plan_base is not None:
            st.markdown("### Run Focus Area Scenarios")

            st.info(
                "This will run all 10 scenarios: Baseline, All-Subcatchments, "
                "Upstream, Downstream, High-Runoff 2x where the Tide Gate and Pump are On/Off, respectively."
            )

            if st.button("Run All Scenarios", key=f"{prefix}_run_all_scenarios"):

                st.info("Running all 10 scenarios...")

                prefix = st.session_state["scenario_prefix"]
                rain_variant = "future" if st.session_state.get("rain_variant") == "future" else "current"

                # Choose rainfall lines
                use_current = (rain_variant == "current")
                rain_lines = st.session_state["rain_lines_cur"] if use_current else st.session_state["rain_lines_fut"]

                # Available scenario groups
                plan_sets = {
                    "baseline": None,
                    "all": plan_all,
                    "upstream": plan_upstream,
                    "downstream": plan_downstr,
                    "highrunoff": plan_highro,
                }

                # Gate options
                gate_opts = {
                    "nogate": "NO",
                    "gate": "YES",
                }

                # RESET ALL STORED DATA
                st.session_state["scenario_display_labels"] = {}
                st.session_state["scenario_flood_volume"] = {}
                st.session_state["scenario_infiltration"] = {}
                st.session_state["scenario_runoff"] = {}

                # -----------------------------
                # RUN ALL 10 SCENARIOS
                # -----------------------------
                try:
                    for subset_key, plan_dict in plan_sets.items():
                        for gate_key, gate_flag in gate_opts.items():

                            # --------------------------------------------
                            # 1. Build canonical scenario key (for files & state)
                            # --------------------------------------------
                            scen_key = make_scenario_key(
                                prefix,
                                subset=subset_key,
                                gate_flag=gate_flag,
                                rain_variant=rain_variant,
                            )

                            # --------------------------------------------
                            # 2. Readable label (no rain variant)
                            # --------------------------------------------
                            display_label = make_display_label(
                                subset=subset_key,
                                gate_flag=gate_flag
                            )

                            st.session_state["scenario_display_labels"][scen_key] = display_label

                            # --------------------------------------------
                            # 3. Build LID lines
                            # --------------------------------------------
                            if subset_key == "baseline":
                                lid_lines = [";"]  # no LIDs
                            else:
                                lid_lines = generate_lid_usage_lines(plan_dict, raster_df)

                            # --------------------------------------------
                            # 4. Run SWMM scenario
                            # --------------------------------------------
                            result = run_swmm_senario(
                                scenario_name=scen_key,
                                rain_lines=rain_lines,
                                tide_lines=tide_lines,
                                lid_lines=lid_lines,
                                gate_flag=gate_flag,
                                duration_minutes=duration_minutes,
                                template_path=template_inp,
                            )

                            # --------------------------------------------
                            # 5. Save outputs (uniform storage)
                            # --------------------------------------------
                            flood_df = st.session_state.get(f"{scen_key}_node_flooding_summary")
                            infiltration = st.session_state.get(f"{scen_key}_storm_infiltration", 0.0)

                            # flood volume from runner
                            flood_vol = st.session_state.get(f"{scen_key}_storm_total_flood", 0.0)

                            st.session_state["scenario_flood_volume"][scen_key] = flood_vol
                            st.session_state["scenario_infiltration"][scen_key] = infiltration
                            st.session_state["scenario_runoff"][scen_key] = result.get("df_runoff")

                    # FINALIZE
                    st.success("All 10 scenarios completed successfully.")
                    st.session_state["scenarios_finished"] = True
                    st.session_state["display_summary"] = True

                except Exception as e:
                    st.error(f"Scenario loop failed: {e}")
                    st.stop()

    flooding_summary_ui()
    scenario_comparison_map_ui()

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