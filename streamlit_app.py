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
    g = ws_gdf.copy()
    if rpt_df.empty:
        g["Total_R"] = np.nan
        return g, ("in" if unit_ui == "U.S. Customary" else "cm")

    if unit_ui == "U.S. Customary":
        total = rpt_df["Impervious"] + rpt_df["Pervious"]; unit = "in"
    else:
        total = (rpt_df["Impervious"] + rpt_df["Pervious"]) * 2.54; unit = "cm"

    # normalize names on both sides
    def norm(n):
        if not isinstance(n, str): n = str(n)
        n = n.strip().lower()
        # coerce Sub_01 -> Sub_1 and Sub_1 -> Sub_1
        m = re.fullmatch(r"sub[_\s-]*0*(\d+)", n)
        return f"sub_{m.group(1)}" if m else re.sub(r"\s+", "", n)

    left = rpt_df.assign(NAME_JOIN=rpt_df["Subcatchment"].astype(str).map(norm)).loc[:, ["NAME_JOIN"]]
    left["Total_R"] = total.values

    g["NAME_JOIN"] = g["NAME"].astype(str).map(norm)
    merged = g.merge(left, on="NAME_JOIN", how="left").drop(columns=["NAME_JOIN"])

    # debug: show match stats
    matched = int(merged["Total_R"].notna().sum())
    st.caption(f"Matched {matched}/{len(merged)} subcatchments from report to shapefile.")
    if matched == 0:
        st.warning("Zero matches. Check that shapefile NAMEs truly look like 'Sub_#' (no hidden spaces or zero-padding).")

    return merged, unit

def extract_runoff_and_lid_data(rpt_file: str) -> pd.DataFrame:
    """
    Robustly parse the 'Subcatchment Runoff Summary' table.
    Returns columns: Subcatchment, Impervious, Pervious (values in the report's unit).
    If only a single 'Runoff' column exists, it is placed in Impervious and Pervious=0.
    """
    import re

    if not os.path.exists(rpt_file):
        return pd.DataFrame(columns=["Subcatchment", "Impervious", "Pervious"])

    with open(rpt_file, "r", errors="ignore") as f:
        lines = f.readlines()

    # 1) Find the section header
    sec_i = next((i for i, l in enumerate(lines) if "Subcatchment Runoff Summary" in l), None)
    if sec_i is None:
        return pd.DataFrame(columns=["Subcatchment", "Impervious", "Pervious"])

    # dash line helper
    def is_dash_line(s: str) -> bool:
        s = s.strip()
        return len(s) > 0 and set(s) <= set("- ")

    # 2) Find the header line between two dash lines
    i = sec_i + 1
    while i < len(lines) and not is_dash_line(lines[i]):  # first dashes
        i += 1
    if i >= len(lines):
        return pd.DataFrame(columns=["Subcatchment", "Impervious", "Pervious"])

    # header should be next non-blank, non-dash line
    i += 1
    while i < len(lines) and (is_dash_line(lines[i]) or not lines[i].strip()):
        i += 1
    if i >= len(lines):
        return pd.DataFrame(columns=["Subcatchment", "Impervious", "Pervious"])
    header_line = lines[i].rstrip("\n")

    # find the underline dashes after header
    i += 1
    while i < len(lines) and not is_dash_line(lines[i]):
        i += 1
    if i >= len(lines):
        return pd.DataFrame(columns=["Subcatchment", "Impervious", "Pervious"])

    # 3) Map header columns
    header_cols = re.split(r"\s{2,}", header_line.strip())
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())

    name_idx = next((k for k, c in enumerate(header_cols) if norm(c).startswith("subcatchment")), 0)
    imperv_idx = next((k for k, c in enumerate(header_cols) if "imperv" in norm(c) and "runoff" in norm(c)), None)
    perv_idx   = next((k for k, c in enumerate(header_cols) if "perv"   in norm(c) and "runoff" in norm(c)), None)
    runoff_idx = next((k for k, c in enumerate(header_cols) if norm(c) == "runoff" or norm(c).endswith(" runoff")), None)

    # 4) Iterate data rows
    rows = []
    i += 1  # start after underline
    while i < len(lines):
        s = lines[i].rstrip("\n")
        if not s.strip() or is_dash_line(s):
            break
        if "Analysis Options" in s or "Node Depth Summary" in s:
            break

        parts = re.split(r"\s{2,}", s.strip())
        if len(parts) < 2:
            i += 1
            continue
        if len(parts) < len(header_cols):
            parts = parts + [""] * (len(header_cols) - len(parts))

        sub = parts[name_idx] if name_idx < len(parts) else parts[0]

        def fget(idx):
            try:
                return float(parts[idx]) if idx is not None and parts[idx] not in ("", None) else None
            except Exception:
                return None

        imperv = fget(imperv_idx)
        perv   = fget(perv_idx)

        if imperv is None and perv is None:
            r = fget(runoff_idx)
            if r is not None:
                imperv, perv = r, 0.0

        if imperv is not None or perv is not None:
            rows.append({"Subcatchment": sub, "Impervious": imperv or 0.0, "Pervious": perv or 0.0})

        i += 1

    return pd.DataFrame(rows, columns=["Subcatchment", "Impervious", "Pervious"])

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
    unit_ui: str, ws_shp_path: str, node_shp_path: str, node_name_field_hint: str = "NAME"
):
    ws_gdf = load_ws(ws_shp_path)
    gdf_left,  runoff_unit_L = prep_total_runoff_gdf(left_df_in_inches,  unit_ui, ws_gdf)
    gdf_right, runoff_unit_R = prep_total_runoff_gdf(right_df_in_inches, unit_ui, ws_gdf)
    runoff_unit = runoff_unit_L

    vals = pd.concat([gdf_left["Total_R"], gdf_right["Total_R"]], ignore_index=True)
    if len(vals) == 0 or not np.isfinite(np.nanmin(vals)) or not np.isfinite(np.nanmax(vals)):
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1e-6

    gdf_left["_fill_total"]  = make_color(gdf_left["Total_R"],  vmin, vmax)
    gdf_right["_fill_total"] = make_color(gdf_right["Total_R"], vmin, vmax)
    for g in (gdf_left, gdf_right):
        g["_label"] = g["NAME"]

    centroid = pd.concat([gdf_left.geometry, gdf_right.geometry]).unary_union.centroid
    view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=14.25)

    def _poly_and_label_layers(gdf):
        poly_layer = pdk.Layer(
            "GeoJsonLayer",
            data=gdf.__geo_interface__,
            pickable=True,
            stroked=True,
            filled=True,
            get_fill_color="properties._fill_total",
            get_line_color=[255,255,255,255],
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
            get_color=[0,0,0],
            get_alignment_baseline="'center'"
        )
        return poly_layer, text_layer

    left_poly,  left_text  = _poly_and_label_layers(gdf_left)
    right_poly, right_text = _poly_and_label_layers(gdf_right)

    left_nodes_layer  = node_layer_from_shp(node_shp_path, left_nodes_post5h_dict,  node_name_field_hint)
    right_nodes_layer = node_layer_from_shp(node_shp_path, right_nodes_post5h_dict, node_name_field_hint)

    c1, c2 = st.columns(2, gap="medium")
    tooltip_html = "<b>{NAME}</b><br/>Total runoff: {Total_R} " + runoff_unit

    with c1:
        st.markdown(f"**{left_title}**")
        layers = [left_poly, left_text] + ([left_nodes_layer] if left_nodes_layer is not None else [])
        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                map_provider="carto",
                map_style="light",
                tooltip={"html": tooltip_html, "style": {"backgroundColor": "white", "color": "black"}}
            ),
            use_container_width=True
        )

    with c2:
        st.markdown(f"**{right_title}**")
        layers = [right_poly, right_text] + ([right_nodes_layer] if right_nodes_layer is not None else [])
        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                map_provider="carto",
                map_style="light",
                tooltip={"html": tooltip_html, "style": {"backgroundColor": "white", "color": "black"}}
            ),
            use_container_width=True
        )

    # Legend
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = mpl.colormaps.get_cmap("Blues")
    c0  = [int(v*255) for v in cmap(norm(vmin))[:3]]
    c1b = [int(v*255) for v in cmap(norm(vmax))[:3]]
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

    st.session_state[f"{scenario_name}_total_flood"] = (cumulative_flooding_acft[-1] if cumulative_flooding_acft else 0.0)
    st.session_state[f"{scenario_name}_post5h_total_flood"] = (cum_cuft_post5h / 43560.0) if cum_cuft_post5h > 0 else 0.0
    st.session_state[f"{scenario_name}_node_flood_post5h_cuft"] = node_cuft_post5h
    return cumulative_flooding_acft, timestamps, rpt_path


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
    st.success(f"Logged in as: {st.session_state['email']}")
    ensure_temp_dir()
    user_id = st.session_state.get("user_id", "guest")
    prefix = f"user_{user_id}_"
    st.title("CoastWise: Watershed Design Toolkit (SWMM)")
    st.markdown("[📘 Tutorial](https://docs.google.com/document/d/1xMxoe41xhWPsPlzUIjQP4K9K_vjhjelN0hgvqfoflGY/edit?usp=sharing)")

    simulation_date = "05/31/2025 12:00"
    template_inp    = "swmm_project.inp"

    # ---- Scenario settings: persist across reruns
    if "cfg" not in st.session_state:
        st.session_state.cfg = {
            "unit": "U.S. Customary",
            "moon_phase": list(moon_tide_ranges.keys())[0],
            "duration_minutes": int(pf_df["Duration_Minutes"].iloc[0]),
            "return_period": "1",
            "align_mode": "peak",
            "settings_ready": False,
        }

    cfg = st.session_state.cfg

    with st.form("scenario_form", clear_on_submit=False):
        unit = st.selectbox(
            "Preferred Units",
            ["U.S. Customary", "Metric (SI)"],
            index=(0 if cfg["unit"] == "U.S. Customary" else 1),
        )

        # tide is always: try live; if not available, fall back to synthetic (moon_phase picker only)
        moon_phase = st.selectbox(
            "Synthetic Tide (fallback if live is unavailable)",
            list(moon_tide_ranges.keys()),
            index=max(
                0,
                list(moon_tide_ranges.keys()).index(cfg["moon_phase"])
                if cfg["moon_phase"] in moon_tide_ranges
                else 0,
            ),
        )

        duration_minutes = st.selectbox(
            "Storm Duration",
            options=pf_df["Duration_Minutes"],
            index=int(np.where(pf_df["Duration_Minutes"].values == cfg["duration_minutes"])[0][0])
                if cfg["duration_minutes"] in pf_df["Duration_Minutes"].values else 0,
            format_func=lambda x: f"{x // 60} hr",
        )

        ret_opts = return_period_labels(int(duration_minutes), unit)
        ret_keys = list(ret_opts.keys())
        ret_idx = ret_keys.index(cfg["return_period"]) if cfg["return_period"] in ret_keys else 0
        return_label = st.selectbox("Return Year", list(ret_opts.values()), index=ret_idx)
        return_period = [k for k, v in ret_opts.items() if v == return_label][0]

        tide_align_label = st.radio(
            "Tide Alignment",
            ["Peak aligned with High Tide", "Peak aligned with Low Tide"],
            index=(0 if cfg["align_mode"] == "peak" else 1),
        )

        submitted = st.form_submit_button("Apply Settings")
    # ======================================================================

    if submitted:
        st.session_state.cfg = {
            "unit": unit,
            "moon_phase": moon_phase,
            "duration_minutes": int(duration_minutes),
            "return_period": str(return_period),
            "align_mode": ("peak" if "High" in tide_align_label else "low"),
            "settings_ready": True,
        }
        cfg = st.session_state.cfg
        st.success("Settings applied.")


    # ---- Only proceed if settings were applied at least once
    if not cfg.get("settings_ready", False):
        st.info("Adjust settings and click **Apply Settings**.")
        st.stop()

    # ---- Build tide + rain using PERSISTED settings
    align_mode = cfg["align_mode"]
    unit = cfg["unit"]
    moon_phase = cfg["moon_phase"]
    duration_minutes = int(cfg["duration_minutes"])
    return_period = cfg["return_period"]

    total_inches = float(pf_df.loc[pf_df["Duration_Minutes"] == duration_minutes, return_period].values[0])

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
        minutes_15, tide_curve_ui, used_live = get_tide_real_or_synthetic(
            moon_phase, unit, start_ts=None, navd88_to_sea_level_offset_ft=MSL_OFFSET_NAVD88_FT
        )
        from rainfall_and_tide_generator import align_rainfall_to_tide
        _, rain_curve_in, _ = align_rainfall_to_tide(
            total_inches=total_inches,
            duration_minutes=duration_minutes,
            tide_curve_15min=tide_curve_ui,
            align=align_mode,
            method="Normal",
            target_index=None,
            prominence=None
        )
        tide_source = "synthetic"

    # ---- Display conversions
    if unit == "U.S. Customary":
        display_rain_curve = rain_curve_in               # inches
        display_tide_curve = tide_curve_ui               # ft
        rain_disp_unit = "inches"; tide_disp_unit = "ft"
    else:
        display_rain_curve = rain_curve_in * 2.54        # inches -> cm
        display_tide_curve = tide_curve_ui               # meters
        rain_disp_unit = "centimeters"; tide_disp_unit = "meters"

    # ---- Charts
    time_hours = np.array(minutes_15, dtype=float) / 60.0
    df_rain = pd.DataFrame({
        "Time (hours)": time_hours,
        "Current": display_rain_curve,
        "Future (+20%)": display_rain_curve * 1.2,
    }).melt("Time (hours)", var_name="Scenario", value_name=f"Rainfall ({rain_disp_unit})")

    st.subheader("Rainfall Distribution")
    st.altair_chart(
        alt.Chart(df_rain).mark_line(strokeWidth=5).encode(
            x=alt.X("Time (hours):Q"),
            y=alt.Y(f"Rainfall ({rain_disp_unit}):Q"),
            color=alt.Color(
                "Scenario:N",
                scale=alt.Scale(
                    domain=["Current", "Future (+20%)"],
                    range=["black", "orange"]
                ),
                legend=alt.Legend(
                    orient="top-left",     # inside the plotting area
                    title=None,
                    direction="vertical",
                    fillColor="white",
                    strokeColor="gray",
                    cornerRadius=3,
                    padding=4
                )
            )
        ).properties(width="container"),
        use_container_width=True
    )

    st.subheader("Tide Profile")
    df_tide = pd.DataFrame({"Time (hours)": time_hours, f"Tide ({tide_disp_unit})": display_tide_curve})
    st.altair_chart(
        alt.Chart(df_tide).mark_line(strokeWidth=5).encode(
            x=alt.X("Time (hours):Q"),
            y=alt.Y(f"Tide ({tide_disp_unit}):Q")
        ),
        use_container_width=True
    )
    st.caption("Source: Real-time tide (last 48h) at 15-min resolution." if tide_source=="live"
               else f"Source: Synthetic tide ({moon_phase}).")

    # ---- Persist computed series for scenario runs
    st.session_state.update({
        "rain_minutes": minutes_15,
        "tide_minutes": minutes_15,
        "display_rain_curve_current": display_rain_curve,
        "display_rain_curve_future": display_rain_curve * 1.2,
        "display_tide_curve": display_tide_curve,
        "rain_sim_curve_current_in": rain_curve_in,          # inches
        "rain_sim_curve_future_in": rain_curve_in * 1.2,     # inches
        "rain_disp_unit": rain_disp_unit,
        "tide_disp_unit": tide_disp_unit,
        "unit_ui": unit,
        "tide_source": tide_source,
        "moon_phase": moon_phase,
        "align_mode": align_mode,
    })


    WS_SHP_PATH = "Subcatchments.shp"
    NODE_SHP_PATH = "Nodes.shp"
    ws_gdf = load_ws(WS_SHP_PATH)
    raster_df = load_raster_cells()

    def _rain_lines_pair(sim_minutes, rain_curve_in, sim_start_str):
        cur = format_timeseries("rain_gage_timeseries", sim_minutes, rain_curve_in, sim_start_str)
        fut = format_timeseries("rain_gage_timeseries", sim_minutes, (np.array(rain_curve_in) * 1.2), sim_start_str)
        return cur, fut

    tide_lines = format_timeseries(
        "tide",
        minutes_15,
        tide_to_feet_for_swmm(st.session_state["display_tide_curve"], st.session_state["unit_ui"]),
        simulation_date
    )
    rain_lines_cur, rain_lines_fut = _rain_lines_pair(minutes_15, st.session_state["rain_sim_curve_current_in"], simulation_date)

    # ---------------- Baseline ----------------
    if st.button("Run Baseline Scenario"):
        try:
            lid_lines = [";"]  # none
            fill_nogate_cur, ts, rpt1 = run_swmm_scenario(f"{prefix}baseline_nogate_current", rain_lines_cur, tide_lines, lid_lines, "NO", template_path=template_inp)
            fill_gate_cur,  _,  rpt2 = run_swmm_scenario(f"{prefix}baseline_gate_current",   rain_lines_cur, tide_lines, lid_lines, "YES", template_path=template_inp)
            fill_nogate_fut,_,  rpt3 = run_swmm_scenario(f"{prefix}baseline_nogate_future",  rain_lines_fut, tide_lines, lid_lines, "NO", template_path=template_inp)
            fill_gate_fut,  _,  rpt4 = run_swmm_scenario(f"{prefix}baseline_gate_future",    rain_lines_fut, tide_lines, lid_lines, "YES", template_path=template_inp)

            st.session_state.update({
                f"{prefix}baseline_timestamps": ts,
                f"{prefix}baseline_fill_current": fill_nogate_cur,
                f"{prefix}baseline_gate_fill_current": fill_gate_cur,
                f"{prefix}baseline_fill_future": fill_nogate_fut,
                f"{prefix}baseline_gate_fill_future": fill_gate_fut,
                f"{prefix}df_base_nogate_current": extract_runoff_and_lid_data(rpt1),
                f"{prefix}df_base_gate_current":   extract_runoff_and_lid_data(rpt2),
                f"{prefix}df_base_nogate_future":  extract_runoff_and_lid_data(rpt3),
                f"{prefix}df_base_gate_future":    extract_runoff_and_lid_data(rpt4),
            })
            df_swmm = extract_runoff_and_lid_data(rpt1)
            if len(df_swmm) == 0:
                st.error("No rows parsed from 'Subcatchment Runoff Summary' in the .rpt")
            st.success("Baseline scenarios complete.")
        except Exception as e:
            st.error(f"Baseline simulation failed: {e}")

    # ---------------- LID config ----------------
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
            with c2: rg_val = st.number_input(f"Rain Gardens ({rg_max} max) — {sub}", 0, rg_max, 0, step=5, key=f"rg_{sub}")
            with c3: rb_val = st.number_input(f"Rain Barrels ({rb_max} max) — {sub}", 0, rb_max, 0, step=5, key=f"rb_{sub}")
            st.session_state[f"{prefix}user_lid_config"][sub] = {"rain_gardens": rg_val, "rain_barrels": rb_val}

    def generate_lid_usage_lines(lid_config: Dict[str, Dict[str,int]], excel_df: pd.DataFrame) -> List[str]:
        lines = []
        tpl = ("{sub:<15}{proc:<16}{num:>7}{area:>8}{width:>7}{initsat:>8}"
               "{fromimp:>8}{toperv:>8}{rptfile:>24}{drainto:>16}{fromperv:>9}")
        for sub, cfg in lid_config.items():
            row = excel_df.loc[excel_df["NAME"] == sub]
            if row.empty: continue
            imperv = float(row["Impervious_ft2"].iloc[0]); perv = float(row["Pervious_ft2"].iloc[0])
            rb = int(cfg.get("rain_barrels", 0))
            if rb > 0:
                pct_imp = (rb * 300) / max(imperv, 1e-9) * 100
                lines.append(tpl.format(sub=sub, proc="rain_barrel", num=rb, area=f"{2.58:.2f}", width=0,
                                        initsat=0, fromimp=f"{pct_imp:.2f}", toperv=1, rptfile="*", drainto="*", fromperv=0))
            rg = int(cfg.get("rain_gardens", 0))
            if rg > 0:
                pct_perv = (rg * 500) / max(perv, 1e-9) * 100
                lines.append(tpl.format(sub=sub, proc="rain_garden", num=rg, area=f"{100:.0f}", width=0,
                                        initsat=0, fromimp=0, toperv=1, rptfile="*", drainto="*", fromperv=f"{pct_perv:.2f}"))
        return lines

    # ---------------- Run Custom LID ----------------
    if st.button("Run Custom LID Scenario"):
        lid_cfg = st.session_state[f"{prefix}user_lid_config"]
        if not lid_cfg or all((v["rain_gardens"]==0 and v["rain_barrels"]==0) for v in lid_cfg.values()):
            st.warning("No LIDs selected.")
        else:
            try:
                lid_lines = generate_lid_usage_lines(lid_cfg, raster_df)
                fill_lid_cur, ts, rpt1 = run_swmm_scenario(f"{prefix}lid_nogate_current", rain_lines_cur, tide_lines, lid_lines, "NO", template_path=template_inp)
                fill_lid_gate_cur,_, rpt2 = run_swmm_scenario(f"{prefix}lid_gate_current",   rain_lines_cur, tide_lines, lid_lines, "YES", template_path=template_inp)
                fill_lid_fut,_,      rpt3 = run_swmm_scenario(f"{prefix}lid_nogate_future",  rain_lines_fut, tide_lines, lid_lines, "NO", template_path=template_inp)
                fill_lid_gate_fut,_, rpt4 = run_swmm_scenario(f"{prefix}lid_gate_future",    rain_lines_fut, tide_lines, lid_lines, "YES", template_path=template_inp)
                st.session_state.update({
                    f"{prefix}lid_timestamps": ts,
                    f"{prefix}lid_fill_current": fill_lid_cur,
                    f"{prefix}lid_gate_fill_current": fill_lid_gate_cur,
                    f"{prefix}lid_fill_future": fill_lid_fut,
                    f"{prefix}lid_gate_fill_future": fill_lid_gate_fut,
                    f"{prefix}df_lid_nogate_current": extract_runoff_and_lid_data(rpt1),
                    f"{prefix}df_lid_gate_current":   extract_runoff_and_lid_data(rpt2),
                    f"{prefix}df_lid_nogate_future":  extract_runoff_and_lid_data(rpt3),
                    f"{prefix}df_lid_gate_future":    extract_runoff_and_lid_data(rpt4),
                })
                st.success("Custom LID scenarios complete.")
            except Exception as e:
                st.error(f"LID simulation failed: {e}")

    # ---------------- Run Max LID ----------------
    if st.button("Run Max LID Scenario"):
        lid_cfg = {row["NAME"]: {"rain_gardens": int(row["Max_RG_DEM_Considered"]),
                                 "rain_barrels": int(row["MaxNumber_RB"])}
                   for _, row in raster_df.iterrows()}
        try:
            lid_lines = generate_lid_usage_lines(lid_cfg, raster_df)
            fill_max_cur, ts, rpt1 = run_swmm_scenario(f"{prefix}lid_max_nogate_current", rain_lines_cur, tide_lines, lid_lines, "NO", template_path=template_inp)
            fill_max_gate_cur,_, rpt2 = run_swmm_scenario(f"{prefix}lid_max_gate_current",   rain_lines_cur, tide_lines, lid_lines, "YES", template_path=template_inp)
            fill_max_fut,_,      rpt3 = run_swmm_scenario(f"{prefix}lid_max_nogate_future",  rain_lines_fut, tide_lines, lid_lines, "NO", template_path=template_inp)
            fill_max_gate_fut,_, rpt4 = run_swmm_scenario(f"{prefix}lid_max_gate_future",    rain_lines_fut, tide_lines, lid_lines, "YES", template_path=template_inp)
            st.session_state.update({
                f"{prefix}lid_max_timestamps": ts,
                f"{prefix}lid_max_fill_current": fill_max_cur,
                f"{prefix}lid_max_gate_fill_current": fill_max_gate_cur,
                f"{prefix}lid_max_fill_future": fill_max_fut,
                f"{prefix}lid_max_gate_fill_future": fill_max_gate_fut,
                f"{prefix}df_lid_max_nogate_current": extract_runoff_and_lid_data(rpt1),
                f"{prefix}df_lid_max_gate_current":   extract_runoff_and_lid_data(rpt2),
                f"{prefix}df_lid_max_nogate_future":  extract_runoff_and_lid_data(rpt3),
                f"{prefix}df_lid_max_gate_future":    extract_runoff_and_lid_data(rpt4),
            })
            st.success("Max LID scenarios complete.")
        except Exception as e:
            st.error(f"Max LID simulation failed: {e}")

    st.subheader("Watershed Baseline Runoff Map")
    key = f"{prefix}df_base_nogate_current"
    if key not in st.session_state:
        st.info("Run the Baseline Scenario first.")
    else:
        df_swmm = st.session_state[key]
        gdf, unit_r = prep_total_runoff_gdf(df_swmm, st.session_state["unit_ui"], ws_gdf)
        vals = gdf["Total_R"].values
        vmin = float(np.nanmin(vals)) if np.isfinite(np.nanmin(vals)) else 0.0
        vmax = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else 1.0
        if abs(vmax - vmin) < 1e-9: vmax = vmin + 1e-6
        gdf["_fill"] = make_color(gdf["Total_R"], vmin, vmax)
        gdf["_label"] = gdf["NAME"]
        centroid = gdf.geometry.union_all().centroid
        view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=14.25)
        poly_layer = pdk.Layer("GeoJsonLayer", data=gdf.__geo_interface__, pickable=True, stroked=True, filled=True,
                               get_fill_color="properties._fill", get_line_color=[255,255,255,255], line_width_min_pixels=1)
        reps = gdf.geometry.representative_point()
        labels = pd.DataFrame({"lon": reps.x, "lat": reps.y, "text": gdf["_label"]})
        text_layer = pdk.Layer("TextLayer", data=labels, get_position='[lon, lat]', get_text="text",
                               get_size=12, get_color=[0,0,0], get_alignment_baseline="'center'")
        tooltip = {"html": "<b>{NAME}</b><br/>Total runoff: {Total_R} "+unit_r,
                   "style": {"backgroundColor":"white","color":"black"}}
        st.pydeck_chart(pdk.Deck(layers=[poly_layer, text_layer], initial_view_state=view_state, map_provider="carto",
                                 map_style="light", tooltip=tooltip), use_container_width=True)

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
            node_shp_path=NODE_SHP_PATH,
            node_name_field_hint="NAME",
        )
    else:
        st.info("Run both scenarios: LID (+20%) and Baseline (+20%) to view the comparison maps.")


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
            df_conv["Flooded Volume"] = (df_balance["Flooding (ac-ft)"].apply(to_disp_ft3)
                                         .apply(maybe_m3)).round(0).astype(int)
            st.subheader(f"Summary ({'m³' if convert_to_m3 else 'ft³'})")
            st.dataframe(df_conv)
            st.session_state[f"{prefix}df_balance"] = df_conv

    # Export Excel
    if f"{prefix}df_balance" in st.session_state and st.button("Download Scenario Results (Excel)"):
        df_balance = st.session_state[f"{prefix}df_balance"]
        sim_start = datetime.strptime(simulation_date, "%m/%d/%Y %H:%M")
        rain_minutes = st.session_state.get("rain_minutes", [])
        tide_minutes = st.session_state.get("tide_minutes", [])
        rain_disp_unit = st.session_state.get("rain_disp_unit", "inches")
        tide_disp_unit = st.session_state.get("tide_disp_unit", "ft")

        rain_ts = st.session_state.get("display_rain_curve_current", [])
        rain_ts_f = st.session_state.get("display_rain_curve_future", [])
        tide_ts = st.session_state.get("display_tide_curve", [])

        if len(rain_ts) > 0:
            r_t = [(sim_start + timedelta(minutes=int(m))).strftime("%m/%d/%Y %H:%M") for m in rain_minutes[:len(rain_ts)]]
            df_rain = pd.DataFrame({"Timestamp": r_t,
                                    f"Rainfall – Current ({rain_disp_unit})": rain_ts[:len(r_t)],
                                    f"Rainfall – +20% ({rain_disp_unit})":    rain_ts_f[:len(r_t)]})
        else:
            df_rain = pd.DataFrame(columns=["Timestamp", f"Rainfall – Current ({rain_disp_unit})", f"Rainfall – +20% ({rain_disp_unit})"])

        if len(tide_ts) > 0:
            t_t = [(sim_start + timedelta(minutes=int(m))).strftime("%m/%d/%Y %H:%M") for m in tide_minutes[:len(tide_ts)]]
            df_tide = pd.DataFrame({"Timestamp": t_t, f"Tide ({tide_disp_unit})": tide_ts[:len(t_t)]})
        else:
            df_tide = pd.DataFrame(columns=["Timestamp", f"Tide ({tide_disp_unit})"])

        lid_cfg = st.session_state.get(f"{prefix}user_lid_config", {})
        if lid_cfg:
            rows = [{"Subcatchment": sub, "Selected Rain Gardens": cfg.get("rain_gardens", 0),
                     "Selected Rain Barrels": cfg.get("rain_barrels", 0)} for sub, cfg in lid_cfg.items()]
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
            df_balance.reset_index().rename(columns={"index":"Scenario"}).to_excel(writer, sheet_name="Scenario Summary", index=False)

        st.download_button(
            label="Download",
            data=excel_output.getvalue(),
            file_name="CoastWise_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if st.button("🚪 Logout"):
        try: shutil.rmtree(st.session_state.temp_dir)
        except Exception: pass
        st.session_state.clear()
        st.success("Logged out and cleaned up all files.")
        st.rerun()


if "user_id" not in st.session_state:
    login_ui()
else:
    app_ui()
