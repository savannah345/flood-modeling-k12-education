import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt
import time
import numpy as np
import re
import io
import os
import shutil 
import tempfile
import mpld3
import os
import glob
import subprocess, sys
from auth_supabase import create_user, authenticate_user, reset_password
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
import pydeck as pdk
from shapely.geometry import Point
from matplotlib import cm, colors as mcolors
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from pyswmm import Simulation, Links, Nodes
from rainfall_and_tide_generator import (
    pf_df,
    convert_units,
    generate_rainfall,
    align_rainfall_to_tide,
    generate_tide_curve,
    moon_tide_ranges,
    fetch_greenstream_dataframe,          # <— live data
    build_timestep_and_resample_15min,    # <— 6-min -> 15-min + units
)

# --- Ensure Playwright browsers are installed ---
@st.cache_resource(show_spinner=False)
def ensure_playwright_browsers():
    # Install Chromium once per server session
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

ensure_playwright_browsers()

st.set_page_config(
    page_title="CoastWise",
    layout="centered"
)

# Clean up files
def delete_user_files(user_id):
    patterns = [f"user_{user_id}_*.inp", f"user_{user_id}_*.rpt", f"user_{user_id}_*.out"]
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Could not delete {file}: {e}")

# Login block
if "user_id" not in st.session_state:
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
                st.session_state["login_time"] = time.time()
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

else:
    st.success(f"Logged in as: {st.session_state['email']}")
    user_id = st.session_state.get("user_id", "guest")
    prefix = f"user_{user_id}_"

    # === YOUR MAIN APP GOES BELOW HERE ===

    st.title("CoastWise: A Gamified Watershed Design Toolkit for Coastal Resilience Using the Stormwater Management Model")
    st.markdown("[📘 View the CoastWise Tutorial](https://docs.google.com/document/d/1xMxoe41xhWPsPlzUIjQP4K9K_vjhjelN0hgvqfoflGY/edit?usp=sharing)")

    simulation_date = "05/31/2025 12:00"
    template_inp    = "swmm_project.inp"

    # === Minimal reordering begins ===

    # 1) Units FIRST (needed to compute tides correctly)
    unit = st.selectbox("Preferred Units", ["U.S. Customary", "Metric (SI)"])
    unit_label = "inches" if unit == "U.S. Customary" else "centimeters"

    # 2) TIDES block SECOND (compute tide before any rainfall selections)
    st.subheader("Tides")
    use_live = st.toggle(
        "Use real-time tide Ryan Resilience Lab - Knitting Mill Creek (fallback to synthetic if unavailable)",
        value=True
    )

    @st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
    def _cached_live_tide(unit: str):
        """Fetch once per hour per unit system; returns (tide_sim_minutes, tide_sim_curve)."""
        df_live = fetch_greenstream_dataframe()
        return build_timestep_and_resample_15min(
            df_live,
            water_col="Water Level NAVD88 (ft)",
            unit=unit,  # feet (US) or meters (SI)
            start_ts=None
        )

    tide_source = "synthetic"
    tide_error  = None
    moon_phase  = None  # will set below

    if use_live:
        try:
            tide_sim_minutes, tide_sim_curve = _cached_live_tide(unit)
            tide_source = "live"
            moon_phase = "Real-time tide data"
            st.success("Loaded real-time tide.")
        except Exception as e:
            tide_error = str(e)
            tide_source = "synthetic"

    if tide_source == "synthetic":
        if use_live and tide_error:
            st.warning(f"Real-time tide unavailable; using synthetic. Reason: {tide_error}")

        st.subheader("Neap vs. Spring Tides")

        with open("NASA_Tides.mp4", "rb") as video_file:
            st.video(video_file.read())

        moon_phase = st.selectbox("Tide", list(moon_tide_ranges.keys()))
        st.subheader("When high tide and peak rainfall happen together, tidal backflow can block stormwater from draining, increasing flood risk.")

        # Synthetic tide is already 15-min and in the selected units
        tide_sim_minutes, tide_sim_curve = generate_tide_curve(moon_phase, unit)

    # 3) RAINFALL selections THIRD (now that tide is ready)
    duration_minutes = st.selectbox(
        "Storm Duration",
        options=pf_df["Duration_Minutes"],
        format_func=lambda x: f"{x // 60} hr"
    )

    def generate_return_period_labels(duration, unit_type):
        row = pf_df[pf_df["Duration_Minutes"] == duration]
        if row.empty:
            return {}
        row_data = row.iloc[0]
        label = "inches" if unit_type == "U.S. Customary" else "centimeters"
        factor = 1 if unit_type == "U.S. Customary" else 2.54
        return {
            col: f"{col}-year storm ({100//int(col)}% annual chance): {row_data[col]*factor:.2f} {label}"
            for col in pf_df.columns[1:]
        }

    return_options = generate_return_period_labels(duration_minutes, unit)
    return_label = st.selectbox("Return Year", list(return_options.values()))
    return_period = [k for k, v in return_options.items() if v == return_label][0]

    # pf_df values are in inches
    rain_inches = float(
        pf_df.loc[pf_df["Duration_Minutes"] == duration_minutes, return_period].values[0]
    )

    method = "Normal"

    # 4) Alignment uses the already-computed tide
    tide_align = st.radio(
        "Tide Alignment",
        ["Peak aligned with High Tide", "Peak aligned with Low Tide"]
    )
    align_mode = "peak" if "High" in tide_align else "low"

    # Align rainfall (in inches) to the 15-min tide curve
    rain_sim_minutes, rain_sim_curve = align_rainfall_to_tide(
        rain_inches,
        duration_minutes,
        tide_sim_curve,
        align=align_mode,
        method=method
    )

    # 5) Display conversions
    if unit == "U.S. Customary":
        display_rain_curve = rain_sim_curve            # inches
        display_tide_curve = tide_sim_curve            # feet (live or synthetic already in ft)
        tide_disp_unit     = "ft"
        rain_disp_unit     = "inches"
    else:
        display_rain_curve = rain_sim_curve * 2.54     # inches -> cm
        display_tide_curve = tide_sim_curve            # already meters for live or synthetic
        tide_disp_unit     = "meters"
        rain_disp_unit     = "centimeters"

    # --- Normalize tide arrays (live can be shorter than 48h) ---
    tide_sim_minutes = np.asarray(tide_sim_minutes)
    display_tide_curve = np.asarray(display_tide_curve)

    min_len = min(tide_sim_minutes.shape[0], display_tide_curve.shape[0])
    tide_sim_minutes  = tide_sim_minutes[:min_len]
    display_tide_curve = display_tide_curve[:min_len]

    # 6) Store for export
    st.session_state["rain_minutes"]       = rain_sim_minutes
    st.session_state["tide_minutes"]       = tide_sim_minutes
    st.session_state["display_rain_curve"] = display_rain_curve
    st.session_state["display_tide_curve"] = display_tide_curve
    st.session_state["rain_disp_unit"]     = rain_disp_unit
    st.session_state["tide_disp_unit"]     = tide_disp_unit

    # --- Rainfall chart (current vs +20%) ---
    time_hours = np.array(rain_sim_minutes) / 60.0

    # current (already in display units) and future (+20%)
    future_rain_curve_display = display_rain_curve * 1.2
    future_rain_curve_inches  = rain_sim_curve * 1.2  # base (inches) for SWMM if/when you use it later

    # make a tidy dataframe for Altair
    df_rain = pd.DataFrame({
        "Time (hours)": np.concatenate([time_hours, time_hours]),
        "Rainfall": np.concatenate([display_rain_curve, future_rain_curve_display]),
        "Scenario": (["Current"] * len(time_hours)) + (["Future (+20%)"] * len(time_hours))
    })

    st.subheader("Rainfall Distribution")
    rain_chart = (
        alt.Chart(df_rain)
        .mark_line()
        .encode(
            x=alt.X("Time (hours):Q", title="Time (hours)"),
            y=alt.Y("Rainfall:Q", title=f"Rainfall ({rain_disp_unit})"),
            color=alt.Color("Scenario:N", legend=alt.Legend(title="Rainfall Case")),
            tooltip=[
                alt.Tooltip("Time (hours):Q", format=".2f"),
                alt.Tooltip("Rainfall:Q", title=f"Rainfall ({rain_disp_unit})", format=".3f"),
                alt.Tooltip("Scenario:N")
            ]
        )
    )
    st.altair_chart(rain_chart, use_container_width=True)

    # Totals (display units)
    total_current = float(np.round(display_rain_curve.sum(), 2))
    total_future  = float(np.round(future_rain_curve_display.sum(), 2))
    st.markdown(f"**Total Rainfall – Current:** {total_current} {rain_disp_unit}")
    st.markdown(f"**Total Rainfall – Future (+20%):** {total_future} {rain_disp_unit}")

    # Save for later use in scenarios and exports
    st.session_state["display_rain_curve_current"] = display_rain_curve
    st.session_state["display_rain_curve_future"]  = future_rain_curve_display
    st.session_state["rain_sim_curve_current_in"]  = rain_sim_curve            # inches (for SWMM)
    st.session_state["rain_sim_curve_future_in"]   = future_rain_curve_inches  # inches (for SWMM)

    # Tide chart (15-min; length may be < 192 for live)
    tide_hours = tide_sim_minutes.astype(float) / 60.0

    df_tide = pd.DataFrame({
        "Time (hours)": tide_hours,
        f"Tide ({tide_disp_unit})": display_tide_curve
    })
    st.subheader("Tide Profile")
    tide_chart = (
        alt.Chart(df_tide)
        .mark_line()
        .encode(
            x="Time (hours)",
            y=f"Tide ({tide_disp_unit})"
        )
    )
    st.altair_chart(tide_chart, use_container_width=True)

    # Source label
    if tide_source == "live":
        st.caption("Source: Real-time tide – last 48 hours at 15-min resolution.")
    else:
        st.caption(f"Source: Synthetic tide ({moon_phase}).")

    # Create a persistent temp folder ONCE when user logs in
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()


    def run_swmm_scenario(
        scenario_name,
        rain_lines,
        tide_lines,
        lid_lines,
        gate_flag,
        full_depth=10.0,
        report_interval=timedelta(minutes=5),
        template_path="swmm_project.inp"
    ):
        temp_dir = st.session_state.temp_dir
        inp_path = os.path.join(temp_dir, f"{scenario_name}.inp")
        rpt_path = os.path.join(temp_dir, f"{scenario_name}.rpt")
        out_path = os.path.join(temp_dir, f"{scenario_name}.out")

        # --- 1. Create .inp file ---
        with open(template_path, "r") as f:
            text = f.read()
        text = text.replace("$RAINFALL_TIMESERIES$", "\n".join(rain_lines))
        text = text.replace("$TIDE_TIMESERIES$", "\n".join(tide_lines))
        text = text.replace("$LID_USAGE$", "\n".join(lid_lines))
        text = text.replace("$TIDE_GATE_CONTROL$", gate_flag)

        with open(inp_path, "w") as f:
            f.write(text)

        # --- 2. Run simulation ---
        cumulative_flooding_acft, timestamps = [], []
        cumulative_cuft = 0.0
        last_report_time = None

        with Simulation(inp_path) as sim:
            for step in sim:
                current_time = sim.current_time
                if last_report_time is None or (current_time - last_report_time) >= report_interval:
                    # Instantaneous flooding rate across all nodes (cfs)
                    total_flooding_cfs = sum(node.flooding for node in Nodes(sim))

                    # Convert to volume for this step (cfs × seconds)
                    step_volume_cuft = total_flooding_cfs * report_interval.total_seconds()

                    # Add to running total
                    cumulative_cuft += step_volume_cuft

                    # Convert to acre-feet
                    cumulative_acft = cumulative_cuft / 43560.0

                    cumulative_flooding_acft.append(cumulative_acft)
                    timestamps.append(current_time.strftime("%m-%d %H:%M"))

                    last_report_time = current_time

        # Optional rename if needed
        if os.path.exists(os.path.join(temp_dir, "updated_model.rpt")):
            shutil.move(os.path.join(temp_dir, "updated_model.rpt"), rpt_path)
        if os.path.exists(os.path.join(temp_dir, "updated_model.out")):
            shutil.move(os.path.join(temp_dir, "updated_model.out"), out_path)

        # Store the total flooding for this scenario
        total_flood_acft = cumulative_flooding_acft[-1] if cumulative_flooding_acft else 0.0
        st.session_state[f"{scenario_name}_total_flood"] = total_flood_acft

        return cumulative_flooding_acft, timestamps, rpt_path

    # === Time-series Formatter ===
    def format_timeseries(name, minutes, values, start_datetime):
        """
        Formats SWMM-compatible time series with proper date rollover.
        - `start_datetime`: string like "05/31/2025 12:00"
        """
        if isinstance(start_datetime, str):
            start_dt = datetime.strptime(start_datetime, "%m/%d/%Y %H:%M")
        else:
            start_dt = start_datetime

        lines = []
        for m, v in zip(minutes, values):
            current_dt = start_dt + timedelta(minutes=int(m))
            timestamp = current_dt.strftime("%m/%d/%Y %H:%M")
            lines.append(f"{name} {timestamp} {v:.5f}")
        return lines


    # === Report Parser ===
    def extract_runoff_and_lid_data(rpt_file):
        lines = open(rpt_file).read().splitlines()
        runoff_section = False
        data = []
        for line in lines:
            if "Subcatchment Runoff Summary" in line:
                runoff_section = True
                continue
            if runoff_section and line.strip()=="" and data:
                break
            if not runoff_section:
                continue
            if line.startswith("----") or line.strip().startswith("Subcatchment"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                imperv = float(parts[5])
                perv   = float(parts[6])
            except ValueError:
                continue
            data.append({
                "Subcatchment": parts[0],
                "Impervious Runoff (in)": imperv,
                "Pervious Runoff   (in)": perv
            })
        return pd.DataFrame(data)

    # --- helper to build rain time series for current vs future (+20%) ---
    def _rain_lines_pair(sim_minutes, rain_curve_in, sim_start_str):
        """Returns (rain_lines_current, rain_lines_future) in SWMM base units (inches)."""
        rain_lines_cur = format_timeseries("rain_gage_timeseries", sim_minutes, rain_curve_in, sim_start_str)
        rain_lines_fut = format_timeseries("rain_gage_timeseries", sim_minutes, (np.array(rain_curve_in) * 1.2), sim_start_str)
        return rain_lines_cur, rain_lines_fut

    st.markdown(f"""
    ### Selected Scenario Summary
    - **Storm Duration:** {duration_minutes//60} hr  
    - **Return Period:** {return_period} yr  
    - **Tide:** {moon_phase}  
    - **Tide Alignment:** {'High Tide Peak' if align_mode=='peak' else 'Low Tide Dip'}  
    - **Display Units:** {unit}
    """)

    # === Run Baseline Scenario ===
    if st.button("Run Baseline Scenario"):
        try:
            # rain in SWMM base units (inches)
            rain_lines_cur, rain_lines_fut = _rain_lines_pair(
                rain_sim_minutes,
                st.session_state["rain_sim_curve_current_in"],
                simulation_date
            )
            tide_lines = format_timeseries("tide", tide_sim_minutes, tide_sim_curve, simulation_date)
            lid_lines  = [";"]  # No LIDs in baseline

            # ---- CURRENT rain (2 scenarios) ----
            fill_nogate_cur, time_nogate, rpt1 = run_swmm_scenario(f"{prefix}baseline_nogate_current", rain_lines_cur, tide_lines, lid_lines, "NO")
            fill_gate_cur,   _,           rpt2 = run_swmm_scenario(f"{prefix}baseline_gate_current",   rain_lines_cur, tide_lines, lid_lines, "YES")

            # ---- FUTURE (+20%) rain (2 scenarios) ----
            fill_nogate_fut, _, rpt1f = run_swmm_scenario(f"{prefix}baseline_nogate_future", rain_lines_fut, tide_lines, lid_lines, "NO")
            fill_gate_fut,   _, rpt2f = run_swmm_scenario(f"{prefix}baseline_gate_future",   rain_lines_fut, tide_lines, lid_lines, "YES")

            # Save time series (timestamps shared with first run)
            st.session_state[f"{prefix}baseline_timestamps"] = time_nogate

            st.session_state[f"{prefix}baseline_fill_current"] = fill_nogate_cur
            st.session_state[f"{prefix}baseline_gate_fill_current"] = fill_gate_cur
            st.session_state[f"{prefix}baseline_fill_future"] = fill_nogate_fut
            st.session_state[f"{prefix}baseline_gate_fill_future"] = fill_gate_fut

            # Parse runoff summaries
            st.session_state[f"{prefix}df_base_nogate_current"] = extract_runoff_and_lid_data(rpt1)
            st.session_state[f"{prefix}df_base_gate_current"]   = extract_runoff_and_lid_data(rpt2)
            st.session_state[f"{prefix}df_base_nogate_future"]  = extract_runoff_and_lid_data(rpt1f)
            st.session_state[f"{prefix}df_base_gate_future"]    = extract_runoff_and_lid_data(rpt2f)

            # Display table (keep same display-unit logic; use *_current to drive maps)
            df1 = st.session_state[f"{prefix}df_base_nogate_current"]
            if unit == "U.S. Customary":
                df_disp = df1.rename(columns={
                    "Impervious Runoff (in)": "Impervious Runoff (inches)",
                    "Pervious Runoff   (in)": "Pervious Runoff (inches)"
                })
            else:
                df_disp = pd.DataFrame({
                    "Subcatchment": df1["Subcatchment"],
                    f"Impervious Runoff ({unit})": df1["Impervious Runoff (in)"] * 2.54,
                    f"Pervious Runoff ({unit})":   df1["Pervious Runoff   (in)"] * 2.54
                })
            st.session_state[f"{prefix}baseline_df"] = df_disp

            st.success("Baseline scenarios (Current & +20%) complete!")

        except Exception as e:
            st.error(f"Baseline simulation failed: {e}")


    # ===== Watershed Choropleths (Impervious vs Pervious) =====
    st.subheader("Watershed Baseline Runoff Maps")

    WS_SHP_PATH = "Subcatchments.shp"  # you confirmed this path

    # Need baseline runoff table from RPT
    if f"{prefix}df_base_nogate_current" not in st.session_state:
        st.info("Run the Baseline Scenario first.")
    else:
        # --- 1) SWMM runoff (inches from parser) -> to requested display units ---
        df_swmm = st.session_state[f"{prefix}df_base_nogate_current"].copy()  # has: Subcatchment, Impervious Runoff (in), Pervious Runoff   (in)
        # Normalize join key to match shapefile NAME
        df_swmm["NAME"] = df_swmm["Subcatchment"].astype(str).str.strip()

        # Convert to display units
        if unit == "U.S. Customary":
            df_swmm["Impervious_R"] = df_swmm["Impervious Runoff (in)"]
            df_swmm["Pervious_R"]   = df_swmm["Pervious Runoff   (in)"]
            runoff_unit = "in"
        else:
            df_swmm["Impervious_R"] = df_swmm["Impervious Runoff (in)"] * 2.54
            df_swmm["Pervious_R"]   = df_swmm["Pervious Runoff   (in)"] * 2.54
            runoff_unit = "cm"

        # --- 2) Read shapefile and prep CRS ---
        try:
            gdf_ws = gpd.read_file(WS_SHP_PATH)
        except Exception as e:
            st.error(f"Could not read shapefile: {e}")
            gdf_ws = None

        if gdf_ws is not None and not gdf_ws.empty:
            if gdf_ws.crs is None:
                gdf_ws = gdf_ws.set_crs(4326)
            else:
                gdf_ws = gdf_ws.to_crs(4326)

            # --- 3) Join on NAME ---
            gdf_ws["NAME"] = gdf_ws["NAME"].astype(str).str.strip()
            gdf = gdf_ws.merge(df_swmm[["NAME","Impervious_R","Pervious_R"]], on="NAME", how="left")

            # Compute ONE global scale across both metrics
            vals = pd.concat([gdf["Impervious_R"], gdf["Pervious_R"]], ignore_index=True)
            global_min = float(np.nanmin(vals))
            global_max = float(np.nanmax(vals))

            # Guards: all-NaN or constant field cases
            if not np.isfinite(global_min) or not np.isfinite(global_max):
                global_min, global_max = 0.0, 1.0
            if abs(global_max - global_min) < 1e-9:
                global_max = global_min + 1e-6

            def make_color(values, vmin, vmax, a=0.9):
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                cmap = cm.get_cmap("Blues")
                out = []
                for v in values:
                    if pd.isna(v):
                        out.append([200,200,200,int(0.4*255)])  # gray for missing
                    else:
                        r,g,b,_ = cmap(norm(float(v)))
                        out.append([int(r*255), int(g*255), int(b*255), int(a*255)])
                return out

            # Apply shared scale to both
            gdf["_fill_imp"]  = make_color(gdf["Impervious_R"], global_min, global_max)
            gdf["_fill_perv"] = make_color(gdf["Pervious_R"], global_min, global_max)
            gdf["_label"] = gdf["NAME"]  # <-- add this

            # --- 5) View + layers ---

            centroid = gdf.geometry.union_all().centroid
            view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=14.25)

            geojson = gdf.__geo_interface__

            imp_layer = pdk.Layer(
                "GeoJsonLayer",
                data=geojson,
                pickable=True,
                stroked=True,
                filled=True,
                get_fill_color="properties._fill_imp",   #  impervious colors
                get_line_color=[255, 255, 255, 255],
                line_width_min_pixels=1,
            )

            perv_layer = pdk.Layer(
                "GeoJsonLayer",
                data=geojson,
                pickable=True,
                stroked=True,
                filled=True,
                get_fill_color="properties._fill_perv",  # pervious colors (fixed)
                get_line_color=[255, 255, 255, 255],
                line_width_min_pixels=1,
            )


            # Labels at representative points
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

            # two columns of maps
            c1, c2 = st.columns(2, gap="medium")

            tooltip = {
                "html": (
                    "<b>{NAME}</b><br/>"
                    "Impervious: {Impervious_R} " + runoff_unit + "<br/>"
                    "Pervious: {Pervious_R} " + runoff_unit
                ),
                "style": {"backgroundColor": "white", "color": "black"}
            }


            with c1:
                st.markdown(f"**Impervious Runoff ({runoff_unit})**")
                st.pydeck_chart(pdk.Deck(layers=[imp_layer, text_layer],
                                        initial_view_state=view_state,
                                        map_style="light",
                                        tooltip=tooltip),
                                use_container_width=True)

            with c2:
                st.markdown(f"**Pervious Runoff ({runoff_unit})**")
                st.pydeck_chart(pdk.Deck(layers=[perv_layer, text_layer],
                                        initial_view_state=view_state,
                                        map_style="light",
                                        tooltip=tooltip),
                                use_container_width=True)

            # === ONE shared legend centered under both maps ===
            from matplotlib import colormaps
            norm = mcolors.Normalize(vmin=global_min, vmax=global_max, clip=True)
            cmap = colormaps.get_cmap("Blues")
            c0  = [int(v*255) for v in cmap(norm(global_min))[:3]]
            c1b = [int(v*255) for v in cmap(norm(global_max))[:3]]

            st.markdown(
                f"""
                <div style="display:flex; justify-content:center; margin-top:6px;">
                <div style="min-width:260px; max-width:640px; width:60%;">
                    <div style="text-align:center; font-size:13px;"><b>Runoff Legend ({runoff_unit})</b></div>
                    <div style="display:flex; align-items:center; gap:10px;">
                    <span>{global_min:.2f}</span>
                    <div style="flex:1; height:12px;
                        background:linear-gradient(to right,
                        rgb({c0[0]},{c0[1]},{c0[2]}),
                        rgb({c1b[0]},{c1b[1]},{c1b[2]}));
                        border:1px solid #888;"></div>
                    <span>{global_max:.2f}</span>
                    </div>
                    <div style="color:#555; font-size:12px; text-align:center; margin-top:6px;">
                    Same scale for both maps
                    </div>
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- Cost and Sizing Assumptions ---
    data = {
        "Infrastructure": [
            "100 sq.ft. Rain Garden",
            "55 gallon Rain Barrel",
            "Tide Gate (10'x5')"
        ],
        "Estimated Installation Cost": [
            "$250",
            "$100",
            "$250,000"
        ]
    }

    cost_df = pd.DataFrame(data)

    # --- Display in Streamlit ---
    st.subheader("Assumed Costs for Flood Mitigation Options")
    st.table(cost_df)


    # Load & sort subcatchments
    def extract_number(name):
        if not isinstance(name, str):
            return float('inf')  # Push non-string or missing names to the bottom
        m = re.search(r"_(\d+)", name)
        return int(m.group(1)) if m else float('inf')

    df = pd.read_excel("raster_cells_per_sub.xlsx")
    df = df.sort_values(by="NAME", key=lambda x: x.map(extract_number)).reset_index(drop=True)

    def generate_lid_usage_lines(lid_config,
                                excel_path="raster_cells_per_sub.xlsx"):
        df = pd.read_excel(excel_path)
        lines = []

        # template matching your desired column widths
        tpl = (
            "{sub:<15}"     # Subcatchment, left-justified width 15
            "{proc:<16}"    # LID Process, left-justified width 16
            "{num:>7}"      # Number, right-justified width 7
            "{area:>8}"     # Area, right-justified width 8
            "{width:>7}"    # Width, right-justified width 7
            "{initsat:>8}"  # InitSat, right-justified width 8
            "{fromimp:>8}"  # FromImp, right-justified width 8
            "{toperv:>8}"   # ToPerv, right-justified width 8
            "{rptfile:>24}" # RptFile, right-justified width 24
            "{drainto:>16}" # DrainTo, right-justified width 16
            "{fromperv:>9}" # FromPerv, right-justified width 9
        )

        for sub, cfg in lid_config.items():
            # find the subcatchment row
            row = df.loc[df["NAME"] == sub]
            if row.empty:
                continue
            imperv = float(row["Impervious_ft2"].iloc[0])
            perv   = float(row["Pervious_ft2"].iloc[0])

            # rain barrels
            rb = cfg.get("rain_barrels", 0)
            if rb > 0:
                pct_imp = (rb * 300) / imperv * 100
                lines.append(tpl.format(
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
                    fromperv=0
                ))

            # rain gardens
            rg = cfg.get("rain_gardens", 0)
            if rg > 0:
                pct_perv = (rg * 500) / perv * 100
                lines.append(tpl.format(
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
                    fromperv=f"{pct_perv:.2f}"
                ))

        return lines

    if st.button("Run Max LID Scenario"):
        # Build max LID config
        max_lid_config = {}
        for idx, row in df.iterrows():
            sub = row["NAME"]
            rg_max = int(row["Max_RG_DEM_Considered"])
            rb_max = int(row["MaxNumber_RB"])
            max_lid_config[sub] = {"rain_gardens": rg_max, "rain_barrels": rb_max}

        lid_lines = generate_lid_usage_lines(max_lid_config)
        tide_lines = format_timeseries("tide", tide_sim_minutes, tide_sim_curve, simulation_date)

        # rain pairs
        rain_lines_cur, rain_lines_fut = _rain_lines_pair(
            rain_sim_minutes,
            st.session_state["rain_sim_curve_current_in"],
            simulation_date
        )

        # CURRENT (2)
        fill_max_cur, time_max, rpt5 = run_swmm_scenario(f"{prefix}lid_max_nogate_current", rain_lines_cur, tide_lines, lid_lines, "NO")
        fill_max_gate_cur, _,   rpt6 = run_swmm_scenario(f"{prefix}lid_max_gate_current",   rain_lines_cur, tide_lines, lid_lines, "YES")

        # FUTURE (2)
        fill_max_fut,  _, rpt5f = run_swmm_scenario(f"{prefix}lid_max_nogate_future", rain_lines_fut, tide_lines, lid_lines, "NO")
        fill_max_gate_fut,_, rpt6f = run_swmm_scenario(f"{prefix}lid_max_gate_future",   rain_lines_fut, tide_lines, lid_lines, "YES")

        # Store time series
        st.session_state[f"{prefix}lid_max_timestamps"] = time_max
        st.session_state[f"{prefix}lid_max_fill_current"] = fill_max_cur
        st.session_state[f"{prefix}lid_max_gate_fill_current"] = fill_max_gate_cur
        st.session_state[f"{prefix}lid_max_fill_future"] = fill_max_fut
        st.session_state[f"{prefix}lid_max_gate_fill_future"] = fill_max_gate_fut

        # Runoff summaries
        st.session_state[f"{prefix}df_lid_max_nogate_current"] = extract_runoff_and_lid_data(rpt5)
        st.session_state[f"{prefix}df_lid_max_gate_current"]   = extract_runoff_and_lid_data(rpt6)
        st.session_state[f"{prefix}df_lid_max_nogate_future"]  = extract_runoff_and_lid_data(rpt5f)
        st.session_state[f"{prefix}df_lid_max_gate_future"]    = extract_runoff_and_lid_data(rpt6f)

        # Cost (unchanged)
        max_total_cost = (df["Max_RG_DEM_Considered"].sum() * 250 +
                        df["MaxNumber_RB"].sum() * 100 +
                        250000)
        st.session_state[f"{prefix}max_total_cost"] = max_total_cost

        st.success("Max LID scenarios (Current & +20%) complete!")

    if st.button("Clear LID Selections"):
        st.session_state[f"{prefix}user_lid_config"] = {}
        for sub in df["NAME"]:
            if f"rg_{sub}" in st.session_state:
                del st.session_state[f"rg_{sub}"]
            if f"rb_{sub}" in st.session_state:
                del st.session_state[f"rb_{sub}"]
        st.rerun()


    if f"{prefix}user_lid_config" not in st.session_state:
        st.session_state[f"{prefix}user_lid_config"] = {}

    available_subs = df["NAME"].tolist()

    st.subheader("Add LID features")
                
    selected_subs = st.multiselect(
        "Select subcatchments to add rain gardens or rain barrels",
        options=available_subs,
        help="Pick one or more to configure LIDs."
    )

    if selected_subs:
        st.markdown("""
            <style>
            .lid-table-container { max-height:600px; overflow-y:scroll; border:1px solid #ccc; }
            .lid-row { display:grid; grid-template-columns:1fr 1.2fr 1.2fr 1.2fr 1.2fr; padding:6px; font-size:16px; align-items:center; border-bottom:1px solid #ccc; }
            .alt-row { background-color:#f9f9f9; }
            </style>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div class="lid-row" style="font-weight:bold;background:#ddd;">
            <div>Subcatchment</div><div>Max Rain Gardens</div><div>Your Rain Gardens</div>
            <div>Max Rain Barrels</div><div>Your Rain Barrels</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="lid-table-container">', unsafe_allow_html=True)
        for idx, row in df[df["NAME"].isin(selected_subs)].iterrows():
            sub    = row["NAME"]
            rg_max = int(row["Max_RG_DEM_Considered"])
            rb_max = int(row["MaxNumber_RB"])
            cls    = "lid-row alt-row" if idx%2==0 else "lid-row"
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns([1,1.2,1.2,1.2,1.2])
            with c1: st.markdown(f"**{sub}**", unsafe_allow_html=True)
            with c2: st.markdown(f"<div style='text-align:center'>{rg_max}</div>", unsafe_allow_html=True)
            with c3: rg_val = st.number_input("Rain Garden", 0, rg_max, 0, step=5, key=f"rg_{sub}", label_visibility="collapsed")
            with c4: st.markdown(f"<div style='text-align:center'>{rb_max}</div>", unsafe_allow_html=True)
            with c5: rb_val = st.number_input("Rain Barrel", 0, rb_max, 0, step=5, key=f"rb_{sub}", label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)
            st.session_state[f"{prefix}user_lid_config"][sub] = {
                "rain_gardens": rg_val,
                "rain_barrels": rb_val
            }
        st.markdown('</div>', unsafe_allow_html=True)

            # === Compute & Display LID Costs ===
        total_cost = 0
        cost_breakdown = []

        # 1) Per‐subcatchment costs
        for sub, cfg in st.session_state[f"{prefix}user_lid_config"].items():
            rg = cfg.get("rain_gardens", 0)
            rb = cfg.get("rain_barrels", 0)
            if rg > 0:
                cost = rg * 250
                cost_breakdown.append({
                    "Subcatchment": sub,
                    "LID Type": "Rain Garden",
                    "Cost": cost
                })
                total_cost += cost
            if rb > 0:
                cost = rb * 100
                cost_breakdown.append({
                    "Subcatchment": sub,
                    "LID Type": "Rain Barrel",
                    "Cost": cost
                })
                total_cost += cost

        # 2) Tide gate cost
        # If user placed any LIDs, we assume tide gate is used because both LID runs include it
        if any(v["rain_gardens"] > 0 or v["rain_barrels"] > 0 for v in st.session_state[f"{prefix}user_lid_config"].values()):
            tide_cost = 250000
            cost_breakdown.append({
                "Subcatchment": "Watershed Outfall",
                "LID Type": "Tide Gate",
                "Cost": tide_cost
            })
            total_cost += tide_cost
        st.session_state[f"{prefix}user_total_cost"] = total_cost

        if cost_breakdown:
            cost_df = pd.DataFrame(cost_breakdown)

            # Stacked bar chart of Cost by Subcatchment & LID Type
            chart = (
                alt.Chart(cost_df)
                .mark_bar()
                .encode(
                    x=alt.X("Subcatchment:N", title="Subcatchment"),
                    y=alt.Y("Cost:Q", title="Cost", stack="zero"),
                    color=alt.Color("LID Type:N", legend=alt.Legend(title="LID Type")),
                    tooltip=["Subcatchment","LID Type","Cost"]
                )
                .properties(width=650, height=350)
            )
            st.altair_chart(chart, use_container_width=True)

            # Total cost below the chart
            st.markdown(f"**Total Estimated Cost with Tide Gate: ${total_cost:}**")
            st.markdown(f"**Total Estimated Cost LID only: ${(total_cost - 250000):}**")

        else:
            st.info("No infrastructure improvements have been added.")

    else:
        st.info("You haven't selected any subcatchments to add LIDs.")

    # === Run Scenario With Selected LID Improvements ===
    if st.button("Run Custom LID Scenario"):
        if f"{prefix}baseline_df" not in st.session_state:
            st.error("Please run the baseline scenario first.")
        else:
            try:
                lid_lines = generate_lid_usage_lines(st.session_state[f"{prefix}user_lid_config"])
                if not lid_lines:
                    st.warning("No LIDs selected.")
                    st.stop()

                tide_lines = format_timeseries("tide", tide_sim_minutes, tide_sim_curve, simulation_date)
                rain_lines_cur, rain_lines_fut = _rain_lines_pair(
                    rain_sim_minutes,
                    st.session_state["rain_sim_curve_current_in"],
                    simulation_date
                )

                # CURRENT (2)
                fill_lid_cur, time_lid, rpt3 = run_swmm_scenario(f"{prefix}lid_nogate_current", rain_lines_cur, tide_lines, lid_lines, "NO")
                fill_lid_gate_cur, _,  rpt4  = run_swmm_scenario(f"{prefix}lid_gate_current",   rain_lines_cur, tide_lines, lid_lines, "YES")

                # FUTURE (2)
                fill_lid_fut, _,  rpt3f = run_swmm_scenario(f"{prefix}lid_nogate_future", rain_lines_fut, tide_lines, lid_lines, "NO")
                fill_lid_gate_fut, _, rpt4f = run_swmm_scenario(f"{prefix}lid_gate_future",   rain_lines_fut, tide_lines, lid_lines, "YES")

                # Store
                st.session_state[f"{prefix}lid_timestamps"] = time_lid
                st.session_state[f"{prefix}lid_fill_current"] = fill_lid_cur
                st.session_state[f"{prefix}lid_gate_fill_current"] = fill_lid_gate_cur
                st.session_state[f"{prefix}lid_fill_future"] = fill_lid_fut
                st.session_state[f"{prefix}lid_gate_fill_future"] = fill_lid_gate_fut

                # Runoff summaries
                st.session_state[f"{prefix}df_lid_nogate_current"] = extract_runoff_and_lid_data(rpt3)
                st.session_state[f"{prefix}df_lid_gate_current"]   = extract_runoff_and_lid_data(rpt4)
                st.session_state[f"{prefix}df_lid_nogate_future"]  = extract_runoff_and_lid_data(rpt3f)
                st.session_state[f"{prefix}df_lid_gate_future"]    = extract_runoff_and_lid_data(rpt4f)

                st.success("Custom LID scenarios (Current & +20%) complete!")

            except Exception as e:
                st.error(f"LID simulation failed: {e}")

    required_flood_keys = [
        f"{prefix}baseline_fill_current",
        f"{prefix}baseline_gate_fill_current",
        f"{prefix}lid_fill_current",
        f"{prefix}lid_gate_fill_current",
        f"{prefix}lid_max_fill_current",
        f"{prefix}lid_max_gate_fill_current",
        f"{prefix}baseline_fill_future",
        f"{prefix}baseline_gate_fill_future",
        f"{prefix}lid_fill_future",
        f"{prefix}lid_gate_fill_future",
        f"{prefix}lid_max_fill_future",
        f"{prefix}lid_max_gate_fill_future",
        f"{prefix}baseline_timestamps"
    ]


    if all(k in st.session_state for k in required_flood_keys):
        # === Retrieve and truncate ===
        ts = st.session_state[f"{prefix}baseline_timestamps"]

        # Baseline
        baseline_cur       = st.session_state[f"{prefix}baseline_fill_current"]
        baseline_gate_cur  = st.session_state[f"{prefix}baseline_gate_fill_current"]
        baseline_fut       = st.session_state[f"{prefix}baseline_fill_future"]
        baseline_gate_fut  = st.session_state[f"{prefix}baseline_gate_fill_future"]

        # LID
        lid_cur            = st.session_state[f"{prefix}lid_fill_current"]
        lid_gate_cur       = st.session_state[f"{prefix}lid_gate_fill_current"]
        lid_fut            = st.session_state[f"{prefix}lid_fill_future"]
        lid_gate_fut       = st.session_state[f"{prefix}lid_gate_fill_future"]

        # Max LID
        lid_max_cur        = st.session_state[f"{prefix}lid_max_fill_current"]
        lid_max_gate_cur   = st.session_state[f"{prefix}lid_max_gate_fill_current"]
        lid_max_fut        = st.session_state[f"{prefix}lid_max_fill_future"]
        lid_max_gate_fut   = st.session_state[f"{prefix}lid_max_gate_fill_future"]

        # Truncate to shortest
        min_len = min(
            len(ts),
            len(baseline_cur), len(baseline_gate_cur),
            len(baseline_fut), len(baseline_gate_fut),
            len(lid_cur), len(lid_gate_cur),
            len(lid_fut), len(lid_gate_fut),
            len(lid_max_cur), len(lid_max_gate_cur),
            len(lid_max_fut), len(lid_max_gate_fut)
        )



    def extract_volumes_from_rpt(rpt_path, scenario_name=None):
        """
        Reads a SWMM .rpt file and extracts key volume metrics, including
        total flooding volume for the whole simulation.
        Returns volumes in acre-feet where applicable.
        """
        flooding = None

        # --- 1. Prefer model-calculated total from session_state ---
        if scenario_name:
            model_total = st.session_state.get(f"{scenario_name}_total_flood", None)
            if model_total is not None:
                flooding = model_total

        try:
            with open(rpt_path, 'r') as f:
                lines = f.readlines()

            infiltration = None
            runoff = None
            rainfall = None

            in_runoff_continuity_section = False

            for line in lines:
                # Runoff Quantity Continuity → precipitation, infiltration, runoff
                if "Runoff Quantity Continuity" in line:
                    in_runoff_continuity_section = True
                    continue
                if in_runoff_continuity_section:
                    if "Total Precipitation" in line:
                        rainfall = float(line.split()[-2])
                    elif "Infiltration Loss" in line:
                        infiltration = float(line.split()[-2])
                    elif "Surface Runoff" in line:
                        runoff = float(line.split()[-2])
                    elif "Continuity Error" in line:
                        in_runoff_continuity_section = False

            return {
                "Rainfall (ac-ft)": rainfall,
                "Infiltration (ac-ft)": infiltration,
                "Runoff (ac-ft)": runoff,
                "Flooding (ac-ft)": flooding
            }

        except Exception as e:
            print(f"Failed to parse RPT file {rpt_path}: {e}")
            return {k: None for k in [
                "Rainfall (ac-ft)", "Infiltration (ac-ft)",
                "Runoff (ac-ft)", "Flooding (ac-ft)"
            ]}

    # === Water Balance Summary (Only show if RPT data exists) ===
    df_balance = None
    results = []
    temp_dir = st.session_state.temp_dir

    rpt_scenarios = {
        # Baseline
        "Baseline (No Tide Gate) – Current": os.path.join(temp_dir, f"{prefix}baseline_nogate_current.rpt"),
        "Baseline + Tide Gate – Current":    os.path.join(temp_dir, f"{prefix}baseline_gate_current.rpt"),
        "Baseline (No Tide Gate) – +20%":    os.path.join(temp_dir, f"{prefix}baseline_nogate_future.rpt"),
        "Baseline + Tide Gate – +20%":       os.path.join(temp_dir, f"{prefix}baseline_gate_future.rpt"),
        # Custom LID
        "LID (No Tide Gate) – Current":      os.path.join(temp_dir, f"{prefix}lid_nogate_current.rpt"),
        "LID + Tide Gate – Current":         os.path.join(temp_dir, f"{prefix}lid_gate_current.rpt"),
        "LID (No Tide Gate) – +20%":         os.path.join(temp_dir, f"{prefix}lid_nogate_future.rpt"),
        "LID + Tide Gate – +20%":            os.path.join(temp_dir, f"{prefix}lid_gate_future.rpt"),
        # Max LID
        "Max LID (No Tide Gate) – Current":  os.path.join(temp_dir, f"{prefix}lid_max_nogate_current.rpt"),
        "Max LID + Tide Gate – Current":     os.path.join(temp_dir, f"{prefix}lid_max_gate_current.rpt"),
        "Max LID (No Tide Gate) – +20%":     os.path.join(temp_dir, f"{prefix}lid_max_nogate_future.rpt"),
        "Max LID + Tide Gate – +20%":        os.path.join(temp_dir, f"{prefix}lid_max_gate_future.rpt"),
    }


    # Only attempt to extract results if ALL expected .rpt files exist
    if all(os.path.exists(path) for path in rpt_scenarios.values()):
        results = []
        for name, path in rpt_scenarios.items():
            try:
                # Make a scenario key that matches the one used in run_swmm_scenario()
                friendly_to_session_prefix = {
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
                scenario_name_for_lookup = f"{prefix}{friendly_to_session_prefix[name]}"

                # Prefer the stored model total flooding value over parsing RPT
                metrics = extract_volumes_from_rpt(path, scenario_name=scenario_name_for_lookup)

                if any(v is not None for k, v in metrics.items() if k != "Scenario"):
                    metrics["Scenario"] = name
                    results.append(metrics)
                    st.session_state[f"{prefix}flood_{friendly_to_session_prefix[name]}"] = metrics["Flooding (ac-ft)"]

            except Exception as e:
                print(f"Could not process {name}: {e}")
                continue
    else:
        results = []

    if results:
        df_balance = pd.DataFrame(results).set_index("Scenario")

        # Force the summary table to use the graph's final cumulative flooding totals
        scenario_to_key = {
            "Baseline (No Tide Gate) – Current":  f"{prefix}baseline_fill_current",
            "Baseline + Tide Gate – Current":     f"{prefix}baseline_gate_fill_current",
            "Baseline (No Tide Gate) – +20%":     f"{prefix}baseline_fill_future",
            "Baseline + Tide Gate – +20%":        f"{prefix}baseline_gate_fill_future",
            "LID (No Tide Gate) – Current":       f"{prefix}lid_fill_current",
            "LID + Tide Gate – Current":          f"{prefix}lid_gate_fill_current",
            "LID (No Tide Gate) – +20%":          f"{prefix}lid_fill_future",
            "LID + Tide Gate – +20%":             f"{prefix}lid_gate_fill_future",
            "Max LID (No Tide Gate) – Current":   f"{prefix}lid_max_fill_current",
            "Max LID + Tide Gate – Current":      f"{prefix}lid_max_gate_fill_current",
            "Max LID (No Tide Gate) – +20%":      f"{prefix}lid_max_fill_future",
            "Max LID + Tide Gate – +20%":         f"{prefix}lid_max_gate_fill_future",
        }


        for scenario, fill_key in scenario_to_key.items():
            if fill_key in st.session_state and len(st.session_state[fill_key]) > 0:
                df_balance.loc[scenario, "Flooding (ac-ft)"] = st.session_state[fill_key][-1]

        # Convert + format logic remains the same
        convert_to_m3 = unit == "Metric (SI)"
        GAL_TO_FT3 = 0.133681
        ACF_TO_FT3 = 43560
        FT3_TO_M3 = 0.0283168

        def convert(val, from_unit):
            if val is None:
                return 0  # or np.nan if you want to mark missing
            if from_unit == "gallons":
                val_ft3 = val * GAL_TO_FT3
            elif from_unit == "ac-ft":
                val_ft3 = val * ACF_TO_FT3
            else:
                val_ft3 = val
            return val_ft3 * FT3_TO_M3 if convert_to_m3 else val_ft3

        df_converted = pd.DataFrame(index=df_balance.index)
        df_converted["Flooded Volume (Event Total)"] = df_balance["Flooding (ac-ft)"].apply(lambda x: convert(x, "ac-ft"))
        df_converted["Infiltration"] = df_balance["Infiltration (ac-ft)"].apply(lambda x: convert(x, "ac-ft"))
        df_converted["Surface Runoff"] = df_balance["Runoff (ac-ft)"].apply(lambda x: convert(x, "ac-ft"))
        df_converted = df_converted.round(0).astype(int)

        cost_lookup = {
            # Baseline
            "Baseline (No Tide Gate) – Current": 0,
            "Baseline + Tide Gate – Current":    250000,
            "Baseline (No Tide Gate) – +20%":    0,
            "Baseline + Tide Gate – +20%":       250000,
            # LID
            "LID (No Tide Gate) – Current":      st.session_state.get(f"{prefix}user_total_cost", 0) - 250000,
            "LID + Tide Gate – Current":         st.session_state.get(f"{prefix}user_total_cost", 0),
            "LID (No Tide Gate) – +20%":         st.session_state.get(f"{prefix}user_total_cost", 0) - 250000,
            "LID + Tide Gate – +20%":            st.session_state.get(f"{prefix}user_total_cost", 0),
            # Max LID
            "Max LID (No Tide Gate) – Current":  st.session_state.get(f"{prefix}max_total_cost", 0) - 250000,
            "Max LID + Tide Gate – Current":     st.session_state.get(f"{prefix}max_total_cost", 0),
            "Max LID (No Tide Gate) – +20%":     st.session_state.get(f"{prefix}max_total_cost", 0) - 250000,
            "Max LID + Tide Gate – +20%":        st.session_state.get(f"{prefix}max_total_cost", 0),
        }

        df_converted["Total Cost ($)"] = df_converted.index.map(cost_lookup).astype(int)

        st.session_state[f"{prefix}df_balance"] = df_converted

        # === Show summary only if user clicks ===
        if st.button("Show Water Balance Summary Table"):
            if df_converted is not None:
                unit_label = "m³" if convert_to_m3 else "ft³"
                st.subheader(f"Summary ({unit_label})")
                st.dataframe(df_converted)
            else:
                st.info("Please run at least one simulation to view the summary table.")


    # === Export Excel Report (Single Click Download Button) ===
    required_excel_keys = [
        "df_balance",
        "baseline_timestamps",

        # Baseline – Current & Future
        "baseline_fill_current",
        "baseline_gate_fill_current",
        "baseline_fill_future",
        "baseline_gate_fill_future",

        # LID – Current & Future
        "lid_fill_current",
        "lid_gate_fill_current",
        "lid_fill_future",
        "lid_gate_fill_future",

        # Max LID – Current & Future
        "lid_max_fill_current",
        "lid_max_gate_fill_current",
        "lid_max_fill_future",
        "lid_max_gate_fill_future",
    ]


    if all(f"{prefix}{k}" in st.session_state for k in required_excel_keys):

        # Step 3: Retrieve water balance summary
        df_balance = st.session_state[f"{prefix}df_balance"]

        # Step 4: Build rainfall and tide time series if available
        rain_time_series = st.session_state.get("display_rain_curve", [])
        tide_time_series = st.session_state.get("display_tide_curve", [])

        rain_disp_unit = st.session_state.get(f"{prefix}rain_disp_unit", "inches")
        tide_disp_unit = st.session_state.get(f"{prefix}tide_disp_unit", "ft")

        # Prepare user's LID selections (if any)
        lid_config = st.session_state.get(f"{prefix}user_lid_config", {})
        if lid_config:
            lid_rows = []
            for sub, cfg in lid_config.items():
                rg = cfg.get("rain_gardens", 0)
                rb = cfg.get("rain_barrels", 0)
                lid_rows.append({
                    "Subcatchment": sub,
                    "Selected Rain Gardens": rg,
                    "Selected Rain Barrels": rb
                })
            df_user_lid = pd.DataFrame(lid_rows)
        else:
            df_user_lid = pd.DataFrame(columns=["Subcatchment", "Selected Rain Gardens", "Selected Rain Barrels"])

        # Step 5: Write everything to Excel
        excel_output = io.BytesIO()
        with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
            # === Scenario Metadata Summary ===
            scenario_summary = pd.DataFrame([{
                "Storm Duration (hr)": duration_minutes // 60,
                "Return Period (yr)": return_period,
                "Tide": moon_phase,
                "Tide Alignment": "High Tide Peak" if align_mode == "peak" else "Low Tide Dip",
                "Units": unit
            }])
            scenario_summary.to_excel(writer, sheet_name="Scenario Settings", index=False)

            # Convert simulation_date string to datetime
            sim_start = datetime.strptime(simulation_date, "%m/%d/%Y %H:%M")

            # === Rainfall Time Series ===
            rain_minutes = st.session_state.get("rain_minutes", [])
            tide_minutes = st.session_state.get("tide_minutes", [])
            if isinstance(rain_time_series, (list, np.ndarray)) and len(rain_time_series) > 0:
                rain_timestamps = [
                    (sim_start + timedelta(minutes=int(m))).strftime("%m/%d/%Y %H:%M")
                    for m in rain_minutes[:len(rain_time_series)]
                ]
                df_rain = pd.DataFrame({
                    "Timestamp": rain_timestamps,
                    f"Rainfall – Current ({rain_disp_unit})": st.session_state["display_rain_curve_current"][:len(rain_timestamps)],
                    f"Rainfall – +20% ({rain_disp_unit})":    st.session_state["display_rain_curve_future"][:len(rain_timestamps)],
                })
                df_rain.to_excel(writer, sheet_name="Rainfall Event", index=False)

            # === Tide Time Series ===
            if isinstance(tide_time_series, (list, np.ndarray)) and len(tide_time_series) > 0:
                tide_timestamps = [
                    (sim_start + timedelta(minutes=int(m))).strftime("%m/%d/%Y %H:%M")
                    for m in tide_minutes[:len(tide_time_series)]
                ]
                df_tide = pd.DataFrame({
                    "Timestamp": tide_timestamps,
                    f"Tide ({tide_disp_unit}": tide_time_series
                })
                df_tide.to_excel(writer, sheet_name="Tide Event", index=False)
            
            # Write water balance and culvert sheets
            df_balance = df_balance.reset_index().rename(columns={"index": "Scenario"})

            # Write custom LID selection (if any)
            df_user_lid.to_excel(writer, sheet_name="User LID Selections", index=False)

            df_balance.to_excel(writer, sheet_name="Scenario Summary", index=False)


        # Download button
        st.download_button(
            label="Download Scenario Results (Excel)",
            data=excel_output.getvalue(),
            file_name="CoastWise_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if st.button("🚪 Logout"):
            # Delete temp_dir and all files inside it
            try:
                shutil.rmtree(st.session_state.temp_dir)
            except Exception as e:
                print(f"Failed to delete temp folder: {e}")

            st.session_state.clear()
            st.success("Logged out and cleaned up all files.")
            st.rerun()

