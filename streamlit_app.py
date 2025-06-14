import streamlit as st
import pandas as pd
import altair as alt
import time
import numpy as np
import re
import io
import os
import shutil 
import matplotlib
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.dates as mdates
from datetime import datetime
from pyswmm import Simulation, Links, Nodes
from rainfall_and_tide_generator import (
    generate_rainfall,
    convert_units,
    pf_df,
    generate_tide_curve,
    align_rainfall_to_tide,
    moon_tide_ranges
)

# --- Configuration ---
simulation_date = "06/01/2025"
template_inp     = "swmm_project.inp"

st.set_page_config(
    page_title="CoastWise",
    layout="centered"
)
st.title("CoastWise: A Gamified Watershed Design Toolkit for Coastal Resilience Using the Stormwater Management Model")

# === User Inputs ===
duration_minutes = st.selectbox(
    "Storm Duration",
    options=pf_df["Duration_Minutes"],
    format_func=lambda x: f"{x // 60} hr"
)

st.subheader("Return Year: think of it like rolling dice - a 10-year storm is like rolling a 10 on a 10-sided die, you might roll it once in 10 tries… or twice… or not at all. But the chance (1/10 = 10%) is always the same each year.") 

return_period = st.selectbox("Return Year", pf_df.columns[1:])
# pf_df values are in inches
rain_inches = float(pf_df.loc[
    pf_df["Duration_Minutes"] == duration_minutes,
    return_period
].values[0])

unit        = st.selectbox("Preferred Units", ["U.S. Customary", "Metric (SI)"])
method      = st.radio("Rainfall Shape", ["Normal", "Randomized"])

st.subheader("How to remember the Tides: Spring tides surge higher than Neap tides that nap (lower)") 

# Load and display video
video_file = open('NASA_Tides.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)


moon_phase  = st.selectbox("Moon Phase", list(moon_tide_ranges.keys()))

st.subheader("When high tide and peak rainfall happen together, tidal backflow can block stormwater from draining, increasing flood risk. If rain falls during low tide, water drains more easily and flooding is less likely.")

tide_align  = st.radio(
    "Tide Alignment",
    ["Peak aligned with High Tide", "Peak aligned with Low Tide"]
)
align_mode  = "peak" if "High" in tide_align else "low"

# --- Generate SWMM-curves (always inches & feet) ---
tide_sim_minutes, tide_sim_curve = generate_tide_curve(moon_phase, "U.S. Customary")
rain_sim_minutes, rain_sim_curve = align_rainfall_to_tide(
    rain_inches,
    duration_minutes,
    tide_sim_curve,
    align=align_mode,
    method=method
)

# --- Generate display-curves based on selected unit ---
if unit == "U.S. Customary":
    display_rain_curve = rain_sim_curve
    display_tide_curve = tide_sim_curve
    tide_disp_unit    = "ft"
    rain_disp_unit = "inches"
elif unit == "Metric (SI)":
    display_rain_curve = rain_sim_curve * 2.54
    display_tide_curve = tide_sim_curve * 0.3048 * 100
    tide_disp_unit    = "meters"
    rain_disp_unit = "centimeters"

# --- Store display values and timestamps for Excel export ---
st.session_state["rain_minutes"] = rain_sim_minutes
st.session_state["tide_minutes"] = tide_sim_minutes
st.session_state["display_rain_curve"] = display_rain_curve
st.session_state["display_tide_curve"] = display_tide_curve

time_hours = np.array(rain_sim_minutes) / 60

# === Display Rainfall Chart ===
df_rain = pd.DataFrame({
    "Time (hours)": time_hours,
    f"Rainfall ({rain_disp_unit})": display_rain_curve
})
st.subheader("Rainfall Distribution")
rain_chart = (
    alt.Chart(df_rain)
       .mark_line()
       .encode(
           x="Time (hours)",
           y=f"Rainfall ({rain_disp_unit})"
       )
)
st.altair_chart(rain_chart, use_container_width=True)

# === Display Tide Chart ===
df_tide = pd.DataFrame({
    "Time (hours)": time_hours,
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
    inp_file = f"{scenario_name}.inp"
    rpt_file = f"{scenario_name}.rpt"
    out_file = f"{scenario_name}.out"

    # --- 1. Create .inp file ---
    with open(template_path, "r") as f:
        text = f.read()
    text = text.replace("$RAINFALL_TIMESERIES$", "\n".join(rain_lines))
    text = text.replace("$TIDE_TIMESERIES$", "\n".join(tide_lines))
    text = text.replace("$LID_USAGE$", "\n".join(lid_lines))
    text = text.replace("$TIDE_GATE_CONTROL$", gate_flag)
    with open(inp_file, "w") as f:
        f.write(text)

    # --- 2. Run simulation ---
    depth_pct, timestamps = [], []
    with Simulation(inp_file) as sim:
        link = Links(sim)["C70_1"]
        last_report_time = None
        for step in sim:
            current_time = sim.current_time
            if last_report_time is None or (current_time - last_report_time) >= report_interval:
                pct = (link.depth / full_depth) * 100
                depth_pct.append(pct)
                timestamps.append(current_time.strftime("%H:%M"))
                last_report_time = current_time

    # --- 3. Move .rpt/.out to unique names ---
    if os.path.exists("updated_model.rpt"):
        shutil.move("updated_model.rpt", rpt_file)
    if os.path.exists("updated_model.out"):
        shutil.move("updated_model.out", out_file)

    return depth_pct, timestamps, rpt_file

# === Time-series Formatter ===
def format_timeseries(name, minutes, values, date):
    lines = []
    for m, v in zip(minutes, values):
        hh = int(m) // 60
        mm = int(m) % 60
        lines.append(f"{name} {date} {hh:02d}:{mm:02d} {v:.5f}")
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

st.markdown(f"""
### Selected Scenario Summary
- **Storm Duration:** {duration_minutes//60} hr  
- **Return Period:** {return_period} yr  
- **Moon Phase:** {moon_phase}  
- **Tide Alignment:** {'High Tide Peak' if align_mode=='peak' else 'Low Tide Dip'}  
- **Display Units:** {unit}
""")

# === Run Baseline Scenario ===
if st.button("Run Baseline Scenario SWMM Simulation"):
    try:
        rain_lines = format_timeseries("rain_gage_timeseries", rain_sim_minutes, rain_sim_curve, simulation_date)
        tide_lines = format_timeseries("tide", tide_sim_minutes, tide_sim_curve, simulation_date)
        lid_lines  = [";"]  # No LIDs in baseline

        # Run both baseline scenarios
        fill_nogate, time_nogate, rpt1 = run_swmm_scenario("baseline_nogate", rain_lines, tide_lines, lid_lines, "NO")
        fill_gate,   time_gate,   rpt2 = run_swmm_scenario("baseline_gate",   rain_lines, tide_lines, lid_lines, "YES")

        # Save culvert depth fills + time
        st.session_state["baseline_fill"] = fill_nogate
        st.session_state["baseline_timestamps"] = time_nogate
        st.session_state["baseline_gate_fill"] = fill_gate

        # Parse runoff data from RPTs
        df1 = extract_runoff_and_lid_data(rpt1)
        df2 = extract_runoff_and_lid_data(rpt2)
        st.session_state["df_base_nogate"] = df1
        st.session_state["df_base_gate"] = df2

        # Unit conversion for final display table
        if unit == "U.S. Customary":
            df_disp = df1.rename(columns={
                "Impervious Runoff (in)": "Impervious Runoff (inches)",
                "Pervious Runoff   (in)": "Pervious Runoff (inches)"
            })
        else:
            factor = 2.54 if unit == "Metric (SI)" else 25.4
            df_disp = pd.DataFrame({
                "Subcatchment": df1["Subcatchment"],
                f"Impervious Runoff ({unit})": df1["Impervious Runoff (in)"] * factor,
                f"Pervious Runoff ({unit})": df1["Pervious Runoff   (in)"] * factor
            })

        st.session_state["baseline_df"] = df_disp
        st.success("Baseline scenarios complete!")

    except Exception as e:
        st.error(f"Baseline simulation failed: {e}")

# === Low Impact Developments (LIDs) UI & Cost ===
st.subheader("Low Impact Developments (LIDs) options")
st.image(
    "green_infrastructure_options.png",
    use_container_width=True
)

st.subheader("Low Impact Developments (LIDs) Storage")
st.image(
    "vols.png",
    use_container_width=True
)

st.subheader("Land Use Land Cover used to determine the maximum number of LIDs for each subcatchment")
st.image(
    "lulc_ledgend.png",
    use_container_width=True
)

st.subheader("Elevation showing old creek bed & Purple areas indicate where infiltration-based green infrastructure (rain gardens) are not recommended... Low lying areas where water accumulates.")
st.image(
    "comare_dem_infil_stor.png",
    use_container_width=True
)

st.subheader("Stormwater Pipes, Inlets, and Outlet & Watershed with labeled Subcatchments. Interesting that some subcatchments do not have any nearby infrastructure.")
st.image(
    "pipe_watersheds.png",
    use_container_width=True
)

if "df_base_nogate" in st.session_state:
    st.subheader("Baseline Runoff (No Tide Gate)")

    df_no = st.session_state["df_base_nogate"].copy()

    # Optional unit conversion
    if unit != "U.S. Customary":
        factor = 2.54 if unit == "Metric (SI)" else 25.4
        df_no["Impervious Runoff (in)"] *= factor
        df_no["Pervious Runoff   (in)"] *= factor
        df_no.columns = ["Subcatchment",
                         f"Impervious Runoff ({unit})",
                         f"Pervious Runoff ({unit})"]
    else:
        df_no.columns = ["Subcatchment",
                         "Impervious Runoff (in)",
                         "Pervious Runoff (in)"]

    st.dataframe(df_no, use_container_width=True)

# --- Cost and Sizing Assumptions ---
data = {
    "Infrastructure": [
        "100 sq.ft. Rain Garden",
        "50 gallon Rain Barrel",
        "Tide Gate (10'x5')"
    ],
    "Contributing Area Assumption": [
        "1 per 500 sq.ft. of herbaceous turfgrass",
        "1 per 300 sq.ft. of rooftop",
        "Protects system from tidal backflow"
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

st.subheader("Steps to Explore LID Design Options:")
st.markdown("""
1. Run the scenario where the LID features (rain gardens, rain barrels) are **MAXED** out for ALL subcatchments.
2. Find subcatchments **unsuitable** for rain gardens. Why can’t rain gardens be used there? (e.g., poor infiltration, former creek bed).
3. Identify subcatchments with **high runoff** but no nearby stormwater pipes or culverts.
4. Observe how your choices impact outflow, flooding, and infiltration.
""")

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
                area=f"{2.95:.2f}",
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
    # Load max values
    max_lid_config = {}
    for idx, row in df.iterrows():
        sub = row["NAME"]
        rg_max = int(row["Max_RG_DEM_Considered"])
        rb_max = int(row["MaxNumber_RB"])
        max_lid_config[sub] = {"rain_gardens": rg_max, "rain_barrels": rb_max}

    rain_lines = format_timeseries("rain_gage_timeseries", rain_sim_minutes, rain_sim_curve, simulation_date)
    tide_lines = format_timeseries("tide", tide_sim_minutes, tide_sim_curve, simulation_date)
    lid_lines = generate_lid_usage_lines(max_lid_config)

    # Run SWMM with and without tide gate
    fill_max_lid, time_max_lid, rpt5 = run_swmm_scenario("lid_max_nogate", rain_lines, tide_lines, lid_lines, "NO")
    fill_max_gate, _, rpt6 = run_swmm_scenario("lid_max_gate", rain_lines, tide_lines, lid_lines, "YES")

    # Store time series
    st.session_state["lid_max_fill"] = fill_max_lid
    st.session_state["lid_max_timestamps"] = time_max_lid
    st.session_state["lid_max_gate_fill"] = fill_max_gate

    # Extract runoff summary
    st.session_state["df_lid_max_nogate"] = extract_runoff_and_lid_data(rpt5)
    st.session_state["df_lid_max_gate"] = extract_runoff_and_lid_data(rpt6)

    # Compute total cost
    max_total_cost = (df["Max_RG_DEM_Considered"].sum() * 250 +
                      df["MaxNumber_RB"].sum() * 100 +
                      250000)  # include tide gate
    st.session_state["max_total_cost"] = max_total_cost

    st.success(f"Max LID scenarios complete!")

if "user_lid_config" not in st.session_state:
    st.session_state["user_lid_config"] = {}

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
        with c3: rg_val = st.number_input("", 0, rg_max, 0, step=5, key=f"rg_{sub}", label_visibility="collapsed")
        with c4: st.markdown(f"<div style='text-align:center'>{rb_max}</div>", unsafe_allow_html=True)
        with c5: rb_val = st.number_input("", 0, rb_max, 0, step=5, key=f"rb_{sub}", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state["user_lid_config"][sub] = {
            "rain_gardens": rg_val,
            "rain_barrels": rb_val
        }
    st.markdown('</div>', unsafe_allow_html=True)

        # === Compute & Display LID Costs ===
    total_cost = 0
    cost_breakdown = []

    # 1) Per‐subcatchment costs
    for sub, cfg in st.session_state["user_lid_config"].items():
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
    if any(v["rain_gardens"] > 0 or v["rain_barrels"] > 0 for v in st.session_state["user_lid_config"].values()):
        tide_cost = 250000
        cost_breakdown.append({
            "Subcatchment": "Watershed Outfall",
            "LID Type": "Tide Gate",
            "Cost": tide_cost
        })
        total_cost += tide_cost
    st.session_state["user_total_cost"] = total_cost

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
        st.markdown(f"**Total Estimated Cost: ${total_cost:,.2f}**")
    else:
        st.info("No infrastructure improvements have been added.")

else:
    st.info("You haven't selected any subcatchments to add LIDs.")

# === Run Scenario With Selected LID Improvements ===
if st.button("Run Scenario With Selected LID Improvements"):
    if "baseline_df" not in st.session_state:
        st.error("Please run the baseline scenario first.")
    else:
        try:
            rain_lines = format_timeseries("rain_gage_timeseries", rain_sim_minutes, rain_sim_curve, simulation_date)
            tide_lines = format_timeseries("tide", tide_sim_minutes, tide_sim_curve, simulation_date)
            lid_lines = generate_lid_usage_lines(st.session_state["user_lid_config"])
            if not lid_lines:
                st.warning("No LIDs selected.")
                st.stop()

            # Run LID-only and LID+Gate scenarios
            fill_lid, time_lid, rpt3 = run_swmm_scenario("lid_nogate", rain_lines, tide_lines, lid_lines, "NO")
            fill_lid_gate, _, rpt4   = run_swmm_scenario("lid_gate", rain_lines, tide_lines, lid_lines, "YES")

            # Store simulation results
            st.session_state["lid_fill"] = fill_lid
            st.session_state["lid_timestamps"] = time_lid
            st.session_state["lid_gate_fill"] = fill_lid_gate

            df_lid = extract_runoff_and_lid_data(rpt3)
            df_lid_gate = extract_runoff_and_lid_data(rpt4)
            st.session_state["df_lid_nogate"] = df_lid
            st.session_state["df_lid_gate"] = df_lid_gate

            st.success("LID scenarios complete!")

        except Exception as e:
            st.error(f"LID simulation failed: {e}")

if all(key in st.session_state for key in [
    "baseline_fill", "baseline_gate_fill",
    "lid_fill", "lid_gate_fill",
    "lid_max_fill", "lid_max_gate_fill",
    "baseline_timestamps"
]):

    st.subheader("The Capacity of the Pipe Closest to the Outlet over Time (All Scenarios)")
    st.markdown("""
### Six Scenario Overview: Tide Gates + Green Infrastructure

In this project, we tested six scenarios combining two types of flood interventions: **tide gates** and **LID (Low Impact Development) features** like rain gardens and rain barrels.

#### How Tide Gates Work
The **tide gate** acts like a one-way door:
- It **blocks tidal water** from backing up into the system during high tide.
- It **lets stormwater exit** when pressure inside the pipe is higher than the tide level.

#### How LID Features Help
**Rain gardens and barrels** reduce stormwater earlier in the system:
- They **slow down** and **store** runoff close to where it falls.
- This reduces how much water reaches the pipes — especially during small and moderate storms.
- During **large storms**, their benefit at the outfall is smaller, but still helpful.

Together, these six scenarios show how **both local solutions (like LIDs)** and **system-wide controls (like tide gates)** are needed to manage flooding—especially in coastal areas where rainfall and tides can overlap.

""")


    time_labels = st.session_state["baseline_timestamps"]
    
    # Convert to datetime for proper tick formatting
    time_objects = [datetime.strptime(t, "%H:%M") for t in time_labels]

    # Define a color palette (or use any you prefer)
    colors = {
        "Baseline": "#141413",             
        "Baseline + Tide Gate": "#5c5dc6", 
        "With LIDs": "#459145",            
        "LIDs + Tide Gate": "#e7ef46",     
        "Max LIDs": "#10f527",             
        "Max LIDs + Tide Gate": "#ea1717"  
    }

    # Define a distinct style per line
    styles = {
        "Baseline": "-",
        "Baseline + Tide Gate": "-",
        "With LIDs": "-.",
        "LIDs + Tide Gate": "-.",
        "Max LIDs": "--",     
        "Max LIDs + Tide Gate": "-" 
    }

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each line with custom style
    ax.plot(time_objects, st.session_state["baseline_fill"],
            label="Baseline", color=colors["Baseline"], linestyle=styles["Baseline"], linewidth=4)

    ax.plot(time_objects, st.session_state["baseline_gate_fill"],
            label="Baseline + Tide Gate", color=colors["Baseline + Tide Gate"], linestyle=styles["Baseline + Tide Gate"], linewidth=4)

    ax.plot(time_objects, st.session_state["lid_fill"],
            label="With LIDs", color=colors["With LIDs"], linestyle=styles["With LIDs"], linewidth=4)

    ax.plot(time_objects, st.session_state["lid_gate_fill"],
            label="LIDs + Tide Gate", color=colors["LIDs + Tide Gate"], linestyle=styles["LIDs + Tide Gate"], linewidth=4)

    ax.plot(time_objects, st.session_state["lid_max_fill"],
            label="Max LIDs", color=colors["Max LIDs"], linestyle=styles["Max LIDs"], linewidth=4)

    ax.plot(time_objects, st.session_state["lid_max_gate_fill"],
            label="Max LIDs + Tide Gate", color=colors["Max LIDs + Tide Gate"], linestyle=styles["Max LIDs + Tide Gate"], linewidth=4)


    ax.set_ylabel("Culvert Fill (%)")
    ax.set_xlabel("Time")
    ax.set_ylim(0, 110)
    ax.set_title("Culvert Capacity Comparison")

    # Show only hourly ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%#I %p'))
    ax.xaxis.set_major_locator(mdates.HourLocator())

    start_time = datetime.strptime("01:00", "%H:%M")
    end_time = start_time + timedelta(hours=23)
    ax.set_xlim([start_time, end_time])

    ax.legend()
    ax.grid(False)
    fig.autofmt_xdate()  # Auto-format x-axis for better label spacing
    st.pyplot(fig)

def extract_volumes_from_rpt(rpt_path):
    try:
        with open(rpt_path, 'r') as f:
            lines = f.readlines()

        outflow = None
        infiltration = None
        runoff = None
        rainfall = None
        flooding = 0.0

        in_outfall_section = False
        in_continuity_section = False
        in_flooding_section = False

        for i, line in enumerate(lines):
            # === Outfall Loading Summary ===
            if "Outfall Loading Summary" in line:
                in_outfall_section = True
                continue
            if in_outfall_section:
                if line.strip().startswith("System"):
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            outflow = float(parts[-1]) * 1e6  # Convert from MG to gallons
                        except ValueError:
                            outflow = None
                    in_outfall_section = False  # Done after "System" line

            # === Runoff Quantity Continuity ===
            if "Runoff Quantity Continuity" in line:
                in_continuity_section = True
                continue
            if in_continuity_section:
                if "Total Precipitation" in line:
                    rainfall = float(line.split()[-2])
                elif "Infiltration Loss" in line:
                    infiltration = float(line.split()[-2])
                elif "Surface Runoff" in line:
                    runoff = float(line.split()[-2])
                elif "Continuity Error" in line:
                    in_continuity_section = False

            # === Node Flooding Summary ===
            if "Flow Routing Continuity" in line:
                in_continuity_section = True
                continue
            if in_continuity_section:
                if "Flooding Loss" in line:
                    flooding = float(line.split()[-2])
                elif "Continuity Error" in line:
                    in_continuity_section = False

        return {
            "Outflow (gallons)": outflow,
            "Rainfall (ac-ft)": rainfall,
            "Infiltration (ac-ft)": infiltration,
            "Runoff (ac-ft)": runoff,
            "Flooding (ac-ft)": flooding
        }

    except Exception as e:
        print(f"Failed to parse RPT file: {e}")
        return {
            "Outflow (gallons)": None,
            "Rainfall (ac-ft)": None,
            "Infiltration (ac-ft)": None,
            "Runoff (ac-ft)": None,
            "Flooding (ac-ft)": None
        }

# === Water Balance Summary (Only show if RPT data exists) ===
df_balance = None
results = []

rpt_scenarios = {
    "Baseline (No Tide Gate)": "baseline_nogate.rpt",
    "Baseline + Tide Gate": "baseline_gate.rpt",
    "LID (No Tide Gate)": "lid_nogate.rpt",
    "LID + Tide Gate": "lid_gate.rpt",
    "Max LID (No Tide Gate)": "lid_max_nogate.rpt",
    "Max LID + Tide Gate": "lid_max_gate.rpt"
}

results = []
for name, path in rpt_scenarios.items():
    try:
        metrics = extract_volumes_from_rpt(path)
        if any(v is not None for k, v in metrics.items() if k != "Scenario"):
            metrics["Scenario"] = name
            results.append(metrics)

            # Generate a clean key: "baseline_nogate", "lid_max_gate", etc.
            key = (
                name.lower()
                .replace("(", "")
                .replace(")", "")
                .replace(" + ", "_plus_")
                .replace(" ", "_")
            )

            st.session_state[f"outflow_{key}"] = metrics["Outflow (gallons)"]
            st.session_state[f"flood_{key}"] = metrics["Flooding (ac-ft)"]

    except Exception as e:
        print(f"Could not process {name}: {e}")
        continue


st.subheader("How Much Water Each Unit Holds")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 1 cubic meter (m³)
    - 1,000 liters   
    - 35.3 ft³  
    - About the size of a washing machine  
    """)

with col2:
    st.markdown("""
    ### 1 cubic foot (ft³)
    - 7.5 gallons  
    - About the size of a microwave  
    - 4 ft³ = half of a bathtub
    """)

if results:
    df_balance = pd.DataFrame(results).set_index("Scenario")

    # --- Unit Conversion ---
    convert_to_m3 = unit == "Metric (SI)"
    GAL_TO_FT3 = 0.133681
    ACF_TO_FT3 = 43560
    FT3_TO_M3 = 0.0283168

    def convert(val, from_unit):
        if from_unit == "gallons":
            val_ft3 = val * GAL_TO_FT3
        elif from_unit == "ac-ft":
            val_ft3 = val * ACF_TO_FT3
        else:
            val_ft3 = val
        return val_ft3 * FT3_TO_M3 if convert_to_m3 else val_ft3

    # --- Apply Conversions ---
    df_converted = pd.DataFrame(index=df_balance.index)
    df_converted["Flooded Volume"] = df_balance["Flooding (ac-ft)"].apply(lambda x: convert(x, "ac-ft"))
    df_converted["Outflow Volume"] = df_balance["Outflow (gallons)"].apply(lambda x: convert(x, "gallons"))
    df_converted["Infiltration"] = df_balance["Infiltration (ac-ft)"].apply(lambda x: convert(x, "ac-ft"))
    df_converted["Surface Runoff"] = df_balance["Runoff (ac-ft)"].apply(lambda x: convert(x, "ac-ft"))
    
    # --- Round numeric values to nearest whole number ---
    df_converted = df_converted.round(0).astype(int)

    # --- Add Cost BEFORE formatting ---
    cost_lookup = {
        "Baseline (No Tide Gate)": 0,
        "Baseline + Tide Gate": 250000,
        "LID (No Tide Gate)": st.session_state.get("user_total_cost", 0) - 250000,
        "LID + Tide Gate": st.session_state.get("user_total_cost", 0),
        "Max LID (No Tide Gate)": st.session_state.get("max_total_cost", 0) - 250000,
        "Max LID + Tide Gate": st.session_state.get("max_total_cost", 0),
    }
    df_converted["Total Cost ($)"] = df_converted.index.map(cost_lookup).astype(int)

    # --- Format for Display (not for Excel) ---
    unit_label = "m³" if convert_to_m3 else "ft³"
    
    st.subheader(f"Summary ({unit_label})")
    st.dataframe(df_converted)

    # --- Store clean numeric version for export ---
    st.session_state["df_balance"] = df_converted

# === Export Excel Report (Single Click Download Button) ===

st.subheader("Sometimes, the areas that would benefit the most from LID solutions (like reduced flooding) are downstream, while the LID has to be installed upstream, where the runoff begins. This means the benefit (less flooding) happens somewhere else. But the burden (cost, space, maintenance) is on someone upstream.")

output = io.BytesIO()

try:
    # === 1. Scenario Summary Table ===
    df_summary = pd.DataFrame([{
        "Storm Duration (hr)": duration_minutes // 60,
        "Return Period (yr)": return_period,
        "Moon Phase": moon_phase,
        "Tide Alignment": tide_align,
        "Display Units": unit
    }])

    # === 3. Water Balance Table ===
    df_balance = st.session_state.get("df_balance", pd.DataFrame())

    # === 4. Rainfall & Tide Curves ===
    df_rain = pd.DataFrame({
        "Timestamp (min)": st.session_state.get("rain_minutes", []),
        f"Rainfall ({unit})": st.session_state.get("display_rain_curve", [])
    })

    df_tide = pd.DataFrame({
        "Timestamp (min)": st.session_state.get("tide_minutes", []),
        f"Tide ({'ft' if unit == 'U.S. Customary' else 'meters'})": st.session_state.get("display_tide_curve", [])
    })

    # === Write All to Excel ===
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Write summary info
        start_row = 0
        df_summary.to_excel(writer, sheet_name="Scenario Summary", index=False, startrow=start_row)
        start_row += len(df_summary) + 3
        
        if not df_balance.empty:
            unit_label = "m³" if convert_to_m3 else "ft³"
            df_balance_display = df_balance.copy()
            df_balance_display.columns = [f"{col} ({unit_label})" for col in df_balance.columns]
            
            df_balance_display.reset_index().to_excel(
                writer, sheet_name="Scenario Summary", index=False, startrow=start_row
            )

        # Write rainfall and tide curves
        if not df_rain.empty:
            df_rain.to_excel(writer, sheet_name="Rainfall Curve", index=False)

        if not df_tide.empty:
            df_tide.to_excel(writer, sheet_name="Tide Curve", index=False)

        # === 5. Culvert Fill % ===
        culvert_keys = [
            "baseline_timestamps",
            "baseline_fill",
            "baseline_gate_fill",
            "lid_fill",
            "lid_gate_fill",
            "lid_max_fill",
            "lid_max_gate_fill"
        ]

        if all(k in st.session_state for k in culvert_keys):
            lengths = [len(st.session_state[k]) for k in culvert_keys if "timestamps" not in k]
            if len(set(lengths)) == 1 and len(st.session_state["baseline_timestamps"]) == lengths[0]:
                df_culvert = pd.DataFrame({
                    "Timestamp": st.session_state["baseline_timestamps"],
                    "Baseline": st.session_state["baseline_fill"],
                    "Baseline + Tide Gate": st.session_state["baseline_gate_fill"],
                    "With LIDs": st.session_state["lid_fill"],
                    "LIDs + Tide Gate": st.session_state["lid_gate_fill"],
                    "Max LIDs": st.session_state["lid_max_fill"],
                    "Max LIDs + Tide Gate": st.session_state["lid_max_gate_fill"]
                })
                df_culvert.to_excel(writer, sheet_name="Culvert Capacity", index=False)
            else:
                st.warning("Skipped culvert chart export due to mismatched time series lengths.")

    # Reset stream and allow download
    output.seek(0)

    st.download_button(
        label="📥 Download Full Excel Report",
        data=output,
        file_name="FloodSimulationReport.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

except Exception as e:
    st.error(f"Failed to create Excel report: {e}")



