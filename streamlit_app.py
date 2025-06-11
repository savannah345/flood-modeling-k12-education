import streamlit as st
import pandas as pd
import altair as alt
import time
import numpy as np
import re
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
    page_title="High Stakes, High Water: A Watershed Design Challenge for Coastal Resilience",
    layout="centered"
)
st.title("High Stakes, High Water: A Watershed Design Challenge for Coastal Resilience")

# === User Inputs ===
duration_minutes = st.selectbox(
    "Storm Duration",
    options=pf_df["Duration_Minutes"],
    format_func=lambda x: f"{x // 60} hr"
)
return_period = st.selectbox("Return Period (years)", pf_df.columns[1:])
# pf_df values are in inches
rain_inches = float(pf_df.loc[
    pf_df["Duration_Minutes"] == duration_minutes,
    return_period
].values[0])

unit        = st.selectbox("Rainfall Units", ["inches", "centimeters"])
method      = st.radio("Rainfall Shape", ["Normal", "Randomized"])
moon_phase  = st.selectbox("Moon Phase", list(moon_tide_ranges.keys()))
tide_align  = st.radio(
    "Tide Alignment",
    ["Peak aligned with High Tide", "Peak aligned with Low Tide"]
)
align_mode  = "peak" if "High" in tide_align else "low"

# --- Generate SWMM-curves (always inches & feet) ---
tide_sim_minutes, tide_sim_curve = generate_tide_curve(moon_phase, "inches")
rain_sim_minutes, rain_sim_curve = align_rainfall_to_tide(
    rain_inches,
    duration_minutes,
    tide_sim_curve,
    align=align_mode,
    method=method
)

# --- Generate display-curves based on selected unit ---
if unit == "inches":
    display_rain_curve = rain_sim_curve
    display_tide_curve = tide_sim_curve
    tide_disp_unit    = "ft"
elif unit == "centimeters":
    display_rain_curve = rain_sim_curve * 2.54
    display_tide_curve = tide_sim_curve * 0.3048 * 100
    tide_disp_unit    = "m"

# --- Store display values and timestamps for Excel export ---
st.session_state["rain_minutes"] = rain_sim_minutes
st.session_state["tide_minutes"] = tide_sim_minutes
st.session_state["display_rain_curve"] = display_rain_curve
st.session_state["display_tide_curve"] = display_tide_curve

time_hours = np.array(rain_sim_minutes) / 60

# === Display Rainfall Chart ===
df_rain = pd.DataFrame({
    "Time (hours)": time_hours,
    f"Rainfall ({unit})": display_rain_curve
})
st.subheader("Rainfall Distribution")
rain_chart = (
    alt.Chart(df_rain)
       .mark_line()
       .encode(
           x="Time (hours)",
           y=f"Rainfall ({unit})"
       )
       .properties(title="Rainfall Distribution")
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
       .properties(title="Tide Profile")
)
st.altair_chart(tide_chart, use_container_width=True)

# === INP File Updater ===
def update_inp_file(template_path, output_path,
                    rain_lines, tide_lines,
                    lid_lines, gate_flag):
    text = open(template_path).read()
    text = text.replace("$RAINFALL_TIMESERIES$", "\n".join(rain_lines))
    text = text.replace("$TIDE_TIMESERIES$",    "\n".join(tide_lines))
    text = text.replace("$LID_USAGE$",           "\n".join(lid_lines))
    text = text.replace("$TIDE_GATE_CONTROL$",    gate_flag)
    open(output_path, "w").write(text)

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
        rain_lines = format_timeseries(
            "rain_gage_timeseries",
            rain_sim_minutes, rain_sim_curve,
            simulation_date
        )
        tide_lines = format_timeseries(
            "tide",
            tide_sim_minutes, tide_sim_curve,
            simulation_date
        )
        lid_lines = [";"]
        
        full_depth = 10.0  # ft 
        report_interval = timedelta(minutes=5)

        # Run baseline WITHOUT tide gate
        update_inp_file(template_inp, "baseline_nogate.inp", rain_lines, tide_lines, lid_lines, "NO")
        depth_pct_no_gate, timestamps = [], []
        with Simulation("baseline_nogate.inp") as sim:
            link = Links(sim)["C70_2"]
            last_report_time = None
            for step in sim:
                current_time = sim.current_time
                if last_report_time is None or (current_time - last_report_time) >= report_interval:
                    pct = (link.depth / full_depth) * 100
                    depth_pct_no_gate.append(pct)
                    timestamps.append(current_time.strftime("%H:%M"))
                    last_report_time = current_time
        st.session_state["baseline_fill"] = depth_pct_no_gate
        st.session_state["baseline_timestamps"] = timestamps

        # After NO gate simulation
        df_base_nogate = extract_runoff_and_lid_data("updated_model.rpt")
        st.session_state["df_base_nogate"] = df_base_nogate

        df_runoff = extract_runoff_and_lid_data("updated_model.rpt")
        df_base   = df_runoff[df_runoff["Subcatchment"].str.startswith("Sub_")]


        # Run baseline WITH tide gate
        update_inp_file(template_inp, "baseline_gate.inp", rain_lines, tide_lines, lid_lines, "YES")
        depth_pct_gate, _ = [], []
        with Simulation("baseline_gate.inp") as sim:
            link = Links(sim)["C70_2"]
            last_report_time = None
            for step in sim:
                current_time = sim.current_time
                if last_report_time is None or (current_time - last_report_time) >= report_interval:
                    pct = (link.depth / full_depth) * 100
                    depth_pct_gate.append(pct)
                    last_report_time = current_time
        st.session_state["baseline_gate_fill"] = depth_pct_gate

        df_base_gate = extract_runoff_and_lid_data("updated_model.rpt")
        st.session_state["df_base_gate"] = df_base_gate

        # Convert for display
        if unit == "inches":
            df_disp = df_base.rename(columns={
                "Impervious Runoff (in)":    "Impervious Runoff (inches)",
                "Pervious Runoff   (in)":    "Pervious Runoff (inches)"
            })
        else:
            factor = 2.54 if unit=="cm" else 25.4
            df_disp = pd.DataFrame({
                "Subcatchment": df_base["Subcatchment"],
                f"Impervious Runoff ({unit})":
                    df_base["Impervious Runoff (in)"] * factor,
                f"Pervious Runoff ({unit})":
                    df_base["Pervious Runoff   (in)"] * factor
            })

        st.session_state["baseline_df"] = df_disp
        st.success("Baseline scenario complete!")

    except Exception as e:
        st.error(f"Baseline simulation failed: {e}")

if "df_base_nogate" in st.session_state:
    st.subheader("Baseline Runoff (No Tide Gate)")

    df_no = st.session_state["df_base_nogate"].copy()

    # Optional unit conversion
    if unit != "inches":
        factor = 2.54 if unit == "cm" else 25.4
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

st.subheader("Pipes")
st.image(
    "model_with_pipes.png",
    use_container_width=True
)

st.subheader("Watershed with Subcatchments")
st.image(
    "watersheds.png",
    use_container_width=True
)

# === Low Impact Developments (LIDs) UI & Cost ===
st.subheader("Low Impact Developments (LIDs)")
st.image(
    "green_infrastructure_options.png",
    use_container_width=True
)

st.image(
    "amounts.png",
    use_container_width=True
)



# Cost data placeholders
data = {
    "Infrastructure": ["85 sq.ft. Rain Garden", "50 gallon Rain Barrel", "Tide Gate (10'x5')"],
    "Estimated Installation Cost": ["$250", "$100", "60000"]
}
cost_df = pd.DataFrame(data)
st.subheader("Estimated Costs for Flood Mitigation Options")
st.table(cost_df)

# Load & sort subcatchments
def extract_number(name):
    m = re.search(r"_(\d+)", name)
    return int(m.group(1)) if m else float("inf")

df = pd.read_excel("raster_cells_per_sub.xlsx")
df = df.sort_values(by="NAME", key=lambda x: x.map(extract_number)).reset_index(drop=True)

st.title("Add LIDs")
if "user_lid_config" not in st.session_state:
    st.session_state["user_lid_config"] = {}

available_subs = df["NAME"].tolist()
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
        tide_cost = 60000
        cost_breakdown.append({
            "Subcatchment": "Watershed Outfall",
            "LID Type": "Tide Gate",
            "Cost": tide_cost
        })
        total_cost += tide_cost


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
    st.info("You haven't selected any subcatchments to update.")

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
            pct_imp = (rb * 595) / imperv * 100
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
            pct_perv = (rg * 1000) / perv * 100
            lines.append(tpl.format(
                sub=sub,
                proc="rain_garden",
                num=rg,
                area=f"{85:.0f}",
                width=0,
                initsat=0,
                fromimp=0,
                toperv=1,
                rptfile="*",
                drainto="*",
                fromperv=f"{pct_perv:.2f}"
            ))

    return lines


# === Run Scenario With LIDs ===
if st.button("Run Scenario With Selected LID Improvements"):
    if "baseline_df" not in st.session_state:
        st.error("Please run the baseline scenario first.")
    else:
        try:
            rain_lines = format_timeseries(
                "rain_gage_timeseries",
                rain_sim_minutes, rain_sim_curve,
                simulation_date
            )
            tide_lines = format_timeseries(
                "tide",
                tide_sim_minutes, tide_sim_curve,
                simulation_date
            )
            lid_lines = generate_lid_usage_lines(st.session_state["user_lid_config"])
            if not lid_lines:
                st.warning("No LIDs selected.")
                st.stop()

            full_depth = 10.0  # ft
            report_interval = timedelta(minutes=5)

            update_inp_file(template_inp, "lid_nogate.inp", rain_lines, tide_lines, lid_lines, "NO")
            depth_pct_lid, timestamps = [], []
            with Simulation("lid_nogate.inp") as sim:
                link = Links(sim)["C70_2"]
                last_report_time = None
                for step in sim:
                    current_time = sim.current_time
                    if last_report_time is None or (current_time - last_report_time) >= report_interval:
                        pct = (link.depth / full_depth) * 100
                        depth_pct_lid.append(pct)
                        timestamps.append(current_time.strftime("%H:%M"))
                        last_report_time = current_time
            st.session_state["lid_fill"] = depth_pct_lid
            st.session_state["lid_timestamps"] = timestamps

            df_lid_nogate = extract_runoff_and_lid_data("updated_model.rpt")
            st.session_state["df_lid_nogate"] = df_lid_nogate

            update_inp_file(template_inp, "updated_model.inp", rain_lines, tide_lines, lid_lines, "YES")
            depth_pct_lid_gate, _ = [], []
            with Simulation("lid_gate.inp") as sim:
                link = Links(sim)["C70_2"]
                last_report_time = None
                for step in sim:
                    current_time = sim.current_time
                    if last_report_time is None or (current_time - last_report_time) >= report_interval:
                        pct = (link.depth / full_depth) * 100
                        depth_pct_lid_gate.append(pct)
                        last_report_time = current_time
            st.session_state["lid_gate_fill"] = depth_pct_lid_gate
            
            df_lid_gate = extract_runoff_and_lid_data("updated_model.rpt")
            st.session_state["df_lid_gate"] = df_lid_gate

        except Exception as e:
            st.error(f"LID simulation failed: {e}")


if all(key in st.session_state for key in [
    "baseline_fill", "baseline_gate_fill",
    "lid_fill", "lid_gate_fill",
    "baseline_timestamps"
]):
    st.subheader("Culvert Capacity Over Time (All Scenarios)")

    time_labels = st.session_state["baseline_timestamps"]
    
    # Convert to datetime for proper tick formatting
    time_objects = [datetime.strptime(t, "%H:%M") for t in time_labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_objects, st.session_state["baseline_fill"], label="Baseline", linewidth=2)
    ax.plot(time_objects, st.session_state["baseline_gate_fill"], label="Baseline + Tide Gate", linestyle="--", linewidth=2)
    ax.plot(time_objects, st.session_state["lid_fill"], label="With LIDs", linestyle="-.", linewidth=2)
    ax.plot(time_objects, st.session_state["lid_gate_fill"], label="LIDs + Tide Gate", linestyle=":", linewidth=2)

    ax.set_ylabel("Culvert Fill (%)")
    ax.set_xlabel("Time")
    ax.set_ylim(0, 110)
    ax.set_title("Culvert Capacity Comparison")

    # Show only hourly ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator())

    ax.legend()
    ax.grid(False)
    fig.autofmt_xdate()  # Auto-format x-axis for better label spacing
    st.pyplot(fig)

import io

# === Export Excel Report ===
if st.button("Download Full Excel Report"):
    try:
        timestamps_rain = st.session_state.get("rain_minutes", [])  # Use actual rainfall timestamps
        rain_curve = st.session_state.get("display_rain_curve", [])

        timestamps_tide = st.session_state.get("tide_minutes", [])  # Use actual tide timestamps
        tide_curve = st.session_state.get("display_tide_curve", [])

        # === Sanity check lengths
        if len(timestamps_rain) != len(rain_curve):
            st.error("Mismatch in rainfall timestamps and values.")
        elif len(timestamps_tide) != len(tide_curve):
            st.error("Mismatch in tide timestamps and values.")
        else:
            # Summary Info
            summary_data = {
                "Storm Duration (hr)": [duration_minutes // 60],
                "Return Period (yr)": [return_period],
                "Moon Phase": [moon_phase],
                "Tide Alignment": [tide_align],
                "Display Units": [unit]
            }
            df_summary = pd.DataFrame(summary_data)

            # Rainfall
            df_rain = pd.DataFrame({
                "Timestamp (min)": timestamps_rain,
                f"Rainfall ({unit})": rain_curve
            })

            # Tide
            df_tide = pd.DataFrame({
                "Timestamp (min)": timestamps_tide,
                f"Tide ({'ft' if unit == 'inches' else 'm'})": tide_curve
            })

            # Culvert Fill Comparison
            df_capacity = pd.DataFrame({
                "Timestamp": st.session_state.get("baseline_timestamps", []),
                "Baseline": st.session_state.get("baseline_fill", []),
                "Baseline + Tide Gate": st.session_state.get("baseline_gate_fill", []),
                "With LIDs": st.session_state.get("lid_fill", []),
                "LIDs + Tide Gate": st.session_state.get("lid_gate_fill", [])
            })

            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_summary.to_excel(writer, sheet_name="Scenario Summary", index=False)
                df_rain.to_excel(writer, sheet_name="Rainfall Curve", index=False)
                df_tide.to_excel(writer, sheet_name="Tide Curve", index=False)
                df_capacity.to_excel(writer, sheet_name="Culvert Capacity", index=False)

            output.seek(0)
            st.download_button(
                label="📥 Download Excel Report",
                data=output,
                file_name="FloodSimulationReport.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"Failed to create Excel report: {e}")
