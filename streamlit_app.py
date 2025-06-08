import streamlit as st
import pandas as pd
import altair as alt
import time
import numpy as np
import re
from pyswmm import Simulation
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
output_inp       = "updated_model.inp"

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

unit        = st.selectbox("Rainfall Units", ["inches", "cm", "mm"])
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
elif unit == "cm":
    display_rain_curve = rain_sim_curve * 2.54
    display_tide_curve = tide_sim_curve * 0.3048 * 100
    tide_disp_unit    = "m"
else:  # mm
    display_rain_curve = rain_sim_curve * 25.4
    display_tide_curve = tide_sim_curve * 0.3048 * 1000
    tide_disp_unit    = "m"

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

# === Rainfall CSV Download ===
csv_rain = df_rain.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Rainfall Data (CSV)",
    data=csv_rain,
    file_name="rainfall_profile.csv",
    mime="text/csv"
)

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

# === Tide CSV Download ===
csv_tide = df_tide.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Tide Data (CSV)",
    data=csv_tide,
    file_name="tide_profile.csv",
    mime="text/csv"
)

st.subheader("Watershed with Subcatchments")
st.image(
    "/workspaces/flood-modeling-k12-education/Images/watersheds.png",
    use_container_width=True
)

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

# === Scenario Summary ===
tide_gate_enabled = "YES" if st.checkbox("Include Tide Gate at Outfall?", value=False) else "NO"
st.markdown(f"""
### Selected Scenario Summary
- **Storm Duration:** {duration_minutes//60} hr  
- **Return Period:** {return_period} yr  
- **Moon Phase:** {moon_phase}  
- **Tide Alignment:** {'High Tide Peak' if align_mode=='peak' else 'Low Tide Dip'}  
- **Tide Gate Enabled:** {'Yes' if tide_gate_enabled=='YES' else 'No'}  
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
        update_inp_file(
            template_inp, output_inp,
            rain_lines, tide_lines,
            lid_lines, tide_gate_enabled
        )

        with Simulation(output_inp) as sim:
            sim.execute()
            sim.close()
        time.sleep(1)

        df_runoff = extract_runoff_and_lid_data("updated_model.rpt")
        df_base   = df_runoff[df_runoff["Subcatchment"].str.startswith("Sub_")]

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

# === Show Baseline Results ===
if "baseline_df" in st.session_state:
    st.subheader("Baseline Subcatchment Runoff Summary (Pre-intervention)")
    st.dataframe(st.session_state["baseline_df"], use_container_width=True)

# === Low Impact Developments (LIDs) UI & Cost ===
st.subheader("Low Impact Developments (LIDs)")
st.image(
    "/workspaces/flood-modeling-k12-education/Images/green_infrastructure_options.png",
    use_container_width=True
)

# Cost data placeholders
data = {
    "Infrastructure": ["80 sq.ft. Rain Garden", "55 gallon Rain Barrel", "Tide Gate (10'x5')"],
    "Estimated Installation Cost": ["$450", "$200", "60000"]
}
cost_df = pd.DataFrame(data)
st.subheader("Estimated Costs for Flood Mitigation Options")
st.table(cost_df)

# Load & sort subcatchments
def extract_number(name):
    m = re.search(r"_(\d+)", name)
    return int(m.group(1)) if m else float("inf")

df = pd.read_excel("/workspaces/flood-modeling-k12-education/raster_cells_per_sub.xlsx")
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
        rg_max = int(row["MaxNumber_RG_DEM_considered"])
        rb_max = int(row["MaxRainBarrell_number"])
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
            cost = rg * 350
            cost_breakdown.append({
                "Subcatchment": sub,
                "LID Type": "Rain Garden",
                "Cost": cost
            })
            total_cost += cost
        if rb > 0:
            cost = rb * 200
            cost_breakdown.append({
                "Subcatchment": sub,
                "LID Type": "Rain Barrel",
                "Cost": cost
            })
            total_cost += cost

    # 2) Tide gate cost
    if tide_gate_enabled == "YES":
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

def generate_lid_usage_lines(lid_config, excel_path="/workspaces/flood-modeling-k12-education/raster_cells_per_sub.xlsx"):
    df_excel = pd.read_excel(excel_path)
    lines = []
    for sub, cfg in lid_config.items():
        try:
            imperv_ft2 = df_excel.loc[df_excel["NAME"]==sub, "Impervious_ft2"].values[0]
        except IndexError:
            continue
        rb = cfg.get("rain_barrels", 0)
        if rb>0:
            pct = (rb*595)/imperv_ft2*100
            lines.append(f"{sub:<17}rain_barrel      {rb:<7}2.95     0     0     {pct:.2f}     0     *     *     0")
        rg = cfg.get("rain_gardens",0)
        if rg>0:
            pct = (rg*1000)/imperv_ft2*100
            lines.append(f"{sub:<17}rain_garden      {rg:<7}85.0     0     0     {pct:.2f}     0     *     *     0")
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

            update_inp_file(
                template_inp, output_inp,
                rain_lines, tide_lines,
                lid_lines, tide_gate_enabled
            )
            with Simulation(output_inp) as sim:
                sim.execute()
                sim.close()
            time.sleep(1)

            df_runoff = extract_runoff_and_lid_data("updated_model.rpt")
            df_latest = df_runoff[df_runoff["Subcatchment"].str.startswith("Sub_")]

            # Convert for display
            if unit=="inches":
                df_disp = df_latest.rename(columns={
                    "Impervious Runoff (in)": "Impervious Runoff (inches)",
                    "Pervious Runoff   (in)": "Pervious Runoff (inches)"
                })
            else:
                factor = 2.54 if unit=="cm" else 25.4
                df_disp = pd.DataFrame({
                    "Subcatchment": df_latest["Subcatchment"],
                    f"Impervious Runoff ({unit})":
                        df_latest["Impervious Runoff (in)"] * factor,
                    f"Pervious Runoff ({unit})":
                        df_latest["Pervious Runoff   (in)"] * factor
                })

            st.subheader("Updated Runoff Summary with LID Improvements")
            st.dataframe(df_disp, use_container_width=True)

        except Exception as e:
            st.error(f"LID simulation failed: {e}")
