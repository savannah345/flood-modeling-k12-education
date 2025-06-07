import streamlit as st
import pandas as pd
import altair as alt
import time 
import numpy as np
import re
import openpyxl
from pyswmm import Simulation
from rainfall_and_tide_generator import (
    generate_rainfall,
    convert_units,
    pf_df,
    generate_tide_curve,
    align_rainfall_to_tide,
    moon_tide_ranges
)

st.set_page_config(page_title="High Stakes, High Water: A Watershed Design Challenge for Coastal Resilience", layout="centered")

st.title("High Stakes, High Water: A Watershed Design Challenge for Coastal Resilience")

# === User Inputs ===
duration_minutes = st.selectbox("Storm Duration", options=pf_df["Duration_Minutes"],
                                format_func=lambda x: f"{x // 60} hr")

return_period = st.selectbox("Return Period (years)", options=pf_df.columns[1:])
rain_inches = pf_df.loc[pf_df["Duration_Minutes"] == duration_minutes, return_period].values[0]

unit = st.selectbox("Rainfall Units", ["inches", "cm", "mm"])
method = st.radio("Rainfall Shape", ["Normal", "Randomized"])
moon_phase = st.selectbox("Moon Phase", list(moon_tide_ranges.keys()))
tide_align = st.radio("Tide Alignment", ["Peak aligned with High Tide", "Peak aligned with Low Tide"])
align_option = "peak" if "High" in tide_align else "low"
align = "peak" if "High" in tide_align else "dip"

# === Generate Tide & Rainfall ===
converted_inches = convert_units(rain_inches, unit)
tide_minutes, tide_curve = generate_tide_curve(moon_phase, unit)
rain_minutes, rain_curve = align_rainfall_to_tide(converted_inches, duration_minutes, tide_curve, align_option, method)

# === Display Rainfall Chart ===
df_rain = pd.DataFrame({
    "Time (hours)": np.array(rain_minutes) / 60,
    f"Rainfall ({unit})": rain_curve
})
st.subheader("Rainfall Distribution")
rain_chart = alt.Chart(df_rain).mark_line().encode(
    x=alt.X('Time (hours)', title='Time (hours)'),
    y=alt.Y(f'Rainfall ({unit})', title=f'Rainfall ({unit})')
).properties(title="Rainfall Distribution")

st.altair_chart(rain_chart, use_container_width=True)

# === Rainfall CSV Download ===
csv_rain = df_rain.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Rainfall Data (CSV)",
    data=csv_rain,
    file_name="rainfall_profile.csv",
    mime="text/csv"
)


# === Display Tide Chart ===
tide_unit = "meters" if unit in ["cm", "mm"] else "feet"
df_tide = pd.DataFrame({
    "Time (hours)": np.array(tide_minutes) / 60,
    f"Tide ({tide_unit})": tide_curve
})
st.subheader("Tide Profile")
tide_chart = alt.Chart(df_tide).mark_line().encode(
    x=alt.X('Time (hours)', title='Time (hours)'),
    y=alt.Y(f'Tide ({tide_unit})', title=f'Tide ({tide_unit})')
).properties(title="Tide Profile")

st.altair_chart(tide_chart, use_container_width=True)


# === Tide CSV Download ===
csv_tide = df_tide.to_csv(index=False).encode('utf-8')
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


def update_inp_file(template_path, output_path,
                    rain_lines, tide_lines, lid_usage,
                    tide_gate_enabled):
    with open(template_path, "r") as f:
        text = f.read()

    text = text.replace("$RAINFALL_TIMESERIES$", "\n".join(rain_lines))
    text = text.replace("$TIDE_TIMESERIES$", "\n".join(tide_lines))
    text = text.replace("$LID_USAGE$", "\n".join(lid_usage))
    text = text.replace("$TIDE_GATE_CONTROL$", tide_gate_enabled)

    with open(output_path, "w") as f:
        f.write(text)

def format_timeseries(name, minutes, values):
    return [f"{name} 06/01/2025 {int(m)//60:02d}:{int(m)%60:02d} {v:.5f}" for m, v in zip(minutes, values)]

# === Runoff and LID Data Extraction ===
def extract_runoff_and_lid_data(rpt_file):
    with open(rpt_file) as f:
        lines = f.readlines()

    runoff_section = False
    runoff_data = []

    for line in lines:
        # Detect section headers
        if "Subcatchment Runoff Summary" in line:
            runoff_section = True
            lid_section = False
            continue


        # Skip dashed separators and blank lines
        if "----" in line or line.strip() == "":
            continue

        # Skip units row or any text row that contains non-numeric "in" or headers
        if "Subcatchment" in line or "Inflow" in line:
            continue

        # Parse Runoff section
        if runoff_section:
            parts = line.split()
            if len(parts) >= 11 and all(p.replace('.', '', 1).replace('-', '', 1).isdigit() for p in parts[4:7] + [parts[10]]):
                runoff_data.append({
                    "Subcatchment": parts[0],
                    "Imperv Runoff (in)": float(parts[5]),
                    "Perv Runoff (in)": float(parts[6]),
                    "Total Runoff (in)": float(parts[7])
                })

    runoff_df = pd.DataFrame(runoff_data)
    runoff_df.rename(columns={
        "Imperv Runoff (in)": "Impervious Runoff (in)",
        "Perv Runoff (in)": "Pervious Runoff (in)"
    }, inplace=True)
    return runoff_df



st.subheader("Tide Gate")
st.image(
    "/workspaces/flood-modeling-k12-education/Images/tide_gate.png",
    use_container_width=True
)


# === Tide Gate Option ===
st.subheader("Tide Gate Option")
tide_gate_enabled = "YES" if st.checkbox("Include Tide Gate at Outfall?", value=False) else "NO"

scenario_summary = f"""
### 📋 Scenario Summary  
- **Storm Duration:** {duration_minutes // 60} hours  
- **Return Period:** {return_period} year  
- **Moon Phase:** {moon_phase}  
- **Tide Alignment:** {"Rainfall aligned with high tide" if align == "peak" else "Rainfall aligned with low tide"}  
- **Tide Gate Enabled:** {"Yes" if tide_gate_enabled == "YES" else "No"}  
- **Units Displayed to User:** {unit}
"""
st.markdown(scenario_summary)


# === Simulation Trigger
if st.button("Run Baseline Scenario SWMM Simulation"):
    try:
        # Convert rainfall total based on unit
        total_inches = convert_units(rain_inches, unit)

        # Generate rainfall and tide curves
        tide_minutes, tide_vals = generate_tide_curve(moon_phase, unit)
        rain_minutes, rain_vals = align_rainfall_to_tide(
            total_inches, duration_minutes, tide_vals,
            align=align, method=method
        )

        # Format time series
        rain_lines = format_timeseries("rain_gage_timeseries", rain_minutes, rain_vals)
        tide_lines = format_timeseries("tide", tide_minutes, tide_vals)

        # Example LID usage (should be generated from user input)
        lid_usage = [
            "Sub_1 rain_garden 1 100 0 100 0 0 * * 0",
            "Sub_2 rain_barrel 1 50 0 0 0 0 * * 0"
        ]

        # === Update INP File
        update_inp_file(
            "swmm_project.inp", "updated_model.inp",
            rain_lines, tide_lines, lid_usage,
            tide_gate_enabled
        )

        # === Run Simulation
        with Simulation("updated_model.inp") as sim:
            sim.execute()
        st.success("Baseline Scenario Complete!")

        time.sleep(5)  # Wait 5 seconds

        runoff_df = extract_runoff_and_lid_data("updated_model.rpt")

        st.subheader("Subcatchment Runoff Summary")
        # Save the original results under a separate key
        st.session_state['baseline_runoff_df'] = runoff_df.copy()
        st.session_state['latest_runoff_df'] = runoff_df.copy()  # optional if you want a live-updated version for LID runs

    except Exception as e:
        st.error(f"❌ Simulation failed: {e}")


if 'baseline_runoff_df' in st.session_state:
    st.markdown("### 📈 Baseline Subcatchment Runoff Summary (Pre-LID)")
    df_baseline = st.session_state['baseline_runoff_df'].copy()

    if unit in ['cm', 'mm']:
        multiplier = 2.54 if unit == 'cm' else 25.4
        for col in ['Impervious Runoff (in)', 'Pervious Runoff (in)', 'Total Runoff (in)']:
            new_col = col.replace('(in)', f'({unit})')
            df_baseline[new_col] = df_baseline[col] * multiplier
        df_baseline.drop(columns=[col for col in df_baseline.columns if '(in)' in col], inplace=True)

    st.dataframe(df_baseline, use_container_width=True)


st.subheader("Low Impact Developments (LIDs)")
st.image(
    "/workspaces/flood-modeling-k12-education/Images/green_infrastructure_options.png",
    use_container_width=True
)


# Cost data (example values, you can adjust)
data = {
    "Infrastructure": ["Rain Garden", "Permeable Pavement", "Rain Barrel", "Tide Gate (10'x5')"],
    "Estimated Cost": ["$3-5 per sq ft", "$10–40 per sq ft", "$100–300", "$15,000–30,000"]
}

# Create DataFrame
cost_df = pd.DataFrame(data)

# Display in Streamlit
st.subheader("Estimated Costs for Flood Mitigation Options")
st.table(cost_df)  # Use st.dataframe(cost_df) if you want scroll/sort features


        # === Load and sort subcatchments ===
def extract_number(name):
    match = re.search(r"_(\d+)", name)
    return int(match.group(1)) if match else float('inf')

df = pd.read_excel("/workspaces/flood-modeling-k12-education/raster_cells_per_sub.xlsx")
df = df.sort_values(by="NAME", key=lambda x: x.map(extract_number)).reset_index(drop=True)

st.title("Add LIDs")

if "user_lid_config" not in st.session_state:
    st.session_state.user_lid_config = {}

# === Step 1: Let student choose if they want to intervene ===
available_subs = df["NAME"].tolist()
selected_subs = st.multiselect(
    "Optional: Select subcatchments to add rain gardens, rain barrels, and/or permeable pavement",
    options=available_subs,
    help="If you want to test different LID options, choose one or more subcatchments to adjust."
)

# === Step 2: Only show the configuration table if something is selected ===
if selected_subs:
    # === Styling ===
    st.markdown("""
        <style>
        .lid-table-container {
            max-height: 600px;
            overflow-y: scroll;
            border: 1px solid #ccc;
        }
        .lid-row {
            display: grid;
            grid-template-columns: 1.5fr 1.5fr 1.5fr 1.5fr 1.2fr 1.2fr 1.2fr;
            padding: 6px 5px;
            font-size: 16px;
            align-items: center;
            border-bottom: 1px solid #ccc;
        }
        .alt-row {
            background-color: #f9f9f9;
        }
        .lid-cell input {
            font-size: 16px !important;
            text-align: center !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # === Header ===
    st.markdown("""
    <div class="lid-row" style="font-weight:bold; background-color:#ddd;">
        <div>Subcatchment</div>
        <div>Max Rain Garden Area (ft²)</div>
        <div>Your Rain Garden Area</div>
        <div>Max Pavement Area (ft²)</div>
        <div>Your Pavement Area</div>
        <div>Max Rain Barrels</div>
        <div>Your Rain Barrels</div>
    </div>
    """, unsafe_allow_html=True)

    # === Start Table Container ===
    st.markdown('<div class="lid-table-container">', unsafe_allow_html=True)

    # === Table Rows ===
    for idx, row in df[df["NAME"].isin(selected_subs)].iterrows():
        sub = row["NAME"]
        rg_max = int(row["Rain_garden_max_ft2"])
        pp_max = int(row["Permeable_pave_max_ft2"])
        rb_max = int(row["MaxRainBarrell_number"])

        row_class = "lid-row alt-row" if idx % 2 == 0 else "lid-row"

        st.markdown(f'<div class="{row_class}">', unsafe_allow_html=True)
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1.5, 1.5, 1.5, 1.5, 1.2, 1.2, 1.2])

        with col1: st.markdown(f"**{sub}**", unsafe_allow_html=True)
        with col2: st.markdown(f"<div style='text-align: center;'>{rg_max}</div>", unsafe_allow_html=True)
        with col3:
            rg_val = st.number_input("", 0, rg_max, 0, step=10, key=f"rg_{sub}", label_visibility="collapsed")
        with col4: st.markdown(f"<div style='text-align: center;'>{pp_max}</div>", unsafe_allow_html=True)
        with col5:
            pp_val = st.number_input("", 0, pp_max, 0, step=10, key=f"pp_{sub}", label_visibility="collapsed")
        with col6: st.markdown(f"<div style='text-align: center;'>{rb_max}</div>", unsafe_allow_html=True)
        with col7:
            rb_val = st.number_input("", 0, rb_max, 0, step=1, key=f"rb_{sub}", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.user_lid_config[sub] = {
            "rain_garden_area": rg_val,
            "permeable_pave_area": pp_val,
            "rain_barrels": rb_val
        }

    st.markdown('</div>', unsafe_allow_html=True)


else:
    st.info("You haven't selected any subcatchments to update.")

total_cost = 0
cost_breakdown = []

for sub, config in st.session_state.user_lid_config.items():
    rg = config["rain_garden_area"]
    pp = config["permeable_pave_area"]
    rb = config["rain_barrels"]
    
    if rg + pp + rb == 0:
        continue

    area_multiplier = 1 if unit == "inches" else 10.7639
    if rg > 0:
        cost = rg * area_multiplier * 4
        cost_breakdown.append({"Subcatchment": sub, "LID Type": "Rain Garden", "Cost": cost})
        total_cost += cost
    if pp > 0:
        cost = pp * area_multiplier * 20
        cost_breakdown.append({"Subcatchment": sub, "LID Type": "Permeable Pavement", "Cost": cost})
        total_cost += cost
    if rb > 0:
        cost = rb * 200
        cost_breakdown.append({"Subcatchment": sub, "LID Type": "Rain Barrels", "Cost": cost})
        total_cost += cost

# Tide gate cost
if tide_gate_enabled == "YES":
    tide_cost = 22500  # Average estimate
    cost_breakdown.append({"Subcatchment": "Watershed Outfall", "LID Type": "Tide Gate", "Cost": tide_cost})
    total_cost += tide_cost

if cost_breakdown:
    st.markdown("### 📊 Estimated Cost by Subcatchment and LID Type")
    cost_df = pd.DataFrame(cost_breakdown)

    chart = alt.Chart(cost_df).mark_bar().encode(
        x=alt.X("Subcatchment:N", title="Subcatchment"),
        y=alt.Y("Cost:Q", title="Cost ($)", stack="zero"),
        color=alt.Color("LID Type:N", legend=alt.Legend(title="LID Type")),
        tooltip=["Subcatchment", "LID Type", "Cost"]
    ).properties(width=650, height=350)

    st.altair_chart(chart, use_container_width=True)

    st.markdown(f"### 💰 Total Estimated Cost (Including Tide Gate): **${total_cost:,.2f}**")
else:
    st.info("No infrastructure improvements (gray or green) have been added to the simulation yet.")
