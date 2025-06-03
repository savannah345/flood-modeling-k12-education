import streamlit as st
import pandas as pd
import altair as alt
import time 
import numpy as np
from pyswmm import Simulation
from rainfall_and_tide_generator import (
    generate_rainfall,
    convert_units,
    pf_df,
    generate_tide_curve,
    align_rainfall_to_tide,
    moon_tide_ranges
)

st.set_page_config(page_title="🌧️ Rainfall & Tide Simulator", layout="centered")

st.title("High Stakes, High Water: A Watershed Design Challenge for Coastal Resilience")

# Cost data (example values, you can adjust)
data = {
    "Infrastructure": ["Rain Garden", "Permeable Pavement", "Rain Barrel", "Tide Gate (10'x5')"],
    "Estimated Cost": ["$10-40 per sq ft", "$8–15 per sq ft", "$100–300", "$15,000–30,000"]
}

# Create DataFrame
cost_df = pd.DataFrame(data)

# Display in Streamlit
st.subheader("💰 Estimated Costs for Flood Mitigation Options")
st.table(cost_df)  # Use st.dataframe(cost_df) if you want scroll/sort features

# === User Inputs ===
duration_minutes = st.selectbox("Storm Duration", options=pf_df["Duration_Minutes"],
                                format_func=lambda x: f"{x // 60} hr")

return_period = st.selectbox("Return Period (years)", options=pf_df.columns[1:])
rain_inches = pf_df.loc[pf_df["Duration_Minutes"] == duration_minutes, return_period].values[0]

unit = st.selectbox("Rainfall Units", ["inches", "cm", "mm"])
method = st.radio("Rainfall Shape", ["Normal", "Randomized"])
moon_phase = st.selectbox("Moon Phase", list(moon_tide_ranges.keys()))
tide_align = st.radio("Tide Alignment", ["🌊 Peak aligned with High Tide", "🌊 Peak aligned with Low Tide"])
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
st.subheader("🌧️ Rainfall Distribution")
st.line_chart(df_rain.set_index("Time (hours)"))

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
st.subheader("🌊 Tide Profile")
st.line_chart(df_tide.set_index("Time (hours)"))

# === Tide CSV Download ===
csv_tide = df_tide.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Tide Data (CSV)",
    data=csv_tide,
    file_name="tide_profile.csv",
    mime="text/csv"
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
    lid_section = False
    runoff_data = []
    lid_data = []

    for line in lines:
        # Detect section headers
        if "Subcatchment Runoff Summary" in line:
            runoff_section = True
            lid_section = False
            continue
        elif "LID Performance Summary" in line:
            runoff_section = False
            lid_section = True
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
                    "Total Runoff (in)": float(parts[7]),
                    "Runoff Coeff": float(parts[10])
                })

        # Parse LID section
        if lid_section:
            parts = line.split()
            if len(parts) >= 10 and all(p.replace('.', '', 1).replace('-', '', 1).isdigit() for p in parts[2:10]):
                lid_data.append({
                    "Subcatchment": parts[0],
                    "LID Control": parts[1],
                    "Inflow (in)": float(parts[2]),
                    "Infil Loss (in)": float(parts[4]),
                    "Surface Outflow (in)": float(parts[5]),
                })

    runoff_df = pd.DataFrame(runoff_data)
    lid_df = pd.DataFrame(lid_data)
    return runoff_df, lid_df



# === Tide Gate Option ===
st.subheader("🌉 Tide Gate Option")
tide_gate_enabled = "YES" if st.checkbox("Include Tide Gate at Outfall?", value=False) else "NO"


# === Simulation Trigger
if st.button("🌊 Run SWMM Simulation"):
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
        st.success("✅ SWMM simulation complete!")

        time.sleep(5)  # Wait 1 second

        runoff_df, lid_df = extract_runoff_and_lid_data("updated_model.rpt")

        st.subheader("📊 Subcatchment Runoff Summary")
        st.dataframe(runoff_df, use_container_width=True)

        st.subheader("🌱 LID Performance Summary")
        st.dataframe(lid_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Simulation failed: {e}")