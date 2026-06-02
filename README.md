## CoastWise: Compound Flooding Scenario Builder, Simulator, and Mapper

CoastWise is a full workflow for constructing storm scenarios, generating rainfall (design storms from **NOAA Atlas 14**) and tide inputs (real-time data or synthetic if real-time is unavailable), running the EPA's **Stormwater Management Model (SWMM)** simulations, visualizing baseline conditions for subcatchment runoff, and exploring green infrastructure placement and amounts across a **155-acre watershed**.

The Streamlit app is the central user interface for CoastWise. It integrates rainfall/tide generation, land-use–based LID planning, SWMM model execution, flooding extraction, and interactive map/chart visualization.

---

## User Authentication
Login, signup, and password reset using:  
- **Supabase** (if running in free Streamlit Cloud), or  
- **PostgreSQL** (if running in Docker)

---

## Storm Scenario Builder
Users can configure storm and tide conditions, including:

- **Units:** U.S. Customary or SI  
- **Storm Duration:** 2–12 hours  
- **Return Period:** NOAA Atlas 14 rainfall values  
- **Tide Input Source:**  
  - Real-time tide data (via Greenstream)  
  - Synthetic moon-phase–based tide ranges (fallback if real-time is unavailable)  
- **Rainfall-Tide Alignment:**  
  - Align rainfall peak with **high tide**, or  
  - Align rainfall peak with **low tide**

---

## SWMM Simulations
The app automatically:

- Builds rainfall, tide, and LID usage **time-series blocks**  
- Creates per-scenario **SWMM INP files**  
- Executes simulations via **pyswmm**  
- Extracts from the SWMM report:  
  - **Node Flooding Summary**  
  - **Subcatchment Runoff Totals**

---

## Baseline and LID Scenarios
Five spatial layouts are run under the same budget:

1. **Baseline** (no LIDs)  
2. **All subcatchments**  
3. **Upstream areas**  
4. **Downstream areas**  
5. **High-runoff areas**

Each layout is modeled with the **Tide Gate & Pump ON** and **OFF**, producing:

**10 total SWMM scenarios**

---

## Reporting Summary Charts
CoastWise generates summary charts showing:

1. **Total Costs**  
2. **Rain Garden & Rain Barrel Counts**  
3. **Total Storage Volumes**

---

## Flooding Volume Comparison Across 10 Scenarios
For each of the 10 runs, the SWMM model produces a **total flood volume** (cubic feet or cubic meters).  
CoastWise extracts this directly from the **Node Flooding Summary** within the `.rpt` file.

The app then displays a ranked bar chart so users can quickly see:

- Which scenario performs **best** at reducing flooding  
- How much influence the **tide gate** has  
- Which LID layout produces the largest **flood-reduction benefit**

---

## Interactive Scenario‑to‑Scenario Difference Mapping
This tool allows users to compare **any two scenarios visually, node by node**.

For each node, CoastWise loads:

- Its location (from `Nodes.shp`)  
- Flooding depth for **Scenario A**  
- Flooding depth for **Scenario B**

### Map Visualization
Nodes are displayed as circles with:

- **Red:** Scenario A is worse (deeper flooding)  
- **Blue:** Scenario B is worse  
- **Gray:** No meaningful change  

Circle **size** indicates the magnitude of the flooding difference.

This allows users to clearly see:

- Where LIDs reduce flooding  
- Which areas see improvements or regressions based on LID placement 
