# 🌊 CoastWise: A Gamified Coastal Flood Simulation Tool

**CoastWise** is an interactive Streamlit application that simulates coastal flooding under different rainfall and tidal scenarios using the **EPA Storm Water Management Model (SWMM)**. Users can explore how **Low Impact Development (LID)** practices—like rain gardens and rain barrels—alongside **tide gates**, affect flooding and system performance across six different scenarios.

## 📁 Project Structure

Make sure your project directory includes the following:

```
📂 your_project_directory/
├── streamlit_app.py                # Main Streamlit app
├── rainfall_and_tide_generator.py  # Rainfall & tide generation functions
├── swmm_project.inp                # Base SWMM template with placeholders
├── raster_cells_per_sub.xlsx       # Subcatchment-level LID eligibility
├── requirements.txt                # List of required Python packages
├── [optional images]               # PNGs for land cover, DEM, LID visuals
├── [optional videos]               # MP4 for tide explanation (e.g., NASA)
```

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/savannah345/flood-modeling-k12-education.git
cd flood-modeling-k12-education
```

### 2. Install dependencies

Use the included `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run streamlit_app.py
```

## 🧰 Scenario Setup

Within the app, users can:

* Select storm **duration**, **return period**, and **rainfall shape**
* Choose **moon phase** and **tide alignment** (peak rain vs. tide)
* View generated rainfall and tide curves
* Run simulations with:

  * No infrastructure (baseline)
  * Tide gate only
  * Custom LID placement
  * Tide gate + Custom LID placement
  * Max LID across all eligible areas without Tide Gate
  * Max LID across all eligible areas with Tide Gate

## 📦 Outputs

* Culvert fill (%) time series for six scenarios
* Subcatchment runoff (impervious and pervious)
* Total infiltration, runoff, flooding, and outflow
* Cost estimates for selected LID + tide gate strategies
* Downloadable Excel workbook:

  * Scenario summary
  * Rainfall and tide curves
  * Water balance volumes (ft³ or m³)
  * Culvert performance

## 📌 Notes for Users

* **You must use the provided `swmm_project.inp`** template. The app fills in time series, LID usage, and gate logic automatically.
* Only **four files are needed to start**:

  1. `streamlit_app.py`
  2. `rainfall_and_tide_generator.py`
  3. `swmm_project.inp`
  4. `raster_cells_per_sub.xlsx`

All other output files (`.inp`, `.rpt`, `.out`) are created dynamically during simulation runs.

## 🧪 Educational Goals

This tool is designed for students, planners, and engineers to:

* Understand how rainfall and tides interact in coastal flooding
* Explore spatial and structural trade-offs in green infrastructure
* Evaluate costs and benefits of upstream vs. downstream interventions

## 👩‍🔬 License & Attribution

* Built on top of the **EPA SWMM** engine (via `pyswmm`)
* Developed for education and outreach in coastal resilience planning
* Demo data includes Norfolk, VA watershed as a case study
