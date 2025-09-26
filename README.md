CoastWise is a browser app that runs real SWMM simulations (via PySWMM) so people can test coastal resilience strategies quickly. You pick a storm, align it with high/low tide, turn a tide gate on/off, and add LID (rain gardens, rain barrels). The app shows flooded nodes, subcatchment runoff, and watershed totals (flooding, infiltration, runoff). It exports a simple Excel summary. No desktop GIS or SWMM user interface (UI) required.

**What it does**
  Builds a 48-hour rainfall + tide timeline for your scenario.
  Writes those inputs into a SWMM template file.
  Runs a small set of scenarios (baseline, ± tide gate, ± LID, ± +20% rain).
  Maps subcatchment runoff and flooded stormwater junction nodes. 
  Shows watershed comparison of infiltraiton, surface runoff, and flooding
  Gives you an Excel with inputs and results for multiple run comparisons.

**How it works**
  UI: Streamlit.
  Engine: SWMM via PySWMM.
  Model file: one SWMM template that has placeholders for rainfall, tides, LID, and tide-gate control.
  Data: basic watershed shapefiles (subcatchments and nodes) and a table for LID upper bounds (per subcatchment).

**Adapting to your watershed** 
  Start with your watershed as a standard SWMM INP.
  Add four placeholders anywhere appropriate in the INP:

      RAINFALL_TIMESERIES; 
      TIDE_TIMESERIES; 
      LID_USAGE; 
      TIDE_GATE_CONTROL
   
  Save this as your template INP. To check your placement download the SWMM file from this GitHub repository.  

3) Provide minimal spatial layers
  Subcatchments polygon layer with a field that matches your INP subcatchment IDs.
  Nodes point layer with IDs that match your INP node IDs.
  Conduits for context (Optional).

4) Define your design storms, tides, and LIDs
  Rainfall: supply return-period depths for a few durations (e.g., 2–24 h). Use your local IDF/Atlas values into the rainfall_and_tide_generator.py file.
  Tide boundary: choose either a live gage you trust or a simple semidiurnal curve with reasonable min/max levels in the rainfall_and_tide_generator.py file.
  LID rules: simple table per subcatchment with “max rain gardens” and “max rain barrels,” based on your land cover assumptions (create your own Excel file that matches the one located in this repository).
