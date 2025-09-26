CoastWise is a browser app that runs real SWMM simulations (via PySWMM) so people can test coastal flood strategies quickly. You pick a storm, align it with high/low tide, turn a tide gate on/off, and add LID (rain gardens, rain barrels). The app shows flooded nodes, subcatchment runoff, and watershed totals (flooding, infiltration, runoff). It exports a simple Excel summary. No desktop GIS or SWMM UI required.

**What it does**
  Builds a 48-hour rainfall + tide timeline for your scenario.
  Writes those inputs into a SWMM template file.
  Runs a small set of scenarios (baseline, ± tide gate, ± LID, ± +20% rain).
  Maps where water piles up and how much flows or infiltrates.
  Gives you a one-click Excel with inputs and results.

**How it works**
  UI: Streamlit.
  Engine: SWMM via PySWMM.
  Model file: one SWMM template that has placeholders for rainfall, tides, LID, and tide-gate control.
  Data: basic watershed shapefiles (subcatchments and nodes) and a lightweight table for LID limits.

**Adapting to your watershed** 
1) Bring your SWMM model
  Start with your watershed as a standard SWMM INP.
  Replace IDs to something clean and stable if needed.
  Add four placeholders anywhere appropriate in the INP:

      $RAINFALL_TIMESERIES$
   
      $TIDE_TIMESERIES$

      $LID_USAGE$

      $TIDE_GATE_CONTROL$
   
  Save this as your template INP.

3) Provide minimal spatial layers
  Subcatchments polygon layer with a field that matches your INP subcatchment IDs.
  Nodes point layer with IDs that match your INP node IDs.
  (Optional) Conduits for context.

4) Define your design knobs
  Rainfall: supply return-period depths for a few durations (e.g., 2–24 h). Use your local IDF/Atlas values.
  Tide boundary: choose either a live gage you trust or a simple semidiurnal curve with reasonable min/max levels.
  LID rules: simple table per subcatchment with “max rain gardens” and “max rain barrels,” based on your land cover assumptions.

5) Check one end-to-end run
  Pick one storm and run baseline.
  Add a small LID count in 1–2 subcatchments and confirm outputs change as expected.
  Toggle tide gate and confirm the outlet boundary condition reacts.

6) Scale up
  Turn on the full scenario set (baseline, +tide gate, custom/max LID, and optional +20% rainfall sensitivity).
  Export Excel and spot-check totals and maps.

**What you need to change**
  IDF/return-period table → your city/region.
  Tide source or ranges → your coastline/estuary (or set fixed synthetic values).
  LID limits → your sizing rules and land cover (simple per-subcatchment caps are enough).
  Shapefile IDs → must match the INP IDs.
  Template INP → must include the four placeholders above.
