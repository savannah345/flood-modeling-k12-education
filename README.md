CoastWise performs fully dynamic rainfall–runoff and tidal simulations using the Stormwater Management Model (SWMM) in real time. By incorporating live tide data, the platform enables users to explore compound flooding scenarios, add low-impact development (LID) features interactively, and receive immediate feedback on runoff and infiltration outcomes. 

Built on a real urban watershed in Norfolk, Virginia, CoastWise demonstrates how rainfall–tide–infrastructure interactions can be visualized and simplified for education and decision support. The framework bridges critical gaps in serious gaming by integrating hydrologic modeling, coastal dynamics, and infrastructure planning within a freely accessible web environment. 

Adaptable to other watersheds through minimal input changes, CoastWise provides a practical tool for classrooms, local governments, and community planners to promote flood literacy and support data-informed resilience planning.

**What it does**
  Builds a 48-hour rainfall + tide timeline for your scenario.
  Writes those inputs into a SWMM template file.
  Runs a small set of scenarios (baseline, ± tide gate, ± LID, ± +20% rain).
  Maps subcatchment runoff and flooded stormwater junction nodes. 
  Shows watershed comparison of infiltraiton, surface runoff, and flooding
  Gives you an Excel with inputs and results for multiple run comparisons.

**How it works**
  User Interface: Streamlit.
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
