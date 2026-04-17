# bit o f a jankkkkk attempt at 1D Regen Cooling Simulator

## Setup
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Run
python main.py

## Run APP to run through sims quickly
python app.py


## Required Inputs
This solver serves as a combined thermal & stress analysis. Requires input geometry, flow rates and performance characteristics from RPA

Inputs:
- RPA radius contour (can import direct .txt file from RPA)
- Chamber pressure Pc
- c*
- Basic Cooling channel geometry (a, H)
- Wall thickness 
- Number of channels
- Coolant MFR (fuel)
- Mixture ratio MF (O/F)
- Ox name
- Fuel name

Optional (currently wip):
- Film cooling & associated params
- Helix angle for cooling channels
