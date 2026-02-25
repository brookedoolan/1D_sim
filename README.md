# 1D Regen Cooling Simulator

## Setup

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Run
python main.py


## Required Inputs
This solver serves as a combined thermal & stress analysis. Requires input geometry, flow rates and performance characteristics from RPA (eventually 'stage 1' optimisation).

Inputs:
- RPA radius contour (can import direct .txt file from RPA)
- Chamber pressure Pc
- c*
- Cooling channel geometry (a, H)
- Wall thickness 
- Number of channels
- Coolant MFR (fuel)
- Mixture ratio MF (O/F)
- Ox name
- Fuel name