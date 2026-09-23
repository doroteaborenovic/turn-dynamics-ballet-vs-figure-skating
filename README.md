# Comparative Physics of Turns in Classical Ballet and Figure Skating

This project presents a comparative analysis of the physical and biomechanical principles governing rotational movements in classical ballet and figure skating.

The main objective is to investigate how physical laws and biomechanical factors influence the execution, rotation, balance, and stability of athletes during turns. Ballet dancers and figure skaters are analyzed using video-based motion analysis and biomechanical models, allowing their movement characteristics to be quantified and compared.

The analysis includes parameters such as angular displacement, angular velocity, center of mass (CoM), inclination angle, effective pendulum length, extrapolated center of mass (XCoM), and margin of stability (MoS). The project combines principles of rotational dynamics, mechanics, and human biomechanics to evaluate differences between the two disciplines.

Motion data are extracted from video recordings using computer vision and pose estimation techniques. Anthropometric parameters are incorporated into the biomechanical model to obtain more realistic estimates of the athletes' center of mass and stability characteristics.

The goal is to determine how the different physical environments and techniques of classical ballet and figure skating affect rotational motion and dynamic balance, and to examine whether the same physical principles can be used to describe both types of movement.

## Project goals

- compare turn dynamics between ballet and figure skating
- analyze angular velocity and rotational stability
- estimate center-of-mass motion and balance margins
- evaluate how inclination and body configuration affect movement
- connect motion analysis with biomechanical modeling

## Main scripts

The repository currently contains the following Python scripts relevant to the project:

- `rezultatizarad.py` — main analysis pipeline for producing plots and summary tables
- `koordinateai.py` — coordinate-related analysis and processing
- `coords.py` — coordinate utilities and helper functions
- `coord_extraction.py` — extraction and preparation of motion coordinates
- `modelobrnutogklatna.py` — simplified inverted-pendulum/dynamic model analysis
- `prikazivanjevidea.py` — video preview/visualization support

## Repository structure

```text
projekat.fizika/
├── README.md
├── .gitignore
├── bitnikodovi.zip
├── coords.py
├── coord_extraction.py
├── koordinateai.py
├── modelobrnutogklatna.py
├── prikazivanjevidea.py
├── rezultatizarad.py
├── rezultati_obrnuto_klatno/
├── kinematika_rezultati_v2/
├── kinematika_videi/
├── konacne_koordinate/
├── konacnirezultati/
├── konacni_videi/
└── venv/
```

## Data and outputs

### Input data folders

These folders currently contain coordinate data and intermediate project outputs:

- `kinematika_rezultati_v2/`
- `kinematika_videi/`
- `konacne_koordinate/`
- `konacni_videi/`

### Output folders

Results are exported to:

- `konacnirezultati/`
- `rezultati_obrnuto_klatno/`

This directory contains generated plots and summary tables for:

- balance and stability analysis
- inertia comparisons
- angular velocity and acceleration
- momentum and torque evaluation
- center-of-mass trajectories

## Requirements

The project requires Python and scientific libraries.

### Recommended setup

```powershell
cd C:\Users\PC\gitara\projekat.fizika
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas scipy matplotlib
```

If the environment already exists, activate it with:

```powershell
.\venv\Scripts\Activate.ps1
```

## Running the project

From the repository root:

```powershell
python .\rezultatizarad.py
```

Additional supporting scripts may be run separately when needed:

```powershell
python .\coord_extraction.py
python .\modelobrnutogklatna.py
python .\prikazivanjevidea.py
```

## Research workflow

The project typically follows this path:

1. load motion data from CSV files
2. clean and interpolate noisy signals
3. estimate body orientation and center-of-mass motion
4. calculate angular and dynamic variables
5. compare ballet and skating movement mechanics
6. save figures and tables for reporting

## Notes

- This project is research-oriented and stores many generated files in the repository.
- Some folders contain both intermediate data and final output.
- The scripts assume execution from the project root.
- Some files include Windows-specific path handling for local analysis.

## Summary

This repository is a biomechanical and motion-analysis project focused on comparing turns in classical ballet and figure skating. It combines coordinate processing, numerical modeling, and visualization to quantify differences in rotational behavior and dynamic stability between the two disciplines.

