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

The repository contains several Python scripts for processing motion data and generating results:

- `rezultatizarad.py` — main analysis pipeline for producing plots and summary tables
- `koordinateai.py` — coordinate-related analysis and processing
- `koordinateana.py` — additional analytical work on extracted motion data
- `coords.py` — coordinate utilities and helper functions
- `modelobrnutogklatna.py` — simplified inverted-pendulum/dynamic model analysis
- `modelvsosoba.py` — comparison between model-based predictions and actual motion
- `momentsile.py` — torque and moment calculations
- `konacnekoordinate.py` — final coordinate preparation and output generation

## Repository structure

```text
projekat.fizika/
├── README.md
├── .gitignore
├── venv/
├── coords.py
├── konacnekoordinate.py
├── koordinateai.py
├── koordinateana.py
├── modelobrnutogklatna.py
├── modelvsosoba.py
├── momentsile.py
├── rezultatizarad.py
├── kinematika_rezultati_v2/
├── kinematika_videi/
├── koordinate_sredjene/
├── obradjene_koordinate/
├── popravljene_koordinate/
├── popravljene_koordinate_deleva/
├── skracene_koordinate/
├── konacne_koordinate/
├── konacnirezultati/
├── rezultati_lott_laws_komparacija/
├── rezultati_prosirena_analiza_grafici/
├── rezultati_prosireni_model_konstantna_brzina/
├── renderovani_videi/
├── konacni_videi/
├── bitnikodovi.zip
└── .git/
```

## Data and outputs

### Input data folders

These folders contain coordinate data and intermediate pipeline outputs:

- `kinematika_rezultati_v2/`
- `koordinate_sredjene/`
- `obradjene_koordinate/`
- `popravljene_koordinate/`
- `konacne_koordinate/`
- `skracene_koordinate/`

### Output folders

Results are exported to:

- `konacnirezultati/`

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

You can also run some of the supporting scripts separately:

```powershell
python .\momentsile.py
python .\modelvsosoba.py
python .\modelobrnutogklatna.py
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

## GitHub workflow

To publish this project on GitHub, use the following commands:

```bash
git status
git add .
git commit -m "Initial project import"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repository-name>.git
git push -u origin main
```

If the repository is already connected to GitHub, just use:

```bash
git push origin main
```

If the remote branch has newer commits, resolve the rebase/merge first and then push again.

## Summary

This repository is a biomechanical and motion-analysis project focused on comparing turns in classical ballet and figure skating. It combines coordinate processing, numerical modeling, and visualization to quantify differences in rotational behavior and dynamic stability between the two disciplines.


### 7. Future updates

After changes are made:

```bash
git add .
git commit -m "Update analysis pipeline and results"
git push
```

## Recommended GitHub repository naming

Good names for this project include:

- `biomechanics-ballet-skating-analysis`
- `dancer-skater-balance-analysis`
- `kinematics-balance-torque-project`
- `athlete-biomechanics-analysis`

## License

This project currently does not include an explicit license file. If you plan to publish it publicly, it is recommended to add one such as MIT or Apache-2.0.

## Suggested next steps

- add a `requirements.txt` file for easier setup
- add a `data/` folder with a clear description of input formats
- split code into modules for better maintainability
- add a `LICENSE` file
- document the coordinate CSV schema used by each script
- clean up duplicate or legacy scripts if they are no longer needed

## Summary

This repository contains a computational biomechanics research workflow for analyzing human movement in ballet and figure skating. It combines scientific signal processing, body-segment mechanics, and result plotting into a single local analysis environment suitable for experimentation, reporting, and research documentation.
