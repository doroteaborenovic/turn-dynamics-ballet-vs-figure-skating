# Comparative Biomechanics of Turn Execution in Ballet and Figure Skating

This repository contains a research-oriented motion-analysis pipeline for comparing rotational movement in classical ballet and figure skating. The code is built around video-based pose estimation, kinematic filtering, and biomechanical evaluation of turn dynamics.

The project is not a packaged application; it is a scientific analysis workflow executed directly from Python scripts. The main objective is to quantify how athletes differ in movement execution, balance, angular velocity, and stability during turning actions.

## What the code actually does

The project performs the following stages:

1. Reads video input from local folders.
2. Extracts 2D/3D pose landmarks using MediaPipe Pose.
3. Applies ROI tracking and adaptive smoothing to reduce noise and jitter.
4. Computes body orientation, angular displacement, angular velocity, and center-of-mass-related quantities.
5. Evaluates biomechanical indicators such as stability, inertia, torque, energy, and inverted-pendulum balance behavior.
6. Produces plots and CSV tables that compare ballet with figure skating performance metrics.

## Main scripts in this repository

- `coord_extraction.py` — video processing and pose extraction pipeline; applies MediaPipe tracking, ROI cropping, and smoothing filters.
- `koordinateai.py` — coordinate processing and analysis of extracted motion data.
- `coords.py` — helper functions for coordinate processing and utility logic.
- `rezultatizarad.py` — main analysis script that generates the comparative kinematic and dynamic results.
- `modelobrnutogklatna.py` — inverted-pendulum / balance model used to evaluate stability and tipping behavior.
- `prikazivanjevidea.py` — support script for video preview and visualization.

## Repository structure

```text
projekat.fizika/
├── README.md
├── .gitignore
├── bitnikodovi.zip
├── coords.py
├── koordinateai.py
├── modelobrnutogklatna.py
├── prikazivanjevidea.py
├── rezultatizarad.py
├── rezultati_obrnuto_klatno/
├── kinematika_videi/
├── konacnirezultati/
├── konacni_videi/
├── venv/
└── .git/
```

## Input and output folders

### Input folders

- `kinematika_videi/` — raw video recordings
- `kinematika_rezultati_v2/` — processed motion results and extracted kinematic data
- `konacne_koordinate/` — final coordinate datasets used for analysis
- `konacni_videi/` — processed or final video outputs

### Output folders

- `konacnirezultati/` — main result directory for plots and tables
- `rezultati_obrnuto_klatno/` — inverted-pendulum / stability comparison outputs

## Analysis focus

The scripts are designed to evaluate:

- angular displacement and angular velocity
- body orientation and turning behavior
- center-of-mass motion and balance margin
- inertia and rotational dynamics
- torque and momentum trends
- kinetic energy and stability comparisons
- differences between ballet and skating turn mechanics

## Requirements

This project depends on scientific and computer-vision libraries:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas scipy matplotlib opencv-python mediapipe
```

The project is currently built for a Windows local workflow and uses Windows-specific path handling in several scripts.

## How to run the project

From the project root:

```powershell
python .\koordianteai.py
python .\rezultatizarad.py
```

If you want to run the balance model separately:

```powershell
python .\modelobrnutogklatna.py
```

## Typical workflow

1. Put video files into the project video folders.
2. Run `coord_extraction.py` to detect pose landmarks and export coordinates.
3. Run `rezultatizarad.py` to compute kinematic and dynamic metrics.
4. Inspect the generated plots and tables in `konacnirezultati/`.
5. Use the inverted-pendulum model outputs for additional stability interpretation.

## Scope of the project

This repository is a biomechanics and motion-analysis project focused on comparing turn execution in ballet and figure skating. It combines pose estimation, signal filtering, numerical modeling, and result visualization into a single research pipeline for investigating rotational mechanics and balance during turning movements.

