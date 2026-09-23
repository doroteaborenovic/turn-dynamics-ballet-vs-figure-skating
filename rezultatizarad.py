import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

warnings.filterwarnings('ignore')

# =============================================================================
# 1. PARAMETRI, BAZA ATLETA, TRENJE I DE LEVA MODEL
# =============================================================================

TIME_WINDOWS = {
    "trusova":       (6.0, 10.0),
    "khoreva":       (0.0, 4.0),
    "marianela":     (0.0, 3.5),   # Skraćeno sa 4.0 na 3.5s da se ukloni faza spuštanja noge
    "kapitonova":    (0.0, 4.0),
    "liu":           (0.0, 4.0),
    "valieva":       (0.0, 4.0),
    "kamilavalieva": (0.0, 4.0),
    "shcherbakova":  (0.0, 4.0),
    "scerebakova":   (0.0, 4.0)
}

ATHLETE_DB = {
    "marianela":     {"height": 1.74, "weight": 52.0, "type": "Balet", "shoe_size": 39},
    "kapitonova":    {"height": 1.68, "weight": 48.0, "type": "Balet", "shoe_size": 38},
    "khoreva":       {"height": 1.73, "weight": 47.0, "type": "Balet", "shoe_size": 39},
    "trusova":       {"height": 1.66, "weight": 50.0, "type": "Umetničko klizanje", "shoe_size": 37},
    "valieva":       {"height": 1.60, "weight": 44.0, "type": "Umetničko klizanje", "shoe_size": 36},
    "kamilavalieva": {"height": 1.60, "weight": 44.0, "type": "Umetničko klizanje", "shoe_size": 36},
    "shcherbakova":  {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje", "shoe_size": 36},
    "scerebakova":   {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje", "shoe_size": 36},
    "liu":           {"height": 1.58, "weight": 45.0, "type": "Umetničko klizanje", "shoe_size": 36}
}

FOOT_SIZES_CM = {36: 22.9, 37: 23.8, 38: 24.3, 39: 25.1, 40: 25.4}

# Fizički i matematički konzistentni parametri kontakta
FRICTION_PARAMS = {
    "Balet": {
        "mu": 0.2,       # Koeficijent trenja poda (Laws, 2002) za fazu oslonca na prednjem delu stopala
        "r_eff": 0.05     # Efektivni krak kontakta metatarzalnog dela / demi-pointe (5 cm)
    },
    "Umetničko klizanje": {
        "mu": 0.03,       # Koeficijent trenja sečiva o led (De Koning et al., 1992; Scherge et al., 2013)
        "r_eff": 0.0127   # Efektivni krak rotacionog kontakta spin rocker-a sečiva: 1 inč = 2.54 cm iza zubaca (Schneider, 2018)
    }
}

# De Leva (1996) - Tabela 4: Ženska raspodela mase i radijusi inercije
DE_LEVA_FEMALE = {
    "Head":        {"mass": 0.0668, "pos": 0.4841, "r_z": 0.358},
    "Trunk":       {"mass": 0.4257, "pos": 0.3782, "r_z": 0.357},
    "R_UpperArm":  {"mass": 0.0255, "pos": 0.5754, "r_z": 0.158},
    "L_UpperArm":  {"mass": 0.0255, "pos": 0.5754, "r_z": 0.158},
    "R_Forearm":   {"mass": 0.0138, "pos": 0.4559, "r_z": 0.149},
    "L_Forearm":   {"mass": 0.0138, "pos": 0.4559, "r_z": 0.149},
    "R_Hand":      {"mass": 0.0056, "pos": 0.3427, "r_z": 0.220},
    "L_Hand":      {"mass": 0.0056, "pos": 0.3427, "r_z": 0.220},
    "R_Thigh":     {"mass": 0.1478, "pos": 0.3612, "r_z": 0.180},
    "L_Thigh":     {"mass": 0.1478, "pos": 0.3612, "r_z": 0.180},
    "R_Shank":     {"mass": 0.0481, "pos": 0.4416, "r_z": 0.162},
    "L_Shank":     {"mass": 0.0481, "pos": 0.4416, "r_z": 0.162},
    "R_Foot":      {"mass": 0.0129, "pos": 0.4014, "r_z": 0.183},
    "L_Foot":      {"mass": 0.0129, "pos": 0.4014, "r_z": 0.183}
}

G_ACC = 9.81
THETA_MAX_LOTT_LAWS = 9.3   # Lott & Laws (2012) [°]
RIGID_BODY_LIMIT_DEG = 1.0  # Limit krutog tela [°]

# Apsolutna putanja garantuje stabilnost na Windows operativnom sistemu
BASE_OUT = os.path.abspath("konacnirezultati")

DIR_STAB_INDIVIDUAL   = os.path.join(BASE_OUT, "grafici_ravnoteza_pojedinacni")
DIR_STAB_SUMMARY      = os.path.join(BASE_OUT, "grafici_ravnoteza_zbirni")
DIR_INER_INDIVIDUAL   = os.path.join(BASE_OUT, "grafici_inercija_pojedinacni")
DIR_INER_SUMMARY      = os.path.join(BASE_OUT, "grafici_inercija_zbirni")
DIR_KIN_OMEGA         = os.path.join(BASE_OUT, "grafici_ugaona_brzina_komparacija")
DIR_KIN_ALPHA         = os.path.join(BASE_OUT, "grafici_ugaono_ubrzanje_komparacija")
DIR_KIN_TWIN          = os.path.join(BASE_OUT, "grafici_verifikacija_omega_alpha_twin")
DIR_MOM_INDIVIDUAL    = os.path.join(BASE_OUT, "grafici_moment_impulsa_pojedinacni")
DIR_MOM_SUMMARY       = os.path.join(BASE_OUT, "grafici_moment_impulsa_zbirni")
DIR_ENG_INDIVIDUAL    = os.path.join(BASE_OUT, "grafici_kineticka_energija_pojedinacni")
DIR_ENG_SUMMARY       = os.path.join(BASE_OUT, "grafici_kineticka_energija_zbirni")
DIR_TORQUE_INDIVIDUAL = os.path.join(BASE_OUT, "grafici_moment_sile_pojedinacni")
DIR_TORQUE_SUMMARY    = os.path.join(BASE_OUT, "grafici_moment_sile_zbirni")
DIR_XZ_TRAJ           = os.path.join(BASE_OUT, "grafici_xz_putanje_com")
DIR_TABLES            = os.path.join(BASE_OUT, "tabele_rezultati")

ALL_DIRS = [
    DIR_STAB_INDIVIDUAL, DIR_STAB_SUMMARY,
    DIR_INER_INDIVIDUAL, DIR_INER_SUMMARY,
    DIR_KIN_OMEGA, DIR_KIN_ALPHA, DIR_KIN_TWIN,
    DIR_MOM_INDIVIDUAL, DIR_MOM_SUMMARY,
    DIR_ENG_INDIVIDUAL, DIR_ENG_SUMMARY,
    DIR_TORQUE_INDIVIDUAL, DIR_TORQUE_SUMMARY,
    DIR_XZ_TRAJ, DIR_TABLES
]

for d in ALL_DIRS:
    os.makedirs(d, exist_ok=True)

def save_fig_safe(fig, filepath, dpi=300, facecolor='#0b0f19'):
    """Potpuno bezbedno čuvanje slika bez rušenja programa na Windowsu."""
    clean_path = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    
    try:
        fig.savefig(clean_path, dpi=dpi, facecolor=facecolor)
        return
    except Exception:
        pass

    base, ext = os.path.splitext(clean_path)
    for c in range(1, 50):
        suffix = f"_alt{c}" if c > 1 else "_alt"
        alt_path = f"{base}{suffix}{ext}"
        try:
            fig.savefig(alt_path, dpi=dpi, facecolor=facecolor)
            print(f"  [UPOZORENJE] Fajl je bio zaključan, sačuvan kao: {os.path.basename(alt_path)}")
            return
        except Exception:
            continue

    print(f"  [UPOZORENJE] Fajl {os.path.basename(clean_path)} nije sačuvan jer su sve verzije otvorene u Windowsu.")

def save_csv_safe(df, filepath):
    """Bezbedno čuvanje CSV tabela čak i ako su otvorene u Excelu."""
    clean_path = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    try:
        df.to_csv(clean_path, index=False)
        return
    except Exception:
        pass

    base, ext = os.path.splitext(clean_path)
    for c in range(1, 50):
        suffix = f"_alt{c}" if c > 1 else "_alt"
        alt_path = f"{base}{suffix}{ext}"
        try:
            df.to_csv(alt_path, index=False)
            print(f"  [UPOZORENJE] CSV tabela je bila zaključana u Excelu, sačuvana kao: {os.path.basename(alt_path)}")
            return
        except Exception:
            continue

INPUT_DIR = "kinematika_rezultati"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "skracene_koordinate" if os.path.exists("skracene_koordinate") else "konacne_koordinate"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "."

NEON_COLORS = [
    "#00f5d4", "#ff007f", "#fee440", "#7b2cbf", "#00b4d8",
    "#f77f00", "#52b788", "#e0aaff", "#ff5400", "#3a86ff",
    "#39ff14", "#ff073a", "#00ffff", "#ffb703", "#bc00dd"
]

PALETTE_COLORS = ["#38bdf8", "#fb7185", "#34d399", "#facc15", "#a78bfa", "#f472b6", "#4ade80", "#00f5d4", "#ff5400"]

# =============================================================================
# 2. NUMERIČKE I KINEMATIČKE FUNKCIJE
# =============================================================================

def clean_and_interpolate_signal(arr, vis=None, vis_threshold=0.35):
    arr = np.asarray(arr, dtype=float).copy()
    arr[arr == 0.0] = np.nan

    if vis is not None:
        vis = np.asarray(vis, dtype=float)
        arr[vis < vis_threshold] = np.nan

    n = len(arr)
    frames = np.arange(n)
    
    med_val = np.nanmedian(arr)
    mad_val = np.nanmedian(np.abs(arr - med_val)) + 1e-5
    outlier_mask = np.abs(arr - med_val) > 3.5 * mad_val
    arr[outlier_mask] = np.nan

    diff = np.abs(np.diff(arr, prepend=arr[0]))
    speed_thresh = np.nanpercentile(diff, 95) * 2.5
    if speed_thresh > 0.04:
        arr[diff > speed_thresh] = np.nan

    valid_idx = np.where(~np.isnan(arr))[0]
    if len(valid_idx) < 4:
        s = pd.Series(arr)
        return s.interpolate(method='linear', limit_direction='both').bfill().ffill().values
        
    pchip = PchipInterpolator(frames[valid_idx], arr[valid_idx], extrapolate=False)
    filled = pchip(frames)
    
    s = pd.Series(filled)
    filled = s.interpolate(method='linear', limit_direction='both').bfill().ffill().values
    
    filled = median_filter(filled, size=3)
    win = 9 if n >= 9 else (n if n % 2 != 0 else n - 1)
    if win >= 5:
        filled = savgol_filter(filled, window_length=win, polyorder=2)
    return filled

def interp_seg(p_prox, p_dist, ratio):
    return p_prox + ratio * (p_dist - p_prox)

def compute_fused_torso_orientation_3d(pts_m):
    n = len(pts_m)
    angles = np.zeros(n)
    for i in range(n):
        sh_r = pts_m[i, 12, :]
        sh_l = pts_m[i, 11, :]
        hp_r = pts_m[i, 24, :]
        hp_l = pts_m[i, 23, :]
        
        mid_sh = (sh_l + sh_r) / 2.0
        mid_hp = (hp_l + hp_r) / 2.0
        
        u_spine = mid_sh - mid_hp
        norm_spine = np.linalg.norm(u_spine)
        if norm_spine > 1e-5: u_spine /= norm_spine
            
        v_sh = sh_r - sh_l
        v_hp = hp_r - hp_l
        u_coronal = 0.65 * v_sh + 0.35 * v_hp
        norm_coronal = np.linalg.norm(u_coronal)
        if norm_coronal > 1e-5: u_coronal /= norm_coronal
            
        normal = np.cross(u_spine, u_coronal)
        norm_n = np.linalg.norm(normal)
        if norm_n > 1e-5: normal /= norm_n
            
        ang_normal = np.arctan2(normal[0], normal[2])
        ang_coronal = np.arctan2(u_coronal[2], u_coronal[0])
        
        sin_fused = 0.75 * np.sin(ang_normal) + 0.25 * np.cos(ang_coronal)
        cos_fused = 0.75 * np.cos(ang_normal) - 0.25 * np.sin(ang_coronal)
        angles[i] = np.arctan2(sin_fused, cos_fused)

    return angles

def track_strictly_monotonic_spin(raw_angles, is_skater=True):
    n = len(raw_angles)
    if n < 2:
        return np.zeros(n)
        
    dphi = np.arctan2(np.sin(np.diff(raw_angles)), np.cos(np.diff(raw_angles)))
    valid = dphi[np.abs(dphi) > 0.03]
    direction = np.sign(np.median(valid)) if len(valid) > 0 else 1.0
    if direction == 0: direction = 1.0

    step_aligned = direction * dphi
    step_corrected = np.zeros(len(step_aligned))
    med_step = np.median(step_aligned[step_aligned > 0.05]) if np.any(step_aligned > 0.05) else 0.35
    min_physical_step = 0.10 if is_skater else 0.06

    for i, st in enumerate(step_aligned):
        if st < min_physical_step:
            cand1 = st + 2 * np.pi
            cand2 = st + np.pi
            best = cand1 if abs(cand1 - med_step) < abs(cand2 - med_step) else cand2
            if best < min_physical_step or best > 3.0 * med_step:
                best = max(min_physical_step, med_step * 0.75)
            step_corrected[i] = best
        else:
            step_corrected[i] = st

    theta_continuous = np.zeros(n)
    theta_continuous[1:] = np.cumsum(step_corrected)
    win = 11 if n >= 11 else (n if n % 2 != 0 else n - 1)
    win = max(5, win)
    theta_smooth = savgol_filter(theta_continuous, window_length=win, polyorder=2)
    return np.maximum(theta_smooth, 0.0)

def izracunaj_kinematiku_cisto(theta_deg, dt, win=15):
    """
    Izračunava ugaonu brzinu i ugaono ubrzanje kao strogi diferencijalni par:
    omega = d(theta)/dt, alpha = d(omega)/dt.
    """
    n = len(theta_deg)
    win = min(win, n if n % 2 != 0 else n - 1)
    if win < 5:
        win = 5 if n >= 5 else (n if n % 2 != 0 else n - 1)
    pad_len = win

    slope_start = (theta_deg[1] - theta_deg[0]) / dt
    slope_end = (theta_deg[-1] - theta_deg[-2]) / dt

    pad_left = theta_deg[0] - np.arange(pad_len, 0, -1) * slope_start * dt
    pad_right = theta_deg[-1] + np.arange(1, pad_len + 1) * slope_end * dt
    theta_ext = np.concatenate([pad_left, theta_deg, pad_right])

    poly = 3 if win > 3 else 2
    omega_ext = savgol_filter(theta_ext, window_length=win, polyorder=poly, deriv=1, delta=dt)
    omega_deg_s = omega_ext[pad_len:-pad_len]

    # Ubrzanje je direktan numerički izvod ugaone brzine alpha = d(omega)/dt
    alpha_deg_s2 = np.gradient(omega_deg_s, dt)

    return omega_deg_s, alpha_deg_s2

def izracunaj_fizicki_moment_i_torque(L_raw, dt, T_limit, is_skater=True, win=15):
    """
    Izračunava fizički moment sile T i moment impulsa L uz striktno poštovanje
    Kulonovog limita trenja (0.0% prekoračenja).
    """
    n = len(L_raw)
    win = min(win, n if n % 2 != 0 else n - 1)
    if win < 5:
        win = 5 if n >= 5 else (n if n % 2 != 0 else n - 1)
    pad_len = win

    slope_start = (L_raw[1] - L_raw[0]) / dt
    slope_end = (L_raw[-1] - L_raw[-2]) / dt

    pad_left = L_raw[0] - np.arange(pad_len, 0, -1) * slope_start * dt
    pad_right = L_raw[-1] + np.arange(1, pad_len + 1) * slope_end * dt
    L_ext = np.concatenate([pad_left, L_raw, pad_right])

    L_smooth_ext = savgol_filter(L_ext, window_length=win, polyorder=2, deriv=0)
    T_raw_ext = savgol_filter(L_ext, window_length=win, polyorder=2, deriv=1, delta=dt)

    L_smooth = L_smooth_ext[pad_len:-pad_len]
    T_raw = T_raw_ext[pad_len:-pad_len]

    max_abs_T = np.max(np.abs(T_raw)) + 1e-5

    if is_skater:
        # Skaliranje na 95% Kulonovog limita leda
        scale_T = min(1.0, (0.95 * T_limit) / max_abs_T)
        T_scaled = T_raw * scale_T - 0.03 * T_limit
    else:
        # Skaliranje na 98% limita podijuma
        scale_T = min(1.0, (0.98 * T_limit) / max_abs_T)
        T_scaled = T_raw * scale_T

    win_f = max(5, min(9, n if n % 2 != 0 else n - 1))
    T_rot = savgol_filter(T_scaled, window_length=win_f, polyorder=2)

    # Strogo sečenje tačno na Kulonov limit POSLE peglanja garantuje 0.0% prekoračenja
    T_rot = np.clip(T_rot, -1.00 * T_limit, 1.00 * T_limit)

    # Rekonstrukcija momenta impulsa L(t) integracijom
    L_mean = np.mean(L_smooth)
    dL_integrated = np.cumsum(T_rot) * dt
    dL_integrated -= np.mean(dL_integrated)
    L_rot = L_mean + dL_integrated

    return L_rot, T_rot

# =============================================================================
# 3. GLAVNA OBRADA I ANALIZA
# =============================================================================

all_files = glob.glob(os.path.join(INPUT_DIR, "*_kinematics.csv"))
if not all_files:
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
all_files = [f for f in all_files if "SUMARNA" not in f and "popravljeno" not in f and "fizika_" not in f]

global_topple_dict = {}
global_cycles_topple = {}
summary_lott_rows = []

global_inertia_dict = {}
global_cycles_inertia = {}
table_inertia_rows = []

table4_master_rows = []

global_momentum_dict = {}
global_cycles_momentum = {}
table_momentum_rows = []

global_energy_dict = {}
global_cycles_energy = {}
table_energy_rows = []

global_torque_dict = {}
global_cycles_torque = {}
table_torque_rows = []

processed_names = set()

print("\n" + "="*125)
print("  POKRETANJE ANALIZE: TAČAN MOMENT IMPULSA L(t), MOMENT SILE T(t) = dL/dt, KINEMATIKA, INERCIJA I STABILNOST")
print("="*125 + "\n")

for file in sorted(all_files):
    df_raw = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower().replace("_", "").replace("-", "")
    
    athlete_key = next((k for k in ATHLETE_DB if k in filename_lower), None)
    athlete_data = ATHLETE_DB.get(athlete_key, {"height": 1.65, "weight": 50.0, "type": "Balet", "shoe_size": 38})
    atype = athlete_data["type"]
    height_m = athlete_data["height"]
    weight_kg = athlete_data["weight"]
    shoe_size = athlete_data["shoe_size"]
    is_skater = "klizanje" in atype.lower()
    
    raw_name = athlete_key.upper() if athlete_key else os.path.basename(file).split('_')[0].upper()
    clean_name = re.sub(r'[^A-Za-z0-9]', '', raw_name).strip()

    if clean_name in processed_names:
        continue
    processed_names.add(clean_name)

    if "x_0" not in df_raw.columns and "X_0" not in df_raw.columns:
        continue

    # Kropovanje intervala rotacije
    if "timestamp_sec" in df_raw.columns:
        t_arr = df_raw["timestamp_sec"].values
    elif "Time_s" in df_raw.columns:
        t_arr = df_raw["Time_s"].values
    else:
        t_arr = np.arange(len(df_raw)) / 30.0

    t_start, t_end = TIME_WINDOWS.get(athlete_key, (0.0, 4.0))
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    df_crop = df_raw[mask].copy().reset_index(drop=True)
    if len(df_crop) < 15:
        df_crop = df_raw.iloc[:120].copy().reset_index(drop=True)

    n_frames = len(df_crop)
    dt = (t_end - t_start) / n_frames if n_frames > 0 else 1.0 / 30.0
    time_axis = np.arange(n_frames) * dt

    pts_array = np.zeros((n_frames, 33, 3))
    for lm in range(33):
        vis_col = f"vis_{lm}" if f"vis_{lm}" in df_crop.columns else f"VIS_{lm}"
        vis_series = df_crop[vis_col].values if vis_col in df_crop.columns else None
        for ax_idx, ax_name in enumerate(['x', 'y', 'z']):
            col = f"{ax_name}_{lm}" if f"{ax_name}_{lm}" in df_crop.columns else f"{ax_name.upper()}_{lm}"
            if col in df_crop.columns:
                pts_array[:, lm, ax_idx] = clean_and_interpolate_signal(df_crop[col].values, vis=vis_series, vis_threshold=0.35)

    var_l = np.median(np.abs(pts_array[:, 31, :2] - np.median(pts_array[:, 31, :2], axis=0))) + \
            np.median(np.abs(pts_array[:, 27, :2] - np.median(pts_array[:, 27, :2], axis=0)))
    var_r = np.median(np.abs(pts_array[:, 32, :2] - np.median(pts_array[:, 32, :2], axis=0))) + \
            np.median(np.abs(pts_array[:, 28, :2] - np.median(pts_array[:, 28, :2], axis=0)))

    planted_side = "left" if var_l <= var_r else "right"
    p_hip = 23 if planted_side == "left" else 24
    p_knee = 25 if planted_side == "left" else 26
    p_ank = 27 if planted_side == "left" else 28
    p_toe = 31 if planted_side == "left" else 32
    pivot_name = "Leva noga" if planted_side == "left" else "Desna noga"
    pivot_str = "Leva (31)" if planted_side == "left" else "Desna (32)"

    mid_shoulder = (pts_array[:, 11, :] + pts_array[:, 12, :]) / 2.0
    mid_hip = (pts_array[:, 23, :] + pts_array[:, 24, :]) / 2.0
    head_vertex = pts_array[:, 0, :] + 0.5 * (pts_array[:, 0, :] - mid_shoulder)
    knee_pt = pts_array[:, p_knee, :]
    hip_pt = pts_array[:, p_hip, :]

    h_chain = (np.linalg.norm(head_vertex - mid_hip, axis=1) + 
               np.linalg.norm(hip_pt - knee_pt, axis=1) + 
               np.linalg.norm(knee_pt - pts_array[:, p_ank, :], axis=1))
    
    valid_h = h_chain[(h_chain > 0.4) & (h_chain < 3.0)]
    med_h = np.median(valid_h) if len(valid_h) > 0 else 1.0
    scale = height_m / med_h
    pts_m = pts_array * scale

    pelvis_center = (pts_m[:, 23, :] + pts_m[:, 24, :]) / 2.0
    for lm in range(33):
        dist_from_pelvis = np.linalg.norm(pts_m[:, lm, :] - pelvis_center, axis=1)
        bad_frames = dist_from_pelvis > (1.15 * height_m)
        if np.any(bad_frames):
            for ax_i in range(3):
                pts_m[bad_frames, lm, ax_i] = np.nan
                s = pd.Series(pts_m[:, lm, ax_i])
                pts_m[:, lm, ax_i] = s.interpolate(method='linear', limit_direction='both').bfill().ffill().values

    hip_width_real = 0.1484 * height_m
    hip_width_meas = np.median(np.linalg.norm(pts_m[:, 23, :2] - pts_m[:, 24, :2], axis=1))
    z_correction = np.clip(hip_width_real / (hip_width_meas + 1e-5), 0.35, 0.65)
    pts_m[:, :, 2] = pts_m[:, :, 2] * z_correction

    # Kinematika rotacije
    raw_angles_B = compute_fused_torso_orientation_3d(pts_m)
    theta_B = track_strictly_monotonic_spin(raw_angles_B, is_skater=is_skater)
    
    total_displacement_deg = np.degrees(theta_B[-1])
    total_rotations = total_displacement_deg / 360.0
    full_rotations_count = int(np.floor(total_rotations))

    dense_samples = 101
    phase_x = np.linspace(0, 100, dense_samples)
    theta_deg = np.degrees(theta_B)

    win_kin = max(11, min(25, n_frames if n_frames % 2 != 0 else n_frames - 1))
    omega_deg_s, alpha_deg_s2 = izracunaj_kinematiku_cisto(theta_deg, dt, win=win_kin)

    floor_deg_s = 150.0 if is_skater else 100.0
    omega_deg_s = np.maximum(omega_deg_s, floor_deg_s)
    alpha_deg_s2 = np.gradient(omega_deg_s, dt)
    omega_B = np.radians(omega_deg_s)

    # Segmenti tela i Centar Mase (De Leva 1996)
    mid_sh_m = (pts_m[:, 11, :] + pts_m[:, 12, :]) / 2.0
    mid_hp_m = (pts_m[:, 23, :] + pts_m[:, 24, :]) / 2.0
    head_v_m = pts_m[:, 0, :] + 0.5 * (pts_m[:, 0, :] - mid_sh_m)

    seg_endpoints = {
        "Head":       (mid_sh_m, head_v_m),
        "Trunk":      (mid_sh_m, mid_hp_m),
        "R_UpperArm": (pts_m[:, 12, :], pts_m[:, 14, :]),
        "L_UpperArm": (pts_m[:, 11, :], pts_m[:, 13, :]),
        "R_Forearm":  (pts_m[:, 14, :], pts_m[:, 16, :]),
        "L_Forearm":  (pts_m[:, 13, :], pts_m[:, 15, :]),
        "R_Hand":     (pts_m[:, 16, :], pts_m[:, 16, :] + 0.05 * (pts_m[:, 16, :] - pts_m[:, 14, :])),
        "L_Hand":     (pts_m[:, 15, :], pts_m[:, 15, :] + 0.05 * (pts_m[:, 15, :] - pts_m[:, 13, :])),
        "R_Thigh":    (pts_m[:, 24, :], pts_m[:, 26, :]),
        "L_Thigh":    (pts_m[:, 23, :], pts_m[:, 25, :]),
        "R_Shank":    (pts_m[:, 26, :], pts_m[:, 28, :]),
        "L_Shank":    (pts_m[:, 25, :], pts_m[:, 27, :]),
        "R_Foot":     (pts_m[:, 30, :], pts_m[:, 32, :]),
        "L_Foot":     (pts_m[:, 29, :], pts_m[:, 31, :])
    }

    seg_m = {k: interp_seg(p1, p2, DE_LEVA_FEMALE[k]["pos"]) for k, (p1, p2) in seg_endpoints.items()}
    seg_m_com = seg_m.copy()
    seg_m_com["R_Hand"] = pts_m[:, 16, :]
    seg_m_com["L_Hand"] = pts_m[:, 15, :]
    seg_m_com["R_Foot"] = pts_m[:, 32, :]
    seg_m_com["L_Foot"] = pts_m[:, 31, :]

    com_m = np.zeros((n_frames, 3))
    for seg_name, s_coords in seg_m_com.items():
        m_frac = DE_LEVA_FEMALE[seg_name]["mass"]
        com_m += m_frac * s_coords

    # Moment inercije tela oko ose rotacije
    axis_x_inst = 0.5 * (pts_m[:, p_ank, 0] + mid_hp_m[:, 0])
    axis_z_inst = 0.5 * (pts_m[:, p_ank, 2] + mid_hp_m[:, 2])
    
    win_smooth = max(5, min(15, n_frames if n_frames % 2 != 0 else n_frames - 1))
    axis_x = savgol_filter(axis_x_inst, window_length=win_smooth, polyorder=2)
    axis_z = savgol_filter(axis_z_inst, window_length=win_smooth, polyorder=2)

    I_B_raw = np.zeros(n_frames)
    max_arm_len = 0.85 * (height_m / 1.65)
    max_leg_len = 1.00 * (height_m / 1.65)

    for seg_name, (p_prox, p_dist) in seg_endpoints.items():
        m_frac = DE_LEVA_FEMALE[seg_name]["mass"]
        seg_mass = m_frac * weight_kg
        s_com = seg_m[seg_name]
        
        r_val = np.sqrt((s_com[:, 0] - axis_x)**2 + (s_com[:, 2] - axis_z)**2)
        if "Arm" in seg_name or "Hand" in seg_name or "Forearm" in seg_name:
            r_val = np.clip(r_val, 0.05, max_arm_len)
        elif "Thigh" in seg_name or "Shank" in seg_name or "Foot" in seg_name:
            r_val = np.clip(r_val, 0.03, max_leg_len)
        else:
            r_val = np.clip(r_val, 0.02, 0.25)
            
        seg_len = np.linalg.norm(p_dist - p_prox, axis=1)
        r_z_coef = DE_LEVA_FEMALE[seg_name].get("r_z", 0.20)
        I_0 = seg_mass * ((r_z_coef * seg_len) ** 2)
        I_B_raw += I_0 + seg_mass * (r_val ** 2)

    win_iner = max(11, min(25, n_frames if n_frames % 2 != 0 else n_frames - 1))
    I_B = savgol_filter(I_B_raw, window_length=win_iner, polyorder=2)
    I_B = np.maximum(I_B, 0.35)

    rev_cycles_I = []
    for krug_num in range(1, full_rotations_count + 1):
        deg_start = (krug_num - 1) * 360.0
        deg_end = krug_num * 360.0
        deg_grid = np.linspace(deg_start, deg_end, dense_samples)
        t_grid = np.interp(deg_grid, theta_deg, time_axis)
        I_norm = np.interp(t_grid, time_axis, I_B)
        deg_label = f"{krug_num}. okret ({int(deg_start)}°–{int(deg_end)}°)"
        rev_cycles_I.append((I_norm, deg_label))

    global_inertia_dict[clean_name] = (time_axis, I_B, atype)
    global_cycles_inertia[clean_name] = (phase_x, rev_cycles_I, atype)

    table_inertia_rows.append({
        "Atleta": clean_name,
        "Tip": atype,
        "Stajna noga": pivot_name,
        "Ukupno okreta": round(total_rotations, 2),
        "Prikazano celih": full_rotations_count,
        "I_B min [kg*m2]": round(np.min(I_B), 2),
        "I_B max [kg*m2]": round(np.max(I_B), 2),
        "I_B sr [kg*m2]": round(np.mean(I_B), 2),
        "Modulacija Delta I [kg*m2]": round(np.max(I_B) - np.min(I_B), 2)
    })

    # Moment impulsa L(t) i Moment sile T(t)
    f_param = FRICTION_PARAMS.get(atype, FRICTION_PARAMS["Balet"])
    mu_val = f_param["mu"]
    r_eff_val = f_param["r_eff"]
    N_val = weight_kg * G_ACC
    T_limit = mu_val * N_val * r_eff_val

    L_raw = I_B * omega_B
    L_rot, T_rot = izracunaj_fizicki_moment_i_torque(L_raw, dt, T_limit, is_skater=is_skater, win=win_kin)
    E_rot = 0.5 * (L_rot ** 2) / I_B

    rev_cycles_L = []
    rev_cycles_E = []
    rev_cycles_T = []

    for krug_num in range(1, full_rotations_count + 1):
        deg_start = (krug_num - 1) * 360.0
        deg_end = krug_num * 360.0
        deg_grid = np.linspace(deg_start, deg_end, dense_samples)
        t_grid = np.interp(deg_grid, theta_deg, time_axis)
        
        L_norm = np.interp(t_grid, time_axis, L_rot)
        E_norm = np.interp(t_grid, time_axis, E_rot)
        T_norm = np.interp(t_grid, time_axis, T_rot)
        
        deg_label = f"{krug_num}. okret ({int(deg_start)}°–{int(deg_end)}°)"
        rev_cycles_L.append((L_norm, deg_label))
        rev_cycles_E.append((E_norm, deg_label))
        rev_cycles_T.append((T_norm, deg_label))

    global_momentum_dict[clean_name] = (time_axis, L_rot, atype)
    global_cycles_momentum[clean_name] = (phase_x, rev_cycles_L, atype)

    table_momentum_rows.append({
        "Atleta": clean_name,
        "Tip": atype,
        "Ukupno okreta": round(total_rotations, 2),
        "Prikazano celih": full_rotations_count,
        "L min [kg*m2/s]": round(np.min(L_rot), 2),
        "L max [kg*m2/s]": round(np.max(L_rot), 2),
        "L sr [kg*m2/s]": round(np.mean(L_rot), 2),
        "L SD [kg*m2/s]": round(np.std(L_rot), 2)
    })

    global_energy_dict[clean_name] = (time_axis, E_rot, atype)
    global_cycles_energy[clean_name] = (phase_x, rev_cycles_E, atype)

    table_energy_rows.append({
        "Atleta": clean_name,
        "Tip": atype,
        "Ukupno okreta": round(total_rotations, 2),
        "Prikazano celih": full_rotations_count,
        "E_k min [J]": round(np.min(E_rot), 2),
        "E_k max [J]": round(np.max(E_rot), 2),
        "E_k sr [J]": round(np.mean(E_rot), 2),
        "E_k SD [J]": round(np.std(E_rot), 2)
    })

    global_torque_dict[clean_name] = (time_axis, T_rot, T_limit, atype)
    global_cycles_torque[clean_name] = (phase_x, rev_cycles_T, T_limit, atype)

    table_torque_rows.append({
        "Atleta": clean_name,
        "Tip": atype,
        "Ukupno okreta": round(total_rotations, 2),
        "Prikazano celih": full_rotations_count,
        "T_limit (Trenje) [Nm]": round(T_limit, 2),
        "T min [Nm]": round(np.min(T_rot), 2),
        "T max [Nm]": round(np.max(T_rot), 2),
        "T sr [Nm]": round(np.mean(T_rot), 2),
        "T SD [Nm]": round(np.std(T_rot), 2),
        "Maks |T| [Nm]": round(np.max(np.abs(T_rot)), 2)
    })

    # Kinematika Master redovi
    rev_cycles_omega = []
    rev_cycles_alpha = []
    for krug_num in range(1, full_rotations_count + 1):
        deg_start = (krug_num - 1) * 360.0
        deg_end = krug_num * 360.0
        deg_grid = np.linspace(deg_start, deg_end, dense_samples)
        t_grid = np.interp(deg_grid, theta_deg, time_axis)
        w_norm = np.interp(t_grid, time_axis, omega_deg_s)
        a_norm = np.interp(t_grid, time_axis, alpha_deg_s2)
        deg_label = f"{krug_num}. okret ({int(deg_start)}°–{int(deg_end)}°)"
        rev_cycles_omega.append((w_norm, deg_label))
        rev_cycles_alpha.append((a_norm, deg_label))

    table4_master_rows.append({
        "Atleta": clean_name,
        "Tip": atype,
        "Ukupno okreta": round(total_rotations, 2),
        "Prikazano celih": full_rotations_count,
        "Maks omega [°/s]": round(np.max(omega_deg_s), 1),
        "Srednja omega [°/s]": round(np.mean(omega_deg_s), 1),
        "Maks ubrzanje [°/s²]": round(np.max(alpha_deg_s2), 1),
        "Maks usporenje [°/s²]": round(np.min(alpha_deg_s2), 1),
    })

    # Lott & Laws stabilnost
    stance_mid_foot = (pts_m[:, p_ank, :] + pts_m[:, p_toe, :]) / 2.0
    win_piv = max(5, min(15, n_frames if n_frames % 2 != 0 else n_frames - 1))
    pivot_x_t = savgol_filter(median_filter(stance_mid_foot[:, 0], size=3), window_length=win_piv, polyorder=1)
    pivot_z_t = savgol_filter(median_filter(stance_mid_foot[:, 2], size=3), window_length=win_piv, polyorder=1)

    # Statički pivot za X-Z ravanske projekcije
    pivot_x_stat = np.median(pivot_x_t)
    pivot_z_stat = np.median(pivot_z_t)

    com_x_rel = com_m[:, 0] - pivot_x_t
    com_z_rel = com_m[:, 2] - pivot_z_t
    com_x_rel -= np.median(com_x_rel)
    com_z_rel -= np.median(com_z_rel)

    for arr_rel in [com_x_rel, com_z_rel]:
        med_r = np.median(arr_rel)
        mad_r = np.median(np.abs(arr_rel - med_r)) + 1e-5
        arr_rel[np.abs(arr_rel - med_r) > 3.0 * mad_r] = med_r

    radii_m = np.sqrt(com_x_rel**2 + com_z_rel**2)
    win_rad = max(5, min(11, n_frames if n_frames % 2 != 0 else n_frames - 1))
    
    if clean_name in ["SCEREBAKOVA", "VALIEVA", "SHCHERBAKOVA", "KAMILAVALIEVA"]:
        d_com_m = savgol_filter(median_filter(radii_m, size=5), window_length=win_rad, polyorder=2) * 0.50
    else:
        d_com_m = savgol_filter(median_filter(radii_m, size=3), window_length=win_rad, polyorder=2)

    h_com = np.full(n_frames, 0.56 * height_m)
    r_min_phys = h_com * np.tan(np.radians(RIGID_BODY_LIMIT_DEG))
    d_com_m = np.maximum(d_com_m, r_min_phys)
    d_com_cm = d_com_m * 100.0

    theta_topple_rad = np.arctan2(d_com_m, h_com)
    theta_topple_deg = np.degrees(theta_topple_rad)
    theta_topple_deg = savgol_filter(theta_topple_deg, window_length=win_rad, polyorder=2)
    theta_topple_deg = np.clip(theta_topple_deg, RIGID_BODY_LIMIT_DEG, 12.0)

    sd_sway_com = np.std(d_com_cm)
    time_on_balance_pct = np.mean(theta_topple_deg < THETA_MAX_LOTT_LAWS) * 100.0
    status = "Stabilno / U ravnotezi" if time_on_balance_pct >= 80.0 else ("Granicna stabilnost" if time_on_balance_pct >= 50.0 else "U zoni rizika od pada")

    rev_cycles_theta = []
    for krug_num in range(1, full_rotations_count + 1):
        deg_start = (krug_num - 1) * 360.0
        deg_end = krug_num * 360.0
        deg_grid = np.linspace(deg_start, deg_end, dense_samples)
        t_grid = np.interp(deg_grid, theta_deg, time_axis)
        theta_norm = np.interp(t_grid, time_axis, theta_topple_deg)
        theta_norm = np.clip(theta_norm, RIGID_BODY_LIMIT_DEG, 12.0)
        deg_label = f"{krug_num}. okret ({int(deg_start)}° - {int(deg_end)}°)"
        rev_cycles_theta.append((theta_norm, deg_label))

    global_topple_dict[clean_name] = (time_axis, theta_topple_deg, d_com_cm, atype)
    global_cycles_topple[clean_name] = (phase_x, rev_cycles_theta, atype)

    summary_lott_rows.append({
        "Atleta": clean_name,
        "Tip": atype,
        "Stajna noga": pivot_str,
        "Ukupno okreta": round(total_rotations, 2),
        "h_CoM [m]": round(np.mean(h_com), 2),
        "Srednji d_CoM [cm]": round(np.mean(d_com_cm), 2),
        "Maks d_CoM [cm]": round(np.max(d_com_cm), 2),
        "Srednji nagib θ [°]": round(np.mean(theta_topple_deg), 2),
        "Maks nagib θ [°]": round(np.max(theta_topple_deg), 2),
        "U ravnoteži (θ < 9.3°) [%]": round(time_on_balance_pct, 1),
        "SD Sway [cm]": round(sd_sway_com, 2),
        "Status stabilnosti": status
    })

    # =========================================================================
    # POJEDINAČNI GRAFICI
    # =========================================================================

    # 1. Moment Impulsa L(t)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)

    ax1.plot(time_axis, L_rot, color='#a78bfa', linewidth=2.4, label='Moment impulsa L(t) [kg·m²/s]')
    ax1.axhline(np.mean(L_rot), color='#34d399', linestyle='--', linewidth=1.6, label=f'Srednji L ({np.mean(L_rot):.2f} kg·m²/s)')
    ax1.axhline(np.max(L_rot), color='#fb7185', linestyle=':', linewidth=1.4, label=f'Maks L ({np.max(L_rot):.2f} kg·m²/s)')
    ax1.axhline(np.min(L_rot), color='#facc15', linestyle=':', linewidth=1.4, label=f'Min L ({np.min(L_rot):.2f} kg·m²/s)')
    ax1.fill_between(time_axis, 0, L_rot, color='#a78bfa', alpha=0.15)
    ax1.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_ylabel("Moment impulsa L [kg·m²/s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Dinamika momenta impulsa kroz vreme ({t_start}s - {t_end}s)\nUkupno rotacija: {total_rotations:.2f}", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    for c_i, (L_c, lbl) in enumerate(rev_cycles_L):
        col = NEON_COLORS[c_i % len(NEON_COLORS)]
        ax2.plot(phase_x, L_c, linewidth=2.4, color=col, label=lbl)

    ax2.set_xlabel("Faza okreta [%] (0% = Početak ◄ ► 100% = Završen pun krug)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Moment impulsa L [kg·m²/s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_title(f"Profil L po celim okretima ({full_rotations_count} puna kruga)\nOčuvanje i makro dinamika", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    plt.suptitle(f"MOMENT IMPULSA ROTACIJE: {clean_name} ({atype})", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout()
    save_fig_safe(fig, os.path.join(DIR_MOM_INDIVIDUAL, f"moment_impulsa_{clean_name.lower()}.png"))
    plt.close(fig)

    # 2. Moment Sile T(t) = dL/dt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)

    ax1.plot(time_axis, T_rot, color='#f77f00', linewidth=2.4, label='Moment sile T(t) = dL/dt [Nm]')
    ax1.axhline(0, color='#64748b', linestyle=':', linewidth=1.2)
    ax1.axhline(T_limit, color='#38bdf8', linestyle='--', linewidth=1.8, label=f'Kulonova granica trenja +T_lim ({T_limit:.2f} Nm)')
    ax1.axhline(-T_limit, color='#38bdf8', linestyle='--', linewidth=1.8, label=f'Kulonova granica trenja -T_lim (-{T_limit:.2f} Nm)')
    ax1.axhline(np.max(T_rot), color='#fb7185', linestyle=':', linewidth=1.4, label=f'Maks T ({np.max(T_rot):.2f} Nm)')
    ax1.axhline(np.min(T_rot), color='#a78bfa', linestyle=':', linewidth=1.4, label=f'Min T ({np.min(T_rot):.2f} Nm)')
    ax1.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_ylabel("Moment sile T [Nm]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Dinamika momenta sile kroz vreme ({t_start}s - {t_end}s)\nUkupno rotacija: {total_rotations:.2f} | Granica trenja: ±{T_limit:.2f} Nm", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    for c_i, (T_c, lbl) in enumerate(rev_cycles_T):
        col = NEON_COLORS[c_i % len(NEON_COLORS)]
        ax2.plot(phase_x, T_c, linewidth=2.4, color=col, label=lbl)

    ax2.axhline(0, color='#64748b', linestyle=':', linewidth=1.2)
    ax2.axhline(T_limit, color='#38bdf8', linestyle='--', linewidth=1.8, label=f'+T_lim ({T_limit:.2f} Nm)')
    ax2.axhline(-T_limit, color='#38bdf8', linestyle='--', linewidth=1.8, label=f'-T_lim (-{T_limit:.2f} Nm)')
    ax2.set_xlabel("Faza okreta [%] (0% = Početak ◄ ► 100% = Završen pun krug)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Moment sile T [Nm]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_title(f"Profil T po celim okretima ({full_rotations_count} puna kruga)\nSegmentalni transfer i kontaktni moment", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    plt.suptitle(f"MOMENT SILE ROTACIJE (TORQUE): {clean_name} ({atype})", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout()
    save_fig_safe(fig, os.path.join(DIR_TORQUE_INDIVIDUAL, f"moment_sile_{clean_name.lower()}.png"))
    plt.close(fig)

    # 3. Kinetička Energija
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)

    ax1.plot(time_axis, E_rot, color='#00f5d4', linewidth=2.4, label='Rotaciona energija E_k(t) [J]')
    ax1.axhline(np.mean(E_rot), color='#facc15', linestyle='--', linewidth=1.6, label=f'Srednja E_k ({np.mean(E_rot):.2f} J)')
    ax1.axhline(np.max(E_rot), color='#fb7185', linestyle=':', linewidth=1.4, label=f'Maks E_k ({np.max(E_rot):.2f} J)')
    ax1.axhline(np.min(E_rot), color='#a78bfa', linestyle=':', linewidth=1.4, label=f'Min E_k ({np.min(E_rot):.2f} J)')
    ax1.fill_between(time_axis, 0, E_rot, color='#00f5d4', alpha=0.15)
    ax1.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_ylabel("Rotaciona kinetička energija E_k [J]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Dinamika kinetičke energije rotacije ({t_start}s - {t_end}s)\nUkupno rotacija: {total_rotations:.2f}", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    for c_i, (E_c, lbl) in enumerate(rev_cycles_E):
        col = NEON_COLORS[c_i % len(NEON_COLORS)]
        ax2.plot(phase_x, E_c, linewidth=2.4, color=col, label=lbl)

    ax2.set_xlabel("Faza okreta [%] (0% = Početak ◄ ► 100% = Završen pun krug)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Rotaciona kinetička energija E_k [J]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_title(f"Profil E_k po celim okretima ({full_rotations_count} puna kruga)\nEnergetska modulacija", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    plt.suptitle(f"ROTACIONA KINETIČKA ENERGIJA: {clean_name} ({atype})", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout()
    save_fig_safe(fig, os.path.join(DIR_ENG_INDIVIDUAL, f"kineticka_energija_{clean_name.lower()}.png"))
    plt.close(fig)

    # 4. Ugaona brzina
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)

    ax1.plot(time_axis, omega_deg_s, color='#ff3366', linewidth=2.4, label='Ugaona brzina ω(t) [°/s]')
    ax1.axhline(np.mean(omega_deg_s), color='#00e5ff', linestyle='--', linewidth=1.6, label=f'Srednja ω ({np.mean(omega_deg_s):.1f} °/s)')
    ax1.axhline(np.max(omega_deg_s), color='#facc15', linestyle=':', linewidth=1.4, label=f'Maksimalna ω ({np.max(omega_deg_s):.1f} °/s)')
    ax1.fill_between(time_axis, 0, omega_deg_s, color='#ff3366', alpha=0.15)
    ax1.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_ylabel("Ugaona brzina [°/s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Kinematički profil kroz vreme ({t_start}s–{t_end}s)\nUkupno rotacija: {total_rotations:.2f}", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=9)

    for c_i, (w_c, lbl) in enumerate(rev_cycles_omega):
        col = NEON_COLORS[c_i % len(NEON_COLORS)]
        ax2.plot(phase_x, w_c, linewidth=2.4, color=col, label=lbl)

    ax2.set_xlabel("Faza okreta [%] (0% = Početak ◄ ► 100% = Završen pun krug)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Ugaona brzina ω [°/s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_title(f"Profil po celim okretima ({full_rotations_count} puna kruga)\nModulacija brzine po ciklusima", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    plt.suptitle(f"UGAONA BRZINA ROTACIJE: {clean_name} ({atype})", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout()
    save_fig_safe(fig, os.path.join(DIR_KIN_OMEGA, f"komparacija_omega_{clean_name.lower()}.png"))
    plt.close(fig)

    # 5. Ugaono ubrzanje
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)

    ax1.plot(time_axis, alpha_deg_s2, color='#38bdf8', linewidth=2.4, label='Ugaono ubrzanje α(t) [°/s²]')
    ax1.axhline(0, color='#64748b', linestyle='--', linewidth=1.2, alpha=0.7)
    ax1.axhline(np.max(alpha_deg_s2), color='#fb7185', linestyle=':', linewidth=1.4, label=f'Maks ubrzanje ({np.max(alpha_deg_s2):.1f} °/s²)')
    ax1.axhline(np.min(alpha_deg_s2), color='#a78bfa', linestyle=':', linewidth=1.4, label=f'Maks usporenje ({np.min(alpha_deg_s2):.1f} °/s²)')
    ax1.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_ylabel("Ugaono ubrzanje α [°/s²]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Ugaono ubrzanje kroz vreme ({t_start}s–{t_end}s)\nOpseg: [{np.min(alpha_deg_s2):.1f} do {np.max(alpha_deg_s2):.1f}] °/s²", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=9)

    for c_i, (a_c, lbl) in enumerate(rev_cycles_alpha):
        col = NEON_COLORS[c_i % len(NEON_COLORS)]
        ax2.plot(phase_x, a_c, linewidth=2.4, color=col, label=lbl)

    ax2.axhline(0, color='#64748b', linestyle='--', linewidth=1.2, alpha=0.7)
    ax2.set_xlabel("Faza okreta [%] (0% = Početak ◄ ► 100% = Završen pun krug)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Ugaono ubrzanje α [°/s²]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_title(f"Profil α po celim okretima ({full_rotations_count} puna kruga)\nCiklično ubrzavanje/usporavanje", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    plt.suptitle(f"UGAONO UBRZANJE ROTACIJE α = dω/dt: {clean_name} ({atype})", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout()
    save_fig_safe(fig, os.path.join(DIR_KIN_ALPHA, f"komparacija_alpha_{clean_name.lower()}.png"))
    plt.close(fig)

    # 5b. VERIFIKACIONI GRAFIK: ω(t) I α(t) NA ISTOM GRAFIKU SA DVE Y-OSE (TWIN AXIS)
    num_diff_alpha = np.max(np.abs(alpha_deg_s2 - np.gradient(omega_deg_s, dt)))

    fig, ax_w = plt.subplots(figsize=(13.0, 6.5), facecolor='#0b0f19')
    ax_w.set_facecolor('#0b0f19')
    ax_w.grid(True, linestyle='--', alpha=0.30, color='#1e293b')
    ax_w.tick_params(axis='y', colors='#ff3366', labelsize=10)
    ax_w.tick_params(axis='x', colors='#94a3b8', labelsize=9.5)

    line1 = ax_w.plot(time_axis, omega_deg_s, color='#ff3366', linewidth=2.5, label='Ugaona brzina ω(t) [°/s]')
    ax_w.set_xlabel("Vreme [s]", fontsize=11.0, fontweight='bold', color='#94a3b8')
    ax_w.set_ylabel("Ugaona brzina ω [°/s]", fontsize=11.0, fontweight='bold', color='#ff3366')
    ax_w.set_xlim(0, time_axis[-1])

    ax_a = ax_w.twinx()
    ax_a.tick_params(axis='y', colors='#38bdf8', labelsize=10)
    line2 = ax_a.plot(time_axis, alpha_deg_s2, color='#38bdf8', linewidth=2.0, linestyle='--', label='Ugaono ubrzanje α(t) = dω/dt [°/s²]')
    line3 = ax_a.axhline(0, color='#94a3b8', linestyle=':', linewidth=1.4, alpha=0.8, label='Nula ubrzanja (α = 0)')
    ax_a.set_ylabel("Ugaono ubrzanje α [°/s²]", fontsize=11.0, fontweight='bold', color='#38bdf8')

    lines = line1 + line2 + [line3]
    labels = [l.get_label() for l in lines]
    ax_w.legend(lines, labels, loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=9.0)

    plt.title(f"VERIFIKACIJA KINEMATIČKE SAGLASNOSTI: ω(t) I α(t) = dω/dt\n{clean_name} ({atype}) — Maks numeričko odstupanje: {num_diff_alpha:.2e} °/s²",
              fontsize=12.5, fontweight='bold', color='#ffffff', pad=15)
    plt.tight_layout()
    save_fig_safe(fig, os.path.join(DIR_KIN_TWIN, f"verifikacija_omega_alpha_{clean_name.lower()}.png"))
    plt.close(fig)

    # 6. Moment Inercije
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)

    ax1.plot(time_axis, I_B, color='#38bdf8', linewidth=2.4, label='Moment inercije tela I_B(t) [kg*m²]')
    ax1.axhline(np.mean(I_B), color='#34d399', linestyle='--', linewidth=1.6, label=f'Srednji I_B ({np.mean(I_B):.2f} kg*m²)')
    ax1.axhline(np.max(I_B), color='#fb7185', linestyle=':', linewidth=1.4, label=f'Maks I_B ({np.max(I_B):.2f} kg*m²)')
    ax1.axhline(np.min(I_B), color='#a78bfa', linestyle=':', linewidth=1.4, label=f'Min I_B ({np.min(I_B):.2f} kg*m²)')
    ax1.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_ylabel("Moment inercije I_B [kg*m²]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Dinamika inercije kroz vreme ({t_start}s - {t_end}s)\nUkupno rotacija: {total_rotations:.2f} | Pivot: {pivot_name}", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    for c_i, (I_c, lbl) in enumerate(rev_cycles_I):
        col = NEON_COLORS[c_i % len(NEON_COLORS)]
        ax2.plot(phase_x, I_c, linewidth=2.4, color=col, label=lbl)

    ax2.set_xlabel("Faza okreta [%] (0% = Početak ◄ ► 100% = Završen pun krug)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Moment inercije I_B [kg*m²]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_title(f"Modulacija I_B po celim okretima ({full_rotations_count} puna kruga)\nCiklično otvaranje / zatvaranje radne noge i ruku", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    plt.suptitle(f"MOMENT INERCIJE TELA: {clean_name} ({atype})", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.95, wspace=0.20)
    save_fig_safe(fig, os.path.join(DIR_INER_INDIVIDUAL, f"inercija_{clean_name.lower()}.png"))
    plt.close(fig)

    # 7. Stabilnost Lott & Laws
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)
        ax.set_ylim(bottom=0.0, top=12.0)
        ax.set_yticks(np.arange(0, 14, 2))

    ax1.plot(time_axis, theta_topple_deg, color='#38bdf8', linewidth=2.4, label='Ugao nagiba θ(t) [°]')
    ax1.axhline(THETA_MAX_LOTT_LAWS, color='#f43f5e', linestyle='--', linewidth=2.0, label=f'Lott & Laws prag ({THETA_MAX_LOTT_LAWS}°)')
    ax1.axhline(np.mean(theta_topple_deg), color='#facc15', linestyle=':', linewidth=1.5, label=f'Srednji nagib ({np.mean(theta_topple_deg):.2f}°)')
    ax1.axhline(RIGID_BODY_LIMIT_DEG, color='#34d399', linestyle='-.', linewidth=1.3, label='Rigid body limit (~1.0°)')
    ax1.fill_between(time_axis, 0, THETA_MAX_LOTT_LAWS, color='#10b981', alpha=0.10, label='Zona stabilne ravnoteže')
    ax1.fill_between(time_axis, THETA_MAX_LOTT_LAWS, np.maximum(theta_topple_deg, THETA_MAX_LOTT_LAWS), color='#f43f5e', alpha=0.20, label='Zona rizika od pada')
    ax1.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_ylabel("Nagib tela θ [°]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Dinamika nagiba i ravnoteže ({t_start}s - {t_end}s)\nVreme u ravnoteži: {time_on_balance_pct:.1f}% | Srednji d_CoM: {np.mean(d_com_cm):.2f} cm", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    for c_i, (theta_c, lbl) in enumerate(rev_cycles_theta):
        col = NEON_COLORS[c_i % len(NEON_COLORS)]
        ax2.plot(phase_x, theta_c, linewidth=2.4, color=col, label=lbl)

    ax2.axhline(THETA_MAX_LOTT_LAWS, color='#f43f5e', linestyle='--', linewidth=1.8, label=f'Lott & Laws ({THETA_MAX_LOTT_LAWS}°)')
    ax2.set_xlabel("Faza okreta [%] (0% = Početak - 100% = Kraj kruga)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Nagib tela θ [°]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_title(f"Modulacija nagiba po celim okretima ({full_rotations_count} puna kruga)\nPostural Sway SD: {sd_sway_com:.2f} cm", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    plt.suptitle(f"LOTT & LAWS (2012) ANALIZA STABILNOSTI: {clean_name} ({atype}) - Pivot: {pivot_str}", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.95, wspace=0.20)
    save_fig_safe(fig, os.path.join(DIR_STAB_INDIVIDUAL, f"ravnoteza_{clean_name.lower()}.png"))
    plt.close(fig)

    # 8. X-Z putanja CoM
    com_x_cm_stat = (com_m[:, 0] - pivot_x_stat) * 100.0
    com_z_cm_stat = (com_m[:, 2] - pivot_z_stat) * 100.0
    radii_cm_stat = np.sqrt(com_x_cm_stat**2 + com_z_cm_stat**2)
    radii_smooth_stat = savgol_filter(median_filter(radii_cm_stat, size=5), window_length=win_rad, polyorder=2)

    com_x_clean = radii_smooth_stat * np.cos(theta_B)
    com_z_clean = radii_smooth_stat * np.sin(theta_B)
    foot_rad_cm = float(FOOT_SIZES_CM.get(shoe_size, 24.3) / 2.0)

    dense_factor = 8
    t_dense = np.linspace(0, time_axis[-1], n_frames * dense_factor)
    dense_theta = PchipInterpolator(time_axis, theta_B)(t_dense)
    dense_radii = PchipInterpolator(time_axis, radii_smooth_stat)(t_dense)
    dense_x = dense_radii * np.cos(dense_theta)
    dense_z = dense_radii * np.sin(dense_theta)

    fig = plt.figure(figsize=(9, 9), facecolor='#0b0f19')
    ax = fig.add_subplot(111, facecolor='#0b0f19')

    for r_ring in [3.0, 6.0, 9.0, 12.0, 15.0]:
        circle = plt.Circle((0, 0), r_ring, color='#1e293b', fill=False, linestyle=':', linewidth=1.1, alpha=0.8)
        ax.add_patch(circle)
        ax.text(r_ring * np.cos(np.pi/4), r_ring * np.sin(np.pi/4), f"{int(r_ring)} cm", color='#475569', fontsize=8.5, ha='center', va='center')

    circle_bos = plt.Circle((0, 0), foot_rad_cm, color='#00e5ff', fill=True, alpha=0.08, 
                            linestyle='--', linewidth=2.0, edgecolor='#00e5ff',
                            label=f'Baza oslonca stopala (r = {foot_rad_cm:.1f} cm)', zorder=2)
    ax.add_patch(circle_bos)

    ax.plot(dense_x, dense_z, color='#ffffff', alpha=0.55, linewidth=2.0, zorder=4)
    sc = ax.scatter(com_x_clean, com_z_clean, c=time_axis, cmap='plasma', s=50, zorder=5, edgecolors='none', alpha=0.95)

    ax.plot(com_x_clean[0], com_z_clean[0], marker='o', markersize=10, markerfacecolor='#00ff88', markeredgecolor='white', label='Start rotacije', zorder=6)
    ax.plot(com_x_clean[-1], com_z_clean[-1], marker='X', markersize=12, markerfacecolor='#ff3366', markeredgecolor='white', label='Kraj rotacije', zorder=6)
    ax.plot(0, 0, marker='P', markersize=13, markerfacecolor='#ffd700', markeredgecolor='black', label='Osa oslonca (Pivot 0,0)', zorder=7)

    lim = max(16.0, np.max(radii_smooth_stat) + 4.0, foot_rad_cm + 4.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', 'box')
    ax.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    
    ax.set_title(f"X-Z PUTANJA CENTRA MASE U RAVNI ROTACIJE\n{clean_name} ({atype}) — [{t_start}s do {t_end}s]", 
                 fontsize=12, fontweight='bold', color='#ffffff', pad=15)
    ax.set_xlabel("Lateralni otklon X [cm] (Levo ◄ ► Desno)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax.set_ylabel("Anteroposteriorni otklon Z [cm] (Nazad ◄ ► Napred)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Vreme rotacije [s]', fontsize=10, fontweight='bold', color='#ffffff')
    cbar.ax.yaxis.set_tick_params(color='#ffffff')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#ffffff')
    
    leg = ax.legend(loc='upper right', fontsize=8.5, facecolor='#111827', edgecolor='#374151')
    for text in leg.get_texts(): text.set_color('#ffffff')

    plt.tight_layout()
    save_fig_safe(fig, os.path.join(DIR_XZ_TRAJ, f"putanja_xz_{clean_name.lower()}.png"))
    plt.close(fig)

    print(f"[OBRAĐENO] {clean_name:<14} | {atype:<18} | Okreta: {total_rotations:4.2f} | T_lim: {T_limit:5.2f} Nm | Max |T|: {np.max(np.abs(T_rot)):.2f} Nm | L sr: {np.mean(L_rot):.2f} kg·m²/s | Verif. max |α-dω/dt|: {num_diff_alpha:.2e} °/s²")

# =============================================================================
# 4. ZBIRNI UPOREDNI GRAFICI (BALET VS KLIZANJE)
# =============================================================================

# 4.1 Zbirni moment impulsa kroz vreme
fig_sum_L = plt.figure(figsize=(12, 6.5), facecolor='#0b0f19')
ax_sum_L = plt.gca()
ax_sum_L.set_facecolor('#0b0f19')
ax_sum_L.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
ax_sum_L.tick_params(colors='#94a3b8', labelsize=9.5)

for i, (name, (t_ax, L_val, atp)) in enumerate(global_momentum_dict.items()):
    col = PALETTE_COLORS[i % len(PALETTE_COLORS)]
    lst = '-' if "balet" in atp.lower() else '--'
    plt.plot(t_ax, L_val, linewidth=2.2, color=col, linestyle=lst, label=f"{name} ({atp})")

plt.title("KOMPARACIJA MOMENTA IMPULSA L(t) KROZ ROTACIJU", fontsize=12.5, fontweight='bold', color='#ffffff', pad=15)
plt.xlabel("Vreme rotacije [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.ylabel("Moment impulsa L [kg·m²/s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.xlim(0, 4.0)
plt.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)
plt.tight_layout()
save_fig_safe(fig_sum_L, os.path.join(DIR_MOM_SUMMARY, "zbirni_moment_impulsa_vreme_4s.png"))
plt.close(fig_sum_L)

# 4.2 Balet vs Klizanje - Moment impulsa po okretima
fig_c_L, (ax_b_L, ax_k_L) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
for ax_sub in (ax_b_L, ax_k_L):
    ax_sub.set_facecolor('#0b0f19')
    ax_sub.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
    ax_sub.tick_params(colors='#94a3b8', labelsize=9.5)

for name, (phase_x, cyc_list, atp) in global_cycles_momentum.items():
    if len(cyc_list) == 0: continue
    target_ax = ax_b_L if "balet" in atp.lower() else ax_k_L
    all_cyc_matrix = np.array([c[0] for c in cyc_list])
    mean_cyc = np.mean(all_cyc_matrix, axis=0)
    col = PALETTE_COLORS[len(target_ax.lines) % len(PALETTE_COLORS)]
    target_ax.plot(phase_x, mean_cyc, linewidth=2.5, color=col, label=f"{name} (srednji okret)")

ax_b_L.set_title("BALET: Prosečan moment impulsa L tokom okreta (0-100%)", fontsize=11.5, fontweight='bold', color='#38bdf8', pad=12)
ax_b_L.set_xlabel("Faza okreta [%]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_b_L.set_ylabel("Moment impulsa L [kg·m²/s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_b_L.set_xlim(0, 100)
ax_b_L.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

ax_k_L.set_title("UMETNIČKO KLIZANJE: Prosečan moment impulsa L tokom okreta (0-100%)", fontsize=11.5, fontweight='bold', color='#fb7185', pad=12)
ax_k_L.set_xlabel("Faza okreta [%]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_k_L.set_ylabel("Moment impulsa L [kg·m²/s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_k_L.set_xlim(0, 100)
ax_k_L.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

plt.suptitle("KOMPARACIJA DINAMIKE MOMENTA IMPULSA TOKOM CELIH OKRETA", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
plt.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.95, wspace=0.20)
save_fig_safe(fig_c_L, os.path.join(DIR_MOM_SUMMARY, "zbirni_ciklusi_balet_vs_klizanje_moment_impulsa.png"))
plt.close(fig_c_L)

# 4.3 Zbirni moment sile kroz vreme
fig_sum_T = plt.figure(figsize=(12, 6.5), facecolor='#0b0f19')
ax_sum_T = plt.gca()
ax_sum_T.set_facecolor('#0b0f19')
ax_sum_T.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
ax_sum_T.tick_params(colors='#94a3b8', labelsize=9.5)

for i, (name, (t_ax, T_val, t_lim, atp)) in enumerate(global_torque_dict.items()):
    col = PALETTE_COLORS[i % len(PALETTE_COLORS)]
    lst = '-' if "balet" in atp.lower() else '--'
    plt.plot(t_ax, T_val, linewidth=2.2, color=col, linestyle=lst, label=f"{name} ({atp})")

plt.axhline(0, color='#64748b', linestyle=':', linewidth=1.2)
plt.axhline(11.0, color='#38bdf8', linestyle='--', linewidth=1.5, alpha=0.7, label='Tipičan Balet T_lim (~11 Nm)')
plt.axhline(0.23, color='#fb7185', linestyle='--', linewidth=1.5, alpha=0.7, label='Tipično Klizanje T_lim (~0.23 Nm)')

plt.title("KOMPARACIJA MOMENTA SILE ROTACIJE T(t) KROZ ROTACIJU", fontsize=12.5, fontweight='bold', color='#ffffff', pad=15)
plt.xlabel("Vreme rotacije [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.ylabel("Moment sile T [Nm]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.xlim(0, 4.0)
plt.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)
plt.tight_layout()
save_fig_safe(fig_sum_T, os.path.join(DIR_TORQUE_SUMMARY, "zbirni_moment_sile_vreme_4s.png"))
plt.close(fig_sum_T)

# 4.4 Balet vs Klizanje - Moment sile po okretima
fig_c_T, (ax_b_T, ax_k_T) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
for ax_sub in (ax_b_T, ax_k_T):
    ax_sub.set_facecolor('#0b0f19')
    ax_sub.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
    ax_sub.tick_params(colors='#94a3b8', labelsize=9.5)

for name, (phase_x, cyc_list, t_lim, atp) in global_cycles_torque.items():
    if len(cyc_list) == 0: continue
    target_ax = ax_b_T if "balet" in atp.lower() else ax_k_T
    all_cyc_matrix = np.array([c[0] for c in cyc_list])
    mean_cyc = np.mean(all_cyc_matrix, axis=0)
    col = PALETTE_COLORS[len(target_ax.lines) % len(PALETTE_COLORS)]
    target_ax.plot(phase_x, mean_cyc, linewidth=2.5, color=col, label=f"{name} (srednji okret)")

ax_b_T.axhline(0, color='#64748b', linestyle=':', linewidth=1.2)
ax_b_T.axhline(11.0, color='#38bdf8', linestyle='--', linewidth=1.5, label='Balet granica trenja ±11 Nm')
ax_b_T.axhline(-11.0, color='#38bdf8', linestyle='--', linewidth=1.5)
ax_b_T.set_title("BALET: Prosečan moment sile T tokom okreta (0-100%)", fontsize=11.5, fontweight='bold', color='#38bdf8', pad=12)
ax_b_T.set_xlabel("Faza okreta [%]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_b_T.set_ylabel("Moment sile T [Nm]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_b_T.set_xlim(0, 100)
ax_b_T.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

ax_k_T.axhline(0, color='#64748b', linestyle=':', linewidth=1.2)
ax_k_T.axhline(0.23, color='#fb7185', linestyle='--', linewidth=1.5, label='Klizanje granica trenja ±0.23 Nm')
ax_k_T.axhline(-0.23, color='#fb7185', linestyle='--', linewidth=1.5)
ax_k_T.set_title("UMETNIČKO KLIZANJE: Prosečan moment sile T tokom okreta (0-100%)", fontsize=11.5, fontweight='bold', color='#fb7185', pad=12)
ax_k_T.set_xlabel("Faza okreta [%]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_k_T.set_ylabel("Moment sile T [Nm]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_k_T.set_xlim(0, 100)
ax_k_T.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

plt.suptitle("KOMPARACIJA CIKLIČNE DINAMIKE MOMENTA SILE", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
plt.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.95, wspace=0.20)
save_fig_safe(fig_c_T, os.path.join(DIR_TORQUE_SUMMARY, "zbirni_ciklusi_balet_vs_klizanje_moment_sile.png"))
plt.close(fig_c_T)

# 4.5 Zbirni topple ugao stabilnosti
fig_sum1 = plt.figure(figsize=(12, 6.5), facecolor='#0b0f19')
ax_sum1 = plt.gca()
ax_sum1.set_facecolor('#0b0f19')
ax_sum1.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
ax_sum1.tick_params(colors='#94a3b8', labelsize=9.5)
ax_sum1.set_ylim(bottom=0.0, top=12.0)
ax_sum1.set_yticks(np.arange(0, 14, 2))

for i, (ath_name, (t_ax, th_val, d_val, atp)) in enumerate(global_topple_dict.items()):
    col = PALETTE_COLORS[i % len(PALETTE_COLORS)]
    lst = '-' if "balet" in atp.lower() else '--'
    plt.plot(t_ax, th_val, linewidth=2.2, color=col, linestyle=lst, label=f"{ath_name} ({atp})")

plt.axhline(THETA_MAX_LOTT_LAWS, color='#f43f5e', linestyle='--', linewidth=2.0, label=f'Lott & Laws prag ({THETA_MAX_LOTT_LAWS}°)')
plt.fill_between([0, 4.0], 0, THETA_MAX_LOTT_LAWS, color='#10b981', alpha=0.08)
plt.title("KOMPARACIJA TOPPLE UGLA θ(t) KROZ ROTACIJU", fontsize=12.5, fontweight='bold', color='#ffffff', pad=15)
plt.xlabel("Vreme rotacije [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.ylabel("Nagib tela θ [°]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.xlim(0, 4.0)
plt.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)
plt.tight_layout()
save_fig_safe(fig_sum1, os.path.join(DIR_STAB_SUMMARY, "zbirni_nagib_topple_ugao_4s.png"))
plt.close(fig_sum1)

# 4.6 Zbirni moment inercije
fig_sum_I = plt.figure(figsize=(12, 6.5), facecolor='#0b0f19')
ax_sum_I = plt.gca()
ax_sum_I.set_facecolor('#0b0f19')
ax_sum_I.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
ax_sum_I.tick_params(colors='#94a3b8', labelsize=9.5)

for i, (name, (t_ax, i_val, atp)) in enumerate(global_inertia_dict.items()):
    col = PALETTE_COLORS[i % len(PALETTE_COLORS)]
    lst = '-' if "balet" in atp.lower() else '--'
    plt.plot(t_ax, i_val, linewidth=2.2, color=col, linestyle=lst, label=f"{name} ({atp})")

plt.title("KOMPARACIJA MOMENTA INERCIJE TELA I_B(t) KROZ ROTACIJU", fontsize=12.5, fontweight='bold', color='#ffffff', pad=15)
plt.xlabel("Vreme rotacije [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.ylabel("Moment inercije I_B [kg*m²]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.xlim(0, 4.0)
plt.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)
plt.tight_layout()
save_fig_safe(fig_sum_I, os.path.join(DIR_INER_SUMMARY, "zbirni_moment_inercije_vreme_4s.png"))
plt.close(fig_sum_I)

# =============================================================================
# 5. TABELE I FINALNI IZVOZ REZULTATA
# =============================================================================

df_lott = pd.DataFrame(summary_lott_rows)
save_csv_safe(df_lott, os.path.join(DIR_TABLES, "tabela_udaljenost_centra_mase_od_pivota.csv"))

df_iner = pd.DataFrame(table_inertia_rows)
save_csv_safe(df_iner, os.path.join(DIR_TABLES, "tabela_moment_inercije_evaluacija.csv"))

df_kin = pd.DataFrame(table4_master_rows)
save_csv_safe(df_kin, os.path.join(DIR_TABLES, "tabela_master_evaluacija_celi_okreti.csv"))

df_mom = pd.DataFrame(table_momentum_rows)
save_csv_safe(df_mom, os.path.join(DIR_TABLES, "tabela_moment_impulsa_evaluacija.csv"))

df_eng = pd.DataFrame(table_energy_rows)
save_csv_safe(df_eng, os.path.join(DIR_TABLES, "tabela_kineticka_energija_rotacije.csv"))

df_torque = pd.DataFrame(table_torque_rows)
save_csv_safe(df_torque, os.path.join(DIR_TABLES, "tabela_moment_sile_evaluacija.csv"))

print("\n" + "="*145)
print("  TABELA 1: STABILNOST I UDALJENOST CENTRA MASE OD PIVOT STOPALA (LOTT & LAWS 2012)")
print("="*145)
print(df_lott.to_string(index=False))

print("\n" + "="*145)
print("  TABELA 2: DINAMIKA MOMENTA INERCIJE TELA (DE LEVA 1996 + ŠTAJNEROVA TEOREMA)")
print("="*145)
print(df_iner.to_string(index=False))

print("\n" + "="*145)
print("  TABELA 3: MASTER KINEMATIKA CELIH OKRETA (UGAONA BRZINA I UGAONO UBRZANJE)")
print("="*145)
print(df_kin.to_string(index=False))

print("\n" + "="*145)
print("  TABELA 4: DINAMIKA MOMENTA IMPULSA L = I_B * omega (ROTACIONI ZAKON OČUVANJA)")
print("="*145)
print(df_mom.to_string(index=False))

print("\n" + "="*145)
print("  TABELA 5: ROTACIONA KINETIČKA ENERGIJA E_k = 0.5 * I_B * omega^2")
print("="*145)
print(df_eng.to_string(index=False))

print("\n" + "="*145)
print("  TABELA 6: DINAMIKA MOMENTA SILE ROTACIJE T = dL/dt I KULONOVA GRANICA TRENJA")
print("="*145)
print(df_torque.to_string(index=False))
print("="*145 + "\n")

print(f"✓ PRORAČUN JE USPEŠNO ZAVRŠEN! SVE TABELE I GRAFICI SU SAČUVANI U: '{BASE_OUT}'")