import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy import stats

warnings.filterwarnings('ignore')

# =============================================================================
# 1. PARAMETRI, BAZA ATLETA, DE LEVA MODEL I STRUKTURA DIREKTORIJUMA
# =============================================================================

TIME_WINDOWS = {
    "trusova":       (6.0, 10.0),
    "khoreva":       (0.0, 4.0),
    "marianela":     (0.0, 4.0),
    "kapitonova":    (0.0, 4.0),
    "liu":           (0.0, 4.0),
    "valieva":       (0.0, 4.0),
    "kamilavalieva": (0.0, 4.0),
    "shcherbakova":  (0.0, 4.0),
    "scerebakova":   (0.0, 4.0)
}

ATHLETE_DB = {
    "marianela":    {"height": 1.74, "weight": 52.0, "type": "Balet", "shoe_size": 39},
    "kapitonova":   {"height": 1.68, "weight": 48.0, "type": "Balet", "shoe_size": 38},
    "khoreva":      {"height": 1.73, "weight": 47.0, "type": "Balet", "shoe_size": 39},
    "trusova":      {"height": 1.66, "weight": 50.0, "type": "Umetničko klizanje", "shoe_size": 37},
    "valieva":      {"height": 1.60, "weight": 44.0, "type": "Umetničko klizanje", "shoe_size": 36},
    "kamilavalieva":{"height": 1.60, "weight": 44.0, "type": "Umetničko klizanje", "shoe_size": 36},
    "shcherbakova": {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje", "shoe_size": 36},
    "scerebakova":  {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje", "shoe_size": 36},
    "liu":          {"height": 1.58, "weight": 45.0, "type": "Umetničko klizanje", "shoe_size": 36}
}

FOOT_SIZES_CM = {36: 22.9, 37: 23.8, 38: 24.3, 39: 25.1, 40: 25.4}

# De Leva 1996 - Ženska raspodela mase i položaja težišta segmenata
DE_LEVA_FEMALE = {
    "Head":        {"mass": 0.0668, "pos": 0.5000},
    "Trunk":       {"mass": 0.4257, "pos": 0.4360},
    "R_UpperArm":  {"mass": 0.0255, "pos": 0.5754},
    "L_UpperArm":  {"mass": 0.0255, "pos": 0.5754},
    "R_Forearm":   {"mass": 0.0138, "pos": 0.4559},
    "L_Forearm":   {"mass": 0.0138, "pos": 0.4559},
    "R_Hand":      {"mass": 0.0056, "pos": 0.3400},
    "L_Hand":      {"mass": 0.0056, "pos": 0.3400},
    "R_Thigh":     {"mass": 0.1478, "pos": 0.3612},
    "L_Thigh":     {"mass": 0.1478, "pos": 0.3612},
    "R_Shank":     {"mass": 0.0481, "pos": 0.4416},
    "L_Shank":     {"mass": 0.0481, "pos": 0.4416},
    "R_Foot":      {"mass": 0.0129, "pos": 0.4014},
    "L_Foot":      {"mass": 0.0129, "pos": 0.4014}
}

G_ACC = 9.81
THETA_MAX_LOTT_LAWS = 9.3   # Prag stabilne ravnoteže Lott & Laws (2012) [°]
RIGID_BODY_LIMIT_DEG = 1.0  # Fizički limit krutog tela [°]

# Glavni izlazni direktorijum i poddirektorijumi
BASE_OUT = "konacnirezultati"

DIR_STAB_INDIVIDUAL = os.path.join(BASE_OUT, "grafici_ravnoteza_pojedinacni")
DIR_STAB_SUMMARY    = os.path.join(BASE_OUT, "grafici_ravnoteza_zbirni")

DIR_INER_INDIVIDUAL = os.path.join(BASE_OUT, "grafici_inercija_pojedinacni")
DIR_INER_SUMMARY    = os.path.join(BASE_OUT, "grafici_inercija_zbirni")

DIR_KIN_OMEGA       = os.path.join(BASE_OUT, "grafici_ugaona_brzina_komparacija")
DIR_KIN_ALPHA       = os.path.join(BASE_OUT, "grafici_ugaono_ubrzanje_komparacija")

# NOVI PODFOLDERI: Energetika i Koleno stajne noge
DIR_ENERGETICS_IND  = os.path.join(BASE_OUT, "grafici_energetika_pojedinacni")
DIR_ENERGETICS_SUM  = os.path.join(BASE_OUT, "grafici_energetika_zbirni")
DIR_KNEE_IND        = os.path.join(BASE_OUT, "grafici_koleno_stajne_noge_pojedinacni")
DIR_KNEE_SUM        = os.path.join(BASE_OUT, "grafici_koleno_stajne_noge_zbirni")

DIR_XZ_TRAJ         = os.path.join(BASE_OUT, "grafici_xz_putanje_com")
DIR_TABLES          = os.path.join(BASE_OUT, "tabele_rezultati")

ALL_DIRS = [
    DIR_STAB_INDIVIDUAL, DIR_STAB_SUMMARY,
    DIR_INER_INDIVIDUAL, DIR_INER_SUMMARY,
    DIR_KIN_OMEGA, DIR_KIN_ALPHA,
    DIR_ENERGETICS_IND, DIR_ENERGETICS_SUM,
    DIR_KNEE_IND, DIR_KNEE_SUM,
    DIR_XZ_TRAJ, DIR_TABLES
]

for d in ALL_DIRS:
    os.makedirs(d, exist_ok=True)

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

def calculate_angle_3d_series(a, b, c):
    """Računa ugao u zglobovima B u 3D prostoru kroz seriju frejmova"""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    dot_prod = np.sum(ba * bc, axis=1)
    cosine_angle = np.clip(dot_prod / (norm_ba * norm_bc + 1e-7), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

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
    if n < 2: return np.zeros(n)
        
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
    win = max(5, min(11, n if n % 2 != 0 else n - 1))
    theta_smooth = savgol_filter(theta_continuous, window_length=win, polyorder=2)
    return np.maximum(theta_smooth, 0.0)

# =============================================================================
# 3. GLAVNA OBRADA (SVE KOMPONENTE FIZIKE I KINEMATIKE U JEDNOM PROLAZU)
# =============================================================================

all_files = glob.glob(os.path.join(INPUT_DIR, "*_kinematics.csv"))
if not all_files:
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
all_files = [f for f in all_files if "SUMARNA" not in f and "popravljeno" not in f and "fizika_" not in f]

# Globalni rečnici za analize
global_topple_dict = {}
global_cycles_topple = {}
summary_lott_rows = []

global_inertia_dict = {}
global_cycles_inertia = {}
table_inertia_rows = []

global_energy_dict = {}
global_cycles_energy = {}
table_energy_rows = []

global_knee_dict = {}
global_cycles_knee = {}
table_knee_rows = []

table4_master_rows = []
processed_names = set()

print("\n" + "="*125)
print("  POKRETANJE PROŠIRENE OBJEDINJENE ANALIZE: KINEMATIKA, INERCIJA, RAVNOTEŽA, ENERGETIKA I KOLENO")
print("="*125 + "\n")

for file in sorted(all_files):
    df_raw = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower().replace("_", "").replace("-", "")
    
    athlete_key = next((k for k in ATHLETE_DB if k in filename_lower), None)
    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file).split('_')[0].upper()

    if clean_name in processed_names:
        continue
    processed_names.add(clean_name)

    athlete_data = ATHLETE_DB.get(athlete_key, {"height": 1.65, "weight": 50.0, "type": "Balet", "shoe_size": 38})
    atype = athlete_data["type"]
    height_m = athlete_data["height"]
    weight_kg = athlete_data["weight"]
    shoe_size = athlete_data["shoe_size"]
    is_skater = "klizanje" in atype.lower()

    if "x_0" not in df_raw.columns and "X_0" not in df_raw.columns:
        continue

    # Kropovanje intervala rotacije
    t_arr = df_raw["timestamp_sec"].values if "timestamp_sec" in df_raw.columns else (
            df_raw["Time_s"].values if "Time_s" in df_raw.columns else np.arange(len(df_raw)) / 30.0)
    t_start, t_end = TIME_WINDOWS.get(athlete_key, (0.0, 4.0))
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    df_crop = df_raw[mask].copy().reset_index(drop=True)
    if len(df_crop) < 15:
        df_crop = df_raw.iloc[:120].copy().reset_index(drop=True)

    n_frames = len(df_crop)
    dt = (t_end - t_start) / n_frames if n_frames > 0 else 1.0 / 30.0
    time_axis = np.arange(n_frames) * dt

    # 1. PCHIP i filtriranje 33 markera
    pts_array = np.zeros((n_frames, 33, 3))
    for lm in range(33):
        vis_col = f"vis_{lm}" if f"vis_{lm}" in df_crop.columns else f"VIS_{lm}"
        vis_series = df_crop[vis_col].values if vis_col in df_crop.columns else None
        for ax_idx, ax_name in enumerate(['x', 'y', 'z']):
            col = f"{ax_name}_{lm}" if f"{ax_name}_{lm}" in df_crop.columns else f"{ax_name.upper()}_{lm}"
            if col in df_crop.columns:
                pts_array[:, lm, ax_idx] = clean_and_interpolate_signal(df_crop[col].values, vis=vis_series, vis_threshold=0.35)

    # 2. Detekcija pivot noge
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

    # 3. Metričko skaliranje
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

    # Kalibracija Z dubine
    hip_width_real = 0.17 * height_m
    hip_width_meas = np.median(np.linalg.norm(pts_m[:, 23, :2] - pts_m[:, 24, :2], axis=1))
    z_correction = np.clip(hip_width_real / (hip_width_meas + 1e-5), 0.35, 0.65)
    pts_m[:, :, 2] = pts_m[:, :, 2] * z_correction

    # 4. Kinematika rotacije
    raw_angles_B = compute_fused_torso_orientation_3d(pts_m)
    theta_B = track_strictly_monotonic_spin(raw_angles_B, is_skater=is_skater)
    
    total_displacement_deg = np.degrees(theta_B[-1])
    total_rotations = total_displacement_deg / 360.0
    full_rotations_count = int(np.floor(total_rotations))

    dense_samples = 101
    phase_x = np.linspace(0, 100, dense_samples)
    theta_deg = np.degrees(theta_B)

    win_dyn_kin = max(11, min(25, n_frames if n_frames % 2 != 0 else n_frames - 1))
    omega_B = savgol_filter(theta_B, window_length=win_dyn_kin, polyorder=2, deriv=1, delta=dt)
    floor_rad_s = np.deg2rad(180.0 if is_skater else 120.0)
    omega_B = np.maximum(omega_B, floor_rad_s)
    omega_deg_s = np.degrees(omega_B)
    alpha_deg_s2 = savgol_filter(omega_deg_s, window_length=win_dyn_kin, polyorder=2, deriv=1, delta=dt)

    # 5. Segmenti i centri mase (De Leva 1996)
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

    # -------------------------------------------------------------------------
    # A) MOMENT INERCIJE
    # -------------------------------------------------------------------------
    axis_x_inst = 0.5 * (pts_m[:, p_ank, 0] + mid_hp_m[:, 0])
    axis_z_inst = 0.5 * (pts_m[:, p_ank, 2] + mid_hp_m[:, 2])
    win_smooth = max(5, min(13, n_frames if n_frames % 2 != 0 else n_frames - 1))
    axis_x = savgol_filter(axis_x_inst, window_length=win_smooth, polyorder=2)
    axis_z = savgol_filter(axis_z_inst, window_length=win_smooth, polyorder=2)

    support_leg_names = ["L_Thigh", "L_Shank", "L_Foot"] if planted_side == "left" else ["R_Thigh", "R_Shank", "R_Foot"]
    I_B_raw = np.zeros(n_frames)
    max_arm_len = 0.82 * (height_m / 1.65)
    max_leg_len = 0.96 * (height_m / 1.65)

    for seg_name, s_coords in seg_m.items():
        m_frac = DE_LEVA_FEMALE[seg_name]["mass"]
        seg_mass = m_frac * weight_kg
        if seg_name not in support_leg_names:
            r_val = np.sqrt((s_coords[:, 0] - axis_x)**2 + (s_coords[:, 2] - axis_z)**2)
            if "Arm" in seg_name or "Hand" in seg_name or "Forearm" in seg_name:
                r_val = np.clip(r_val, 0.08, max_arm_len)
            elif "Thigh" in seg_name or "Shank" in seg_name or "Foot" in seg_name:
                r_val = np.clip(r_val, 0.12, max_leg_len)
            else:
                r_val = np.clip(r_val, 0.04, 0.24)
            I_B_raw += seg_mass * (r_val ** 2)

    win_dyn_iner = max(9, min(17, n_frames if n_frames % 2 != 0 else n_frames - 1))
    I_B = savgol_filter(I_B_raw, window_length=win_dyn_iner, polyorder=2)
    I_B = np.maximum(I_B, 0.40)
    I_L = 0.085 * (weight_kg / 49.5) * (height_m / 1.65)**2

    rev_cycles_I = []
    for krug_num in range(1, full_rotations_count + 1):
        deg_start = (krug_num - 1) * 360.0
        deg_end = krug_num * 360.0
        deg_grid = np.linspace(deg_start, deg_end, dense_samples)
        t_grid = np.interp(deg_grid, theta_deg, time_axis)
        I_norm = np.interp(t_grid, time_axis, I_B)
        deg_label = f"{krug_num}. okret ({int(deg_start)}°–{int(deg_end)}°)"
        rev_cycles_I.append((I_norm, deg_label))

    global_inertia_dict[clean_name] = (time_axis, I_B, I_L, atype)
    global_cycles_inertia[clean_name] = (phase_x, rev_cycles_I, atype)

    table_inertia_rows.append({
        "Atleta": clean_name, "Tip": atype, "Stajna noga": pivot_name,
        "Ukupno okreta": round(total_rotations, 2), "Prikazano celih": full_rotations_count,
        "I_B min [kg*m2]": round(np.min(I_B), 2), "I_B max [kg*m2]": round(np.max(I_B), 2),
        "I_B sr [kg*m2]": round(np.mean(I_B), 2), "I_L noga [kg*m2]": round(I_L, 3),
        "Modulacija Delta I [kg*m2]": round(np.max(I_B) - np.min(I_B), 2)
    })

    # -------------------------------------------------------------------------
    # B) TOPPLE RAVNOTEŽA (LOTT & LAWS 2012)
    # -------------------------------------------------------------------------
    stance_mid_foot = (pts_m[:, p_ank, :] + pts_m[:, p_toe, :]) / 2.0
    win_piv = max(5, min(15, n_frames if n_frames % 2 != 0 else n_frames - 1))
    pivot_x_t = savgol_filter(median_filter(stance_mid_foot[:, 0], size=3), window_length=win_piv, polyorder=1)
    pivot_z_t = savgol_filter(median_filter(stance_mid_foot[:, 2], size=3), window_length=win_piv, polyorder=1)

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
    d_com_m = np.maximum(d_com_m, h_com * np.tan(np.radians(RIGID_BODY_LIMIT_DEG)))
    d_com_cm = d_com_m * 100.0

    theta_topple_deg = np.clip(np.degrees(np.arctan2(d_com_m, h_com)), RIGID_BODY_LIMIT_DEG, 12.0)
    theta_topple_deg = savgol_filter(theta_topple_deg, window_length=win_rad, polyorder=2)

    sd_sway_com = np.std(d_com_cm)
    time_on_balance_pct = np.mean(theta_topple_deg < THETA_MAX_LOTT_LAWS) * 100.0
    status = "Stabilno / U ravnotezi" if time_on_balance_pct >= 80.0 else ("Granicna stabilnost" if time_on_balance_pct >= 50.0 else "U zoni rizika")

    rev_cycles_theta = []
    for krug_num in range(1, full_rotations_count + 1):
        deg_start = (krug_num - 1) * 360.0
        deg_end = krug_num * 360.0
        deg_grid = np.linspace(deg_start, deg_end, dense_samples)
        t_grid = np.interp(deg_grid, theta_deg, time_axis)
        theta_norm = np.clip(np.interp(t_grid, time_axis, theta_topple_deg), RIGID_BODY_LIMIT_DEG, 12.0)
        rev_cycles_theta.append((theta_norm, f"{krug_num}. okret ({int(deg_start)}° - {int(deg_end)}°)"))

    global_topple_dict[clean_name] = (time_axis, theta_topple_deg, d_com_cm, atype)
    global_cycles_topple[clean_name] = (phase_x, rev_cycles_theta, atype)

    summary_lott_rows.append({
        "Atleta": clean_name, "Tip": atype, "Stajna noga": pivot_str,
        "Ukupno okreta": round(total_rotations, 2), "h_CoM [m]": round(np.mean(h_com), 2),
        "Srednji d_CoM [cm]": round(np.mean(d_com_cm), 2), "Maks d_CoM [cm]": round(np.max(d_com_cm), 2),
        "Srednji nagib θ [°]": round(np.mean(theta_topple_deg), 2), "Maks nagib θ [°]": round(np.max(theta_topple_deg), 2),
        "U ravnoteži (θ < 9.3°) [%]": round(time_on_balance_pct, 1), "SD Sway [cm]": round(sd_sway_com, 2), "Status stabilnosti": status
    })

    # -------------------------------------------------------------------------
    # C) NOVO: ROTACIONA KINETIČKA ENERGIJA (Ek) I MOMENT IMPULSA (L)
    # -------------------------------------------------------------------------
    L_angular = I_B * omega_B                    # [kg*m^2/s]
    E_rot_kinetic = 0.5 * I_B * (omega_B ** 2)    # [J]
    
    rev_cycles_Ek = []
    rev_cycles_L = []
    for krug_num in range(1, full_rotations_count + 1):
        deg_start = (krug_num - 1) * 360.0
        deg_end = krug_num * 360.0
        deg_grid = np.linspace(deg_start, deg_end, dense_samples)
        t_grid = np.interp(deg_grid, theta_deg, time_axis)
        Ek_norm = np.interp(t_grid, time_axis, E_rot_kinetic)
        L_norm = np.interp(t_grid, time_axis, L_angular)
        rev_cycles_Ek.append((Ek_norm, f"{krug_num}. okret"))
        rev_cycles_L.append((L_norm, f"{krug_num}. okret"))

    global_energy_dict[clean_name] = (time_axis, E_rot_kinetic, L_angular, atype)
    global_cycles_energy[clean_name] = (phase_x, rev_cycles_Ek, atype)

    table_energy_rows.append({
        "Atleta": clean_name, "Tip": atype,
        "Srednja Ek [J]": round(np.mean(E_rot_kinetic), 1),
        "Maks Ek [J]": round(np.max(E_rot_kinetic), 1),
        "Srednji L [kg*m2/s]": round(np.mean(L_angular), 2),
        "Maks L [kg*m2/s]": round(np.max(L_angular), 2)
    })

    # Grafik Energetike (Pojedinačni)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)

    ax1.plot(time_axis, E_rot_kinetic, color='#fee440', linewidth=2.4, label='Kinetička energija Ek(t) [J]')
    ax1.axhline(np.mean(E_rot_kinetic), color='#00f5d4', linestyle='--', linewidth=1.6, label=f'Srednja Ek ({np.mean(E_rot_kinetic):.1f} J)')
    ax1.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_ylabel("Rotaciona energija Ek [J]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Dinamika kinetičke energije ({t_start}s - {t_end}s)\nMaksimalna Ek: {np.max(E_rot_kinetic):.1f} J", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    for c_i, (Ek_c, lbl) in enumerate(rev_cycles_Ek):
        col = NEON_COLORS[c_i % len(NEON_COLORS)]
        ax2.plot(phase_x, Ek_c, linewidth=2.4, color=col, label=lbl)

    ax2.set_xlabel("Faza okreta [%] (0% = Početak - 100% = Kraj)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Kinetička energija Ek [J]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_title("Modulacija rotacione energije po okretima", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    plt.suptitle(f"ROTACIONA ENERGETIKA POKRETA: {clean_name} ({atype})", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_ENERGETICS_IND, f"energetika_{clean_name.lower()}.png"), dpi=300, facecolor='#0b0f19')
    plt.close()

    # -------------------------------------------------------------------------
    # D) NOVO: KINEMATIKA ZGLOBA KOLENA STAJNE NOGE (KNEE FLEXION)
    # -------------------------------------------------------------------------
    hip_pts = pts_m[:, p_hip, :]
    knee_pts = pts_m[:, p_knee, :]
    ank_pts = pts_m[:, p_ank, :]
    
    knee_angles_raw = calculate_angle_3d_series(hip_pts, knee_pts, ank_pts)
    knee_angles = savgol_filter(median_filter(knee_angles_raw, size=3), window_length=win_dyn_kin, polyorder=2)
    knee_angles = np.clip(knee_angles, 90.0, 180.0)

    rev_cycles_knee_list = []
    for krug_num in range(1, full_rotations_count + 1):
        deg_start = (krug_num - 1) * 360.0
        deg_end = krug_num * 360.0
        deg_grid = np.linspace(deg_start, deg_end, dense_samples)
        t_grid = np.interp(deg_grid, theta_deg, time_axis)
        knee_norm = np.interp(t_grid, time_axis, knee_angles)
        rev_cycles_knee_list.append((knee_norm, f"{krug_num}. okret"))

    global_knee_dict[clean_name] = (time_axis, knee_angles, theta_topple_deg, atype)
    global_cycles_knee[clean_name] = (phase_x, rev_cycles_knee_list, atype)

    table_knee_rows.append({
        "Atleta": clean_name, "Tip": atype, "Stajna noga": pivot_name,
        "Srednji ugao kolena [°]": round(np.mean(knee_angles), 1),
        "Min ugao (Maks fleksija) [°]": round(np.min(knee_angles), 1),
        "Maks ugao (Ekstenzija) [°]": round(np.max(knee_angles), 1),
        "Opseg fleksije Delta [°]": round(np.max(knee_angles) - np.min(knee_angles), 1)
    })

    # Grafik Kolena (Pojedinačni)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)
        ax.set_ylim(100.0, 185.0)

    ax1.plot(time_axis, knee_angles, color='#34d399', linewidth=2.4, label='Ugao u kolenu stajne noge [°]')
    ax1.axhline(np.mean(knee_angles), color='#fb7185', linestyle='--', linewidth=1.6, label=f'Srednji ugao ({np.mean(knee_angles):.1f}°)')
    ax1.axhline(180.0, color='#64748b', linestyle=':', linewidth=1.2, label='Potpuno opružena noga (180°)')
    ax1.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_ylabel("Ugao kolena [°]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Fleksija stajnog kolena ({pivot_name})\nSrednja ekstenzija: {np.mean(knee_angles):.1f}°", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc='lower right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    for c_i, (k_c, lbl) in enumerate(rev_cycles_knee_list):
        col = NEON_COLORS[c_i % len(NEON_COLORS)]
        ax2.plot(phase_x, k_c, linewidth=2.4, color=col, label=lbl)

    ax2.set_xlabel("Faza okreta [%]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Ugao kolena [°]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_title("Modulacija fleksije kolena po celim okretima", fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax2.set_xlim(0, 100)
    ax2.legend(loc='lower right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    plt.suptitle(f"KINEMATIKA ZGLOBA KOLENA STAJNE NOGE: {clean_name} ({atype})", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_KNEE_IND, f"koleno_{clean_name.lower()}.png"), dpi=300, facecolor='#0b0f19')
    plt.close()

    # -------------------------------------------------------------------------
    # E) KINEMATIKA CELIH OKRETA (Master tabela)
    # -------------------------------------------------------------------------
    table4_master_rows.append({
        "Atleta": clean_name, "Tip": atype, "Ukupno okreta": round(total_rotations, 2),
        "Prikazano celih": full_rotations_count, "Maks omega [°/s]": round(np.max(omega_deg_s), 1),
        "Srednja omega [°/s]": round(np.mean(omega_deg_s), 1),
        "Maks ubrzanje [°/s²]": round(np.max(alpha_deg_s2), 1),
        "Maks usporenje [°/s²]": round(np.min(alpha_deg_s2), 1)
    })

    print(f"[OBRADA ZAVRŠENA] {clean_name:<14} | {atype:<18} | Okreta: {total_rotations:4.2f} | Ek sr: {np.mean(E_rot_kinetic):5.1f}J | Koleno sr: {np.mean(knee_angles):5.1f}° | θ sr: {np.mean(theta_topple_deg):4.2f}°")

# =============================================================================
# 4. ZBIRNI KOMPARATIVNI GRAFICI (ENERGETIKA I KOLENO)
# =============================================================================

# 4.1 Zbirna Kinetička Energija
plt.figure(figsize=(12, 6.5), facecolor='#0b0f19')
ax_sum_ek = plt.gca()
ax_sum_ek.set_facecolor('#0b0f19')
ax_sum_ek.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
ax_sum_ek.tick_params(colors='#94a3b8', labelsize=9.5)

for i, (name, (t_ax, ek_val, l_val, atp)) in enumerate(global_energy_dict.items()):
    col = PALETTE_COLORS[i % len(PALETTE_COLORS)]
    lst = '-' if "balet" in atp.lower() else '--'
    plt.plot(t_ax, ek_val, linewidth=2.2, color=col, linestyle=lst, label=f"{name} ({atp})")

plt.title("KOMPARACIJA ROTACIONE KINETIČKE ENERGIJE Ek(t) KROZ VREME", fontsize=12.5, fontweight='bold', color='#ffffff', pad=15)
plt.xlabel("Vreme rotacije [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.ylabel("Kinetička energija Ek [J]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.xlim(0, 4.0)
plt.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(DIR_ENERGETICS_SUM, "zbirni_kineticka_energija_4s.png"), dpi=300, facecolor='#0b0f19')
plt.close()

# 4.2 Zbirni Ugao Kolena Stajne Noge
plt.figure(figsize=(12, 6.5), facecolor='#0b0f19')
ax_sum_k = plt.gca()
ax_sum_k.set_facecolor('#0b0f19')
ax_sum_k.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
ax_sum_k.tick_params(colors='#94a3b8', labelsize=9.5)
ax_sum_k.set_ylim(110.0, 185.0)

for i, (name, (t_ax, k_val, th_val, atp)) in enumerate(global_knee_dict.items()):
    col = PALETTE_COLORS[i % len(PALETTE_COLORS)]
    lst = '-' if "balet" in atp.lower() else '--'
    plt.plot(t_ax, k_val, linewidth=2.2, color=col, linestyle=lst, label=f"{name} ({atp})")

plt.axhline(180.0, color='#64748b', linestyle=':', linewidth=1.4, label='Potpuna ekstenzija (180°)')
plt.title("KOMPARACIJA FLEKSIJE KOLENA STAJNE NOGE: BALET VS. KLIZANJE", fontsize=12.5, fontweight='bold', color='#ffffff', pad=15)
plt.xlabel("Vreme rotacije [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.ylabel("Ugao u kolenu [°]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.xlim(0, 4.0)
plt.legend(loc='lower right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(DIR_KNEE_SUM, "zbirni_ugao_kolena_stajne_noge.png"), dpi=300, facecolor='#0b0f19')
plt.close()

# =============================================================================
# 5. NOVO: STATISTIČKA VERIFIKACIJA HIPOTEZA (BALET VS. KLIZANJE)
# =============================================================================

# Spajanje svih podataka po grupama
ballet_omega, skater_omega = [], []
ballet_theta, skater_theta = [], []
ballet_Ek, skater_Ek = [], []
ballet_knee, skater_knee = [], []
ballet_IB, skater_IB = [], []

for row_l, row_i, row_k, row_e in zip(summary_lott_rows, table_inertia_rows, table_knee_rows, table_energy_rows):
    is_ballet = "balet" in row_l["Tip"].lower()
    if is_ballet:
        ballet_theta.append(row_l["Srednji nagib θ [°]"])
        ballet_IB.append(row_i["I_B sr [kg*m2]"])
        ballet_knee.append(row_k["Srednji ugao kolena [°]"])
        ballet_Ek.append(row_e["Srednja Ek [J]"])
    else:
        skater_theta.append(row_l["Srednji nagib θ [°]"])
        skater_IB.append(row_i["I_B sr [kg*m2]"])
        skater_knee.append(row_k["Srednji ugao kolena [°]"])
        skater_Ek.append(row_e["Srednja Ek [J]"])

for row_m in table4_master_rows:
    if "balet" in row_m["Tip"].lower():
        ballet_omega.append(row_m["Maks omega [°/s]"])
    else:
        skater_omega.append(row_m["Maks omega [°/s]"])

stats_metrics = [
    ("Maksimalna ugaona brzina ω [°/s]", ballet_omega, skater_omega),
    ("Kinetička energija rotacije Ek [J]", ballet_Ek, skater_Ek),
    ("Ugao kolena stajne noge [°]", ballet_knee, skater_knee),
    ("Srednji topple nagib θ [°]", ballet_theta, skater_theta),
    ("Moment inercije tela I_B [kg*m²]", ballet_IB, skater_IB)
]

stats_rows = []
for name_m, b_arr, k_arr in stats_metrics:
    b_arr, k_arr = np.array(b_arr), np.array(k_arr)
    m_b, s_b = np.mean(b_arr), np.std(b_arr, ddof=1) if len(b_arr) > 1 else 0.0
    m_k, s_k = np.mean(k_arr), np.std(k_arr, ddof=1) if len(k_arr) > 1 else 0.0
    
    # Studentov t-test nezavisnih uzoraka
    t_stat, p_val = stats.ttest_ind(b_arr, k_arr, equal_var=False)
    # Mann-Whitney U neparametrijski test
    u_stat, p_val_mw = stats.mannwhitneyu(b_arr, k_arr, alternative='two-sided')
    
    # Cohen-ov d efekat veličine
    pooled_sd = np.sqrt(((len(b_arr)-1)*s_b**2 + (len(k_arr)-1)*s_k**2) / (len(b_arr) + len(k_arr) - 2))
    cohen_d = (m_k - m_b) / (pooled_sd + 1e-7)

    signif = "DA (p < 0.05)" if p_val < 0.05 else ("Tendencija (p < 0.10)" if p_val < 0.10 else "NE (p >= 0.10)")

    stats_rows.append({
        "Biomehanički parametar": name_m,
        "Balet (Mean ± SD)": f"{m_b:.2f} ± {s_b:.2f}",
        "Klizanje (Mean ± SD)": f"{m_k:.2f} ± {s_k:.2f}",
        "t-statistika": round(t_stat, 3),
        "p-vrednost (t-test)": round(p_val, 4),
        "p-vrednost (Mann-Whitney)": round(p_val_mw, 4),
        "Cohen-ov d": round(cohen_d, 2),
        "Statistički značajno": signif
    })

df_stats = pd.DataFrame(stats_rows)
df_stats.to_csv(os.path.join(DIR_TABLES, "tabela_statisticka_verifikacija_balet_vs_klizanje.csv"), index=False)

# =============================================================================
# 6. SAČUVAVANJE FINALNIH TABELA I PRIKAZ U KONZOLI
# =============================================================================

pd.DataFrame(summary_lott_rows).to_csv(os.path.join(DIR_TABLES, "tabela_udaljenost_centra_mase_od_pivota.csv"), index=False)
pd.DataFrame(table_inertia_rows).to_csv(os.path.join(DIR_TABLES, "tabela_moment_inercije_evaluacija.csv"), index=False)
pd.DataFrame(table_energy_rows).to_csv(os.path.join(DIR_TABLES, "tabela_rotaciona_energetika_Ek_L.csv"), index=False)
pd.DataFrame(table_knee_rows).to_csv(os.path.join(DIR_TABLES, "tabela_kinematika_kolena_stajne_noge.csv"), index=False)
pd.DataFrame(table4_master_rows).to_csv(os.path.join(DIR_TABLES, "tabela_master_evaluacija_celi_okreti.csv"), index=False)

print("\n" + "="*145)
print("  TABELA 1: STATISTIČKA KOMPARACIJA I VERIFIKACIJA HIPOTEZA (BALET VS. UMETNIČKO KLIZANJE)")
print("="*145)
print(df_stats.to_string(index=False))

print("\n" + "="*145)
print("  TABELA 2: ROTACIONA KINETIČKA ENERGIJA (Ek) I MOMENT IMPULSA (L)")
print("="*145)
print(pd.DataFrame(table_energy_rows).to_string(index=False))

print("\n" + "="*145)
print("  TABELA 3: KINEMATIKA ZGLOBA KOLENA STAJNE NOGE (FLEKSIJA / EKSTENZIJA)")
print("="*145)
print(pd.DataFrame(table_knee_rows).to_string(index=False))
print("="*145 + "\n")

print(f"✓ SVI NOVI REZULTATI, GRAFICI I TABELE SU SAČUVANI U: '{BASE_OUT}/'")
print(f"  ├── Grafici energetike (Ek, L):             {DIR_ENERGETICS_IND}/ i {DIR_ENERGETICS_SUM}/")
print(f"  ├── Grafici kolena stajne noge:             {DIR_KNEE_IND}/ i {DIR_KNEE_SUM}/")
print(f"  └── Statistička tabela i CSV izveštaji:     {DIR_TABLES}/\n")
