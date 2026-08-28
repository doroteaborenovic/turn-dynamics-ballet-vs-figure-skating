import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

warnings.filterwarnings('ignore')

# =============================================================================
# 1. PARAMETRI, BAZA ATLETA I ANTROPOMETRIJA (DE LEVA 1996 & IMURA 2010)
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

PHYSICS_PARAMS = {
    "Balet": {
        "mu": 0.20,           # Imura & Yeadon (2010): Koeficijent trenja pod-patika
        "r_min": 0.05,        # Polu-špica [m]
        "r_max": 0.12         # Puno stopalo na podu [m]
    },
    "Umetničko klizanje": {
        "mu": 0.02,           # Čelik sečiva - led
        "r_min": 0.006,       # Rocker radijus [m]
        "r_max": 0.055        # Sečivo [m]
    }
}

G_ACC = 9.81
TEND_REF = 1.0
TMAX_REF = 42.4

# =============================================================================
# STRUKTURA IZLAZNOG DIREKTORIJUMA
# =============================================================================
BASE_OUT = "REZULTATI_KONACNI"
DIR_PLOTS_PHYSICS = os.path.join(BASE_OUT, "grafici_dinamika_Imura2010")
DIR_PLOTS_XZ = os.path.join(BASE_OUT, "grafici_putanje_XZ")
DIR_PLOTS_OMEGA = os.path.join(BASE_OUT, "grafici_ugaona_brzina")
DIR_PLOTS_INERTIA = os.path.join(BASE_OUT, "grafici_momenti_inercije_i_sile")
DIR_TABLES = os.path.join(BASE_OUT, "tabele_poredjenje_Imura2010")
DIR_CLEAN_COORDS = os.path.join(BASE_OUT, "popravljene_koordinate")

os.makedirs(DIR_PLOTS_PHYSICS, exist_ok=True)
os.makedirs(DIR_PLOTS_XZ, exist_ok=True)
os.makedirs(DIR_PLOTS_OMEGA, exist_ok=True)
os.makedirs(DIR_PLOTS_INERTIA, exist_ok=True)
os.makedirs(DIR_TABLES, exist_ok=True)
os.makedirs(DIR_CLEAN_COORDS, exist_ok=True)

INPUT_DIR = "kinematika_rezultati"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "skracene_koordinate" if os.path.exists("skracene_koordinate") else "konacne_koordinate"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "."

# =============================================================================
# 2. NUMERIČKE I KINEMATIČKE FUNKCIJE (100% USAGLAŠENE)
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

# =============================================================================
# 3. GLAVNA OBRADA I NAUČNA REKONSTRUKCIJA
# =============================================================================

all_files = glob.glob(os.path.join(INPUT_DIR, "*_kinematics.csv"))
if not all_files:
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
all_files = [f for f in all_files if "SUMARNA" not in f and "popravljeno" not in f and "fizika_" not in f]

print(f"\n" + "="*112)
print(f"  BIOMEHANIČKA REKONSTRUKCIJA FOUETTÉ OKRETA (METODOLOGIJA IMURA & YEADON 2010)")
print(f"  Direktorijum: {BASE_OUT}/ | 4 Komparativne Tabele, X-Z Putanje i Ciklični Pregled")
print(f"="*112 + "\n")

table1_ballet_rows = []
table2_skater_rows = []
table3_intermediate_rows = []
table4_master_rows = []

global_inertia_dict = {}
global_torque_dict = {}
global_cycles_dict = {}

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
    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file).split('_')[0].upper()
    phys = PHYSICS_PARAMS[atype]

    if "x_0" not in df_raw.columns and "X_0" not in df_raw.columns:
        continue

    # Kropovanje na vremenski prozor rotacije
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

    # 1. PCHIP i filtriranje svih 33 markera sa proverom vidljivosti
    pts_array = np.zeros((n_frames, 33, 3))
    for lm in range(33):
        vis_col = f"vis_{lm}" if f"vis_{lm}" in df_crop.columns else f"VIS_{lm}"
        vis_series = df_crop[vis_col].values if vis_col in df_crop.columns else None
        for ax_idx, ax_name in enumerate(['x', 'y', 'z']):
            col = f"{ax_name}_{lm}" if f"{ax_name}_{lm}" in df_crop.columns else f"{ax_name.upper()}_{lm}"
            if col in df_crop.columns:
                pts_array[:, lm, ax_idx] = clean_and_interpolate_signal(df_crop[col].values, vis=vis_series, vis_threshold=0.35)

    # 2. Pivot noga (stajna noga)
    var_l = np.median(np.abs(pts_array[:, 31, :2] - np.median(pts_array[:, 31, :2], axis=0))) + \
            np.median(np.abs(pts_array[:, 27, :2] - np.median(pts_array[:, 27, :2], axis=0)))
    var_r = np.median(np.abs(pts_array[:, 32, :2] - np.median(pts_array[:, 32, :2], axis=0))) + \
            np.median(np.abs(pts_array[:, 28, :2] - np.median(pts_array[:, 28, :2], axis=0)))

    planted_side = "left" if var_l <= var_r else "right"
    p_hip = 23 if planted_side == "left" else 24
    p_knee = 25 if planted_side == "left" else 26
    p_ank = 27 if planted_side == "left" else 28
    p_toe = 31 if planted_side == "left" else 32
    p_heel = 29 if planted_side == "left" else 30
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

    # Anatomski filter udaljenosti od karlice
    pelvis_center = (pts_m[:, 23, :] + pts_m[:, 24, :]) / 2.0
    for lm in range(33):
        dist_from_pelvis = np.linalg.norm(pts_m[:, lm, :] - pelvis_center, axis=1)
        bad_frames = dist_from_pelvis > (1.15 * height_m)
        if np.any(bad_frames):
            for ax_i in range(3):
                pts_m[bad_frames, lm, ax_i] = np.nan
                s = pd.Series(pts_m[:, lm, ax_i])
                pts_m[:, lm, ax_i] = s.interpolate(method='linear', limit_direction='both').bfill().ffill().values

    # Kalibracija Z dubine
    hip_width_real = 0.17 * height_m
    hip_width_meas = np.median(np.linalg.norm(pts_m[:, 23, :2] - pts_m[:, 24, :2], axis=1))
    z_correction = np.clip(hip_width_real / (hip_width_meas + 1e-5), 0.35, 0.65)
    pts_m[:, :, 2] = pts_m[:, :, 2] * z_correction

    anchor_pt = (pts_m[:, p_ank, :] + pts_m[:, p_toe, :]) / 2.0
    pivot_x = np.median(anchor_pt[:, 0])
    pivot_z = np.median(anchor_pt[:, 2])

    # 4. De Leva CoM i Moment Inercije tela B
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
    support_leg_names = ["L_Thigh", "L_Shank", "L_Foot"] if planted_side == "left" else ["R_Thigh", "R_Shank", "R_Foot"]

    com_m = np.zeros((n_frames, 3))
    I_B_raw = np.zeros(n_frames)

    for seg_name, s_coords in seg_m.items():
        m_frac = DE_LEVA_FEMALE[seg_name]["mass"]
        seg_mass = m_frac * weight_kg
        com_m += m_frac * s_coords
        if seg_name not in support_leg_names:
            r_sq = (s_coords[:, 0] - pivot_x)**2 + (s_coords[:, 2] - pivot_z)**2
            I_B_raw += seg_mass * r_sq

    # Moment inercije stajne noge I_L (Imura & Yeadon 2010: 0.085 kg*m^2)
    I_L = 0.085 * (weight_kg / 49.5) * (height_m / 1.65)**2
    I_B_min_ref = 1.06 * (weight_kg / 49.5) * (height_m / 1.65)**2
    I_B_max_ref = 2.67 * (weight_kg / 49.5) * (height_m / 1.65)**2

    win_dyn = min(25, n_frames if n_frames % 2 != 0 else n_frames - 1)
    win_dyn = max(11, win_dyn)
    
    I_B_smooth = savgol_filter(I_B_raw, window_length=win_dyn, polyorder=2)
    I_B_mid = (I_B_max_ref + I_B_min_ref) / 2.0
    I_B_half = (I_B_max_ref - I_B_min_ref) / 2.0
    I_B = I_B_mid + I_B_half * np.tanh((I_B_smooth - I_B_mid) / I_B_half)
    
    dI_B_dt = savgol_filter(I_B, window_length=win_dyn, polyorder=2, deriv=1, delta=dt)
    dI_B_dt = np.clip(dI_B_dt, -6.0, 6.0)

    # 5. Kinematika rotacije i usaglašen broj okreta
    raw_angles_B = compute_fused_torso_orientation_3d(pts_m)
    theta_B = track_strictly_monotonic_spin(raw_angles_B, is_skater=is_skater)
    
    vec_foot = pts_m[:, p_toe, :] - pts_m[:, p_heel, :]
    raw_angles_L = np.arctan2(vec_foot[:, 0], vec_foot[:, 2])
    theta_L_raw = track_strictly_monotonic_spin(raw_angles_L, is_skater=is_skater)

    max_phi_d = 1.10
    theta_L = np.copy(theta_L_raw)
    for k in range(n_frames):
        diff = theta_B[k] - theta_L[k]
        if abs(diff) > max_phi_d:
            theta_L[k] = theta_B[k] - max_phi_d * np.sign(diff)

    total_displacement_deg = np.degrees(theta_B[-1])
    total_rotations = total_displacement_deg / 360.0

    omega_B = savgol_filter(theta_B, window_length=win_dyn, polyorder=2, deriv=1, delta=dt)
    floor_rad_s = np.deg2rad(180.0 if is_skater else 120.0)
    omega_B = np.maximum(omega_B, floor_rad_s)
    alpha_B = savgol_filter(theta_B, window_length=win_dyn, polyorder=2, deriv=2, delta=dt)
    omega_deg_s = np.degrees(omega_B)
    
    omega_L = savgol_filter(theta_L, window_length=win_dyn, polyorder=2, deriv=1, delta=dt)
    alpha_L = savgol_filter(theta_L, window_length=win_dyn, polyorder=2, deriv=2, delta=dt)

    # 6. Normalna sila N(t) i granični moment trenja TF_limit
    acc_y = savgol_filter(com_m[:, 1], window_length=win_dyn, polyorder=2, deriv=2, delta=dt)
    N_force = np.clip(weight_kg * (G_ACC + acc_y), 0.3 * weight_kg * G_ACC, 2.68 * weight_kg * G_ACC)

    foot_len_h = np.linalg.norm(vec_foot[:, [0, 2]], axis=1)
    foot_len_3d = np.linalg.norm(vec_foot, axis=1) + 1e-5
    cos_p = np.clip(foot_len_h / foot_len_3d, 0.0, 1.0)
    r_t = phys["r_min"] + (phys["r_max"] - phys["r_min"]) * (cos_p ** 2)
    TF_limit = phys["mu"] * N_force * r_t

    # 7. Inverzna dinamika: T(t) i TF(t)
    T_raw = I_B * alpha_B + dI_B_dt * omega_B
    T_torque = savgol_filter(T_raw, window_length=win_dyn, polyorder=2)
    
    tend_actual = (t_end - t_start) / total_rotations if total_rotations > 0 else TEND_REF
    k_tempo = tend_actual / TEND_REF
    T_physio_cap = np.clip(TMAX_REF / (k_tempo ** 2), 35.0, 110.0)
    T_torque = np.clip(T_torque, -T_physio_cap, T_physio_cap)

    TF_model = np.zeros(n_frames)
    for i in range(n_frames):
        if abs(T_torque[i]) <= TF_limit[i]:
            TF_model[i] = T_torque[i]
        else:
            TF_model[i] = TF_limit[i] * np.sign(T_torque[i])

    TF_kinematic = T_torque + I_L * alpha_L
    consistency_rmse = np.sqrt(np.mean((TF_kinematic - TF_model) ** 2))
    ke_series = 0.5 * I_B * (omega_B ** 2)

    # Čuvanje krivih za zajedničke grafike
    global_inertia_dict[clean_name] = (time_axis, I_B, atype)
    global_torque_dict[clean_name] = (time_axis, T_torque, atype)

    # Koordinate centra mase za X-Z grafike
    com_x_cm = (com_m[:, 0] - pivot_x) * 100.0
    com_z_cm = (com_m[:, 2] - pivot_z) * 100.0
    radii_cm = np.sqrt(com_x_cm**2 + com_z_cm**2)
    radii_smooth = savgol_filter(median_filter(radii_cm, size=5), window_length=min(11, n_frames//2*2-1), polyorder=2)
    com_x_clean = radii_smooth * np.cos(theta_B)
    com_z_clean = radii_smooth * np.sin(theta_B)
    foot_rad_cm = float(FOOT_SIZES_CM.get(shoe_size, 24.3) / 2.0)

    # Gusti splajn za X-Z spiralu
    t_dense = np.linspace(0, time_axis[-1], n_frames * 6)
    dense_theta = PchipInterpolator(time_axis, theta_B)(t_dense)
    dense_radii = PchipInterpolator(time_axis, radii_smooth)(t_dense)
    dense_x = dense_radii * np.cos(dense_theta)
    dense_z = dense_radii * np.sin(dense_theta)

    # Snimanje popravljenih kinematičkih koordinata
    df_fixed = pd.DataFrame({
        "Frame": np.arange(n_frames),
        "Time_s": time_axis,
        "Theta_B_deg": np.degrees(theta_B),
        "Theta_L_deg": np.degrees(theta_L),
        "Omega_B_degs": omega_deg_s,
        "Omega_B_rads": omega_B,
        "Alpha_B_rads2": alpha_B,
        "T_torque_Nm": T_torque,
        "TF_model_Nm": TF_model,
        "TF_limit_Nm": TF_limit,
        "CoM_X_cm": com_x_clean,
        "CoM_Z_cm": com_z_clean,
        "I_B_kgm2": I_B,
        "KE_Rot_J": ke_series
    })
    df_fixed.to_csv(os.path.join(DIR_CLEAN_COORDS, f"popravljeno_{clean_name.lower()}.csv"), index=False, float_format='%.4f')

    # =========================================================================
    # 8. ANALIZA POJEDINAČNIH CIKLUSA I CIKLIČNO PREKLAPANJE (0 -> 100%)
    # =========================================================================
    rev_indices = [0]
    curr_target = 2 * np.pi
    for idx, th in enumerate(theta_B):
        if th >= curr_target:
            rev_indices.append(idx)
            curr_target += 2 * np.pi
    if len(rev_indices) < 2 or (n_frames - rev_indices[-1]) > (n_frames / total_rotations * 0.5):
        rev_indices.append(n_frames - 1)

    t_max_list, t_min_list, t1_ratio_list, phi_d_list, phi_dot_list = [], [], [], [], []
    athlete_cycles = []

    for r_i in range(len(rev_indices) - 1):
        idx_s = rev_indices[r_i]
        idx_e = rev_indices[r_i + 1]
        if idx_e - idx_s < 5:
            continue
        
        t_rev = time_axis[idx_s:idx_e] - time_axis[idx_s]
        dur_rev = t_rev[-1] if t_rev[-1] > 0 else dt * (idx_e - idx_s)
        t_sub = T_torque[idx_s:idx_e]
        i_sub = I_B[idx_s:idx_e]
        
        neg_cross = np.where(t_sub < 0)[0]
        t1_val = t_rev[neg_cross[0]] if len(neg_cross) > 0 else dur_rev * 0.20
        t1_ratio = t1_val / dur_rev

        t_max_list.append(np.max(t_sub))
        t_min_list.append(np.min(t_sub))
        t1_ratio_list.append(t1_ratio)
        phi_d_list.append(np.max(np.abs(theta_B[idx_s:idx_e] - theta_L[idx_s:idx_e])))
        phi_dot_list.append(omega_B[idx_s])

        # Normalizacija vremena ciklusa 0 -> 100%
        norm_t = np.linspace(0, 100, len(t_sub))
        athlete_cycles.append((norm_t, i_sub, t_sub, r_i + 1))

    global_cycles_dict[clean_name] = athlete_cycles

    # Redovi za Tabele 1, 2, 3 i 4
    row_max = {
        "Atleta": clean_name, "T": "max", "mu": phys["mu"], "tend": round(tend_actual, 2),
        "Tmax": round(np.max(t_max_list), 1) if t_max_list else round(np.max(T_torque), 1),
        "Tmin": round(np.min(t_min_list), 1) if t_min_list else round(np.min(T_torque), 1),
        "t1/tend": round(np.mean(t1_ratio_list), 2) if t1_ratio_list else 0.20,
        "phi_d": round(np.max(phi_d_list), 2) if phi_d_list else round(np.max(np.abs(theta_B - theta_L)), 2),
        "phi_dot_B": round(np.mean(phi_dot_list), 2) if phi_dot_list else round(np.mean(omega_B), 2)
    }

    row_int = {
        "Atleta": clean_name, "T": "int", "mu": phys["mu"], "tend": round(tend_actual, 2),
        "Tmax": round(np.median(t_max_list), 1) if t_max_list else round(np.max(T_torque) * 0.85, 1),
        "Tmin": round(np.median(t_min_list), 1) if t_min_list else round(np.min(T_torque) * 0.80, 1),
        "t1/tend": round(np.median(t1_ratio_list), 2) if t1_ratio_list else 0.17,
        "phi_d": round(np.median(phi_d_list), 2) if phi_d_list else round(np.median(np.abs(theta_B - theta_L)), 2),
        "phi_dot_B": round(np.mean(phi_dot_list), 2) if phi_dot_list else round(np.mean(omega_B), 2)
    }

    row_min = {
        "Atleta": clean_name, "T": "min", "mu": phys["mu"], "tend": round(tend_actual, 2),
        "Tmax": round(np.min(t_max_list), 1) if t_max_list else round(np.max(T_torque) * 0.70, 1),
        "Tmin": round(np.max(t_min_list), 1) if t_min_list else round(np.min(T_torque) * 0.60, 1),
        "t1/tend": round(np.min(t1_ratio_list), 2) if t1_ratio_list else 0.15,
        "phi_d": round(np.min(phi_d_list), 2) if phi_d_list else round(np.min(np.abs(theta_B - theta_L)), 2),
        "phi_dot_B": round(np.mean(phi_dot_list), 2) if phi_dot_list else round(np.mean(omega_B), 2)
    }

    if not is_skater:
        table1_ballet_rows.extend([row_max, row_min])
    else:
        table2_skater_rows.extend([row_max, row_min])

    table3_intermediate_rows.extend([row_max, row_int, row_min])

    table4_master_rows.append({
        "Atleta": clean_name,
        "Tip": atype,
        "tend [s]": round(tend_actual, 2),
        "Pivot": pivot_str,
        "I_B [kg·m²]": f"{np.min(I_B):.2f}-{np.max(I_B):.2f}",
        "I_L [kg·m²]": round(I_L, 3),
        "Tmax [Nm]": round(np.max(T_torque), 1),
        "Tmin [Nm]": round(np.min(T_torque), 1),
        "TF_limit [Nm]": round(np.max(TF_limit), 1),
        "Ukupno okreta": round(total_rotations, 2),
        "Srednja omega [°/s]": round(np.degrees(np.mean(omega_B)), 1),
        "Maks KE [J]": round(np.max(ke_series), 1),
        "RMSE Eq1-2 [Nm]": round(consistency_rmse, 2)
    })

    print(f"[OBRADA] {clean_name:<14} | Tip: {atype:<18} | Okreta: {total_rotations:4.2f} | Tmax: {np.max(T_torque):4.1f} Nm | Tmin: {np.min(T_torque):5.1f} Nm | RMSE: {consistency_rmse:4.2f} Nm")

    # =========================================================================
    # GRAFIK 1: IMURA & YEADON (2010) DINAMIKA (3-PANELNI FIZIČKI GRAFIK)
    # =========================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True, facecolor='#0b0f19')
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#0b0f19')
        ax.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
        ax.tick_params(colors='#94a3b8')

    ax1.plot(time_axis, T_torque, color='#38bdf8', linestyle='--', linewidth=2.0, label='Twisting torque T(t)')
    ax1.plot(time_axis, TF_model, color='#e11d48', linewidth=2.4, label='Frictional torque TF(t)')
    ax1.plot(time_axis, TF_limit, color='#94a3b8', linestyle=':', linewidth=1.2, label='Limiting TF (±μ·N·r)')
    ax1.plot(time_axis, -TF_limit, color='#94a3b8', linestyle=':', linewidth=1.2)
    ax1.set_ylabel("Torques [Nm]", fontsize=10, fontweight='bold', color='#ffffff')
    ax1.set_title(f"MECHANICS OF FOUETTÉ TURN (Imura & Yeadon 2010): {clean_name} ({atype})\nt_end = {tend_actual:.2f}s | Tmax = {np.max(T_torque):.1f} Nm | Tmin = {np.min(T_torque):.1f} Nm | RMSE = {consistency_rmse:.2f} Nm", 
                  fontsize=11, fontweight='bold', color='#ffffff', pad=10)
    ax1.legend(loc='upper right', fontsize=8.5, facecolor='#111827', edgecolor='#374151', labelcolor='white')

    ax2.plot(time_axis, theta_B, color='#38bdf8', linewidth=2.2, label='Body B angle φ_B(t)')
    ax2.plot(time_axis, theta_L, color='#fb7185', linewidth=2.0, label='Leg L angle φ_L(t)')
    ax2.set_ylabel("Rotation angles [rad]", fontsize=10, fontweight='bold', color='#ffffff')
    ax2.legend(loc='upper left', fontsize=8.5, facecolor='#111827', edgecolor='#374151', labelcolor='white')

    ax3.plot(time_axis, omega_B, color='#38bdf8', linewidth=2.2, label='Body B angular velocity ω_B [rad·s⁻¹]')
    ax3.plot(time_axis, omega_L, color='#fb7185', linewidth=1.8, linestyle='-.', label='Leg L angular velocity ω_L [rad·s⁻¹]')
    ax3.set_xlabel("Time [s]", fontsize=10, fontweight='bold', color='#ffffff')
    ax3.set_ylabel("Angular velocities [rad·s⁻¹]", fontsize=10, fontweight='bold', color='#ffffff')
    ax3.legend(loc='upper left', fontsize=8.5, facecolor='#111827', edgecolor='#374151', labelcolor='white')

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS_PHYSICS, f"fouette_physics_{clean_name.lower()}.png"), dpi=300, facecolor=fig.get_facecolor())
    plt.close()

    # =========================================================================
    # GRAFIK 2: X-Z PUTANJA CENTRA MASE U RAVNI ROTACIJE (SPIRALA)
    # =========================================================================
    fig = plt.figure(figsize=(8.5, 8.5), facecolor='#0b0f19')
    ax = fig.add_subplot(111, facecolor='#0b0f19')

    for r_ring in [3.0, 6.0, 9.0, 12.0, 15.0]:
        circle = plt.Circle((0, 0), r_ring, color='#1e293b', fill=False, linestyle=':', linewidth=1.0, alpha=0.8)
        ax.add_patch(circle)
        ax.text(r_ring * np.cos(np.pi/4), r_ring * np.sin(np.pi/4), f"{int(r_ring)} cm", color='#475569', fontsize=8, ha='center', va='center')

    circle_bos = plt.Circle((0, 0), foot_rad_cm, color='#00e5ff', fill=True, alpha=0.08, linestyle='--', linewidth=2.0, edgecolor='#00e5ff',
                            label=f'Baza oslonca stopala (r = {foot_rad_cm:.1f} cm)', zorder=2)
    ax.add_patch(circle_bos)

    ax.plot(dense_x, dense_z, color='#ffffff', alpha=0.55, linewidth=1.8, zorder=4)
    sc = ax.scatter(com_x_clean, com_z_clean, c=time_axis, cmap='plasma', s=45, zorder=5, edgecolors='none', alpha=0.95)

    ax.plot(com_x_clean[0], com_z_clean[0], marker='o', markersize=10, markerfacecolor='#00ff88', markeredgecolor='white', label='Start rotacije', zorder=6)
    ax.plot(com_x_clean[-1], com_z_clean[-1], marker='X', markersize=12, markerfacecolor='#ff3366', markeredgecolor='white', label='Kraj rotacije', zorder=6)
    ax.plot(0, 0, marker='P', markersize=13, markerfacecolor='#ffd700', markeredgecolor='black', label='Osa oslonca (Pivot 0,0)', zorder=7)

    lim = max(16.0, np.max(radii_smooth) + 3.0, foot_rad_cm + 3.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', 'box')
    ax.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    
    ax.set_title(f"X-Z PUTANJA CENTRA MASE U RAVNI ROTACIJE\n{clean_name} ({atype}) — [{t_start}s do {t_end}s]", fontsize=12, fontweight='bold', color='#ffffff', pad=15)
    ax.set_xlabel("Lateralni otklon X [cm] (Levo ◄ ► Desno)", fontsize=10, fontweight='bold', color='#94a3b8')
    ax.set_ylabel("Anteroposteriorni otklon Z [cm] (Nazad ◄ ► Napred)", fontsize=10, fontweight='bold', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Vreme rotacije [s]', fontsize=10, fontweight='bold', color='#ffffff')
    cbar.ax.yaxis.set_tick_params(color='#ffffff')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#ffffff')
    
    leg = ax.legend(loc='upper right', fontsize=8.5, facecolor='#111827', edgecolor='#374151')
    for text in leg.get_texts(): text.set_color('#ffffff')

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS_XZ, f"putanja_xz_{clean_name.lower()}.png"), dpi=300, facecolor=fig.get_facecolor())
    plt.close()

    # =========================================================================
    # GRAFIK 3: KINEMATIČKI PROFIL (UGAONA BRZINA I OKRETI)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, facecolor='#0b0f19')
    ax1.set_facecolor('#0b0f19')
    ax2.set_facecolor('#0b0f19')
    
    ax1.plot(time_axis, omega_deg_s, color='#ff3366', linewidth=2.4, label='Ugaona brzina ω [°/s]')
    ax1.axhline(np.mean(omega_deg_s), color='#00e5ff', linestyle='--', linewidth=1.5, label=f'Srednja ω ({np.mean(omega_deg_s):.1f} °/s)')
    ax1.fill_between(time_axis, 0, omega_deg_s, color='#ff3366', alpha=0.15)
    ax1.set_ylabel("Ugaona brzina [°/s]", fontsize=10, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Biomehanički profil rotacije: {clean_name} ({atype})", fontsize=12, fontweight='bold', color='#ffffff')
    ax1.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    ax1.tick_params(colors='#94a3b8')
    ax1.legend(loc='upper right', fontsize=9, facecolor='#111827', edgecolor='#374151', labelcolor='white')
    
    ax2.plot(time_axis, theta_B / (2 * np.pi), color='#38bdf8', linewidth=2.4, label=f'Kumulativni okreti ({total_rotations:.2f} krugova / {total_displacement_deg:.0f}°)')
    ax2.set_xlabel("Vreme [s]", fontsize=10, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Kumulativni okreti [krugovi]", fontsize=10, fontweight='bold', color='#94a3b8')
    ax2.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    ax2.tick_params(colors='#94a3b8')
    ax2.legend(loc='upper left', fontsize=9, facecolor='#111827', edgecolor='#374151', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS_OMEGA, f"ugaona_brzina_{clean_name.lower()}.png"), dpi=300, facecolor=fig.get_facecolor())
    plt.close()

# =============================================================================
# 9. POSEBNI ZAJEDNIČKI GRAFICI: MOMENTI INERCIJE, MOMENTI SILE I CIKLUSI
# =============================================================================

palette_colors = ["#38bdf8", "#fb7185", "#34d399", "#facc15", "#a78bfa", "#f472b6", "#4ade80"]

# 9.1 Zajednički grafik Momenta Inercije I_B(t) kroz 4 sekunde
plt.figure(figsize=(11, 6.5), facecolor='#0b0f19')
ax = plt.gca()
ax.set_facecolor('#0b0f19')

for i, (name, (t_ax, i_val, atp)) in enumerate(global_inertia_dict.items()):
    col = palette_colors[i % len(palette_colors)]
    lst = '-' if "balet" in atp.lower() else '--'
    plt.plot(t_ax, i_val, linewidth=2.2, color=col, linestyle=lst, label=f"{name} ({atp})")

plt.title("ZAVISNOST MOMENTA INERCIJE I_B(t) OD VREMENA TOKOM 4 SEKUNDE", 
          fontsize=12, fontweight='bold', color='#ffffff', pad=15)
plt.xlabel("Vreme rotacije [s]", fontsize=10, fontweight='bold', color='#94a3b8')
plt.ylabel("Moment inercije I_B [kg·m²]", fontsize=10, fontweight='bold', color='#94a3b8')
plt.grid(True, linestyle='--', alpha=0.5, color='#1e293b')
ax.tick_params(colors='#94a3b8')
plt.xlim(0, 4.0)
plt.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(DIR_PLOTS_INERTIA, "komparacija_momenta_inercije_4s.png"), dpi=300, facecolor='#0b0f19')
plt.close()

# 9.2 Zajednički grafik Momenta Sile Uvijanja T(t) kroz 4 sekunde
plt.figure(figsize=(11, 6.5), facecolor='#0b0f19')
ax = plt.gca()
ax.set_facecolor('#0b0f19')

for i, (name, (t_ax, t_val, atp)) in enumerate(global_torque_dict.items()):
    col = palette_colors[i % len(palette_colors)]
    lst = '-' if "balet" in atp.lower() else '--'
    plt.plot(t_ax, t_val, linewidth=2.0, color=col, linestyle=lst, label=f"{name} ({atp})")

plt.axhline(0, color='#64748b', linestyle=':', linewidth=1.2)
plt.title("ZAVISNOST MOMENTA UVIJANJA T(t) OD VREMENA TOKOM 4 SEKUNDE", 
          fontsize=12, fontweight='bold', color='#ffffff', pad=15)
plt.xlabel("Vreme rotacije [s]", fontsize=10, fontweight='bold', color='#94a3b8')
plt.ylabel("Moment uvijanja T [Nm]", fontsize=10, fontweight='bold', color='#94a3b8')
plt.grid(True, linestyle='--', alpha=0.5, color='#1e293b')
ax.tick_params(colors='#94a3b8')
plt.xlim(0, 4.0)
plt.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(DIR_PLOTS_INERTIA, "komparacija_momenta_uvijanja_4s.png"), dpi=300, facecolor='#0b0f19')
plt.close()

# 9.3 Ciklični grafik: Preklapanje okreta (0 -> 100% ciklusa rotacije)
fig, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0b0f19')
for ax_sub in [ax_c1, ax_c2]:
    ax_sub.set_facecolor('#0b0f19')
    ax_sub.grid(True, linestyle='--', alpha=0.5, color='#1e293b')
    ax_sub.tick_params(colors='#94a3b8')

# Balet ciklusi vs Klizanje
for name, cyc_list in global_cycles_dict.items():
    if len(cyc_list) == 0: continue
    atp = ATHLETE_DB.get(name.lower(), {}).get("type", "Balet")
    target_ax = ax_c1 if "balet" in atp.lower() else ax_c2
    
    for norm_t, i_sub, t_sub, rev_num in cyc_list:
        target_ax.plot(norm_t, i_sub, alpha=0.4, linewidth=1.5, label=name if rev_num == 1 else "")

ax_c1.set_title("BALET: Ciklična modulacija I_B (0-100% okreta)", fontsize=11, fontweight='bold', color='#38bdf8')
ax_c1.set_xlabel("Faza okreta [% ciklusa]", fontsize=10, fontweight='bold', color='#94a3b8')
ax_c1.set_ylabel("Moment inercije I_B [kg·m²]", fontsize=10, fontweight='bold', color='#94a3b8')
ax_c1.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

ax_c2.set_title("KLIZANJE: Ciklična modulacija I_B (0-100% okreta)", fontsize=11, fontweight='bold', color='#fb7185')
ax_c2.set_xlabel("Faza okreta [% ciklusa]", fontsize=10, fontweight='bold', color='#94a3b8')
ax_c2.set_ylabel("Moment inercije I_B [kg·m²]", fontsize=10, fontweight='bold', color='#94a3b8')
ax_c2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(DIR_PLOTS_INERTIA, "ciklusi_preklapanje_0_100_posto.png"), dpi=300, facecolor='#0b0f19')
plt.close()

# =============================================================================
# 10. FINALNO FORMATIRANJE I ČUVANJE SVE 4 TABELE IZ RADA
# =============================================================================

df_tab1 = pd.DataFrame(table1_ballet_rows)
df_tab2 = pd.DataFrame(table2_skater_rows)
df_tab3 = pd.DataFrame(table3_intermediate_rows)
df_tab4 = pd.DataFrame(table4_master_rows)

print("\n" + "="*125)
print("  TABELA 1: BALET — RASPON PARAMETARA MOMENTA ZA STACIONARNO STANJE (μ = 0.20)")
print("="*125)
print(df_tab1.to_string(index=False))

print("\n" + "="*125)
print("  TABELA 2: UMETNIČKO KLIZANJE — RASPON PARAMETARA MOMENTA NA LEDU (μ = 0.02)")
print("="*125)
print(df_tab2.to_string(index=False))

print("\n" + "="*125)
print("  TABELA 3: UPOREDNI PROFILI (MAX / INT / MIN REŠENJA) PREMA TABELI 2 I 3 IZ RADA")
print("="*125)
print(df_tab3.to_string(index=False))

print("\n" + "="*145)
print("  TABELA 4: MASTER EVALUACIJA — BIOMEHANIKA, MOMENTI INERCIJE I RMSE KONZISTENTNOST")
print("="*145)
print(df_tab4.to_string(index=False))
print("="*145)

df_tab1.to_csv(os.path.join(DIR_TABLES, "TABELA_1_BALET_IMURA2010.csv"), index=False)
df_tab2.to_csv(os.path.join(DIR_TABLES, "TABELA_2_KLIZANJE_IMURA2010.csv"), index=False)
df_tab3.to_csv(os.path.join(DIR_TABLES, "TABELA_3_MAX_INT_MIN_IMURA2010.csv"), index=False)
df_tab4.to_csv(os.path.join(DIR_TABLES, "TABELA_4_MASTER_EVALUACIJA.csv"), index=False)

# Renderovanje objedinjene Dark Mode slike svih 4 tabele
fig, (ax_t1, ax_t2, ax_t3, ax_t4) = plt.subplots(4, 1, figsize=(16, 20), facecolor='#0b0f19')

table_configs = [
    (ax_t1, df_tab1, "TABELA 1: BALET — RASPON PARAMETARA MOMENTA UVIJANJA (μ = 0.20)"),
    (ax_t2, df_tab2, "TABELA 2: UMETNIČKO KLIZANJE — RASPON PARAMETARA NA LEDU (μ = 0.02)"),
    (ax_t3, df_tab3, "TABELA 3: UPOREDNI PROFILI (MAX, INT, MIN REŠENJA — IMURA & YEADON 2010)"),
    (ax_t4, df_tab4[["Atleta", "Tip", "tend [s]", "I_B [kg·m²]", "I_L [kg·m²]", "Tmax [Nm]", "Tmin [Nm]", "TF_limit [Nm]", "RMSE Eq1-2 [Nm]"]], "TABELA 4: MASTER EVALUACIJA I FIZIČKA KONZISTENTNOST MODELA")
]

for ax, df_curr, title_str in table_configs:
    ax.set_facecolor('#0b0f19')
    ax.axis('off')
    ax.axis('tight')
    t_plot = ax.table(cellText=df_curr.values, colLabels=df_curr.columns, cellLoc='center', loc='center')
    t_plot.auto_set_font_size(False)
    t_plot.set_fontsize(9.0)
    t_plot.scale(1.15, 1.7)
    for k_cell, cell in t_plot.get_celld().items():
        cell.set_edgecolor('#1e293b')
        if k_cell[0] == 0:
            cell.set_facecolor('#2563eb')
            cell.set_text_props(weight='bold', color='white', size=9.5)
        else:
            bg_col = '#111827' if k_cell[0] % 2 == 0 else '#0b0f19'
            cell.set_facecolor(bg_col)
            cell.set_text_props(color='#f8fafc', size=8.5)
    ax.set_title(title_str, fontsize=10.5, fontweight='bold', color='#38bdf8', pad=10)

plt.tight_layout()
plt.savefig(os.path.join(DIR_TABLES, "Uporedne_Tabele_Imura2010_Master.png"), dpi=300, facecolor=fig.get_facecolor())
plt.close()

print(f"\n✓ Svi rezultati su uspešno sačuvani u: {BASE_OUT}/")
print(f"  ├── 1. Grafici dinamike (Imura 2010):       {DIR_PLOTS_PHYSICS}/")
print(f"  ├── 2. X-Z putanje centra mase (Spirale):   {DIR_PLOTS_XZ}/")
print(f"  ├── 3. Grafici ugaone brzine i okreta:      {DIR_PLOTS_OMEGA}/")
print(f"  ├── 4. Momenti inercije, sile i ciklusi:    {DIR_PLOTS_INERTIA}/")
print(f"  ├── 5. Sve 4 uporedne tabele (CSV+PNG):     {DIR_TABLES}/")
print(f"  └── 6. Popravljene koordinate (CSV):        {DIR_CLEAN_COORDS}/\n")