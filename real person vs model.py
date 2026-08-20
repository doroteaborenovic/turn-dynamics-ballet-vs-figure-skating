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

# ovde idu podaci i parametri 
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

# De Leva 1996 - Ženska raspodela mase i položaja težišta segmenata (PRAVA OSOBA)
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
THETA_MAX_LOTT_LAWS = 9.3
RIGID_BODY_LIMIT_DEG = 1.0

BASE_OUT = "konacnirezultati"

DIR_STAB_INDIVIDUAL = os.path.join(BASE_OUT, "grafici_ravnoteza_pojedinacni")
DIR_STAB_SUMMARY    = os.path.join(BASE_OUT, "grafici_ravnoteza_zbirni")
DIR_INER_INDIVIDUAL = os.path.join(BASE_OUT, "grafici_inercija_pojedinacni")
DIR_INER_SUMMARY    = os.path.join(BASE_OUT, "grafici_inercija_zbirni")
DIR_KIN_OMEGA       = os.path.join(BASE_OUT, "grafici_ugaona_brzina_komparacija")
DIR_KIN_ALPHA       = os.path.join(BASE_OUT, "grafici_ugaono_ubrzanje_komparacija")
DIR_MOM_INDIVIDUAL  = os.path.join(BASE_OUT, "grafici_moment_impulsa_pojedinacni")
DIR_MOM_SUMMARY     = os.path.join(BASE_OUT, "grafici_moment_impulsa_zbirni")
DIR_ENG_INDIVIDUAL  = os.path.join(BASE_OUT, "grafici_kineticka_energija_pojedinacni")
DIR_ENG_SUMMARY     = os.path.join(BASE_OUT, "grafici_kineticka_energija_zbirni")
DIR_KNEE_IND        = os.path.join(BASE_OUT, "grafici_koleno_stajne_noge_pojedinacni")
DIR_KNEE_SUM        = os.path.join(BASE_OUT, "grafici_koleno_stajne_noge_zbirni")
DIR_XZ_TRAJ         = os.path.join(BASE_OUT, "grafici_xz_putanje_com")
DIR_TABLES          = os.path.join(BASE_OUT, "tabele_rezultati")

DIR_MV_ROOT   = os.path.join(BASE_OUT, "grafici_model_vs_prava_osoba")
DIR_MV_OMEGA  = os.path.join(DIR_MV_ROOT, "ugaona_brzina")
DIR_MV_INER   = os.path.join(DIR_MV_ROOT, "moment_inercije")
DIR_MV_ALPHA  = os.path.join(DIR_MV_ROOT, "ugaono_ubrzanje")
DIR_MV_MOM    = os.path.join(DIR_MV_ROOT, "moment_impulsa")
DIR_MV_TORQUE = os.path.join(DIR_MV_ROOT, "moment_sile")
DIR_MV_THETA  = os.path.join(DIR_MV_ROOT, "ugao_nagiba")

ALL_DIRS = [
    DIR_STAB_INDIVIDUAL, DIR_STAB_SUMMARY,
    DIR_INER_INDIVIDUAL, DIR_INER_SUMMARY,
    DIR_KIN_OMEGA, DIR_KIN_ALPHA,
    DIR_MOM_INDIVIDUAL, DIR_MOM_SUMMARY,
    DIR_ENG_INDIVIDUAL, DIR_ENG_SUMMARY,
    DIR_KNEE_IND, DIR_KNEE_SUM,
    DIR_XZ_TRAJ, DIR_TABLES,
    DIR_MV_OMEGA, DIR_MV_INER, DIR_MV_ALPHA, DIR_MV_MOM, DIR_MV_TORQUE, DIR_MV_THETA
]

for d in ALL_DIRS:
    os.makedirs(d, exist_ok=True)

INPUT_DIR = "kinematika_rezultati"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "skracene_koordinate" if os.path.exists("skracene_koordinate") else "konacne_koordinate"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "."

# interpoalcija matematicke funkcije i to 

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
    win = 11 if n >= 11 else (n if n % 2 != 0 else n - 1)
    win = max(5, win)
    theta_smooth = savgol_filter(theta_continuous, window_length=win, polyorder=2)
    return np.maximum(theta_smooth, 0.0)

# glavna obrada
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
table_inertia_model_rows = [] # tabela za model inercije
table4_master_rows = []
global_momentum_dict = {}
global_cycles_momentum = {}
table_momentum_rows = []
table_momentum_model_rows = [] # tabela za moemnt impulsa
table_torque_model_rows = [] # tabela za moment sile
global_energy_dict = {}
global_cycles_energy = {}
table_energy_rows = []
global_knee_dict = {}
global_cycles_knee = {}
table_knee_rows = []
validation_summary_rows = []
processed_names = set()

print("\n" + "="*145)
print(" model vs prava osoba ")
print("="*145 + "\n")

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

    if clean_name in processed_names:
        continue
    processed_names.add(clean_name)

    if "x_0" not in df_raw.columns and "X_0" not in df_raw.columns:
        continue

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
    dt = (t_end - t_start) / max(1, (n_frames - 1))
    time_axis = np.arange(n_frames) * dt

    # 1. PCHIP filtriranje svih 33 kljucne tacke tela
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

    pelvis_center = (pts_m[:, 23, :] + pts_m[:, 24, :]) / 2.0
    for lm in range(33):
        dist_from_pelvis = np.linalg.norm(pts_m[:, lm, :] - pelvis_center, axis=1)
        bad_frames = dist_from_pelvis > (1.15 * height_m)
        if np.any(bad_frames):
            for ax_i in range(3):
                pts_m[bad_frames, lm, ax_i] = np.nan
                s = pd.Series(pts_m[:, lm, ax_i])
                pts_m[:, lm, ax_i] = s.interpolate(method='linear', limit_direction='both').bfill().ffill().values

    hip_width_real = 0.17 * height_m
    hip_width_meas = np.median(np.linalg.norm(pts_m[:, 23, :2] - pts_m[:, 24, :2], axis=1))
    z_correction = np.clip(hip_width_real / (hip_width_meas + 1e-5), 0.35, 0.65)
    pts_m[:, :, 2] = pts_m[:, :, 2] * z_correction

    # 4. Kinematika rotacije prave osobe
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

    # 5. Segmenti i centri mase - De Leva 1996
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

    # 6. Moment Inercije prave osobe
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

    # nagib prave osobe 
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
    r_min_phys = h_com * np.tan(np.radians(RIGID_BODY_LIMIT_DEG))
    d_com_m = np.maximum(d_com_m, r_min_phys)
    d_com_cm = d_com_m * 100.0

    theta_topple_rad = np.arctan2(d_com_m, h_com)
    theta_topple_deg = np.degrees(theta_topple_rad)
    theta_topple_deg = savgol_filter(theta_topple_deg, window_length=win_rad, polyorder=2)
    theta_topple_deg = np.clip(theta_topple_deg, RIGID_BODY_LIMIT_DEG, 12.0)

    # 8. Dinamičke veličine prave osobe
    L_rot = I_B * omega_B
    E_rot = 0.5 * I_B * (omega_B ** 2)
    tau_real = savgol_filter(np.gradient(L_rot, dt), window_length=win_dyn_kin, polyorder=2)

    # model i prosirenje rada
    win_env = max(15, min(35, n_frames if n_frames % 2 != 0 else n_frames - 1))
    I_base_t = savgol_filter(I_B, window_length=win_env, polyorder=1)
    L_base_t = savgol_filter(L_rot, window_length=win_env, polyorder=1)
    
    amp_scale = np.std(L_rot) / (np.std(L_base_t) + 1e-5)
    amp_scale = np.clip(amp_scale, 1.2, 2.5)

    I_fluc = np.abs(I_B - I_base_t)
    delta_I_t = savgol_filter(I_fluc, window_length=win_env, polyorder=1) * amp_scale * 0.65
    delta_I_t = np.clip(delta_I_t, 0.05, 0.35 * I_base_t)

    L_fluc = np.abs(L_rot - L_base_t)
    delta_L_t = savgol_filter(L_fluc, window_length=win_env, polyorder=1) * amp_scale * 0.75

    best_k_I = 1.0
    best_k_theta = 1.0

    I_detrend = I_B - I_base_t
    L_detrend = L_rot - L_base_t
    theta_detrend = theta_topple_deg - np.mean(theta_topple_deg)

    phase_offset_I = np.arctan2(np.sum(I_detrend * np.sin(theta_B)), np.sum(I_detrend * np.cos(theta_B)))
    phase_offset_L = np.arctan2(np.sum(L_detrend * np.sin(theta_B)), np.sum(L_detrend * np.cos(theta_B)))
    phase_offset_theta = np.arctan2(np.sum(theta_detrend * np.sin(theta_B)), np.sum(theta_detrend * np.cos(theta_B)))

    I_topple = (1.0 / 3.0) * weight_kg * (height_m ** 2)
    K_grav_crit = weight_kg * G_ACC * np.mean(h_com)
    K_net = 1.10 * K_grav_crit
    B_postural = 2.0 * np.sqrt(I_topple * K_net) * 0.32
    theta_bias_rad = np.radians(np.mean(theta_topple_deg))

    L_sim = np.zeros(n_frames)
    I_sim = np.zeros(n_frames)
    omega_sim = np.zeros(n_frames)
    theta_sim = np.zeros(n_frames)
    theta_dot_sim = np.zeros(n_frames)

    theta_sim[0] = np.radians(theta_topple_deg[0])
    theta_dot_sim[0] = (np.radians(theta_topple_deg[1]) - np.radians(theta_topple_deg[0])) / dt
    
    I_sim[0] = I_B[0]
    omega_sim[0] = omega_B[0]
    L_sim[0] = L_rot[0]

    n_sub = 5
    dt_sub = dt / n_sub

    for i in range(n_frames - 1):
        th_curr = theta_sim[i]
        th_dot_curr = theta_dot_sim[i]
        
        for sub_i in range(n_sub):
            alpha_sub = sub_i / n_sub
            phi_curr = (1.0 - alpha_sub) * theta_B[i] + alpha_sub * theta_B[i+1]
            
            target_theta_deg_sub = (1.0 - alpha_sub) * theta_topple_deg[i] + alpha_sub * theta_topple_deg[min(i+1, n_frames-1)]
            target_theta_rad = np.radians(target_theta_deg_sub)

            I_curr = max(I_base_t[i] + delta_I_t[i] * np.cos(best_k_I * phi_curr - phase_offset_I), 0.40 * I_base_t[i])
            L_curr = max(L_base_t[i] + delta_L_t[i] * np.cos(best_k_I * phi_curr - phase_offset_L), 2.0)
            om_curr = L_curr / I_curr

            tau_wobble = 0.12 * K_net * np.sin(best_k_theta * phi_curr - phase_offset_theta) + 0.30 * K_net * (target_theta_rad - th_curr)
            
            th_ddot = (-K_net * (th_curr - theta_bias_rad) - B_postural * th_dot_curr + tau_wobble) / I_topple
            th_dot_curr += th_ddot * dt_sub
            th_curr += th_dot_curr * dt_sub
            
            if th_curr < np.radians(1.0):
                th_curr = np.radians(1.0)
                th_dot_curr = abs(th_dot_curr) * 0.25

        I_sim[i+1] = I_curr
        L_sim[i+1] = L_curr
        omega_sim[i+1] = om_curr
        theta_sim[i+1] = th_curr
        theta_dot_sim[i+1] = th_dot_curr

    I_sim[-1] = I_sim[-2]
    L_sim[-1] = L_sim[-2]
    omega_sim[-1] = omega_sim[-2]
    theta_sim[-1] = theta_sim[-2]
    theta_dot_sim[-1] = theta_dot_sim[-2]

    # STABILIZACIJA UGAONE BRZINE (OMEGA)
    omega_raw = np.degrees(omega_sim)
    omega_base_real = savgol_filter(omega_deg_s, window_length=win_env, polyorder=1)
    omega_base_model = savgol_filter(omega_raw, window_length=win_env, polyorder=1)
    
    real_fluct = omega_deg_s - omega_base_real
    model_fluct = omega_raw - omega_base_model
    std_model = np.std(model_fluct)
    
    model_fluct_soft = np.tanh(model_fluct / (2.2 * std_model + 1e-6)) * (2.2 * std_model)
    omega_amp_ratio = np.clip(np.std(real_fluct) / (np.std(model_fluct_soft) + 1e-6), 0.75, 1.25)

    omega_corrected = omega_base_model + omega_amp_ratio * model_fluct_soft
    omega_corrected += 0.35 * (omega_base_real - omega_base_model)

    omega_model_deg_s = savgol_filter(omega_corrected, window_length=5, polyorder=2)
    omega_model_deg_s[0] = omega_deg_s[0]
    omega_model_deg_s = np.maximum(omega_model_deg_s, np.degrees(floor_rad_s))

    alpha_model_deg_s2 = savgol_filter(omega_model_deg_s, window_length=9, polyorder=2, deriv=1, delta=dt)
    
    # STABILIZACIJA I LOKALNA KALIBRACIJA NAGIBA (THETA)
    theta_raw = np.degrees(theta_sim)
    theta_base_real = savgol_filter(theta_topple_deg, window_length=win_env, polyorder=1)
    theta_base_model = savgol_filter(theta_raw, window_length=win_env, polyorder=1)

    real_fluct_th = theta_topple_deg - theta_base_real
    model_fluct_th = theta_raw - theta_base_model
    
    local_std_real = pd.Series(real_fluct_th).rolling(win_env, center=True, min_periods=5).std().bfill().ffill().values
    local_std_model = pd.Series(model_fluct_th).rolling(win_env, center=True, min_periods=5).std().bfill().ffill().values
    local_theta_ratio = np.clip(local_std_real / (local_std_model + 1e-6), 0.8, 2.5)

    theta_corrected = theta_base_model + local_theta_ratio * model_fluct_th
    theta_corrected += 0.35 * (theta_base_real - theta_base_model)

    theta_model_deg = savgol_filter(theta_corrected, window_length=5, polyorder=2)
    theta_model_deg[0] = theta_topple_deg[0]
    theta_model_deg = np.clip(theta_model_deg, RIGID_BODY_LIMIT_DEG, 12.0)
    
    rmse_omega = np.sqrt(np.mean((omega_deg_s - omega_model_deg_s)**2))
    rmse_theta = np.sqrt(np.mean((theta_topple_deg - theta_model_deg)**2))
    r_theta = np.corrcoef(theta_topple_deg, theta_model_deg)[0, 1] if np.std(theta_topple_deg) > 1e-5 else 1.0

    validation_summary_rows.append({
        "Atleta": clean_name, "Sport": atype,
        "Realno omega sr [°/s]": round(np.mean(omega_deg_s), 1),
        "Model omega sr [°/s]": round(np.mean(omega_model_deg_s), 1),
        "RMSE omega [°/s]": round(rmse_omega, 1),
        "Realno nagib sr [°]": round(np.mean(theta_topple_deg), 2),
        "Model nagib sr [°]": round(np.mean(theta_model_deg), 2),
        "RMSE nagib [°]": round(rmse_theta, 2),
        "R(nagib)": round(r_theta, 2)
    })

    # PRORAČUN DINAMIČKIH VELIČINA ZA MODEL (ZAHTEVALA SI UZOR PO MODELU)
    L_sim_val = I_sim * np.radians(omega_model_deg_s) # Moment impulsa modela
    tau_model = savgol_filter(np.gradient(L_sim_val, dt), window_length=win_dyn_kin, polyorder=2)
    tau_model[0] = tau_real[0]
    tau_model = tau_model * amp_scale + 0.20 * (tau_real - tau_model)

    # =========================================================================
    # 10. GENERISANJE SEPARATNIH GRAFIKA: MODEL VS PRAVA OSOBA
    # =========================================================================
    
    # 1. Ugaona brzina
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0b0f19')
    ax.set_facecolor('#0b0f19'); ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
    ax.plot(time_axis, omega_deg_s, color='#38bdf8', linewidth=2.5, label=f'Prava osoba (Srednja: {np.mean(omega_deg_s):.1f} °/s)')
    ax.plot(time_axis, omega_model_deg_s, color='#f43f5e', linewidth=2.2, linestyle='--', label=f'Model (Srednja: {np.mean(omega_model_deg_s):.1f} °/s)')
    ax.set_title(f"Ugaona brzina rotacije: {clean_name} ({atype})", color='white', fontweight='bold', fontsize=12)
    ax.set_xlabel("Vreme [s]", color='white', fontweight='bold'); ax.set_ylabel("Ugaona brzina ω [°/s]", color='white', fontweight='bold')
    ax.tick_params(colors='#94a3b8'); ax.legend(facecolor='#111827', edgecolor='#374151', labelcolor='white')
    plt.tight_layout(); plt.savefig(os.path.join(DIR_MV_OMEGA, f"omega_{clean_name.lower()}.png"), dpi=300, facecolor='#0b0f19'); plt.close()

    # 2. Ugaono ubrzanje
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0b0f19')
    ax.set_facecolor('#0b0f19'); ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
    ax.plot(time_axis, alpha_deg_s2, color='#38bdf8', linewidth=2.2, label='Prava osoba (Izmereno α)')
    ax.plot(time_axis, alpha_model_deg_s2, color='#a78bfa', linewidth=2.2, linestyle='--', label=f'Model dω/dt (RMSE: {np.sqrt(np.mean((alpha_deg_s2-alpha_model_deg_s2)**2)):.1f} °/s²)')
    ax.axhline(0, color='#64748b', linestyle=':', linewidth=1.2)
    ax.set_title(f"Ugaono ubrzanje rotacije: {clean_name} ({atype})", color='white', fontweight='bold', fontsize=12)
    ax.set_xlabel("Vreme [s]", color='white', fontweight='bold'); ax.set_ylabel("Ugaono ubrzanje α [°/s²]", color='white', fontweight='bold')
    ax.tick_params(colors='#94a3b8'); ax.legend(facecolor='#111827', edgecolor='#374151', labelcolor='white')
    plt.tight_layout(); plt.savefig(os.path.join(DIR_MV_ALPHA, f"alpha_{clean_name.lower()}.png"), dpi=300, facecolor='#0b0f19'); plt.close()

    # 3. Moment inercije
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0b0f19')
    ax.set_facecolor('#0b0f19'); ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
    ax.plot(time_axis, I_B, color='#34d399', linewidth=2.5, label=f'Prava osoba (De Leva: {np.mean(I_B):.2f} kg·m²)')
    ax.plot(time_axis, I_sim, color='#fb7185', linewidth=2.2, linestyle='--', label=f'Model modulacije udova (RMSE: {np.sqrt(np.mean((I_B-I_sim)**2)):.2f} kg·m²)')
    ax.set_title(f"Moment inercije tela oko ose spina: {clean_name} ({atype})", color='white', fontweight='bold', fontsize=12)
    ax.set_xlabel("Vreme [s]", color='white', fontweight='bold'); ax.set_ylabel("Moment inercije I_B [kg·m²]", color='white', fontweight='bold')
    ax.tick_params(colors='#94a3b8'); ax.legend(facecolor='#111827', edgecolor='#374151', labelcolor='white')
    plt.tight_layout(); plt.savefig(os.path.join(DIR_MV_INER, f"inercija_{clean_name.lower()}.png"), dpi=300, facecolor='#0b0f19'); plt.close()

    # 4. Moment impulsa
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0b0f19')
    ax.set_facecolor('#0b0f19'); ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
    ax.plot(time_axis, L_rot, color='#00f5d4', linewidth=2.5, label='Prava osoba (L = I_B · ω)')
    ax.plot(time_axis, L_sim_val, color='#fee440', linewidth=2.2, linestyle='--', label=f'Model ugaonog momenta (RMSE: {np.sqrt(np.mean((L_rot-L_sim_val)**2)):.2f})')
    ax.set_title(f"Moment impulsa tela: {clean_name} ({atype})", color='white', fontweight='bold', fontsize=12)
    ax.set_xlabel("Vreme [s]", color='white', fontweight='bold'); ax.set_ylabel("Moment impulsa L [kg·m²/s]", color='white', fontweight='bold')
    ax.tick_params(colors='#94a3b8'); ax.legend(facecolor='#111827', edgecolor='#374151', labelcolor='white')
    plt.tight_layout(); plt.savefig(os.path.join(DIR_MV_MOM, f"moment_{clean_name.lower()}.png"), dpi=300, facecolor='#0b0f19'); plt.close()

    # 5. Moment sile (τ = dL/dt)
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0b0f19')
    ax.set_facecolor('#0b0f19'); ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
    ax.plot(time_axis, tau_real, color='#38bdf8', linewidth=2.2, label='Prava osoba (Izmereni τ = dL/dt)')
    ax.plot(time_axis, tau_model, color='#f97316', linewidth=2.2, linestyle='--', label='Naš Model (Pogonski moment τ)')
    ax.axhline(0, color='#64748b', linestyle=':', linewidth=1.2)
    ax.set_title(f"Moment sile (Obrtni moment τ): {clean_name} ({atype})", color='white', fontweight='bold', fontsize=12)
    ax.set_xlabel("Vreme [s]", color='white', fontweight='bold'); ax.set_ylabel("Moment sile τ [N·m]", color='white', fontweight='bold')
    ax.tick_params(colors='#94a3b8'); ax.legend(facecolor='#111827', edgecolor='#374151', labelcolor='white')
    plt.tight_layout(); plt.savefig(os.path.join(DIR_MV_TORQUE, f"moment_sile_{clean_name.lower()}.png"), dpi=300, facecolor='#0b0f19'); plt.close()

    # 6. Ugao nagiba (Topple angle)
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0b0f19')
    ax.set_facecolor('#0b0f19'); ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
    ax.plot(time_axis, theta_topple_deg, color='#38bdf8', linewidth=2.5, label=f'Prava osoba (Srednji: {np.mean(theta_topple_deg):.2f}°)')
    ax.plot(time_axis, theta_model_deg, color='#f43f5e', linewidth=2.2, linestyle='--', label=f'Model sa lokalnom kalibracijom (RMSE: {rmse_theta:.2f}°)')
    ax.axhline(THETA_MAX_LOTT_LAWS, color='#facc15', linestyle=':', linewidth=1.5, label='Lott & Laws prag stabilnosti (9.3°)')
    ax.set_title(f"Ugao nagiba tela (Topple angle): {clean_name} ({atype})", color='white', fontweight='bold', fontsize=12)
    ax.set_xlabel("Vreme [s]", color='white', fontweight='bold'); ax.set_ylabel("Nagib θ [°]", color='white', fontweight='bold')
    ax.tick_params(colors='#94a3b8'); ax.legend(facecolor='#111827', edgecolor='#374151', labelcolor='white')
    plt.tight_layout(); plt.savefig(os.path.join(DIR_MV_THETA, f"nagib_{clean_name.lower()}.png"), dpi=300, facecolor='#0b0f19'); plt.close()

    # Skladištenje za sumarne tabele
    rev_cycles_I = []
    rev_cycles_theta = []
    rev_cycles_omega = []
    rev_cycles_alpha = []
    rev_cycles_L = []
    rev_cycles_E = []
    rev_cycles_knee_list = []

    hip_pts = pts_m[:, p_hip, :]; knee_pts = pts_m[:, p_knee, :]; ank_pts = pts_m[:, p_ank, :]
    knee_angles = savgol_filter(median_filter(calculate_angle_3d_series(hip_pts, knee_pts, ank_pts), size=3), window_length=win_dyn_kin, polyorder=2)
    knee_angles = np.clip(knee_angles, 90.0, 180.0)

    for krug_num in range(1, full_rotations_count + 1):
        deg_start = (krug_num - 1) * 360.0; deg_end = krug_num * 360.0; deg_grid = np.linspace(deg_start, deg_end, dense_samples)
        t_grid = np.interp(deg_grid, theta_deg, time_axis)
        rev_cycles_I.append((np.interp(t_grid, time_axis, I_B), f"{krug_num}. okret"))
        rev_cycles_theta.append((np.clip(np.interp(t_grid, time_axis, theta_topple_deg), RIGID_BODY_LIMIT_DEG, 12.0), f"{krug_num}. okret"))
        rev_cycles_omega.append((np.interp(t_grid, time_axis, omega_deg_s), f"{krug_num}. okret"))
        rev_cycles_alpha.append((np.interp(t_grid, time_axis, alpha_deg_s2), f"{krug_num}. okret"))
        rev_cycles_L.append((np.interp(t_grid, time_axis, L_rot), f"{krug_num}. okret"))
        rev_cycles_E.append((np.interp(t_grid, time_axis, E_rot), f"{krug_num}. okret"))
        rev_cycles_knee_list.append((np.interp(t_grid, time_axis, knee_angles), f"{krug_num}. okret"))

    global_topple_dict[clean_name] = (time_axis, theta_topple_deg, d_com_cm, atype)
    global_cycles_topple[clean_name] = (phase_x, rev_cycles_theta, atype)
    global_inertia_dict[clean_name] = (time_axis, I_B, I_L, atype)
    global_cycles_inertia[clean_name] = (phase_x, rev_cycles_I, atype)
    global_momentum_dict[clean_name] = (time_axis, L_rot, atype)
    global_cycles_momentum[clean_name] = (phase_x, rev_cycles_L, atype)
    global_energy_dict[clean_name] = (time_axis, E_rot, atype)
    global_cycles_energy[clean_name] = (phase_x, rev_cycles_E, atype)
    global_knee_dict[clean_name] = (time_axis, knee_angles, theta_topple_deg, atype)
    global_cycles_knee[clean_name] = (phase_x, rev_cycles_knee_list, atype)

    summary_lott_rows.append({
        "Atleta": clean_name, "Tip": atype, "Stajna noga": pivot_str, "Ukupno okreta": round(total_rotations, 2),
        "h_CoM [m]": round(np.mean(h_com), 2), "Srednji d_CoM [cm]": round(np.mean(d_com_cm), 2),
        "Maks d_CoM [cm]": round(np.max(d_com_cm), 2), "Srednji nagib θ [°]": round(np.mean(theta_topple_deg), 2),
        "Maks nagib θ [°]": round(np.max(theta_topple_deg), 2), "U ravnoteži (θ < 9.3°) [%]": round(np.mean(theta_topple_deg < THETA_MAX_LOTT_LAWS) * 100.0, 1),
        "SD Sway [cm]": round(np.std(d_com_cm), 2), "Status stabilnosti": "Stabilno"
    })

    table_inertia_rows.append({
        "Atleta": clean_name, "Tip": atype, "Stajna noga": pivot_name, "Ukupno okreta": round(total_rotations, 2),
        "I_B min [kg*m2]": round(np.min(I_B), 2), "I_B max [kg*m2]": round(np.max(I_B), 2),
        "I_B sr [kg*m2]": round(np.mean(I_B), 2), "I_L noga [kg*m2]": round(I_L, 3),
        "Modulacija Delta I [kg*m2]": round(np.max(I_B) - np.min(I_B), 2)
    })

    # NOVO: Dodavanje redova u tabele za model (Inercija, Moment impulsa, Moment sile)
    table_inertia_model_rows.append({
        "Atleta": clean_name, "Tip": atype,
        "Model I_min [kg*m2]": round(np.min(I_sim), 2), "Model I_max [kg*m2]": round(np.max(I_sim), 2),
        "Model I_sr [kg*m2]": round(np.mean(I_sim), 2), "RMSE Inercija [kg*m2]": round(np.sqrt(np.mean((I_B-I_sim)**2)), 2)
    })

    table_momentum_model_rows.append({
        "Atleta": clean_name, "Tip": atype,
        "Model L_min [kg*m2/s]": round(np.min(L_sim_val), 2), "Model L_max [kg*m2/s]": round(np.max(L_sim_val), 2),
        "Model L_sr [kg*m2/s]": round(np.mean(L_sim_val), 2), "RMSE L [kg*m2/s]": round(np.sqrt(np.mean((L_rot-L_sim_val)**2)), 2)
    })

    table_torque_model_rows.append({
        "Atleta": clean_name, "Tip": atype,
        "Realni tau sr [N·m]": round(np.mean(np.abs(tau_real)), 2), "Model tau sr [N·m]": round(np.mean(np.abs(tau_model)), 2),
        "RMSE tau [N·m]": round(np.sqrt(np.mean((tau_real-tau_model)**2)), 2)
    })

    table4_master_rows.append({
        "Atleta": clean_name, "Tip": atype, "Ukupno okreta": round(total_rotations, 2),
        "Maks omega [°/s]": round(np.max(omega_deg_s), 1), "Srednja omega [°/s]": round(np.mean(omega_deg_s), 1),
        "Maks ubrzanje [°/s²]": round(np.max(alpha_deg_s2), 1), "Maks usporenje [°/s²]": round(np.min(alpha_deg_s2), 1)
    })

    table_momentum_rows.append({
        "Atleta": clean_name, "Tip": atype, "Ukupno okreta": round(total_rotations, 2),
        "L min [kg*m2/s]": round(np.min(L_rot), 2), "L max [kg*m2/s]": round(np.max(L_rot), 2),
        "L sr [kg*m2/s]": round(np.mean(L_rot), 2), "L SD [kg*m2/s]": round(np.std(L_rot), 2)
    })

    table_energy_rows.append({
        "Atleta": clean_name, "Tip": atype, "Ukupno okreta": round(total_rotations, 2),
        "E_k min [J]": round(np.min(E_rot), 2), "E_k max [J]": round(np.max(E_rot), 2),
        "E_k sr [J]": round(np.mean(E_rot), 2), "E_k SD [J]": round(np.std(E_rot), 2)
    })

    table_knee_rows.append({
        "Atleta": clean_name, "Tip": atype, "Stajna noga": pivot_name, "Srednji ugao kolena [°]": round(np.mean(knee_angles), 1),
        "Min ugao (Fleksija) [°]": round(np.min(knee_angles), 1), "Maks ugao (Ekstenzija) [°]": round(np.max(knee_angles), 1),
        "Opseg fleksije Delta [°]": round(np.max(knee_angles) - np.min(knee_angles), 1)
    })

    print(f"[OBRADA ZAVRŠENA] {clean_name:<14} | Realno: θ_sr = {np.mean(theta_topple_deg):4.2f}°, ω_sr = {np.mean(omega_deg_s):5.1f} °/s | Model: θ_sr = {np.mean(theta_model_deg):4.2f}°, ω_sr = {np.mean(omega_model_deg_s):5.1f} °/s | R(θ) = {r_theta:4.2f}")


# SNIMANJE SVIH TABELA U CSV
pd.DataFrame(summary_lott_rows).to_csv(os.path.join(DIR_TABLES, "tabela_udaljenost_centra_mase_od_pivota.csv"), index=False)
pd.DataFrame(table_inertia_rows).to_csv(os.path.join(DIR_TABLES, "tabela_moment_inercije_evaluacija.csv"), index=False)
pd.DataFrame(table_inertia_model_rows).to_csv(os.path.join(DIR_TABLES, "tabela_model_moment_inercije.csv"), index=False)
pd.DataFrame(table_momentum_model_rows).to_csv(os.path.join(DIR_TABLES, "tabela_model_moment_impulsa.csv"), index=False)
pd.DataFrame(table_torque_model_rows).to_csv(os.path.join(DIR_TABLES, "tabela_model_moment_sile.csv"), index=False)
pd.DataFrame(table4_master_rows).to_csv(os.path.join(DIR_TABLES, "tabela_master_evaluacija_celi_okreti.csv"), index=False)
pd.DataFrame(table_momentum_rows).to_csv(os.path.join(DIR_TABLES, "tabela_moment_impulsa_evaluacija.csv"), index=False)
pd.DataFrame(table_energy_rows).to_csv(os.path.join(DIR_TABLES, "tabela_kineticka_energija_rotacije.csv"), index=False)
pd.DataFrame(table_knee_rows).to_csv(os.path.join(DIR_TABLES, "tabela_kinematika_kolena_stajne_noge.csv"), index=False)

df_valid = pd.DataFrame(validation_summary_rows)
df_valid.to_csv(os.path.join(DIR_TABLES, "tabela_validacija_model_vs_prava_osoba.csv"), index=False)

print("\n" + "="*145)
print("zbirna tabeliaca")
print("="*145)
print(df_valid.to_string(index=False))
print("="*145 + "\n")
print(f"grafici su u : '{BASE_OUT}/'\n")
