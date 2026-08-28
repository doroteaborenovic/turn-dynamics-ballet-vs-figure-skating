import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.integrate import solve_ivp

warnings.filterwarnings('ignore')

# =============================================================================
# 1. PARAMETRI, BAZA ATLETA I LOTT & LAWS (2012) MODEL
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
    "marianela":     {"height": 1.74, "weight": 52.0, "type": "Balet"},
    "kapitonova":    {"height": 1.68, "weight": 48.0, "type": "Balet"},
    "khoreva":       {"height": 1.73, "weight": 47.0, "type": "Balet"},
    "trusova":       {"height": 1.66, "weight": 50.0, "type": "Umetničko klizanje"},
    "valieva":       {"height": 1.60, "weight": 44.0, "type": "Umetničko klizanje"},
    "kamilavalieva": {"height": 1.60, "weight": 44.0, "type": "Umetničko klizanje"},
    "shcherbakova":  {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje"},
    "scerebakova":   {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje"},
    "liu":           {"height": 1.58, "weight": 45.0, "type": "Umetničko klizanje"}
}

# De Leva 1996 - Ženska raspodela mase segmenata
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
RIGID_BODY_LIMIT_DEG = 1.0  # Limit krutog tela [°]

# Referentni parametri ženskog tela (Lott & Laws 2012)
REF_MASS_FEMALE = 45.0      # [kg]
REF_HEIGHT_FEMALE = 1.50    # [m]
REF_IX_FEMALE = 55.3        # [kg*m^2] (topple moment inercije oko vrha stopala)

THETA_MAX_LOTT_LAWS = 9.3   # Nominalni kritični prag pada [°]
THETA_MAX_MIN = 7.4         # Donja granica (9.3° - 1.9°) [°]
THETA_MAX_MAX = 11.2        # Gornja granica (9.3° + 1.9°) [°]

BASE_OUT = "rezultati_obrnuto_klatno"
DIR_ODE_INDIVIDUAL = os.path.join(BASE_OUT, "grafici_pojedinacni_ode")
DIR_ODE_SUMMARY    = os.path.join(BASE_OUT, "grafici_zbirni_ode")
DIR_TABLES         = os.path.join(BASE_OUT, "tabele_komparacija")

for d in [DIR_ODE_INDIVIDUAL, DIR_ODE_SUMMARY, DIR_TABLES]:
    os.makedirs(d, exist_ok=True)

INPUT_DIR = "kinematika_rezultati"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "skracene_koordinate" if os.path.exists("skracene_koordinate") else "konacne_koordinate"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "."

PALETTE_COLORS = ["#38bdf8", "#fb7185", "#34d399", "#facc15", "#a78bfa", "#f472b6", "#4ade80", "#00f5d4", "#ff5400"]

# =============================================================================
# 2. NUMERIČKE I KINEMATIČKE FUNKCIJE (IDENTIČNE GLAVNOM KODU)
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
# 3. GLAVNA OBRADA I INTEGRACIJA OBRNUTOG KLATNA (ODE)
# =============================================================================

all_files = glob.glob(os.path.join(INPUT_DIR, "*_kinematics.csv"))
if not all_files:
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
all_files = [f for f in all_files if "SUMARNA" not in f and "popravljeno" not in f and "fizika_" not in f]

global_ode_dict = {}
table_ode_rows = []
processed_names = set()

print("\n" + "="*125)
print("  POKRETANJE SINHRONIZOVANE ANALIZE MODELA OBRNUTOG KLATNA (LOTT & LAWS 2012)")
print("="*125 + "\n")

for file in sorted(all_files):
    df_raw = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower().replace("_", "").replace("-", "")
    
    athlete_key = next((k for k in ATHLETE_DB if k in filename_lower), None)
    athlete_data = ATHLETE_DB.get(athlete_key, {"height": 1.65, "weight": 50.0, "type": "Balet"})
    atype = athlete_data["type"]
    height_m = athlete_data["height"]
    weight_kg = athlete_data["weight"]
    is_skater = "klizanje" in atype.lower()
    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file).split('_')[0].upper()

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

    # 1. PCHIP i filtriranje svih 33 markera
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
    hip_width_real = 0.1484 * height_m
    hip_width_meas = np.median(np.linalg.norm(pts_m[:, 23, :2] - pts_m[:, 24, :2], axis=1))
    z_correction = np.clip(hip_width_real / (hip_width_meas + 1e-5), 0.35, 0.65)
    pts_m[:, :, 2] = pts_m[:, :, 2] * z_correction

    # 4. Kinematika rotacije i ugaona brzina
    raw_angles_B = compute_fused_torso_orientation_3d(pts_m)
    theta_B = track_strictly_monotonic_spin(raw_angles_B, is_skater=is_skater)
    total_rotations = np.degrees(theta_B[-1]) / 360.0

    win_kin = max(11, min(25, n_frames if n_frames % 2 != 0 else n_frames - 1))
    omega_B = savgol_filter(theta_B, window_length=win_kin, polyorder=2, deriv=1, delta=dt)
    
    floor_rad_s = np.deg2rad(150.0 if is_skater else 100.0)
    omega_B = np.maximum(omega_B, floor_rad_s)
    omega_deg_s = np.degrees(omega_B)
    
    mean_omega_deg_s = np.mean(omega_deg_s)
    mean_omega_rev_s = mean_omega_deg_s / 360.0

    # 5. Centar mase (De Leva 1996)
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

    # 6. Stabilnost i Topple ugao nagiba (potpuno usklađen)
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

    l_arm = 0.56 * height_m
    r_min_phys = l_arm * np.tan(np.radians(RIGID_BODY_LIMIT_DEG))
    d_com_m = np.maximum(d_com_m, r_min_phys)

    theta_topple_rad = np.arctan2(d_com_m, l_arm)
    theta_topple_deg = np.degrees(theta_topple_rad)
    theta_topple_deg = savgol_filter(theta_topple_deg, window_length=win_rad, polyorder=2)
    theta_topple_deg = np.clip(theta_topple_deg, RIGID_BODY_LIMIT_DEG, 12.0)

    # -------------------------------------------------------------------------
    # 7. INTEGRACIJA OBRNUTOG KLATNA (ODE)
    # -------------------------------------------------------------------------
    I_topple = REF_IX_FEMALE * (weight_kg / REF_MASS_FEMALE) * ((height_m / REF_HEIGHT_FEMALE) ** 2)
    theta_0_deg = float(theta_topple_deg[0])
    theta_0_rad = np.radians(theta_0_deg)

    def inverted_pendulum_ode(t, y):
        th, w = y
        if th >= np.pi / 2.0:
            return [0.0, 0.0]
        return [w, (weight_kg * G_ACC * l_arm / I_topple) * np.sin(th)]

    sol = solve_ivp(inverted_pendulum_ode, (time_axis[0], time_axis[-1]), [theta_0_rad, 0.0], t_eval=time_axis, method='RK45')
    theta_rigid_deg = np.degrees(sol.y[0])

    def get_time_to_angle(target_deg):
        idx = np.where(theta_rigid_deg >= target_deg)[0]
        return time_axis[idx[0]] if len(idx) > 0 else time_axis[-1]

    t_fall_nom = get_time_to_angle(THETA_MAX_LOTT_LAWS) # 9.3°
    t_fall_min = get_time_to_angle(THETA_MAX_MIN)       # 7.4°
    t_fall_max = get_time_to_angle(THETA_MAX_MAX)       # 11.2°

    n_ode_nom = t_fall_nom * mean_omega_rev_s
    n_ode_min = t_fall_min * mean_omega_rev_s
    n_ode_max = t_fall_max * mean_omega_rev_s
    n_tol = (n_ode_max - n_ode_min) / 2.0

    global_ode_dict[clean_name] = {
        "time": time_axis, "theta_real": theta_topple_deg, "theta_rigid": theta_rigid_deg,
        "type": atype, "I_topple": I_topple, "t_nom": t_fall_nom, "theta_0": theta_0_deg
    }

    table_ode_rows.append({
        "Sportista": clean_name, "Tip": atype, "theta_0 [deg]": theta_0_deg,
        "omega_rev_s": mean_omega_rev_s, "omega_deg_s": mean_omega_deg_s,
        "t_pad": t_fall_nom, "N_kruto": n_ode_nom, "N_tol": n_tol,
        "N_stvarno": total_rotations, "theta_sr": np.mean(theta_topple_deg)
    })

    # Pojedinačni grafik ODE
    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor='#0b0f19')
    ax.set_facecolor('#0b0f19')
    ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
    ax.plot(time_axis, theta_topple_deg, color='#00f5d4', linewidth=2.5, label='Stvarni sportista θ_real(t)')
    ax.plot(time_axis, theta_rigid_deg, color='#ff007f', linewidth=2.5, linestyle='--', label='Model krutog tela θ_rigid(t)')
    ax.axhline(THETA_MAX_LOTT_LAWS, color='#facc15', linestyle='--', linewidth=1.8, label=f'Nominalni prag ({THETA_MAX_LOTT_LAWS}°)')
    ax.fill_between(time_axis, 0, THETA_MAX_MIN, color='#10b981', alpha=0.08, label='Stabilna zona (< 7.4°)')
    ax.fill_between(time_axis, THETA_MAX_MIN, THETA_MAX_MAX, color='#f59e0b', alpha=0.15, label='Sumnjiva zona (7.4° - 11.2°)')
    ax.fill_between(time_axis, THETA_MAX_MAX, 90.0, color='#ef4444', alpha=0.10, label='Zona pada (> 11.2°)')

    if t_fall_nom < time_axis[-1]:
        ax.scatter([t_fall_nom], [THETA_MAX_LOTT_LAWS], color='#ff007f', s=80, zorder=5, label=f'Pad krutog tela (t = {t_fall_nom:.2f}s)')

    ax.set_ylim(0, max(15.0, min(45.0, np.max(theta_rigid_deg) + 2.0)))
    ax.set_xlim(0, time_axis[-1])
    ax.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax.set_ylabel("Ugao nagiba θ [°]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.set_title(f"OBRNUTO KLATNO: {clean_name} ({atype})\nKruto telo: {n_ode_nom:.2f}±{n_tol:.2f} okreta | Stvarno izvedeno: {total_rotations:.2f} okreta", fontsize=11, fontweight='bold', color='#ffffff', pad=12)
    ax.legend(loc='upper left', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.0)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_ODE_INDIVIDUAL, f"ode_{clean_name.lower()}.png"), dpi=300, facecolor='#0b0f19')
    plt.close()

    print(f"[OBRAĐENO] {clean_name:<14} | θ₀: {theta_0_deg:4.2f}° | ω: {mean_omega_deg_s:5.1f}°/s | t_pad: {t_fall_nom:4.2f}s | N_kruto: {n_ode_nom:4.2f} | N_stvarno: {total_rotations:4.2f}")

# =============================================================================
# 4. ZBIRNI KOMPARATIVNI GRAFIK (STVARNI LJUDI VS. OBRNUTO KLATNO)
# =============================================================================

fig, (ax_real, ax_ode) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor='#0b0f19')
for ax in (ax_real, ax_ode):
    ax.set_facecolor('#0b0f19')
    ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
    ax.tick_params(colors='#94a3b8', labelsize=9.5)
    ax.set_xlim(0, 4.0)

for i, (name, data) in enumerate(global_ode_dict.items()):
    col = PALETTE_COLORS[i % len(PALETTE_COLORS)]
    lst = '-' if "balet" in data["type"].lower() else '--'
    ax_real.plot(data["time"], data["theta_real"], color=col, linestyle=lst, linewidth=2.2, label=f"{name} ({data['type']})")
    ax_ode.plot(data["time"], data["theta_rigid"], color=col, linestyle=lst, linewidth=2.2, label=f"{name} (θ₀={data['theta_0']:.1f}°)")

ax_real.fill_between([0, 4.0], 0, THETA_MAX_MIN, color='#10b981', alpha=0.08)
ax_real.fill_between([0, 4.0], THETA_MAX_MIN, THETA_MAX_MAX, color='#f59e0b', alpha=0.15)
ax_real.fill_between([0, 4.0], THETA_MAX_MAX, 15.0, color='#ef4444', alpha=0.08)
ax_real.axhline(THETA_MAX_LOTT_LAWS, color='#facc15', linestyle='--', linewidth=1.6, label=f'Nominalni prag ({THETA_MAX_LOTT_LAWS}°)')
ax_real.set_ylim(0, 14.0)
ax_real.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_real.set_ylabel("Stvarni nagib θ [°]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_real.set_title("STVARNI SPORTISTI (AKTIVNA KONTROLA)\nKorekcija i održavanje unutar stabilne zone", fontsize=11, fontweight='bold', color='#38bdf8', pad=12)
ax_real.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=7.8)

ax_ode.fill_between([0, 4.0], 0, THETA_MAX_MIN, color='#10b981', alpha=0.08)
ax_ode.fill_between([0, 4.0], THETA_MAX_MIN, THETA_MAX_MAX, color='#f59e0b', alpha=0.15)
ax_ode.axhline(THETA_MAX_LOTT_LAWS, color='#facc15', linestyle='--', linewidth=1.6, label=f'Nominalni prag ({THETA_MAX_LOTT_LAWS}°)')
ax_ode.set_ylim(0, 45.0)
ax_ode.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_ode.set_ylabel("Teorijski nagib krutog tela θ [°]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_ode.set_title("MODEL KRUTOG TELA (OBRNUTO KLATNO — ODE)\nEksponencijalni toppling ka padu", fontsize=11, fontweight='bold', color='#fb7185', pad=12)
ax_ode.legend(loc='upper left', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=7.8)

plt.suptitle("EVALUACIJA OBRNUTOG KLATNA: REALNA KONTROLA VS. PASIVNO KRUTO TELO", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
plt.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.95, wspace=0.20)
plt.savefig(os.path.join(DIR_ODE_SUMMARY, "zbirna_komparacija_ode.png"), dpi=300, facecolor='#0b0f19')
plt.close()

# =============================================================================
# 5. KOMPARATIVNA TABELA: MOJI REZULTATI VS. LOTT & LAWS (2012)
# =============================================================================

comparison_rows = []

# 1. Dodavanje analiziranih sportista
for r in table_ode_rows:
    th0 = r["theta_0 [deg]"]
    if th0 <= 0.5:
        ref_str = "3.3 ± 0.2 okr (θ₀=0.1°, ω=1.7)"
    elif th0 <= 2.5:
        ref_str = "1.8 ± 0.2 okr (θ₀=1.0°, ω=1.7)"
    elif th0 <= 4.0:
        ref_str = "1.1 ± 0.1 okr (θ₀=3.0°, ω=1.7)"
    elif th0 <= 6.0:
        ref_str = "0.8 ± 0.1 okr (θ₀≈4.6°, ω=1.7)"
    else:
        ref_str = "0.0 ± 0.1 okr (θ₀>9.3°, ω=1.7)"

    comparison_rows.append({
        "Kategorija": r["Tip"],
        "Sportista / Model": r["Sportista"],
        "Početni nagib θ₀ [°]": f"{th0:.2f}°",
        "Ugaona brzina ω": f"{r['omega_rev_s']:.2f} rev/s ({r['omega_deg_s']:.1f}°/s)",
        "Vreme pada": f"{r['t_pad']:.2f} s",
        "model obrnutog klatna": f"{r['N_kruto']:.2f} ± {r['N_tol']:.2f}",
        "stvaran broj okreta": f"{r['N_stvarno']:.2f}",
        "Stvarni nagib osobe": f"{r['theta_sr']:.2f}°",
        "Lott & Laws Ref. (Tabela 3)": ref_str
    })

# 2. Referentni podaci iz rada Lott & Laws (2012)
comparison_rows.append({
    "Kategorija": "Lott & Laws (2012)",
    "Sportista / Model": "Ref Model (θ₀=0.1°)",
    "Početni nagib θ₀ [°]": "0.10°",
    "Ugaona brzina ω": "1.70 rev/s (612.0°/s)",
    "Vreme pada": "1.94 s",
    "model obrnutog klatna": "3.30 ± 0.20",
    "stvaran broj okreta": "— (Teorijski)",
    "Stvarni nagib osobe": "—",
    "Lott & Laws Ref. (Tabela 3)": "3.3 ± 0.2 (Referenca)"
})
comparison_rows.append({
    "Kategorija": "Lott & Laws (2012)",
    "Sportista / Model": "Ref Model (θ₀=1.0°)",
    "Početni nagib θ₀ [°]": "1.00°",
    "Ugaona brzina ω": "1.70 rev/s (612.0°/s)",
    "Vreme pada": "1.06 s",
    "model obrnutog klatna": "1.80 ± 0.20",
    "stvaran broj okreta": "— (Teorijski)",
    "Stvarni nagib osobe": "—",
    "Lott & Laws Ref. (Tabela 3)": "1.8 ± 0.2 (Referenca)"
})
comparison_rows.append({
    "Kategorija": "Lott & Laws (2012)",
    "Sportista / Model": "Ref Model (θ₀=3.0°)",
    "Početni nagib θ₀ [°]": "3.00°",
    "Ugaona brzina ω": "1.70 rev/s (612.0°/s)",
    "Vreme pada": "0.65 s",
    "model obrnutog klatna": "1.10 ± 0.10",
    "stvaran broj okreta": "— (Teorijski)",
    "Stvarni nagib osobe": "—",
    "Lott & Laws Ref. (Tabela 3)": "1.1 ± 0.1 (Referenca)"
})

df_comp = pd.DataFrame(comparison_rows)
df_comp.to_csv(os.path.join(DIR_TABLES, "tabela_trostruka_komparacija_balet_klizanje_lott_laws.csv"), index=False)

# Renderovanje slike tabele (.PNG)
fig, ax = plt.subplots(figsize=(19, 9.5), facecolor='#0b0f19')
ax.set_facecolor('#0b0f19')
ax.axis('off')

table_plot = ax.table(cellText=df_comp.values.tolist(), colLabels=list(df_comp.columns), loc='center', cellLoc='center')
table_plot.auto_set_font_size(False)
table_plot.set_fontsize(9.5)
table_plot.scale(1.0, 2.1)

for (row_idx, col_idx), cell in table_plot.get_celld().items():
    cell.set_edgecolor('#1e293b')
    cell.set_linewidth(1.2)
    if row_idx == 0:
        cell.set_facecolor('#1e293b')
        cell.set_text_props(color='#38bdf8', fontweight='bold', fontsize=10.0)
    else:
        kat = df_comp.values.tolist()[row_idx - 1][0]
        bg_col = '#0f172a' if "Balet" in kat else ('#1a1024' if "klizanje" in kat.lower() else '#181b16')
        cell.set_facecolor(bg_col)
        cell.set_text_props(color='#f8fafc', fontsize=9.2)

plt.title("KOMPARATIVNA TABELA: MOJI REZULTATI vs. LOTT & LAWS (2012) MODEL\n"
          "Poređenje početnog nagiba (θ₀), ugaone brzine (ω), modela obrnutog klatna i stvarnog broja okreta", 
          fontsize=12.5, fontweight='bold', color='#ffffff', pad=25)

plt.tight_layout()
table_png_path = os.path.join(DIR_TABLES, "tabela_komparacija_balet_klizanje_lott_laws.png")
plt.savefig(table_png_path, dpi=300, facecolor='#0b0f19', bbox_inches='tight')
plt.close()

# Prikaz tabele u konzoli
print("\n" + "="*160)
print("  FINALNA TABELA: MOJI REZULTATI (BALET / KLIZANJE) VS. LOTT & LAWS (2012)")
print("="*160)
print(df_comp.to_string(index=False))
print("="*160 + "\n")
print(f"✓ Pojedinačni ODE grafici:  {DIR_ODE_INDIVIDUAL}/")
print(f"✓ Zbirni ODE grafik:        {DIR_ODE_SUMMARY}/")
print(f"✓ CSV Tabela:               {os.path.join(DIR_TABLES, 'tabela_trostruka_komparacija_balet_klizanje_lott_laws.csv')}")
print(f"✓ PNG Slika tabele (300DPI): {table_png_path}\n")