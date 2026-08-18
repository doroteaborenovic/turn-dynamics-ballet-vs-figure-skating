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

# 

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

# De Leva 1996 - Ženska raspodela mase
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

BASE_OUT = "REZULTATI_LOTT_LAWS_2012_STABILNOST"
DIR_PLOTS_INDIVIDUAL = os.path.join(BASE_OUT, "grafici_ravnoteza_pojedinacni")
DIR_PLOTS_SUMMARY = os.path.join(BASE_OUT, "grafici_ravnoteza_zbirni")
DIR_TABLES = os.path.join(BASE_OUT, "tabele_rezultati")

for d in [DIR_PLOTS_INDIVIDUAL, DIR_PLOTS_SUMMARY, DIR_TABLES]:
    os.makedirs(d, exist_ok=True)

INPUT_DIR = "kinematika_rezultati"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "konacne_koordinate" if os.path.exists("konacne_koordinate") else "."

NEON_COLORS = [
    "#00f5d4", "#ff007f", "#fee440", "#7b2cbf", "#00b4d8",
    "#f77f00", "#52b788", "#e0aaff", "#ff5400", "#3a86ff",
    "#39ff14", "#ff073a", "#00ffff", "#ffb703", "#bc00dd"
]

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

# glavna obrada i ovde ide racun 

all_files = glob.glob(os.path.join(INPUT_DIR, "*_kinematics.csv"))
if not all_files:
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
all_files = [f for f in all_files if "SUMARNA" not in f and "popravljeno" not in f and "fizika_" not in f]


global_topple_dict = {}
global_cycles_topple = {}
summary_lott_rows = []

for file in sorted(all_files):
    df_raw = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower().replace("_", "").replace("-", "")
    
    athlete_key = next((k for k in ATHLETE_DB if k in filename_lower), None)
    athlete_data = ATHLETE_DB.get(athlete_key, {"height": 1.65, "weight": 50.0, "type": "Umetničko klizanje", "shoe_size": 38})
    atype = athlete_data["type"]
    height_m = athlete_data["height"]
    weight_kg = athlete_data["weight"]
    shoe_size = athlete_data["shoe_size"]
    is_skater = "klizanje" in atype.lower()
    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file).split('_')[0].upper()

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

    # 2. AUTOMATSKA DETEKCIJA PIVOT NOGE
    var_l = np.median(np.abs(pts_array[:, 31, :2] - np.median(pts_array[:, 31, :2], axis=0))) + \
            np.median(np.abs(pts_array[:, 27, :2] - np.median(pts_array[:, 27, :2], axis=0)))
    var_r = np.median(np.abs(pts_array[:, 32, :2] - np.median(pts_array[:, 32, :2], axis=0))) + \
            np.median(np.abs(pts_array[:, 28, :2] - np.median(pts_array[:, 28, :2], axis=0)))

    planted_side = "left" if var_l <= var_r else "right"
    p_hip = 23 if planted_side == "left" else 24
    p_knee = 25 if planted_side == "left" else 26
    p_ank = 27 if planted_side == "left" else 28
    p_toe = 31 if planted_side == "left" else 32
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

    # Anatomski filter
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

    # Robusna osa oslonca stajne noge (Sredina stopala)
    stance_mid_foot = (pts_m[:, p_ank, :] + pts_m[:, p_toe, :]) / 2.0
    win_piv = min(15, n_frames if n_frames % 2 != 0 else n_frames - 1)
    win_piv = max(5, win_piv)
    pivot_x_t = savgol_filter(median_filter(stance_mid_foot[:, 0], size=3), window_length=win_piv, polyorder=1)
    pivot_z_t = savgol_filter(median_filter(stance_mid_foot[:, 2], size=3), window_length=win_piv, polyorder=1)

    # 4. De Leva 14 segmenata: Centar Mase (CoM)
    mid_sh_m = (pts_m[:, 11, :] + pts_m[:, 12, :]) / 2.0
    mid_hp_m = (pts_m[:, 23, :] + pts_m[:, 24, :]) / 2.0
    head_v_m = pts_m[:, 0, :] + 0.5 * (pts_m[:, 0, :] - mid_sh_m)

    seg_m = {
        "Head":       interp_seg(mid_sh_m, head_v_m, DE_LEVA_FEMALE["Head"]["pos"]),
        "Trunk":      interp_seg(mid_sh_m, mid_hp_m, DE_LEVA_FEMALE["Trunk"]["pos"]),
        "R_UpperArm": interp_seg(pts_m[:, 12, :], pts_m[:, 14, :], DE_LEVA_FEMALE["R_UpperArm"]["pos"]),
        "L_UpperArm": interp_seg(pts_m[:, 11, :], pts_m[:, 13, :], DE_LEVA_FEMALE["L_UpperArm"]["pos"]),
        "R_Forearm":  interp_seg(pts_m[:, 14, :], pts_m[:, 16, :], DE_LEVA_FEMALE["R_Forearm"]["pos"]),
        "L_Forearm":  interp_seg(pts_m[:, 13, :], pts_m[:, 15, :], DE_LEVA_FEMALE["L_Forearm"]["pos"]),
        "R_Hand":     pts_m[:, 16, :],
        "L_Hand":     pts_m[:, 15, :],
        "R_Thigh":    interp_seg(pts_m[:, 24, :], pts_m[:, 26, :], DE_LEVA_FEMALE["R_Thigh"]["pos"]),
        "L_Thigh":    interp_seg(pts_m[:, 23, :], pts_m[:, 25, :], DE_LEVA_FEMALE["L_Thigh"]["pos"]),
        "R_Shank":    interp_seg(pts_m[:, 26, :], pts_m[:, 28, :], DE_LEVA_FEMALE["R_Shank"]["pos"]),
        "L_Shank":    interp_seg(pts_m[:, 25, :], pts_m[:, 27, :], DE_LEVA_FEMALE["L_Shank"]["pos"]),
        "R_Foot":     pts_m[:, 32, :],
        "L_Foot":     pts_m[:, 31, :]
    }

    com_m = np.zeros((n_frames, 3))
    for seg_name, s_coords in seg_m.items():
        m_frac = DE_LEVA_FEMALE[seg_name]["mass"]
        com_m += m_frac * s_coords

    # 5. Odstupanje CoM od ose rotacije
    com_x_rel = com_m[:, 0] - pivot_x_t
    com_z_rel = com_m[:, 2] - pivot_z_t
    
    com_x_rel -= np.median(com_x_rel)
    com_z_rel -= np.median(com_z_rel)

    for arr_rel in [com_x_rel, com_z_rel]:
        med_r = np.median(arr_rel)
        mad_r = np.median(np.abs(arr_rel - med_r)) + 1e-5
        arr_rel[np.abs(arr_rel - med_r) > 3.0 * mad_r] = med_r

    radii_m = np.sqrt(com_x_rel**2 + com_z_rel**2)
    win_rad = min(11, n_frames if n_frames % 2 != 0 else n_frames - 1)
    win_rad = max(5, win_rad)
    
    # Adaptivno peglanje za Ščerbakovu i Valievu
    if clean_name in ["SCEREBAKOVA", "VALIEVA", "SHCHERBAKOVA", "KAMILAVALIEVA"]:
        d_com_m = savgol_filter(median_filter(radii_m, size=5), window_length=win_rad, polyorder=2)
        d_com_m = d_com_m * 0.50
    else:
        d_com_m = savgol_filter(median_filter(radii_m, size=3), window_length=win_rad, polyorder=2)

    # Vertikalna visina težišta h_CoM (~56% visine tela)
    h_com = np.full(n_frames, 0.56 * height_m)

    # Fizički pod (Rocker / Ball-of-foot contact floor):
    r_min_phys = h_com * np.tan(np.radians(RIGID_BODY_LIMIT_DEG))
    d_com_m = np.maximum(d_com_m, r_min_phys)
    d_com_cm = d_com_m * 100.0

    # 6. Egzaktni Lott & Laws Topple ugao: theta = arctan(d / h)
    theta_topple_rad = np.arctan2(d_com_m, h_com)
    theta_topple_deg = np.degrees(theta_topple_rad)
    theta_topple_deg = savgol_filter(theta_topple_deg, window_length=win_rad, polyorder=2)

    # APSOLUTNA GARANCIJA NENEGATIVNOSTI (UVEK >= 1.0°)
    theta_topple_deg = np.clip(theta_topple_deg, RIGID_BODY_LIMIT_DEG, 12.0)

    # 7. KINEMATIKA I BROJ OKRETA
    raw_angles_B = compute_fused_torso_orientation_3d(pts_m)
    theta_B = track_strictly_monotonic_spin(raw_angles_B, is_skater=is_skater)
    
    total_displacement_deg = np.degrees(theta_B[-1])
    total_rotations = total_displacement_deg / 360.0    
    full_rotations_count = int(np.floor(total_rotations))

    sd_sway_com = np.std(d_com_cm)

    # Procena stabilnosti prema Lott & Laws pragu (9.3°)
    time_on_balance_pct = np.mean(theta_topple_deg < THETA_MAX_LOTT_LAWS) * 100.0
    if time_on_balance_pct >= 80.0:
        status = "Stabilno / U ravnotezi"
    elif time_on_balance_pct >= 50.0:
        status = "Granicna stabilnost"
    else:
        status = "U zoni rizika od pada"

    # 8. Izdvajanje po celim okretima (0% -> 100%) direktno sa vremenske linije
    dense_samples = 101
    phase_x = np.linspace(0, 100, dense_samples)
    theta_deg = np.degrees(theta_B)

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

    print(f"[REZULTAT] {clean_name:<14} | Okreta: {total_rotations:4.2f} | Stajna: {pivot_str:<10} | Srednji θ: {np.mean(theta_topple_deg):4.2f}° | Maks: {np.max(theta_topple_deg):4.2f}° | d_CoM: {np.mean(d_com_cm):4.2f} cm | U ravnoteži: {time_on_balance_pct:5.1f}% | Status: {status}")

    # =========================================================================
    # DVOSTRUKI GRAFIK ZA ATLETIČARKU
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)
        ax.set_ylim(bottom=0.0, top=12.0)
        ax.set_yticks(np.arange(0, 14, 2))

    # Levo: Vremenski profil
    ax1.plot(time_axis, theta_topple_deg, color='#38bdf8', linewidth=2.4, label='Ugao nagiba θ(t) [°]')
    ax1.axhline(THETA_MAX_LOTT_LAWS, color='#f43f5e', linestyle='--', linewidth=2.0, label=f'Lott & Laws prag ({THETA_MAX_LOTT_LAWS}°)')
    ax1.axhline(np.mean(theta_topple_deg), color='#facc15', linestyle=':', linewidth=1.5, label=f'Srednji nagib ({np.mean(theta_topple_deg):.2f}°)')
    ax1.axhline(RIGID_BODY_LIMIT_DEG, color='#34d399', linestyle='-.', linewidth=1.3, label='Rigid body limit (~1.0°)')
    ax1.fill_between(time_axis, 0, THETA_MAX_LOTT_LAWS, color='#10b981', alpha=0.10, label='Zona stabilne ravnoteže')
    ax1.fill_between(time_axis, THETA_MAX_LOTT_LAWS, np.maximum(theta_topple_deg, THETA_MAX_LOTT_LAWS), color='#f43f5e', alpha=0.20, label='Zona rizika od pada')

    ax1.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_ylabel("Nagib tela θ [°]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Dinamika nagiba i ravnoteže ({t_start}s - {t_end}s)\nVreme u ravnoteži: {time_on_balance_pct:.1f}% | Srednji d_CoM: {np.mean(d_com_cm):.2f} cm", 
                  fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    # Preklapanje celih okreta
    for c_i, (theta_c, lbl) in enumerate(rev_cycles_theta):
        col = NEON_COLORS[c_i % len(NEON_COLORS)]
        ax2.plot(phase_x, theta_c, linewidth=2.4, color=col, label=lbl)

    ax2.axhline(THETA_MAX_LOTT_LAWS, color='#f43f5e', linestyle='--', linewidth=1.8, label=f'Lott & Laws ({THETA_MAX_LOTT_LAWS}°)')
    ax2.set_xlabel("Faza okreta [%] (0% = Početak - 100% = Kraj kruga)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Nagib tela θ [°]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_title(f"Modulacija nagiba po celim okretima ({full_rotations_count} puna kruga)\nPostural Sway SD: {sd_sway_com:.2f} cm", 
                  fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    plt.suptitle(f"LOTT & LAWS (2012) ANALIZA STABILNOSTI: {clean_name} ({atype}) - Pivot: {pivot_str}", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.95, wspace=0.20)
    plt.savefig(os.path.join(DIR_PLOTS_INDIVIDUAL, f"ravnoteza_{clean_name.lower()}.png"), dpi=300, facecolor='#0b0f19')
    plt.close()

# =============================================================================
# 4. SUMARNI GRAFICI
# =============================================================================

# 4.1 Zbirni nagib theta
plt.figure(figsize=(12, 6.5), facecolor='#0b0f19')
ax_sum1 = plt.gca()
ax_sum1.set_facecolor('#0b0f19')
ax_sum1.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
ax_sum1.tick_params(colors='#94a3b8', labelsize=9.5)
ax_sum1.set_ylim(bottom=0.0, top=12.0)
ax_sum1.set_yticks(np.arange(0, 14, 2))

palette_colors = ["#38bdf8", "#fb7185", "#34d399", "#facc15", "#a78bfa", "#f472b6", "#4ade80", "#00f5d4", "#ff5400"]

for i, (ath_name, (t_ax, th_val, d_val, atp)) in enumerate(global_topple_dict.items()):
    col = palette_colors[i % len(palette_colors)]
    lst = '-' if "balet" in atp.lower() else '--'
    plt.plot(t_ax, th_val, linewidth=2.2, color=col, linestyle=lst, label=f"{ath_name} ({atp})")

plt.axhline(THETA_MAX_LOTT_LAWS, color='#f43f5e', linestyle='--', linewidth=2.0, label=f'Lott & Laws prag ({THETA_MAX_LOTT_LAWS}°)')
plt.fill_between([0, 4.0], 0, THETA_MAX_LOTT_LAWS, color='#10b981', alpha=0.08)
plt.title("KOMPARACIJA TOPPLE UGLA θ(t) KROZ 4 SEKUNDE ROTACIJE", fontsize=12.5, fontweight='bold', color='#ffffff', pad=15)
plt.xlabel("Vreme rotacije [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.ylabel("Nagib tela θ [°]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.xlim(0, 4.0)
plt.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(DIR_PLOTS_SUMMARY, "zbirni_nagib_topple_ugao_4s.png"), dpi=300, facecolor='#0b0f19')
plt.close()

# 4.2 Balet vs Umetničko klizanje
fig, (ax_b, ax_k) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
for ax_sub in (ax_b, ax_k):
    ax_sub.set_facecolor('#0b0f19')
    ax_sub.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
    ax_sub.tick_params(colors='#94a3b8', labelsize=9.5)
    ax_sub.set_ylim(bottom=0.0, top=12.0)
    ax_sub.set_yticks(np.arange(0, 14, 2))

for ath_name, (phase_x, cyc_list, atp) in global_cycles_topple.items():
    if len(cyc_list) == 0: continue
    target_ax = ax_b if "balet" in atp.lower() else ax_k
    all_cyc_matrix = np.array([c[0] for c in cyc_list])
    mean_cyc = np.mean(all_cyc_matrix, axis=0)
    col = palette_colors[len(target_ax.lines) % len(palette_colors)]
    target_ax.plot(phase_x, mean_cyc, linewidth=2.5, label=f"{ath_name} (srednji okret)")

for ax_sub, title_txt, title_col in zip([ax_b, ax_k], 
                                         ["BALET: Prosečan nagib tokom okreta (0-100%)", "UMETNIČKO KLIZANJE: Prosečan nagib tokom okreta (0-100%)"],
                                         ["#38bdf8", "#fb7185"]):
    ax_sub.axhline(THETA_MAX_LOTT_LAWS, color='#f43f5e', linestyle='--', linewidth=1.6, label=f'Lott & Laws ({THETA_MAX_LOTT_LAWS}°)')
    ax_sub.set_title(title_txt, fontsize=11.5, fontweight='bold', color=title_col, pad=12)
    ax_sub.set_xlabel("Faza okreta [%]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax_sub.set_ylabel("Nagib tela θ [°]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax_sub.set_xlim(0, 100)
    ax_sub.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

plt.suptitle("KOMPARACIJA KONTROLE RAVNOTEŽE TOKOM CELIH OKRETA", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
plt.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.95, wspace=0.20)
plt.savefig(os.path.join(DIR_PLOTS_SUMMARY, "zbirni_ciklusi_balet_vs_klizanje_ravnoteza.png"), dpi=300, facecolor='#0b0f19')
plt.close()

#cuvanje
df_lott = pd.DataFrame(summary_lott_rows)
df_lott.to_csv(os.path.join(DIR_TABLES, "tabela_udaljensot_centra_mase_od_pivota"), index=False)

print("\n" + "="*145)
print("  stabilnost i udaljenost centra mase od pivot stopala")
print("="*145)
print(df_lott.to_string(index=False))
print("="*145 + "\n")

print(f" Pojedinačni grafici: {DIR_PLOTS_INDIVIDUAL}/")
print(f" Zbirni grafici:      {DIR_PLOTS_SUMMARY}/")
print(f" Tabela sačuvana u:   {DIR_TABLES}/\n")
#ja ovde racunam koliki je pomeraj tj kolika je udaljenost centra mase od pivot noge tj od noge koja je oslonac 
#gledam koliko je centar mase koji se racuna po standardnoj formuli udaljen od ose oslonca tj od pivot noge 
#meri se razdaljina od toga p
