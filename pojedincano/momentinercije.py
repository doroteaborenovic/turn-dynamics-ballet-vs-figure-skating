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

# parametri i javno dostupne info o masi i visini 

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

BASE_OUT = "REZULTATI_MOMENT_INERCIJE"
DIR_PLOTS_INDIVIDUAL = os.path.join(BASE_OUT, "grafici_inercija_pojedinacni")
DIR_PLOTS_SUMMARY = os.path.join(BASE_OUT, "grafici_inercija_zbirni")
DIR_TABLES = os.path.join(BASE_OUT, "tabele_inercija")

for d in [DIR_PLOTS_INDIVIDUAL, DIR_PLOTS_SUMMARY, DIR_TABLES]:
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

# funkcijice za racunanje momenta inercije 

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

# obrada i radcun

all_files = glob.glob(os.path.join(INPUT_DIR, "*_kinematics.csv"))
if not all_files:
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
all_files = [f for f in all_files if "SUMARNA" not in f and "popravljeno" not in f and "fizika_" not in f]

print("\n" + "="*115)
print("moment inercije")
print("="*115 + "\n")

global_inertia_dict = {}
global_cycles_dict = {}
table_inertia_rows = []

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

    if "x_0" not in df_raw.columns and "X_0" not in df_raw.columns:
        continue

    #ide samo 4 skeunde da bi videi bili normalizovani 
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

    # pivot nona 
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

    # 4. Trenutna vertikalna osa rotacije u svakom frejmu (kroz stajno stopalo i trup)
    mid_sh_m = (pts_m[:, 11, :] + pts_m[:, 12, :]) / 2.0
    mid_hp_m = (pts_m[:, 23, :] + pts_m[:, 24, :]) / 2.0
    head_v_m = pts_m[:, 0, :] + 0.5 * (pts_m[:, 0, :] - mid_sh_m)

    axis_x_inst = 0.5 * (pts_m[:, p_ank, 0] + mid_hp_m[:, 0])
    axis_z_inst = 0.5 * (pts_m[:, p_ank, 2] + mid_hp_m[:, 2])
    
    win_smooth = min(13, n_frames if n_frames % 2 != 0 else n_frames - 1)
    win_smooth = max(5, win_smooth)
    axis_x = savgol_filter(axis_x_inst, window_length=win_smooth, polyorder=2)
    axis_z = savgol_filter(axis_z_inst, window_length=win_smooth, polyorder=2)

    # 5. Pozicije centara mase za svih 14 De Leva segmenata
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

    # 6. Dinamički proračun I_B(t) = sum(m_i * r_i^2)
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

    # Glatko filtriranje koje čuva prirodne dinamičke sinusoide okreta
    win_dyn = min(17, n_frames if n_frames % 2 != 0 else n_frames - 1)
    win_dyn = max(9, win_dyn)
    I_B = savgol_filter(I_B_raw, window_length=win_dyn, polyorder=2)
    
    # Zaštita od graničnih šumova na samom kraju: uvek striktno pozitivan broj!
    I_B = np.maximum(I_B, 0.40)

    # Moment inercije stajne noge I_L duž ose (Imura & Yeadon 2010)
    I_L = 0.085 * (weight_kg / 49.5) * (height_m / 1.65)**2

    # 7. Kinematika rotacije i usaglašen broj okreta
    raw_angles_B = compute_fused_torso_orientation_3d(pts_m)
    theta_B = track_strictly_monotonic_spin(raw_angles_B, is_skater=is_skater)
    
    total_displacement_deg = np.degrees(theta_B[-1])
    total_rotations = total_displacement_deg / 360.0
    full_rotations_count = int(np.floor(total_rotations))

    # 8. Izdvajanje po celim okretima (0% -> 100%) direktno sa vremenske linije
    dense_samples = 101
    phase_x = np.linspace(0, 100, dense_samples)
    theta_deg = np.degrees(theta_B)

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
    global_cycles_dict[clean_name] = (phase_x, rev_cycles_I, atype)

    # =========================================================================
    # DVOSTRUKI GRAFIK ZA ATLETIČARKU (LEVO: VREME 0-4s | DESNO: CELI OKRETI 0-100%)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)

    # Levo: Kontinualni vremenski profil
    ax1.plot(time_axis, I_B, color='#38bdf8', linewidth=2.4, label='Moment inercije tela I_B(t) [kg*m²]')
    ax1.axhline(np.mean(I_B), color='#34d399', linestyle='--', linewidth=1.6, 
                label=f'Srednji I_B ({np.mean(I_B):.2f} kg*m²)')
    ax1.axhline(np.max(I_B), color='#fb7185', linestyle=':', linewidth=1.4, 
                label=f'Maks I_B ({np.max(I_B):.2f} kg*m²)')
    ax1.axhline(np.min(I_B), color='#a78bfa', linestyle=':', linewidth=1.4, 
                label=f'Min I_B ({np.min(I_B):.2f} kg*m²)')
    ax1.axhline(I_L, color='#facc15', linestyle='-.', linewidth=1.3, 
                label=f'Stajna noga I_L ({I_L:.3f} kg*m²)')
    ax1.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_ylabel("Moment inercije I_B [kg*m²]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Dinamika inercije kroz vreme ({t_start}s - {t_end}s)\nUkupno rotacija: {total_rotations:.2f} | Pivot: {pivot_name}", 
                  fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    # Desno: Preklapanje celih okreta
    for c_i, (I_c, lbl) in enumerate(rev_cycles_I):
        col = NEON_COLORS[c_i % len(NEON_COLORS)]
        ax2.plot(phase_x, I_c, linewidth=2.4, color=col, label=lbl)

    ax2.set_xlabel("Faza okreta [%] (0% = Početak ◄ ► 100% = Završen pun krug)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Moment inercije I_B [kg*m²]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_title(f"Modulacija I_B po celim okretima ({full_rotations_count} puna kruga)\nCiklično otvaranje / zatvaranje radne noge i ruku", 
                  fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    plt.suptitle(f"MOMENT INERCIJE TELA: {clean_name} ({atype})", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.95, wspace=0.20)
    plt.savefig(os.path.join(DIR_PLOTS_INDIVIDUAL, f"inercija_{clean_name.lower()}.png"), dpi=300, facecolor='#0b0f19')
    plt.close()

    table_inertia_rows.append({
        "Atleta": clean_name,
        "Tip": atype,
        "Stajna noga": pivot_name,
        "Ukupno okreta": round(total_rotations, 2),
        "Prikazano celih": full_rotations_count,
        "I_B min [kg*m2]": round(np.min(I_B), 2),
        "I_B max [kg*m2]": round(np.max(I_B), 2),
        "I_B sr [kg*m2]": round(np.mean(I_B), 2),
        "I_L noga [kg*m2]": round(I_L, 3),
        "Modulacija Delta I [kg*m2]": round(np.max(I_B) - np.min(I_B), 2)
    })

    print(f"[OBRADA] {clean_name:<14} | Tip: {atype:<18} | Celih: {full_rotations_count:2d} (od {total_rotations:4.2f}) | I_B opseg: [{np.min(I_B):.2f} - {np.max(I_B):.2f}] kg*m² | I_L: {I_L:.3f} kg*m²")

# =============================================================================
# 4. ZAJEDNIČKI UPOREDNI GRAFICI MOMENTA INERCIJE
# =============================================================================

# 4.1 Uporedni grafik I_B(t) kroz vreme za sve atletičarke
plt.figure(figsize=(12, 6.5), facecolor='#0b0f19')
ax_sum1 = plt.gca()
ax_sum1.set_facecolor('#0b0f19')
ax_sum1.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
ax_sum1.tick_params(colors='#94a3b8', labelsize=9.5)

palette_colors = ["#38bdf8", "#fb7185", "#34d399", "#facc15", "#a78bfa", "#f472b6", "#4ade80", "#00f5d4", "#ff5400"]

for i, (name, (t_ax, i_val, i_leg, atp)) in enumerate(global_inertia_dict.items()):
    col = palette_colors[i % len(palette_colors)]
    lst = '-' if "balet" in atp.lower() else '--'
    plt.plot(t_ax, i_val, linewidth=2.2, color=col, linestyle=lst, label=f"{name} ({atp})")

plt.title("KOMPARACIJA MOMENTA INERCIJE TELA I_B(t) KROZ 4 SEKUNDE ROTACIJE", 
          fontsize=12.5, fontweight='bold', color='#ffffff', pad=15)
plt.xlabel("Vreme rotacije [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.ylabel("Moment inercije I_B [kg*m²]", fontsize=10.5, fontweight='bold', color='#94a3b8')
plt.xlim(0, 4.0)
plt.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(DIR_PLOTS_SUMMARY, "zbirni_moment_inercije_vreme_4s.png"), dpi=300, facecolor='#0b0f19')
plt.close()

# 4.2 Uporedni pregled po okretima (Balet vs Umetničko klizanje)
fig, (ax_b, ax_k) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
for ax_sub in (ax_b, ax_k):
    ax_sub.set_facecolor('#0b0f19')
    ax_sub.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
    ax_sub.tick_params(colors='#94a3b8', labelsize=9.5)

for name, (phase_x, cyc_list, atp) in global_cycles_dict.items():
    if len(cyc_list) == 0: continue
    target_ax = ax_b if "balet" in atp.lower() else ax_k
    
    all_cyc_matrix = np.array([c[0] for c in cyc_list])
    mean_cyc = np.mean(all_cyc_matrix, axis=0)
    
    col = palette_colors[len(target_ax.lines) % len(palette_colors)]
    target_ax.plot(phase_x, mean_cyc, linewidth=2.5, label=f"{name} (srednji okret)")

ax_b.set_title("BALET: Prosečna modulacija I_B tokom okreta (0-100%)", fontsize=11.5, fontweight='bold', color='#38bdf8', pad=12)
ax_b.set_xlabel("Faza okreta [%]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_b.set_ylabel("Moment inercije I_B [kg*m²]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_b.set_xlim(0, 100)
ax_b.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

ax_k.set_title("UMETNIČKO KLIZANJE: Prosečna modulacija I_B tokom okreta (0-100%)", fontsize=11.5, fontweight='bold', color='#fb7185', pad=12)
ax_k.set_xlabel("Faza okreta [%]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_k.set_ylabel("Moment inercije I_B [kg*m²]", fontsize=10.5, fontweight='bold', color='#94a3b8')
ax_k.set_xlim(0, 100)
ax_k.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

plt.suptitle("KOMPARACIJA CIKLIČNE MODULACIJE MOMENTA INERCIJE", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
plt.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.95, wspace=0.20)
plt.savefig(os.path.join(DIR_PLOTS_SUMMARY, "zbirni_ciklusi_balet_vs_klizanje.png"), dpi=300, facecolor='#0b0f19')
plt.close()

# =============================================================================
# 5. TABELARNI PRIKAZ I ČUVANJE REZULTATA
# =============================================================================

df_tab = pd.DataFrame(table_inertia_rows)
df_tab.to_csv(os.path.join(DIR_TABLES, "momentinercijetabela.csv"), index=False)

print("\n" + "="*135)
print(" tabela momenta inercije")
print("="*135)
print(df_tab.to_string(index=False))
print("="*135 + "\n")

print(f"✓ Pojedinačni grafici inercije (vreme + celi okreti): {DIR_PLOTS_INDIVIDUAL}/")
print(f"✓ Zbirni komparativni grafici inercije:               {DIR_PLOTS_SUMMARY}/")
print(f"✓ Tabela sa rezultatima:                             {DIR_TABLES}/\n")
