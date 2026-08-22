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

# parametri poznati i to 
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

# Balet (Serenade model): r_eff = 2/3 * 1.68 cm = 1.12 cm = 0.0112 m
# Klizanje (1 inch rocker): r_eff = 2/3 * 1.27 cm = 0.85 cm = 0.0085 m
ATHLETE_DB = {
    "marianela":    {"height": 1.74, "weight": 52.0, "type": "Balet", "shoe_size": 39, "mu": 0.30, "r_eff": 0.0112},
    "kapitonova":   {"height": 1.68, "weight": 48.0, "type": "Balet", "shoe_size": 38, "mu": 0.30, "r_eff": 0.0112},
    "khoreva":      {"height": 1.73, "weight": 47.0, "type": "Balet", "shoe_size": 39, "mu": 0.30, "r_eff": 0.0112},
    "trusova":      {"height": 1.66, "weight": 50.0, "type": "Umetničko klizanje", "shoe_size": 37, "mu": 0.006, "r_eff": 0.0085},
    "valieva":      {"height": 1.60, "weight": 44.0, "type": "Umetničko klizanje", "shoe_size": 36, "mu": 0.006, "r_eff": 0.0085},
    "kamilavalieva":{"height": 1.60, "weight": 44.0, "type": "Umetničko klizanje", "shoe_size": 36, "mu": 0.006, "r_eff": 0.0085},
    "shcherbakova": {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje", "shoe_size": 36, "mu": 0.006, "r_eff": 0.0085},
    "scerebakova":  {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje", "shoe_size": 36, "mu": 0.006, "r_eff": 0.0085},
    "liu":          {"height": 1.58, "weight": 45.0, "type": "Umetničko klizanje", "shoe_size": 36, "mu": 0.006, "r_eff": 0.0085}
}

# De Leva model segmenata tela (žene, 1996)
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
THETA_INITIAL_TOP_DEG = 1.5 # Početni nagib čigre [°]
T_SIMULATION_LONG = 650.0

BASE_OUT = "konacnirezultaticigra_trenje"
DIR_MASTER_5PANEL = os.path.join(BASE_OUT, "grafici_5panel_cigra_vs_osoba")
DIR_LONG_SIM      = os.path.join(BASE_OUT, "grafici_dugotrajna_simulacija_300s_5min")
DIR_TABLES        = os.path.join(BASE_OUT, "tabele_rezultati")

for d in [BASE_OUT, DIR_MASTER_5PANEL, DIR_LONG_SIM, DIR_TABLES]:
    os.makedirs(d, exist_ok=True)

INPUT_DIR = "kinematika_rezultati"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "skracene_koordinate" if os.path.exists("skracene_koordinate") else "konacne_koordinate"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "."

PALETTE_COLORS = ["#38bdf8", "#fb7185", "#34d399", "#facc15", "#a78bfa", "#f472b6", "#4ade80", "#00f5d4", "#ff5400"]

# funkcijice

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

# -------------------------------------------------------------------------
# EGZAKTNO REŠAVANJE ČIGRE KOVALJEVSKE SA MOMENTOM TRENJA
# -------------------------------------------------------------------------

def solve_kovalevskaya_exact_friction(theta_0_rad, omega_0_rad_s, J_phys, m_kg, mu_friction, r_contact, time_eval):
    tau_fric = mu_friction * m_kg * G_ACC * r_contact
    beta = tau_fric / max(0.1, (J_phys * omega_0_rad_s))

    n1_0 = np.sin(theta_0_rad)
    n2_0 = 0.0
    n3_0 = np.cos(theta_0_rad)

    m1_0 = 0.0
    m2_0 = 0.0
    m3_0 = 0.5 * omega_0_rad_s

    y0 = [m1_0, m2_0, m3_0, n1_0, n2_0, n3_0, 0.0]

    def deriv(t, y):
        m1, m2, m3, n1, n2, n3, _ = y
        dm1 = m2 * m3 - beta * m1
        dm2 = - (m3 * m1 + 0.5 * n3) - beta * m2
        dm3 = 0.5 * n2 - beta * m3

        dn1 = 2.0 * m3 * n2 - m2 * n3
        dn2 = m1 * n3 - 2.0 * m3 * n1
        dn3 = m2 * n1 - m1 * n2

        d_rot = 2.0 * m3
        return [dm1, dm2, dm3, dn1, dn2, dn3, d_rot]

    def fall_event(t, y):
        th = np.arccos(np.clip(y[5], -1.0, 1.0))
        return th - np.radians(THETA_MAX_LOTT_LAWS)
    fall_event.terminal = True
    fall_event.direction = 1

    sol = solve_ivp(deriv, (time_eval[0], time_eval[-1]), y0, t_eval=time_eval, events=fall_event, method='RK45', rtol=1e-7, atol=1e-7)

    t_sol = sol.t
    m1 = sol.y[0]
    m2 = sol.y[1]
    m3 = sol.y[2]
    n3 = np.clip(sol.y[5], -1.0, 1.0)
    rot_cum = sol.y[6] / (2.0 * np.pi)

    theta_deg_raw = np.degrees(np.arccos(n3))
    omega_top_rad_s_raw = np.sqrt(m1**2 + m2**2 + (2.0 * m3)**2)
    omega_top_deg_s_raw = np.degrees(omega_top_rad_s_raw)

    if len(t_sol) > 1:
        alpha_top_deg_s2_raw = np.gradient(omega_top_deg_s_raw, t_sol)
    else:
        alpha_top_deg_s2_raw = np.zeros_like(omega_top_deg_s_raw)

    L_top_raw = J_phys * omega_top_rad_s_raw
    E_k_top_raw = 0.5 * J_phys * (m1**2 + m2**2 + 2.0 * (m3**2))

    if len(sol.t_events[0]) > 0:
        has_fallen = True
        t_fall = float(sol.t_events[0][0])
        turns_fall = float(np.interp(t_fall, t_sol, rot_cum))
    else:
        has_fallen = False
        t_fall = float(time_eval[-1])
        turns_fall = float(rot_cum[-1])

    delta_E_k = float(E_k_top_raw[0] - E_k_top_raw[-1])

    def reindex_to_time_eval(arr):
        if not has_fallen or (len(t_sol) == len(time_eval) and np.allclose(t_sol, time_eval)):
            return arr
        out = np.full(len(time_eval), np.nan, dtype=float)
        mask = time_eval <= (t_fall + 1e-5)
        if np.any(mask):
            out[mask] = np.interp(time_eval[mask], t_sol, arr)
        return out

    theta_deg = reindex_to_time_eval(theta_deg_raw)
    omega_top_deg_s = reindex_to_time_eval(omega_top_deg_s_raw)
    alpha_top_deg_s2 = reindex_to_time_eval(alpha_top_deg_s2_raw)
    L_top = reindex_to_time_eval(L_top_raw)
    E_k_top = reindex_to_time_eval(E_k_top_raw)

    return theta_deg, omega_top_deg_s, alpha_top_deg_s2, L_top, E_k_top, t_fall, turns_fall, beta, delta_E_k, has_fallen, t_sol, theta_deg_raw, omega_top_deg_s_raw

# obrada i simulacija 

all_files = glob.glob(os.path.join(INPUT_DIR, "*_kinematics.csv"))
if not all_files:
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
all_files = [f for f in all_files if "SUMARNA" not in f and "popravljeno" not in f and "fizika_" not in f]

master_rows_4s = []
master_rows_300s = []
long_sim_data = {}
processed_names = set()

print("\n" + "="*145)
print(f"  POKRETANJE EVALUACIJE: ČIGRA KOVALJEVSKE SA TRENJEM (4s video + {T_SIMULATION_LONG:.0f}s DUGOTRAJNA SIMULACIJA / 5 MINUTA)")
print("="*145 + "\n")

for file in sorted(all_files):
    df_raw = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower().replace("_", "").replace("-", "")
    athlete_key = next((k for k in ATHLETE_DB if k in filename_lower), None)
    athlete_data = ATHLETE_DB.get(athlete_key, {"height": 1.65, "weight": 50.0, "type": "Balet", "shoe_size": 38, "mu": 0.30, "r_eff": 0.0112})
    atype = athlete_data["type"]
    height_m = athlete_data["height"]
    weight_kg = athlete_data["weight"]
    mu_val = athlete_data["mu"]
    r_eff_val = athlete_data["r_eff"]
    is_skater = "klizanje" in atype.lower()
    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file).split('_')[0].upper()

    if clean_name in processed_names or ("x_0" not in df_raw.columns and "X_0" not in df_raw.columns): 
        continue
    processed_names.add(clean_name)

    # Kropovanje intervala rotacije
    if "timestamp_sec" in df_raw.columns:
        t_arr = df_raw["timestamp_sec"].values
    elif "Time_s" in df_raw.columns:
        t_arr = df_raw["Time_s"].values
    else:
        t_arr = np.arange(len(df_raw)) / 30.0

    t_start, t_end = TIME_WINDOWS.get(athlete_key, (0.0, 4.0))
    video_duration = t_end - t_start
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    df_crop = df_raw[mask].copy().reset_index(drop=True)
    if len(df_crop) < 15:
        df_crop = df_raw.iloc[:120].copy().reset_index(drop=True)

    n_frames = len(df_crop)
    dt = video_duration / n_frames if n_frames > 0 else 1.0 / 30.0
    time_axis = np.arange(n_frames) * dt

    # 1. PCHIP filtriranje svih 33 markera
    pts_array = np.zeros((n_frames, 33, 3))
    for lm in range(33):
        vis_col = f"vis_{lm}" if f"vis_{lm}" in df_crop.columns else f"VIS_{lm}"
        vis_series = df_crop[vis_col].values if vis_col in df_crop.columns else None
        for ax_idx, ax_name in enumerate(['x', 'y', 'z']):
            col = f"{ax_name}_{lm}" if f"{ax_name}_{lm}" in df_crop.columns else f"{ax_name.upper()}_{lm}"
            if col in df_crop.columns:
                pts_array[:, lm, ax_idx] = clean_and_interpolate_signal(df_crop[col].values, vis=vis_series, vis_threshold=0.35)

    # 2. Detekcija pivot noge (stajne noge)
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

    # 4. Kinematika rotacije osobe (Ugaona brzina omega i ubrzanje alfa)
    raw_angles_B = compute_fused_torso_orientation_3d(pts_m)
    theta_B = track_strictly_monotonic_spin(raw_angles_B, is_skater=is_skater)
    
    total_displacement_deg = np.degrees(theta_B[-1])
    total_rotations_4s = total_displacement_deg / 360.0

    win_dyn_kin = max(11, min(25, n_frames if n_frames % 2 != 0 else n_frames - 1))
    omega_B = savgol_filter(theta_B, window_length=win_dyn_kin, polyorder=2, deriv=1, delta=dt)
    floor_rad_s = np.deg2rad(180.0 if is_skater else 120.0)
    omega_B = np.maximum(omega_B, floor_rad_s)
    omega_deg_s = np.degrees(omega_B)
    alpha_deg_s2 = savgol_filter(omega_deg_s, window_length=win_dyn_kin, polyorder=2, deriv=1, delta=dt)

    omega_0_rad_s = float(omega_B[0])

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

    # 6. Moment inercije tela (I_B)
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
    J_phys_0 = float(I_B[0])

    # Moment impulsa L i rotaciona kinetička energija E_k realne osobe
    L_rot_person = I_B * omega_B             # [kg*m^2/s]
    E_rot_person = 0.5 * I_B * (omega_B ** 2) # [J]

    # 7. Ugaoni nagib realne osobe (Lott & Laws 2012)
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

    theta_topple_rad = np.arctan2(d_com_m, h_com)
    theta_topple_deg = np.degrees(theta_topple_rad)
    theta_topple_deg = savgol_filter(theta_topple_deg, window_length=win_rad, polyorder=2)
    theta_topple_deg = np.clip(theta_topple_deg, RIGID_BODY_LIMIT_DEG, 12.0)

    # -------------------------------------------------------------------------
    # 8. EGZAKTNO REŠAVANJE ČIGRE (KRATAK HORIZONT - PROZOR VIDEA)
    # -------------------------------------------------------------------------
    theta_deg_top, omega_deg_s_top, alpha_deg_s2_top, L_top, E_k_top, t_fall_4s, turns_fall_4s, beta_calc, delta_E_k_4s, has_fallen_4s, _, _, _ = solve_kovalevskaya_exact_friction(
        theta_0_rad=np.radians(THETA_INITIAL_TOP_DEG),
        omega_0_rad_s=omega_0_rad_s,
        J_phys=J_phys_0,
        m_kg=weight_kg,
        mu_friction=mu_val,
        r_contact=r_eff_val,
        time_eval=time_axis
    )

    # -------------------------------------------------------------------------
    # 9. DUGOTRAJNA SIMULACIJA ČIGRE (HORIZONT: 300 SEKUNDI / 5 MINUTA)
    # -------------------------------------------------------------------------
    time_eval_long = np.linspace(0.0, T_SIMULATION_LONG, 6000)
    _, _, _, _, _, t_fall_300s, turns_fall_300s, _, delta_E_k_300s, has_fallen_300s, t_sol_long, th_raw_long, w_raw_long = solve_kovalevskaya_exact_friction(
        theta_0_rad=np.radians(THETA_INITIAL_TOP_DEG),
        omega_0_rad_s=omega_0_rad_s,
        J_phys=J_phys_0,
        m_kg=weight_kg,
        mu_friction=mu_val,
        r_contact=r_eff_val,
        time_eval=time_eval_long
    )

    long_sim_data[clean_name] = {
        "type": atype, "t_sol": t_sol_long, "theta": th_raw_long, "omega": w_raw_long,
        "t_fall": t_fall_300s, "turns": turns_fall_300s, "has_fallen": has_fallen_300s,
        "mu": mu_val, "beta": beta_calc, "delta_Ek": delta_E_k_300s
    }

    # Grafik komparacije (5 panela)
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), dpi=300, facecolor='#0b0f19', sharex=True)
    for ax in axes:
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8')

    status_4s_str = f"Pad čigre u {t_fall_4s:.2f}s (prelazak {THETA_MAX_LOTT_LAWS}°)" if has_fallen_4s else f"Stabilno (> {video_duration:.1f}s bez pada)"

    axes[0].plot(time_axis, theta_deg_top, color='#38bdf8', linewidth=2.4, label=rf'Čigra Kovaljevske ($\mu={mu_val}$, $r_{{eff}}={r_eff_val*100:.2f}\text{{cm}}$, $\beta={beta_calc:.4f}$) $\theta_{{top}}(t)$ [°]')
    axes[0].plot(time_axis, theta_topple_deg, color='#fb7185', linewidth=2.0, linestyle='--', label=f'Realna osoba ({clean_name}) $\\theta_{{person}}(t)$ [°]')
    axes[0].axhline(THETA_MAX_LOTT_LAWS, color='#facc15', linestyle=':', linewidth=1.5, label=f'Lott & Laws prag pada ({THETA_MAX_LOTT_LAWS}°)')
    axes[0].fill_between(time_axis, 0, THETA_MAX_LOTT_LAWS, color='#10b981', alpha=0.06, label='Zona stabilnosti')
    axes[0].scatter([0], [THETA_INITIAL_TOP_DEG], color='#00ff88', s=70, zorder=6, label=f'Start: $\\theta_0 = {THETA_INITIAL_TOP_DEG:.1f}^\\circ$')
    if has_fallen_4s:
        axes[0].scatter([t_fall_4s], [THETA_MAX_LOTT_LAWS], color='#ff0055', s=90, zorder=7, edgecolors='white', linewidth=1.5, label=f'Pad čigre ($t={t_fall_4s:.2f}$s)')
    axes[0].set_ylabel(r"Nagib $\theta$ [°]", fontsize=10, color='#94a3b8')
    axes[0].set_ylim(0, max(14.0, np.nanmax(theta_topple_deg) + 2.0))
    axes[0].legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.2)
    axes[0].set_title(f"DINAMIKA ROTACIJE I ČIGRA KOVALJEVSKE SA TRENJEM: {clean_name} ({atype.upper()})\n$r_{{eff}} = {r_eff_val*100:.2f}$ cm | Trenje: $\\mu = {mu_val}$ | {status_4s_str} | $\\omega_0 = {np.degrees(omega_0_rad_s):.1f}^\\circ$/s", fontsize=11.5, fontweight='bold', color='white', pad=10)

    axes[1].plot(time_axis, omega_deg_s_top, color='#38bdf8', linewidth=2.4, label=r'Čigra $\omega_{top}(t) = \sqrt{p^2+q^2+r^2}$ [°/s]')
    axes[1].plot(time_axis, omega_deg_s, color='#ff007f', linewidth=2.0, linestyle='--', label=r'Realna osoba $\omega_{person}(t)$ [°/s]')
    axes[1].scatter([0], [np.degrees(omega_0_rad_s)], color='#00ff88', s=70, zorder=6, label=f'Start: $\\omega_0 = {np.degrees(omega_0_rad_s):.1f}^\\circ$/s')
    axes[1].set_ylabel(r"Ugaona brzina $\omega$ [°/s]", fontsize=10, color='#94a3b8')
    axes[1].legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.2)

    axes[2].plot(time_axis, alpha_deg_s2_top, color='#38bdf8', linewidth=2.4, label=r'Čigra $\alpha_{top}(t)$ [°/s²]')
    axes[2].plot(time_axis, alpha_deg_s2, color='#a855f7', linewidth=2.0, linestyle='--', label=r'Realna osoba $\alpha_{person}(t)$ [°/s²]')
    axes[2].axhline(0, color='#64748b', linestyle=':', linewidth=1.0)
    axes[2].set_ylabel(r"Ubrzanje $\alpha$ [°/s²]", fontsize=10, color='#94a3b8')
    axes[2].legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.2)

    axes[3].plot(time_axis, L_top, color='#38bdf8', linewidth=2.4, label=r'Čigra $L_{top}(t)$ [kg·m²/s]')
    axes[3].plot(time_axis, L_rot_person, color='#34d399', linewidth=2.0, linestyle='--', label=r'Realna osoba $L_{person}(t) = I_B\omega$ [kg·m²/s]')
    axes[3].scatter([0], [L_rot_person[0]], color='#00ff88', s=70, zorder=6, label=f'Start: $L_0 = {L_rot_person[0]:.2f}$ kg·m²/s')
    axes[3].set_ylabel(r"Moment $L$ [kg·m²/s]", fontsize=10.5, color='#94a3b8')
    axes[3].legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.2)

    axes[4].plot(time_axis, E_k_top, color='#38bdf8', linewidth=2.4, label=r'Čigra $E_{k,top}(t)$ [J]')
    axes[4].plot(time_axis, E_rot_person, color='#f59e0b', linewidth=2.0, linestyle='--', label=r'Realna osoba $E_{k,person}(t) = \frac{1}{2}I_B\omega^2$ [J]')
    axes[4].scatter([0], [E_rot_person[0]], color='#00ff88', s=70, zorder=6, label=f'Start: $E_{{k,0}} = {E_rot_person[0]:.2f}$ J')
    axes[4].set_xlabel(r"Vreme $t$ [s]", fontsize=10.5, color='#94a3b8')
    axes[4].set_ylabel(r"Energija $E_k$ [J]", fontsize=10, color='#94a3b8')
    axes[4].set_xlim(0, video_duration)
    axes[4].legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.2)

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_MASTER_5PANEL, f"komparacija_trenje_5panel_{clean_name.lower()}.png"), facecolor='#0b0f19')
    plt.close()

    # Tabela 4s
    vreme_4s_str = f"{round(t_fall_4s, 2)} s" if has_fallen_4s else f"> {video_duration:.1f} s (Stabilno)"
    master_rows_4s.append({
        "Atleta": clean_name, "Grupa": atype, "μ trenje": mu_val, "r_eff [cm]": round(r_eff_val * 100, 2), "β koef": round(beta_calc, 4),
        "Realno (Video) [okr]": round(total_rotations_4s, 2),
        "Čigra 4s [okr]": round(turns_fall_4s, 2),
        "Pad čigre (4s prozor)": vreme_4s_str,
        "I_ukupno sr [kg*m2]": round(np.mean(I_B), 2),
        "Omega sr [°/s]": round(np.mean(omega_deg_s), 1),
        "Pad energije čigre 4s [J]": round(delta_E_k_4s, 2)
    })

    # Tabela 300s (5 min)
    vreme_300s_str = f"{round(t_fall_300s, 2)} s" if has_fallen_300s else f"> {T_SIMULATION_LONG:.0f} s (Stabilno)"
    master_rows_300s.append({
        "Atleta": clean_name, "Grupa": atype, "μ trenje": mu_val, "r_eff [cm]": round(r_eff_val * 100, 2), "β koef": round(beta_calc, 4),
        "Omega_0 [°/s]": round(np.degrees(omega_0_rad_s), 1),
        "Vreme pada (do 300s / 5min) [s]": vreme_300s_str,
        "Ukupno okreta do pada": round(turns_fall_300s, 1),
        "Gubitak energije [J]": round(delta_E_k_300s, 2)
    })

    print(f"[OBRAĐENO] {clean_name:<14} | {atype:<18} | μ = {mu_val:<5} | 4s status: {status_4s_str:<28} | Vreme pada (5 min test): {vreme_300s_str}")

# Nagib i ispis za sve atlete 
plt.figure(figsize=(14, 7), dpi=300, facecolor='#0b0f19')
ax_long_th = plt.gca()
ax_long_th.set_facecolor('#0b0f19')
ax_long_th.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
ax_long_th.tick_params(colors='#94a3b8')

for idx, (name, data) in enumerate(long_sim_data.items()):
    col = PALETTE_COLORS[idx % len(PALETTE_COLORS)]
    lst = '-' if "balet" in data["type"].lower() else '--'
    lbl = f"{name} ({data['type']}) — Pad: {data['t_fall']:.1f}s ({data['turns']:.0f} okr)" if data["has_fallen"] else f"{name} ({data['type']}) — Stabilno >300s ({data['turns']:.0f} okr)"
    ax_long_th.plot(data["t_sol"], data["theta"], color=col, linestyle=lst, linewidth=2.2, label=lbl)
    if data["has_fallen"]:
        ax_long_th.scatter([data["t_fall"]], [THETA_MAX_LOTT_LAWS], color=col, s=75, edgecolors='white', zorder=6)

ax_long_th.axhline(THETA_MAX_LOTT_LAWS, color='#facc15', linestyle=':', linewidth=1.8, label=f'Lott & Laws prag pada ({THETA_MAX_LOTT_LAWS}°)')
ax_long_th.fill_between([0, T_SIMULATION_LONG], 0, THETA_MAX_LOTT_LAWS, color='#10b981', alpha=0.06, label='Zona stabilnosti')
ax_long_th.set_xlabel("vreme simulacije $t$ [s]", fontsize=11, fontweight='bold', color='#94a3b8')
ax_long_th.set_ylabel(r"nagib čigre $\theta_{top}(t)$ [°]", fontsize=11, fontweight='bold', color='#94a3b8')
ax_long_th.set_title(f"simulacija stabilnosti nagiba čigre", fontsize=13, fontweight='bold', color='white', pad=12)
ax_long_th.set_xlim(0, T_SIMULATION_LONG)
ax_long_th.set_ylim(0, 14.0)
ax_long_th.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(DIR_LONG_SIM, "dugotrajna_simulacija_300s_nagib_svi.png"), facecolor='#0b0f19')
plt.close()

# Ugaona brzina svih atleta do 650 sekundi 
plt.figure(figsize=(14, 7), dpi=300, facecolor='#0b0f19')
ax_long_w = plt.gca()
ax_long_w.set_facecolor('#0b0f19')
ax_long_w.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
ax_long_w.tick_params(colors='#94a3b8')

for idx, (name, data) in enumerate(long_sim_data.items()):
    col = PALETTE_COLORS[idx % len(PALETTE_COLORS)]
    lst = '-' if "balet" in data["type"].lower() else '--'
    ax_long_w.plot(data["t_sol"], data["omega"], color=col, linestyle=lst, linewidth=2.2, label=f"{name} ({data['type']}) — $\\omega_0={data['omega'][0]:.0f}^\\circ$/s")

ax_long_w.set_xlabel("Vreme simulacije $t$ [s]", fontsize=11, fontweight='bold', color='#94a3b8')
ax_long_w.set_ylabel(r"Ugaona brzina čigre $\omega_{top}(t)$ [°/s]", fontsize=11, fontweight='bold', color='#94a3b8')
ax_long_w.set_title(f"usporenje i smanjenje ugaone brzine", fontsize=13, fontweight='bold', color='white', pad=12)
ax_long_w.set_xlim(0, T_SIMULATION_LONG)
ax_long_w.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(DIR_LONG_SIM, "dugotrajna_simulacija_300s_ugaona_brzina_svi.png"), facecolor='#0b0f19')
plt.close()

df_master_4s = pd.DataFrame(master_rows_4s)
df_master_4s.to_csv(os.path.join(DIR_TABLES, "tabela_cigra_sa_trenjem_evaluacija_4s.csv"), index=False)

df_master_300s = pd.DataFrame(master_rows_300s)
df_master_300s.to_csv(os.path.join(DIR_TABLES, "tabela_cigra_dugotrajna_simulacija_300s_5min.csv"), index=False)

print("\n" + "="*155)
print(" Evaluacija kroz 4 sekunde i realni video")
print("="*155)
print(df_master_4s.to_string(index=False))

print("\n" + "="*155)
print(f"  Simulacija do {T_SIMULATION_LONG:.0f} sekundi")
print("="*155)
print(df_master_300s.to_string(index=False))
print("="*155 + "\n")

print(f" rezultati: '{BASE_OUT}/'")
print(f" - grafici: {DIR_MASTER_5PANEL}/")
print(f" - simulacijice:        {DIR_LONG_SIM}/")
print(f" - tabelice:          {DIR_TABLES}/\n")
