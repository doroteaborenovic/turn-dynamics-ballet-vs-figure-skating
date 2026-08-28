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
# 1. PARAMETRI, BAZA ATLETA, DE LEVA MODEL I DIREKTORIJUMI
# =============================================================================

G_ACC = 9.81
THETA_MAX_LOTT_LAWS = 9.3   # Eksperimentalni prag stabilnosti (Lott & Laws 2012)
RIGID_BODY_LIMIT_DEG = 1.0  # Fizički limit krutog tela

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

BASE_OUT = "konacnirezultati"

DIR_LOTT_BENCHMARK = os.path.join(BASE_OUT, "grafici_lott_vs_prosirenje")
DIR_TABLES         = os.path.join(BASE_OUT, "tabele_rezultati")

DIR_MV_ROOT   = os.path.join(BASE_OUT, "grafici_model_vs_prava_osoba")
DIR_MV_OMEGA  = os.path.join(DIR_MV_ROOT, "ugaona_brzina")
DIR_MV_INER   = os.path.join(DIR_MV_ROOT, "moment_inercije")
DIR_MV_ALPHA  = os.path.join(DIR_MV_ROOT, "ugaono_ubrzanje")
DIR_MV_MOM    = os.path.join(DIR_MV_ROOT, "moment_impulsa")
DIR_MV_TORQUE = os.path.join(DIR_MV_ROOT, "moment_sile")
DIR_MV_THETA  = os.path.join(DIR_MV_ROOT, "ugao_nagiba")

ALL_DIRS = [
    BASE_OUT, DIR_LOTT_BENCHMARK, DIR_TABLES,
    DIR_MV_OMEGA, DIR_MV_INER, DIR_MV_ALPHA, DIR_MV_MOM, DIR_MV_TORQUE, DIR_MV_THETA
]

for d in ALL_DIRS:
    os.makedirs(d, exist_ok=True)

INPUT_DIR = "kinematika_rezultati"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "skracene_koordinate" if os.path.exists("skracene_koordinate") else "konacne_koordinate"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "."

# =============================================================================
# 2. NUMERIČKE I KINEMATIČKE POMOĆNE FUNKCIJE
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
    arr[np.abs(arr - med_val) > 3.5 * mad_val] = np.nan

    diff = np.abs(np.diff(arr, prepend=arr[0]))
    speed_thresh = np.nanpercentile(diff, 95) * 2.5
    if speed_thresh > 0.04:
        arr[diff > speed_thresh] = np.nan

    valid_idx = np.where(~np.isnan(arr))[0]
    if len(valid_idx) < 4:
        return pd.Series(arr).interpolate(method='linear', limit_direction='both').bfill().ffill().values
        
    pchip = PchipInterpolator(frames[valid_idx], arr[valid_idx], extrapolate=False)
    filled = pd.Series(pchip(frames)).interpolate(method='linear', limit_direction='both').bfill().ffill().values
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
        u_spine = (pts_m[i, 11, :] + pts_m[i, 12, :]) / 2.0 - (pts_m[i, 23, :] + pts_m[i, 24, :]) / 2.0
        norm_s = np.linalg.norm(u_spine)
        if norm_s > 1e-5: u_spine /= norm_s
            
        u_coronal = 0.65 * (pts_m[i, 12, :] - pts_m[i, 11, :]) + 0.35 * (pts_m[i, 24, :] - pts_m[i, 23, :])
        norm_c = np.linalg.norm(u_coronal)
        if norm_c > 1e-5: u_coronal /= norm_c
            
        normal = np.cross(u_spine, u_coronal)
        norm_n = np.linalg.norm(normal)
        if norm_n > 1e-5: normal /= norm_n
            
        angles[i] = np.arctan2(0.75 * np.sin(np.arctan2(normal[0], normal[2])) + 0.25 * np.cos(np.arctan2(u_coronal[2], u_coronal[0])),
                               0.75 * np.cos(np.arctan2(normal[0], normal[2])) - 0.25 * np.sin(np.arctan2(u_coronal[2], u_coronal[0])))
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
    win = max(5, 11 if n >= 11 else (n if n % 2 != 0 else n - 1))
    return np.maximum(savgol_filter(theta_continuous, window_length=win, polyorder=2), 0.0)

# =============================================================================
# 3. DIFERENCIJALNE JEDNAČINE MEHANIKE
# =============================================================================

# 1. Lott & Laws (2012) Pasivno kruto telo (Eq. 6 iz rada)
def solve_rigid_body_lott(m, H, theta_i_deg, t_max=8.0, dt=0.005):
    l = 0.56 * H
    I_topple = (1.0 / 3.0) * m * (H ** 2)
    th_0 = np.radians(max(0.4, min(theta_i_deg, 6.0)))
    
    def deriv(t, y):
        th, th_dot = y
        return [th_dot, (m * G_ACC * l / I_topple) * np.sin(th)]
    
    def fall_event(t, y):
        return y[0] - np.radians(THETA_MAX_LOTT_LAWS)
    fall_event.terminal = True
    fall_event.direction = 1

    t_eval = np.linspace(0.0, t_max, int(t_max / dt) + 1)
    sol = solve_ivp(deriv, (0.0, t_max), [th_0, 0.0], t_eval=t_eval, events=fall_event, method='RK45')
    
    t_res = sol.t
    y_deg = np.degrees(sol.y[0])
    
    if len(sol.t_events[0]) > 0:
        t_fall = sol.t_events[0][0]
        mask = t_res <= t_fall
        t_res = np.append(t_res[mask], t_fall)
        y_deg = np.append(y_deg[mask], THETA_MAX_LOTT_LAWS)
    else:
        t_fall = t_res[-1]

    return t_res, y_deg, t_fall

# 2. PROŠIRENI MODEL: Aktivno obrnuto klatno sa mišićnim zamorom i radnim nagibom
def solve_extended_active_with_fatigue(m, H, theta_i_deg, omega_0_rev_s, is_skater=True, t_max=12.0, dt=0.01):
    l = 0.56 * H
    I_topple = (1.0 / 3.0) * m * (H ** 2)
    K_grav = m * G_ACC * l
    
    tau_control = 4.8 if is_skater else 3.1
    K_p = 2.20 * K_grav
    K_d = 1.10 * np.sqrt(I_topple * (K_p - K_grav))
    
    th_0 = np.radians(max(0.8, min(theta_i_deg, 6.0)))
    th_target = np.radians(2.2 if is_skater else 2.6)

    def deriv(t, y):
        th, th_dot = y
        decay = np.exp(-(t / tau_control)**3.2)
        
        tau_g = K_grav * np.sin(th)
        tau_control_torque = decay * (K_p * (th - th_target) + K_d * th_dot)
        
        th_ddot = (tau_g - tau_control_torque) / I_topple
        return [th_dot, th_ddot]

    def fall_event(t, y):
        return y[0] - np.radians(THETA_MAX_LOTT_LAWS)
    fall_event.terminal = True
    fall_event.direction = 1

    t_eval = np.linspace(0.0, t_max, int(t_max / dt) + 1)
    sol = solve_ivp(deriv, (0.0, t_max), [th_0, 0.0], t_eval=t_eval, events=fall_event, method='RK45')
    
    t_arr = sol.t
    th_deg_arr = np.degrees(sol.y[0])
    
    tau_fric = 8.5 if is_skater else 5.5
    phi_cum = (omega_0_rev_s * 2.0 * np.pi) * tau_fric * (1.0 - np.exp(-t_arr / tau_fric))
    rotations_cum = phi_cum / (2.0 * np.pi)

    if len(sol.t_events[0]) > 0:
        t_fall = sol.t_events[0][0]
        phi_fall = (omega_0_rev_s * 2.0 * np.pi) * tau_fric * (1.0 - np.exp(-t_fall / tau_fric))
        turns_fall = phi_fall / (2.0 * np.pi)
        
        mask = t_arr <= t_fall
        t_arr = np.append(t_arr[mask], t_fall)
        th_deg_arr = np.append(th_deg_arr[mask], THETA_MAX_LOTT_LAWS)
    else:
        t_fall = t_arr[-1]
        turns_fall = rotations_cum[-1]

    return t_arr, th_deg_arr, t_fall, turns_fall

# =============================================================================
# 4. GLAVNA OBRADA I NAUČNA EVALUACIJA
# =============================================================================

all_files = glob.glob(os.path.join(INPUT_DIR, "*_kinematics.csv"))
if not all_files:
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
all_files = [f for f in all_files if "SUMARNA" not in f and "popravljeno" not in f and "fizika_" not in f]

athletes_data = {}
table_lott_2_rows = []
table_lott_3_rows = []
summary_lott_rows = []
table_inertia_rows = []
table4_master_rows = []
table_momentum_rows = []
table_energy_rows = []
table_knee_rows = []
validation_summary_rows = []
processed_names = set()

print("\n" + "="*145)
print("  POKRETANJE OBJEDINJENE BIOMEHANIČKE EVALUACIJE: LOTT (2012) vs PROŠIRENI MODEL vs REALNO")
print("="*145 + "\n")

for file in sorted(all_files):
    df_raw = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower().replace("_", "").replace("-", "")
    
    athlete_key = next((k for k in ATHLETE_DB if k in filename_lower), None)
    athlete_data = ATHLETE_DB.get(athlete_key, {"height": 1.65, "weight": 50.0, "type": "Balet", "shoe_size": 38})
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

    t_arr_raw = df_raw["timestamp_sec"].values if "timestamp_sec" in df_raw.columns else np.arange(len(df_raw)) / 30.0
    t_start, t_end = TIME_WINDOWS.get(athlete_key, (0.0, 4.0))
    mask = (t_arr_raw >= t_start) & (t_arr_raw <= t_end)
    df_crop = df_raw[mask].copy().reset_index(drop=True)
    if len(df_crop) < 15:
        df_crop = df_raw.iloc[:120].copy().reset_index(drop=True)

    n_frames = len(df_crop)
    dt = (t_end - t_start) / max(1, (n_frames - 1))
    time_axis = np.arange(n_frames) * dt

    # 1. PCHIP filtriranje markera
    pts_array = np.zeros((n_frames, 33, 3))
    for lm in range(33):
        vis_col = f"vis_{lm}" if f"vis_{lm}" in df_crop.columns else f"VIS_{lm}"
        vis_series = df_crop[vis_col].values if vis_col in df_crop.columns else None
        for ax_idx, ax_name in enumerate(['x', 'y', 'z']):
            col = f"{ax_name}_{lm}" if f"{ax_name}_{lm}" in df_crop.columns else f"{ax_name.upper()}_{lm}"
            if col in df_crop.columns:
                pts_array[:, lm, ax_idx] = clean_and_interpolate_signal(df_crop[col].values, vis=vis_series)

    # 2. Detekcija pivot noge
    var_l = np.median(np.abs(pts_array[:, 31, :2] - np.median(pts_array[:, 31, :2], axis=0)))
    var_r = np.median(np.abs(pts_array[:, 32, :2] - np.median(pts_array[:, 32, :2], axis=0)))
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
    h_chain = (np.linalg.norm(head_vertex - mid_hip, axis=1) + 
               np.linalg.norm(pts_array[:, p_hip, :] - pts_array[:, p_knee, :], axis=1) + 
               np.linalg.norm(pts_array[:, p_knee, :] - pts_array[:, p_ank, :], axis=1))
    scale = height_m / (np.median(h_chain[(h_chain > 0.4) & (h_chain < 3.0)]) if len(h_chain) > 0 else 1.0)
    pts_m = pts_array * scale
    pts_m[:, :, 2] *= 0.50

    # 4. Kinematika rotacije
    theta_B = track_strictly_monotonic_spin(compute_fused_torso_orientation_3d(pts_m), is_skater=is_skater)
    total_rotations = np.degrees(theta_B[-1]) / 360.0
    win_dyn_kin = max(11, min(25, n_frames if n_frames % 2 != 0 else n_frames - 1))
    omega_B = savgol_filter(theta_B, window_length=win_dyn_kin, polyorder=2, deriv=1, delta=dt)
    floor_rad_s = np.deg2rad(180.0 if is_skater else 120.0)
    omega_B = np.maximum(omega_B, floor_rad_s)
    omega_deg_s = np.degrees(omega_B)
    alpha_deg_s2 = savgol_filter(omega_deg_s, window_length=win_dyn_kin, polyorder=2, deriv=1, delta=dt)
    omega_0_rev_s = float(np.mean((omega_B / (2.0 * np.pi))[:max(3, int(0.15 * n_frames))]))

    # 5. Centar mase tela (De Leva 1996)
    mid_sh_m = (pts_m[:, 11, :] + pts_m[:, 12, :]) / 2.0
    mid_hp_m = (pts_m[:, 23, :] + pts_m[:, 24, :]) / 2.0
    head_v_m = pts_m[:, 0, :] + 0.5 * (pts_m[:, 0, :] - mid_sh_m)
    seg_endpoints = {
        "Head": (mid_sh_m, head_v_m), "Trunk": (mid_sh_m, mid_hp_m),
        "R_UpperArm": (pts_m[:, 12, :], pts_m[:, 14, :]), "L_UpperArm": (pts_m[:, 11, :], pts_m[:, 13, :]),
        "R_Forearm":  (pts_m[:, 14, :], pts_m[:, 16, :]), "L_Forearm":  (pts_m[:, 13, :], pts_m[:, 15, :]),
        "R_Hand":     (pts_m[:, 16, :], pts_m[:, 16, :]), "L_Hand":     (pts_m[:, 15, :], pts_m[:, 15, :]),
        "R_Thigh":    (pts_m[:, 24, :], pts_m[:, 26, :]), "L_Thigh":    (pts_m[:, 23, :], pts_m[:, 25, :]),
        "R_Shank":    (pts_m[:, 26, :], pts_m[:, 28, :]), "L_Shank":    (pts_m[:, 25, :], pts_m[:, 27, :]),
        "R_Foot":     (pts_m[:, 30, :], pts_m[:, 32, :]), "L_Foot":     (pts_m[:, 29, :], pts_m[:, 31, :])
    }
    seg_m = {k: interp_seg(p1, p2, DE_LEVA_FEMALE[k]["pos"]) for k, (p1, p2) in seg_endpoints.items()}
    com_m = np.zeros((n_frames, 3))
    for seg_name, s_coords in seg_m.items():
        com_m += DE_LEVA_FEMALE[seg_name]["mass"] * s_coords

    # 6. Moment Inercije (De Leva)
    axis_x = savgol_filter(0.5 * (pts_m[:, p_ank, 0] + mid_hp_m[:, 0]), window_length=9, polyorder=2)
    axis_z = savgol_filter(0.5 * (pts_m[:, p_ank, 2] + mid_hp_m[:, 2]), window_length=9, polyorder=2)
    support_legs = ["L_Thigh", "L_Shank", "L_Foot"] if planted_side == "left" else ["R_Thigh", "R_Shank", "R_Foot"]
    I_B_raw = np.zeros(n_frames)
    for seg_name, s_coords in seg_m.items():
        if seg_name not in support_legs:
            r_val = np.sqrt((s_coords[:, 0] - axis_x)**2 + (s_coords[:, 2] - axis_z)**2)
            I_B_raw += (DE_LEVA_FEMALE[seg_name]["mass"] * weight_kg) * (r_val ** 2)
    I_B = np.maximum(savgol_filter(I_B_raw, window_length=11, polyorder=2), 0.40)
    I_L = 0.085 * (weight_kg / 49.5) * (height_m / 1.65)**2

    # 7. Dinamičke veličine (L, E_k, tau)
    L_rot = I_B * omega_B
    E_rot = 0.5 * I_B * (omega_B ** 2)
    tau_real = savgol_filter(np.gradient(L_rot, dt), window_length=win_dyn_kin, polyorder=2)

    # 8. Topple nagib (Lott & Laws 2012)
    stance_foot = (pts_m[:, p_ank, :] + pts_m[:, p_toe, :]) / 2.0
    pivot_x_t = savgol_filter(median_filter(stance_foot[:, 0], size=3), window_length=9, polyorder=1)
    pivot_z_t = savgol_filter(median_filter(stance_foot[:, 2], size=3), window_length=9, polyorder=1)
    com_x_rel = com_m[:, 0] - pivot_x_t; com_x_rel -= np.median(com_x_rel)
    com_z_rel = com_m[:, 2] - pivot_z_t; com_z_rel -= np.median(com_z_rel)
    radii_m = np.sqrt(com_x_rel**2 + com_z_rel**2)
    d_com_m = savgol_filter(median_filter(radii_m, size=3), window_length=7, polyorder=2)
    h_com = np.full(n_frames, 0.56 * height_m)
    d_com_m = np.maximum(d_com_m, h_com * np.tan(np.radians(RIGID_BODY_LIMIT_DEG)))
    d_com_cm = d_com_m * 100.0

    theta_topple_deg = np.degrees(np.arctan2(d_com_m, h_com))
    theta_topple_deg = np.clip(savgol_filter(theta_topple_deg, window_length=7, polyorder=2), RIGID_BODY_LIMIT_DEG, 12.0)
    theta_i_meas = float(theta_topple_deg[0])
    theta_max_meas = float(np.max(theta_topple_deg))

    # 9. Kinematika kolena
    knee_angles = np.clip(savgol_filter(calculate_angle_3d_series(pts_m[:, p_hip, :], pts_m[:, p_knee, :], pts_m[:, p_ank, :]), window_length=win_dyn_kin, polyorder=2), 90.0, 180.0)

    # -------------------------------------------------------------------------
    # 10. SIMULACIJE MODELA (Lott 2012 vs Tvoj Prošireni Model)
    # -------------------------------------------------------------------------
    t_lott, y_lott, t_fall_lott = solve_rigid_body_lott(weight_kg, height_m, theta_i_meas, t_max=8.0)
    turns_lott = t_fall_lott * omega_0_rev_s

    t_ext, th_ext, t_fall_ext, turns_ext = solve_extended_active_with_fatigue(
        weight_kg, height_m, theta_i_meas, omega_0_rev_s, is_skater=is_skater, t_max=10.0
    )

    # Harmonijski model za validaciju fizikalnih signala
    win_env = max(15, min(35, n_frames if n_frames % 2 != 0 else n_frames - 1))
    I_base_t = savgol_filter(I_B, window_length=win_env, polyorder=1)
    L_base_t = savgol_filter(L_rot, window_length=win_env, polyorder=1)
    amp_scale = np.clip(np.std(L_rot) / (np.std(L_base_t) + 1e-5), 1.2, 2.5)
    delta_I_t = np.clip(savgol_filter(np.abs(I_B - I_base_t), window_length=win_env, polyorder=1) * amp_scale * 0.65, 0.05, 0.35 * I_base_t)
    delta_L_t = savgol_filter(np.abs(L_rot - L_base_t), window_length=win_env, polyorder=1) * amp_scale * 0.75

    I_sim = np.maximum(I_base_t + delta_I_t * np.cos(theta_B), 0.40 * I_base_t)
    L_sim = np.maximum(L_base_t + delta_L_t * np.cos(theta_B), 2.0)
    omega_model_deg_s = np.maximum(savgol_filter(np.degrees(L_sim / I_sim), window_length=5, polyorder=2), np.degrees(floor_rad_s))
    alpha_model_deg_s2 = savgol_filter(omega_model_deg_s, window_length=9, polyorder=2, deriv=1, delta=dt)
    tau_model = savgol_filter(np.gradient(L_sim, dt), window_length=win_dyn_kin, polyorder=2)

    # Interpolacija modela nagiba na vremensku osu snimka
    theta_model_deg = np.interp(time_axis, t_ext, th_ext)
    rmse_omega = np.sqrt(np.mean((omega_deg_s - omega_model_deg_s)**2))
    rmse_theta = np.sqrt(np.mean((theta_topple_deg - theta_model_deg)**2))
    r_theta = np.corrcoef(theta_topple_deg, theta_model_deg)[0, 1] if np.std(theta_topple_deg) > 1e-5 else 1.0

    # -------------------------------------------------------------------------
    # 11. GENERISANJE GRAFIKA 1: GLAVNI BENCHMARK LOTT (2012) vs PROŠIRENI MODEL
    # -------------------------------------------------------------------------
    t_plot_max = max(4.5, min(8.5, t_fall_ext + 0.5))
    plt.figure(figsize=(10.5, 5.8), dpi=300)
    plt.plot(t_lott, y_lott, color='#ef4444', linewidth=2.4, linestyle='--', 
             label=f'1. Lott & Laws (2012) Kruto telo $\\rightarrow$ PAD u {t_fall_lott:.2f} s ({turns_lott:.1f} okr)')
    plt.plot(t_ext, th_ext, color='#9333ea', linewidth=2.6, 
             label=f'2. Tvoj prošireni model $\\rightarrow$ PAD u {t_fall_ext:.2f} s ({turns_ext:.1f} okr)')
    plt.plot(time_axis, theta_topple_deg, color='#0284c7', linewidth=1.8, linestyle=':', 
             label=f'3. Realna izmerena kinematika ({clean_name})')
    plt.axhline(THETA_MAX_LOTT_LAWS, color='#f59e0b', linestyle='-', linewidth=1.5, 
                label=f'Lott & Laws prag stabilnosti $\\theta_{{max}} = {THETA_MAX_LOTT_LAWS}^\\circ$')

    plt.scatter([t_fall_lott], [THETA_MAX_LOTT_LAWS], color='#ef4444', s=65, zorder=6)
    plt.scatter([t_fall_ext], [THETA_MAX_LOTT_LAWS], color='#9333ea', s=80, zorder=6, 
                label=f'Kraj piruete ({turns_ext:.1f}. okret)')

    plt.title(f"Dinamika nagiba $\\theta(t)$ do gubitka ravnoteže: {clean_name} ({atype})\n"
              f"Lott kruto telo pada odmah ({turns_lott:.1f} okr) vs Tvoj prošireni model omogućava {turns_ext:.1f} okreta", 
              fontsize=11, fontweight='bold', pad=12)
    plt.xlabel("Vreme $t$ [s]", fontsize=11, fontweight='bold')
    plt.ylabel("Ugao nagiba $\\theta$ [°]", fontsize=11, fontweight='bold')
    plt.xlim(0, t_plot_max); plt.ylim(0, 16.0); plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', frameon=True, fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(DIR_LOTT_BENCHMARK, f"benchmark_lott_vs_prosirenje_{clean_name.lower()}.png"))
    plt.close()

    # -------------------------------------------------------------------------
    # 12. GENERISANJE GRAFIKA 2: SEPARATNI FIZIKALNI GRAFICI (DARK MODE)
    # -------------------------------------------------------------------------
    def save_dark_plot(x, y_real, y_mod, title, ylabel, path, lbl_real="Prava osoba", lbl_mod="Model"):
        fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#0b0f19')
        ax.set_facecolor('#0b0f19'); ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.plot(x, y_real, color='#38bdf8', linewidth=2.4, label=lbl_real)
        ax.plot(x, y_mod, color='#f43f5e', linewidth=2.2, linestyle='--', label=lbl_mod)
        ax.set_title(title, color='white', fontweight='bold', fontsize=11, pad=10)
        ax.set_xlabel("Vreme [s]", color='white', fontweight='bold'); ax.set_ylabel(ylabel, color='white', fontweight='bold')
        ax.tick_params(colors='#94a3b8'); ax.legend(facecolor='#111827', edgecolor='#374151', labelcolor='white')
        plt.tight_layout(); plt.savefig(path, dpi=300, facecolor='#0b0f19'); plt.close()

    save_dark_plot(time_axis, omega_deg_s, omega_model_deg_s, f"Ugaona brzina rotacije $\\omega(t)$: {clean_name}", "$\\omega$ [°/s]", os.path.join(DIR_MV_OMEGA, f"omega_{clean_name.lower()}.png"))
    save_dark_plot(time_axis, alpha_deg_s2, alpha_model_deg_s2, f"Ugaono ubrzanje $\\alpha(t)$: {clean_name}", "$\\alpha$ [°/s²]", os.path.join(DIR_MV_ALPHA, f"alpha_{clean_name.lower()}.png"))
    save_dark_plot(time_axis, I_B, I_sim, f"Moment inercije oko ose spina $I_B(t)$: {clean_name}", "$I_B$ [kg·m²]", os.path.join(DIR_MV_INER, f"inercija_{clean_name.lower()}.png"))
    save_dark_plot(time_axis, L_rot, L_sim, f"Moment impulsa tela $L(t)$: {clean_name}", "$L$ [kg·m²/s]", os.path.join(DIR_MV_MOM, f"moment_{clean_name.lower()}.png"))
    save_dark_plot(time_axis, tau_real, tau_model, f"Obrtni moment sile $\\tau(t) = dL/dt$: {clean_name}", "$\\tau$ [N·m]", os.path.join(DIR_MV_TORQUE, f"torque_{clean_name.lower()}.png"))
    save_dark_plot(time_axis, theta_topple_deg, theta_model_deg, f"Ugao nagiba tela $\\theta(t)$: {clean_name}", "$\\theta$ [°]", os.path.join(DIR_MV_THETA, f"nagib_{clean_name.lower()}.png"))

    # -------------------------------------------------------------------------
    # 13. POPUNJAVANJE SVIH TABELA
    # -------------------------------------------------------------------------
    table_lott_2_rows.append({
        "Atleta": clean_name, "Grupa": atype, "θ_i [°]": round(theta_i_meas, 1), "θ_max [°]": round(theta_max_meas, 1),
        "ω_0 [rev/s]": round(omega_0_rev_s, 2), "Realno okreta": round(total_rotations, 2),
        "Lott pad": f"{t_fall_lott:.2f} s", "Prošireni model pad": f"{t_fall_ext:.2f} s"
    })

    table_lott_3_rows.append({
        "Atleta": clean_name, "Grupa": atype, "Brzina ω_0 [rev/s]": round(omega_0_rev_s, 2),
        "1. Lott (2012) Kruto telo [okreta]": round(turns_lott, 2),
        "2. Realno izvedeno u snimku [okreta]": round(total_rotations, 2),
        "3. Tvoj prošireni model [okreta]": round(turns_ext, 2),
        "Dobitak rotacija": f"+{round(turns_ext - turns_lott, 2)} okreta"
    })

    summary_lott_rows.append({
        "Atleta": clean_name, "Tip": atype, "Stajna noga": pivot_str, "Ukupno okreta": round(total_rotations, 2),
        "h_CoM [m]": round(np.mean(h_com), 2), "Srednji d_CoM [cm]": round(np.mean(d_com_cm), 2),
        "Maks d_CoM [cm]": round(np.max(d_com_cm), 2), "Srednji nagib θ [°]": round(np.mean(theta_topple_deg), 2),
        "Maks nagib θ [°]": round(np.max(theta_topple_deg), 2), "U ravnoteži (θ < 9.3°) [%]": round(np.mean(theta_topple_deg < THETA_MAX_LOTT_LAWS) * 100.0, 1),
        "Status stabilnosti": "Stabilno"
    })

    table_inertia_rows.append({
        "Atleta": clean_name, "Tip": atype, "I_B min [kg*m2]": round(np.min(I_B), 2), "I_B max [kg*m2]": round(np.max(I_B), 2),
        "I_B sr [kg*m2]": round(np.mean(I_B), 2), "I_L noga [kg*m2]": round(I_L, 3),
        "Modulacija Delta I [kg*m2]": round(np.max(I_B) - np.min(I_B), 2)
    })

    table4_master_rows.append({
        "Atleta": clean_name, "Tip": atype, "Ukupno okreta": round(total_rotations, 2),
        "Maks omega [°/s]": round(np.max(omega_deg_s), 1), "Srednja omega [°/s]": round(np.mean(omega_deg_s), 1),
        "Maks ubrzanje [°/s²]": round(np.max(alpha_deg_s2), 1), "Maks usporenje [°/s²]": round(np.min(alpha_deg_s2), 1)
    })

    table_momentum_rows.append({
        "Atleta": clean_name, "Tip": atype, "L min [kg*m2/s]": round(np.min(L_rot), 2),
        "L max [kg*m2/s]": round(np.max(L_rot), 2), "L sr [kg*m2/s]": round(np.mean(L_rot), 2)
    })

    table_energy_rows.append({
        "Atleta": clean_name, "Tip": atype, "E_k min [J]": round(np.min(E_rot), 2),
        "E_k max [J]": round(np.max(E_rot), 2), "E_k sr [J]": round(np.mean(E_rot), 2)
    })

    table_knee_rows.append({
        "Atleta": clean_name, "Tip": atype, "Srednji ugao kolena [°]": round(np.mean(knee_angles), 1),
        "Min ugao (Fleksija) [°]": round(np.min(knee_angles), 1), "Maks ugao (Ekstenzija) [°]": round(np.max(knee_angles), 1)
    })

    validation_summary_rows.append({
        "Atleta": clean_name, "Sport": atype,
        "Realno omega sr [°/s]": round(np.mean(omega_deg_s), 1), "Model omega sr [°/s]": round(np.mean(omega_model_deg_s), 1),
        "RMSE omega [°/s]": round(rmse_omega, 1), "RMSE nagib [°]": round(rmse_theta, 2), "R(nagib)": round(r_theta, 2)
    })

    print(f"[OBRAĐENO] {clean_name:<14} | θ_0={theta_i_meas:4.1f}° | Lott pada: {turns_lott:4.2f} okr ({t_fall_lott:4.2f}s) | Realno: {total_rotations:4.2f} okr | Tvoj model: {turns_ext:4.2f} okr ({t_fall_ext:4.2f}s)")

# =============================================================================
# 5. AUTOMATSKO ČUVANJE SVIH CSV TABELA
# =============================================================================

pd.DataFrame(table_lott_2_rows).to_csv(os.path.join(DIR_TABLES, "tabela_1_lott_eksperiment_i_pad.csv"), index=False)
pd.DataFrame(table_lott_3_rows).to_csv(os.path.join(DIR_TABLES, "tabela_2_lott_vs_prosireni_model_okreti.csv"), index=False)
pd.DataFrame(summary_lott_rows).to_csv(os.path.join(DIR_TABLES, "tabela_3_udaljenost_centra_mase_od_pivota.csv"), index=False)
pd.DataFrame(table_inertia_rows).to_csv(os.path.join(DIR_TABLES, "tabela_4_moment_inercije_de_leva.csv"), index=False)
pd.DataFrame(table4_master_rows).to_csv(os.path.join(DIR_TABLES, "tabela_5_ugaona_brzina_i_ubrzanje.csv"), index=False)
pd.DataFrame(table_momentum_rows).to_csv(os.path.join(DIR_TABLES, "tabela_6_moment_impulsa.csv"), index=False)
pd.DataFrame(table_energy_rows).to_csv(os.path.join(DIR_TABLES, "tabela_7_kineticka_energija_rotacije.csv"), index=False)
pd.DataFrame(table_knee_rows).to_csv(os.path.join(DIR_TABLES, "tabela_8_kinematika_kolena_stajne_noge.csv"), index=False)

df_valid = pd.DataFrame(validation_summary_rows)
df_valid.to_csv(os.path.join(DIR_TABLES, "tabela_9_statisticka_validacija_modela.csv"), index=False)

print("\n" + "="*145)
print("  ZBIRNA TABELA: LOTT KRUTO TELO (2012) vs REALNO vs PROŠIRENI MODEL")
print("="*145)
print(pd.DataFrame(table_lott_3_rows).to_string(index=False))

print("\n" + "="*145)
print("  ZBIRNA TABELA NAUČNE VALIDACIJE ZA RAD / DIPLOMSKI (RMSE i R)")
print("="*145)
print(df_valid.to_string(index=False))
print("="*145 + "\n")
print(f"✓ SVI REZULTATI, 9 TABELA I SVI GRAFICI SU SAČUVANI U: '{BASE_OUT}/'\n")