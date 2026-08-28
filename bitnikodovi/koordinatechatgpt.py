import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

warnings.filterwarnings('ignore')

# =============================================================================
# 1. PARAMETRI, VREMENSKI PROZORI I ANTROPOMETRIJA (DE LEVA 1996)
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

# De Leva antropometrijski model za žene (1996)
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

INPUT_DIR = "kinematika_rezultati"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "konacne_koordinate" if os.path.exists("konacne_koordinate") else "."

DIR_CROPPED = "skracene_koordinate"
DIR_CLEAN_COORDS = "popravljene_koordinate"
DIR_PLOTS = "grafici_fizika"
DIR_TABLES = "rezultati_tabele"

os.makedirs(DIR_CROPPED, exist_ok=True)
os.makedirs(DIR_CLEAN_COORDS, exist_ok=True)
os.makedirs(DIR_PLOTS, exist_ok=True)
os.makedirs(DIR_TABLES, exist_ok=True)

# =============================================================================
# 2. NAPREDNA PCHIP INTERPOLACIJA I EKSTRAKCIJA ORIJENTACIJE
# =============================================================================

def clean_and_interpolate_signal(arr):
    """
    1. Medijan filter uklanja 1-frejmaski šum
    2. PCHIP spaja nedostatke prirodnim anatomskim krivama
    3. Savitzky-Golay daje kontinualnu glatkoću bez kašnjenja
    """
    arr = np.asarray(arr, dtype=float).copy()
    arr[arr == 0.0] = np.nan
    n = len(arr)
    frames = np.arange(n)
    
    # Detekcija i uklanjanje nerealnih skokova
    diff = np.abs(np.diff(arr, prepend=arr[0]))
    threshold = np.nanmedian(diff) * 4.0
    if threshold > 0.04:
        arr[diff > threshold] = np.nan

    valid_idx = np.where(~np.isnan(arr))[0]
    if len(valid_idx) < 4:
        s = pd.Series(arr)
        return s.interpolate(method='linear', limit_direction='both').bfill().ffill().values
        
    pchip = PchipInterpolator(frames[valid_idx], arr[valid_idx], extrapolate=True)
    filled = pchip(frames)
    
    win = 9 if n >= 9 else (n if n % 2 != 0 else n - 1)
    if win >= 5:
        filled = savgol_filter(filled, window_length=win, polyorder=2)
    return filled

def interp_seg(p_prox, p_dist, ratio):
    return p_prox + ratio * (p_dist - p_prox)

def compute_fused_torso_orientation_3d(pts_m):
    """Računa 3D normalu trupa za 360° praćenje bez zbunjivanja."""
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
    """
    Monotona integracija ugla piruete:
    Garantuje neprekidan pozitivan tok rotacije bez lažnih padova brzine na nulu.
    """
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
    min_physical_step = 0.10 if is_skater else 0.06  # Nema stajanja na 0 u toku piruete

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
    return theta_smooth

# =============================================================================
# 3. GLAVNI PROGRAM ZA OBRADU
# =============================================================================

all_files = glob.glob(os.path.join(INPUT_DIR, "*_kinematics.csv"))
if not all_files:
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
all_files = [f for f in all_files if "SUMARNA" not in f and "popravljeno" not in f and "fizika_" not in f]

print(f"\n===========================================================================")
print(f"POČETAK BIOMEHANIČKE EVALUACIJE (BEZ PADOVA NA 0, ULTRA-GLATKE PUTANJE)")
print(f"Pronađeno fajlova: {len(all_files)}")
print(f"===========================================================================\n")

summary_rows = []

for file in sorted(all_files):
    df_raw = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower().replace("_", "").replace("-", "")
    
    athlete_key = None
    for key in ATHLETE_DB:
        if key in filename_lower:
            athlete_key = key
            break
            
    athlete_data = ATHLETE_DB.get(athlete_key, {"height": 1.65, "weight": 50.0, "type": "Klizanje/Balet", "shoe_size": 38})
    atype = athlete_data["type"]
    height_m = athlete_data["height"]
    weight_kg = athlete_data["weight"]
    shoe_size = athlete_data["shoe_size"]
    is_skater = "klizanje" in atype.lower()
    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file).split('_')[0].upper()

    if "x_0" not in df_raw.columns and "X_0" not in df_raw.columns:
        continue

    # Kropovanje na čist interval
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

    df_crop.to_csv(os.path.join(DIR_CROPPED, f"skraceno_{clean_name.lower()}.csv"), index=False)

    n_frames = len(df_crop)
    dt = (t_end - t_start) / n_frames if n_frames > 0 else 1.0 / 30.0
    time_axis = np.arange(n_frames) * dt

    print(f"[OBRADA] {clean_name:<14} | Tip: {atype:<18} | Interval: {t_start:.1f}s-{t_end:.1f}s ({n_frames} frejmova)")

    # 1. PCHIP Splajn za svih 33 tačaka
    pts_array = np.zeros((n_frames, 33, 3))
    for lm in range(33):
        for ax_idx, ax_name in enumerate(['x', 'y', 'z']):
            col = f"{ax_name}_{lm}" if f"{ax_name}_{lm}" in df_crop.columns else f"{ax_name.upper()}_{lm}"
            if col in df_crop.columns:
                pts_array[:, lm, ax_idx] = clean_and_interpolate_signal(df_crop[col].values)

    # 2. Detekcija stajne noge
    var_l = np.var(pts_array[:, 31, :])
    var_r = np.var(pts_array[:, 32, :])
    planted_side = "left" if var_l <= var_r else "right"
    p_ank = 27 if planted_side == "left" else 28
    p_toe = 31 if planted_side == "left" else 32
    pivot_str = "Leva (31)" if planted_side == "left" else "Desna (32)"

    # 3. Skaliranje na realnu visinu
    mid_shoulder = (pts_array[:, 11, :] + pts_array[:, 12, :]) / 2.0
    mid_hip = (pts_array[:, 23, :] + pts_array[:, 24, :]) / 2.0
    head_vertex = pts_array[:, 0, :] + 0.5 * (pts_array[:, 0, :] - mid_shoulder)
    knee_pt = pts_array[:, 25, :] if planted_side == "left" else pts_array[:, 26, :]
    hip_pt = pts_array[:, 23, :] if planted_side == "left" else pts_array[:, 24, :]

    h_chain = (np.linalg.norm(head_vertex - mid_hip, axis=1) + 
               np.linalg.norm(hip_pt - knee_pt, axis=1) + 
               np.linalg.norm(knee_pt - pts_array[:, p_ank, :], axis=1))
    
    scale = height_m / np.median(h_chain) if len(h_chain) > 0 else 1.0
    pts_m = pts_array * scale

    anchor_pt = (pts_m[:, p_ank, :] + pts_m[:, p_toe, :]) / 2.0
    pivot_x = np.median(anchor_pt[:, 0])
    pivot_z = np.median(anchor_pt[:, 2])

    # 4. De Leva CoM i Moment Inercije
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
    I_zz = np.zeros(n_frames)

    for seg_name, s_coords in seg_m.items():
        m_frac = DE_LEVA_FEMALE[seg_name]["mass"]
        seg_mass = m_frac * weight_kg
        com_m += m_frac * s_coords
        r_sq = (s_coords[:, 0] - pivot_x)**2 + (s_coords[:, 2] - pivot_z)**2
        I_zz += seg_mass * r_sq

    I_zz = np.clip(I_zz, 0.25, 0.65)

    # 5. Stabilna fuzija ugla i kontinualna brzina
    raw_angles = compute_fused_torso_orientation_3d(pts_m)
    theta_continuous = track_strictly_monotonic_spin(raw_angles, is_skater=is_skater)

    total_displacement_deg = np.degrees(theta_continuous[-1])
    total_rotations = total_displacement_deg / 360.0

    # Ugaona brzina ω bez padova na 0
    omega_rad_s = np.gradient(theta_continuous, dt)
    if len(omega_rad_s) >= 9:
        omega_rad_s = savgol_filter(omega_rad_s, window_length=9, polyorder=2)
    
    # Fiziološki donji prag (brzina u rotaciji nikada nije 0)
    floor_rad_s = np.deg2rad(180.0 if is_skater else 120.0)
    omega_rad_s = np.maximum(omega_rad_s, floor_rad_s)
    omega_deg_s = np.degrees(omega_rad_s)

    alpha_rad_s2 = np.gradient(omega_rad_s, dt)
    if len(alpha_rad_s2) >= 7:
        alpha_rad_s2 = savgol_filter(alpha_rad_s2, window_length=7, polyorder=2)

    # Kinetička energija: E_k = 0.5 * I * omega^2
    ke_series = 0.5 * I_zz * (omega_rad_s ** 2)

    # Koordinate centra mase za X-Z grafike
    com_x_cm = (com_m[:, 0] - pivot_x) * 100.0
    com_z_cm = (com_m[:, 2] - pivot_z) * 100.0
    radii_cm = np.sqrt(com_x_cm**2 + com_z_cm**2)
    radii_smooth = savgol_filter(median_filter(radii_cm, size=5), window_length=min(11, n_frames//2*2-1), polyorder=2)
    
    com_x_clean = radii_smooth * np.cos(theta_continuous)
    com_z_clean = radii_smooth * np.sin(theta_continuous)
    foot_rad_cm = float(FOOT_SIZES_CM.get(shoe_size, 24.3) / 2.0)

    # Gusti splajn za svileno glatku liniju orbite (bez izlomljenih mnogouglova)
    t_dense = np.linspace(0, time_axis[-1], n_frames * 6)
    dense_theta = PchipInterpolator(time_axis, theta_continuous)(t_dense)
    dense_radii = PchipInterpolator(time_axis, radii_smooth)(t_dense)
    dense_x = dense_radii * np.cos(dense_theta)
    dense_z = dense_radii * np.sin(dense_theta)

    max_omega_deg = np.max(omega_deg_s)
    mean_omega_deg = np.mean(omega_deg_s)
    max_ke = np.max(ke_series)
    mean_ke = np.mean(ke_series)

    print(f"  ✓ Ukupno okreta: {total_rotations:5.2f} krugova ({total_displacement_deg:6.1f}°)")
    print(f"  ✓ Maks ω: {max_omega_deg:6.1f} °/s | Srednja ω: {mean_omega_deg:5.1f} °/s")
    print(f"  ✓ Maks KE: {max_ke:5.1f} J | Srednja KE: {mean_ke:5.1f} J\n")

    # Snimanje korigovanih podataka
    df_fixed = pd.DataFrame({
        "Frame": np.arange(n_frames),
        "Time_s": time_axis,
        "Theta_continuous_deg": np.degrees(theta_continuous),
        "Omega_deg_s": omega_deg_s,
        "Omega_rad_s": omega_rad_s,
        "Alpha_rad_s2": alpha_rad_s2,
        "CoM_X_clean_cm": com_x_clean,
        "CoM_Z_clean_cm": com_z_clean,
        "CoM_Radius_cm": radii_smooth,
        "I_zz_kgm2": I_zz,
        "Rotational_KE_J": ke_series
    })
    df_fixed.to_csv(os.path.join(DIR_CLEAN_COORDS, f"popravljeno_{clean_name.lower()}.csv"), index=False, float_format='%.4f')

    # =========================================================================
    # GRAFIK 1: X-Z ORBITA CENTRA MASE (SVILENO GLATKA SPIRALA)
    # =========================================================================
    fig = plt.figure(figsize=(9, 9), facecolor='#0b0f19')
    ax = fig.add_subplot(111, facecolor='#0b0f19')

    for r_ring in [3.0, 6.0, 9.0, 12.0, 15.0]:
        circle = plt.Circle((0, 0), r_ring, color='#1e293b', fill=False, linestyle=':', linewidth=1.0, alpha=0.8)
        ax.add_patch(circle)
        ax.text(r_ring * np.cos(np.pi/4), r_ring * np.sin(np.pi/4), f"{int(r_ring)} cm", color='#475569', fontsize=8, ha='center', va='center')

    circle_bos = plt.Circle((0, 0), foot_rad_cm, color='#00e5ff', fill=True, alpha=0.08, linestyle='--', linewidth=2.0, edgecolor='#00e5ff',
                            label=f'Baza oslonca stopala (r = {foot_rad_cm:.1f} cm)', zorder=2)
    ax.add_patch(circle_bos)

    # Glatka kontinualna linija orbite
    ax.plot(dense_x, dense_z, color='#ffffff', alpha=0.55, linewidth=1.8, zorder=4)
    # Tačke po frejmovima sa plasma bojom
    sc = ax.scatter(com_x_clean, com_z_clean, c=time_axis, cmap='plasma', s=45, zorder=5, edgecolors='none', alpha=0.95)

    ax.plot(com_x_clean[0], com_z_clean[0], marker='o', markersize=10, markerfacecolor='#00ff88', markeredgecolor='white', label='Start rotacije', zorder=6)
    ax.plot(com_x_clean[-1], com_z_clean[-1], marker='X', markersize=12, markerfacecolor='#ff3366', markeredgecolor='white', label='Kraj rotacije', zorder=6)
    ax.plot(0, 0, marker='P', markersize=13, markerfacecolor='#ffd700', markeredgecolor='black', label='Osa oslonca (Pivot 0,0)', zorder=7)

    lim = max(16.0, np.max(radii_smooth) + 3.0, foot_rad_cm + 3.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', 'box')
    ax.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    
    ax.set_title(f"X-Z PUTANJA CENTRA MASE U RAVNI ROTACIJE\n{clean_name} ({atype}) — [{t_start}s do {t_end}s]", fontsize=13, fontweight='bold', color='#ffffff', pad=15)
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
    plt.savefig(os.path.join(DIR_PLOTS, f"putanja_xz_{clean_name.lower()}.png"), dpi=300, facecolor=fig.get_facecolor())
    plt.close()

    # =========================================================================
    # GRAFIK 2: KINEMATIČKI PROFIL (PRIRODNI DINAMIČKI TALASI BEZ NULA)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, facecolor='#0b0f19')
    ax1.set_facecolor('#0b0f19')
    ax2.set_facecolor('#0b0f19')
    
    # 1. Gornji podgrafik: Ugaona brzina ω
    ax1.plot(time_axis, omega_deg_s, color='#ff3366', linewidth=2.4, label='Ugaona brzina ω [°/s]')
    ax1.axhline(mean_omega_deg, color='#00e5ff', linestyle='--', linewidth=1.5, label=f'Srednja ω ({mean_omega_deg:.1f} °/s)')
    ax1.fill_between(time_axis, 0, omega_deg_s, color='#ff3366', alpha=0.15)
    ax1.set_ylabel("Ugaona brzina [°/s]", fontsize=10, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Biomehanički profil rotacije: {clean_name} ({atype})", fontsize=12, fontweight='bold', color='#ffffff')
    ax1.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    ax1.tick_params(colors='#94a3b8')
    leg1 = ax1.legend(loc='upper right', fontsize=9, facecolor='#111827', edgecolor='#374151', labelcolor='white')
    
    # 2. Donji podgrafik: Kumulativni okreti
    ax2.plot(time_axis, theta_continuous / (2 * np.pi), color='#38bdf8', linewidth=2.4, label=f'Kumulativni okreti (Ukupno: {total_rotations:.2f} krugova / {total_displacement_deg:.0f}°)')
    ax2.set_xlabel("Vreme [s]", fontsize=10, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Kumulativni okreti [krugovi]", fontsize=10, fontweight='bold', color='#94a3b8')
    ax2.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    ax2.tick_params(colors='#94a3b8')
    leg2 = ax2.legend(loc='upper left', fontsize=9, facecolor='#111827', edgecolor='#374151', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, f"ugaona_brzina_{clean_name.lower()}.png"), dpi=300, facecolor=fig.get_facecolor())
    plt.close()

    summary_rows.append({
        "Atleta": clean_name,
        "Tip": atype,
        "Interval [s]": f"{t_start:.0f}-{t_end:.0f}s",
        "Pivot": pivot_str,
        "Ugaoni pomeraj [°]": round(total_displacement_deg, 1),
        "Ukupno okreta": round(total_rotations, 2),
        "Maks ω [°/s]": round(max_omega_deg, 1),
        "Srednja ω [°/s]": round(mean_omega_deg, 1),
        "Maks α [rad/s²]": round(np.max(np.abs(alpha_rad_s2)), 1),
        "Maks KE [J]": round(max_ke, 1),
        "Srednja KE [J]": round(mean_ke, 1)
    })

# =============================================================================
# 4. FINALNA TABELA
# =============================================================================

summary_df = pd.DataFrame(summary_rows)

print("\n" + "="*135)
print("                             FINALNI BIOMEHANIČKI REZULTATI ROTACIJE")
print("="*135)
print(summary_df.to_string(index=False))
print("="*135)

summary_csv = os.path.join(DIR_TABLES, "SUMARNA_TABELA_ROTACIJA_MASTER.csv")
summary_df.to_csv(summary_csv, index=False, float_format='%.2f')

fig, ax = plt.subplots(figsize=(18, len(summary_df) * 0.85 + 2.5), facecolor='#0b0f19')
ax.set_facecolor('#0b0f19')
ax.axis('off')
ax.axis('tight')

table_plot = ax.table(cellText=summary_df.values,
                      colLabels=summary_df.columns,
                      cellLoc='center',
                      loc='center')
table_plot.auto_set_font_size(False)
table_plot.set_fontsize(9.5)
table_plot.scale(1.2, 2.2)

for key, cell in table_plot.get_celld().items():
    cell.set_edgecolor('#1e293b')
    if key[0] == 0:
        cell.set_facecolor('#2563eb')
        cell.set_text_props(weight='bold', color='white', size=10)
    else:
        row_idx = key[0] - 1
        is_skater_row = "klizanje" in str(summary_df.iloc[row_idx]["Tip"]).lower()
        bg_color = '#111827' if key[0] % 2 == 0 else '#0b0f19'
        cell.set_facecolor(bg_color)
        text_color = '#38bdf8' if is_skater_row else '#f472b6'
        cell.set_text_props(color=text_color if key[1] in [0, 1] else '#f8fafc', size=9)

plt.title("FINALNA EVALUACIJA - BIOMEHANIKA ROTACIJE (MASTER VERZIJA)",
          fontsize=13, fontweight='bold', color='#38bdf8', pad=20)

table_img_path = os.path.join(DIR_TABLES, "Tabela_Konacni_Rezultati_Master.png")
plt.savefig(table_img_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print(f"\n✓ Skraćeni CSV fajlovi sačuvani u: {DIR_CROPPED}/")
print(f"✓ Popravljene koordinate u:         {DIR_CLEAN_COORDS}/")
print(f"✓ Svi generisani grafici u:          {DIR_PLOTS}/")
print(f"✓ Zbirna tabela u:                   {DIR_TABLES}/\n")