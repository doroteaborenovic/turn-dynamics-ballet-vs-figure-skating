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

# parametri za duzinu i javno dostupni parametri za amsu i visinu

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
  #ouptut 
BASE_OUT = "REZULTATI_PO_OKRETIMA_FIZIKA"
DIR_SIDE_OMEGA = os.path.join(BASE_OUT, "grafici_ugaona_brzina_komparacija")
DIR_SIDE_ALPHA = os.path.join(BASE_OUT, "grafici_ugaono_ubrzanje_komparacija")
DIR_TABLES = os.path.join(BASE_OUT, "tabele_kinematika")

for d in [DIR_SIDE_OMEGA, DIR_SIDE_ALPHA, DIR_TABLES]:
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

# funkcije koje se koriste (omega=teta/t i a=omega/t)

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

# obrada i konstrukcija programa 

all_files = glob.glob(os.path.join(INPUT_DIR, "*_kinematics.csv"))
if not all_files:
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
all_files = [f for f in all_files if "SUMARNA" not in f and "popravljeno" not in f and "fizika_" not in f]

print(f"\n" + "="*115)
print(f"rezultatici")
print(f"="*115 + "\n")

table4_master_rows = []

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

    # filktritranje sa proverom vidljivosti 
    pts_array = np.zeros((n_frames, 33, 3))
    for lm in range(33):
        vis_col = f"vis_{lm}" if f"vis_{lm}" in df_crop.columns else f"VIS_{lm}"
        vis_series = df_crop[vis_col].values if vis_col in df_crop.columns else None
        for ax_idx, ax_name in enumerate(['x', 'y', 'z']):
            col = f"{ax_name}_{lm}" if f"{ax_name}_{lm}" in df_crop.columns else f"{ax_name.upper()}_{lm}"
            if col in df_crop.columns:
                pts_array[:, lm, ax_idx] = clean_and_interpolate_signal(df_crop[col].values, vis=vis_series, vis_threshold=0.35)

    # trazenje pivot noge tako sto se gleda koja je noga sa koordiantama najamnje aktivna tj koje koordinete se manje poemraju u prostoru i tako se proglasi pivot noga 
    var_l = np.median(np.abs(pts_array[:, 31, :2] - np.median(pts_array[:, 31, :2], axis=0))) + \
            np.median(np.abs(pts_array[:, 27, :2] - np.median(pts_array[:, 27, :2], axis=0)))
    var_r = np.median(np.abs(pts_array[:, 32, :2] - np.median(pts_array[:, 32, :2], axis=0))) + \
            np.median(np.abs(pts_array[:, 28, :2] - np.median(pts_array[:, 28, :2], axis=0)))

    planted_side = "left" if var_l <= var_r else "right"
    p_hip = 23 if planted_side == "left" else 24
    p_knee = 25 if planted_side == "left" else 26
    p_ank = 27 if planted_side == "left" else 28
    p_toe = 31 if planted_side == "left" else 32

    # skaliranjee
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

    # Kalibracija Z dubine tj procenica
    hip_width_real = 0.17 * height_m
    hip_width_meas = np.median(np.linalg.norm(pts_m[:, 23, :2] - pts_m[:, 24, :2], axis=1))
    z_correction = np.clip(hip_width_real / (hip_width_meas + 1e-5), 0.35, 0.65)
    pts_m[:, :, 2] = pts_m[:, :, 2] * z_correction

    win_dyn = min(25, n_frames if n_frames % 2 != 0 else n_frames - 1)
    win_dyn = max(11, win_dyn)

    # odredjivanje tacnog broja rotacija 
    raw_angles_B = compute_fused_torso_orientation_3d(pts_m)
    theta_B = track_strictly_monotonic_spin(raw_angles_B, is_skater=is_skater)
    
    total_displacement_deg = np.degrees(theta_B[-1])
    total_rotations = total_displacement_deg / 360.0
    full_rotations_count = int(np.floor(total_rotations))

    # 5. Ugaona brzina ω(t) i ugaono ubrzanje α(t)
    omega_B = savgol_filter(theta_B, window_length=win_dyn, polyorder=2, deriv=1, delta=dt)
    floor_rad_s = np.deg2rad(180.0 if is_skater else 120.0)
    omega_B = np.maximum(omega_B, floor_rad_s)
    omega_deg_s = np.degrees(omega_B)

    alpha_deg_s2 = savgol_filter(omega_deg_s, window_length=win_dyn, polyorder=2, deriv=1, delta=dt)

    # 6. Preslikavanje po celim okretima (0% - 100%)
    dense_samples = 101
    phase_x = np.linspace(0, 100, dense_samples)
    theta_deg = np.degrees(theta_B)

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

    # =========================================================================
    # DVOSTRUKI GRAFIK 1: UGAONA BRZINA (LEVO: KROZ VREME | DESNO: CELI OKRETI)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)

    # Levo: Vremenski profil
    ax1.plot(time_axis, omega_deg_s, color='#ff3366', linewidth=2.4, label='Ugaona brzina ω(t) [°/s]')
    ax1.axhline(np.mean(omega_deg_s), color='#00e5ff', linestyle='--', linewidth=1.6, 
                label=f'Srednja ω ({np.mean(omega_deg_s):.1f} °/s)')
    ax1.axhline(np.max(omega_deg_s), color='#facc15', linestyle=':', linewidth=1.4, 
                label=f'Maksimalna ω ({np.max(omega_deg_s):.1f} °/s)')
    ax1.fill_between(time_axis, 0, omega_deg_s, color='#ff3366', alpha=0.15)
    ax1.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_ylabel("Ugaona brzina [°/s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Kinematički profil kroz vreme ({t_start}s–{t_end}s)\nUkupno rotacija: {total_rotations:.2f}", 
                  fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=9)

    # Desno: Preklapanje celih okreta
    for c_i, (w_c, lbl) in enumerate(rev_cycles_omega):
        col = NEON_COLORS[c_i % len(NEON_COLORS)]
        ax2.plot(phase_x, w_c, linewidth=2.4, color=col, label=lbl)

    ax2.set_xlabel("Faza okreta [%] (0% = Početak ◄ ► 100% = Završen pun krug)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Ugaona brzina ω [°/s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_title(f"Profil po celim okretima ({full_rotations_count} puna kruga)\nModulacija brzine po ciklusima", 
                  fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    plt.suptitle(f"UGAONA BRZINA ROTACIJE: {clean_name} ({atype})", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_SIDE_OMEGA, f"komparacija_omega_{clean_name.lower()}.png"), dpi=300, facecolor='#0b0f19')
    plt.close()

    # =========================================================================
    # DVOSTRUKI GRAFIK 2: UGAONO UBRZANJE (LEVO: KROZ VREME | DESNO: CELI OKRETI)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.grid(True, linestyle='--', alpha=0.35, color='#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)

    # Levo: Vremenski profil α(t)
    ax1.plot(time_axis, alpha_deg_s2, color='#38bdf8', linewidth=2.4, label='Ugaono ubrzanje α(t) [°/s²]')
    ax1.axhline(0, color='#64748b', linestyle='--', linewidth=1.2, alpha=0.7)
    ax1.axhline(np.max(alpha_deg_s2), color='#fb7185', linestyle=':', linewidth=1.4, 
                label=f'Maks ubrzanje ({np.max(alpha_deg_s2):.1f} °/s²)')
    ax1.axhline(np.min(alpha_deg_s2), color='#a78bfa', linestyle=':', linewidth=1.4, 
                label=f'Maks usporenje ({np.min(alpha_deg_s2):.1f} °/s²)')
    ax1.set_xlabel("Vreme [s]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_ylabel("Ugaono ubrzanje α [°/s²]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Ugaono ubrzanje kroz vreme ({t_start}s–{t_end}s)\nOpseg: [{np.min(alpha_deg_s2):.1f} do {np.max(alpha_deg_s2):.1f}] °/s²", 
                  fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_xlim(0, time_axis[-1])
    ax1.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=9)

    # Desno: Preklapanje celih okreta α
    for c_i, (a_c, lbl) in enumerate(rev_cycles_alpha):
        col = NEON_COLORS[c_i % len(NEON_COLORS)]
        ax2.plot(phase_x, a_c, linewidth=2.4, color=col, label=lbl)

    ax2.axhline(0, color='#64748b', linestyle='--', linewidth=1.2, alpha=0.7)
    ax2.set_xlabel("Faza okreta [%] (0% = Početak ◄ ► 100% = Završen pun krug)", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Ugaono ubrzanje α [°/s²]", fontsize=10.5, fontweight='bold', color='#94a3b8')
    ax2.set_title(f"Profil α po celim okretima ({full_rotations_count} puna kruga)\nCiklično ubrzavanje/usporavanje", 
                  fontsize=11.5, fontweight='bold', color='#ffffff', pad=12)
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', labelcolor='white', fontsize=8.5)

    plt.suptitle(f"UGAONO UBRZANJE ROTACIJE: {clean_name} ({atype})", fontsize=13.5, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_SIDE_ALPHA, f"komparacija_alpha_{clean_name.lower()}.png"), dpi=300, facecolor='#0b0f19')
    plt.close()

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

    print(f"[OBRADA] {clean_name:<14} | Celih: {full_rotations_count:2d} (od {total_rotations:4.2f}) | Omega sr={np.mean(omega_deg_s):5.1f} °/s | Alpha opseg: [{np.min(alpha_deg_s2):6.1f} do {np.max(alpha_deg_s2):6.1f}] °/s²")

# prikaz i output
df_tab4 = pd.DataFrame(table4_master_rows)
df_tab4.to_csv(os.path.join(DIR_TABLES, "tabela.csv"), index=False)

print("\n" + "="*115)
print("  tabelicaaaa")
print("="*115)
print(df_tab4.to_string(index=False))
print("="*115 + "\n")
print(f"✓ Generisani grafici brzine:    {DIR_SIDE_OMEGA}/")
print(f"✓ Generisani grafici ubrzanja:  {DIR_SIDE_ALPHA}/\n")
