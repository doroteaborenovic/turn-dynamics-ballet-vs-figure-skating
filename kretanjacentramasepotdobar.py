import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

warnings.filterwarnings('ignore')

# =============================================================================
# 1. BAZA SPORTISTA I ANTROPOMETRIJA (DE LEVA 1996)
# =============================================================================

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

# De Leva model za žene
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

DIR_CLEAN_COORDS = "popravljene_koordinate"
DIR_PLOTS = "grafici_fizika"
DIR_TABLES = "rezultati_tabele"

os.makedirs(DIR_CLEAN_COORDS, exist_ok=True)
os.makedirs(DIR_PLOTS, exist_ok=True)
os.makedirs(DIR_TABLES, exist_ok=True)

# =============================================================================
# 2. NAPREDNE BIOMEHANIČKE I 3D FUNKCIJE
# =============================================================================

def interp_seg(p_prox, p_dist, ratio):
    return p_prox + ratio * (p_dist - p_prox)

def compute_torso_orientation_3d(pts):
    """
    Kombinuje 3D vektor kičme, ramena, kukova i lica (nosa) kako bi
    nedvosmisleno dobio tačan ugao gledanja (0 do 360 stepeni) u X-Z ravni.
    """
    if not (11 in pts and 12 in pts and 23 in pts and 24 in pts):
        return None

    mid_sh = (pts[11] + pts[12]) / 2.0
    mid_hip = (pts[23] + pts[24]) / 2.0
    
    spine = mid_sh - mid_hip
    shoulders = pts[12] - pts[11]
    
    # 3D normala grudnog koša (vektorski proizvod kičme i ramena)
    chest_normal = np.cross(spine, shoulders)
    norm = np.linalg.norm(chest_normal)
    if norm < 1e-6:
        return None
    chest_normal = chest_normal / norm

    # Ako je detektovan nos (LM 0), garantujemo da normala gleda napred
    if 0 in pts:
        face_dir = pts[0] - mid_sh
        if np.dot(chest_normal, face_dir) < 0:
            chest_normal = -chest_normal

    # Ugao normale u horizontalnoj ravni (X = lateralno, Z = dubina)
    return np.arctan2(chest_normal[0], chest_normal[2])

def unwrap_high_speed_spin(angles_raw):
    """
    Tačno akumulira rotacije za brze piruete i spinove.
    Ne postavlja veštačka ograničenja na brzinu i ne gubi krugove.
    """
    n = len(angles_raw)
    if n < 2:
        return np.zeros(n)

    # Računanje koraka kroz kružnu razliku [-pi, pi]
    diffs = np.arctan2(np.sin(np.diff(angles_raw)), np.cos(np.diff(angles_raw)))
    
    # Određivanje dominantnog smera rotacije (+1 = suprotno od kazaljke, -1 = u smeru)
    direction = np.sign(np.median(diffs[np.abs(diffs) > 0.05]))
    if direction == 0:
        direction = 1.0

    phi = direction * angles_raw
    phi_fixed = np.zeros(n)
    phi_fixed[0] = phi[0]

    # Nominalni korak rotacije (inicijalna procena)
    step_nom = np.abs(np.median(diffs))
    if step_nom < 0.15:
        step_nom = 0.35  # ~20 deg/frame

    for i in range(1, n):
        # Sirova promena ugla
        raw_step = phi[i] - phi_fixed[i-1]
        
        # Svi mogući prelasci preko kruga i polukruga (usled zamene strana)
        candidates = [
            raw_step,
            raw_step - 2 * np.pi,
            raw_step + 2 * np.pi,
            raw_step - np.pi,
            raw_step + np.pi,
            raw_step - 4 * np.pi,
            raw_step + 4 * np.pi
        ]
        
        # Bira kandidata koji zadržava prirodan rotacioni tok (pozitivan i blizak trenutnoj brzini)
        def score_candidate(c):
            if c <= 0.02:  # Kazna za mirovanje ili okretanje unazad
                return 500.0 + abs(c)
            return abs(c - step_nom)

        best_c = min(candidates, key=score_candidate)
        
        # Ako je detektovan prekid, koristi prethodnu dinamiku
        if best_c <= 0.02:
            best_c = step_nom * 0.95

        phi_fixed[i] = phi_fixed[i-1] + best_c
        
        # Adaptivno praćenje promene brzine (ubrzanje/usporavanje bez sečenja!)
        if 0.15 * step_nom < best_c < 4.0 * step_nom:
            step_nom = 0.80 * step_nom + 0.20 * best_c

    # Glatko filtriranje ugla
    win = 7 if len(phi_fixed) >= 7 else 3
    phi_smooth = savgol_filter(phi_fixed, window_length=win, polyorder=2)
    return phi_smooth - phi_smooth[0]

def smooth_orbit_polar(com_x_cm, com_z_cm, theta_continuous):
    """Gradi čistu polarnu orbitu za X-Z prikaz."""
    n = len(com_x_cm)
    if n < 5:
        return com_x_cm, com_z_cm
        
    r_raw = np.sqrt(com_x_cm**2 + com_z_cm**2)
    r_med = median_filter(r_raw, size=5)
    win_r = min(11, max(5, (n // 8) * 2 + 1))
    r_smooth = savgol_filter(r_med, window_length=win_r, polyorder=2)
    r_smooth = np.maximum(r_smooth, 0.5)
    
    x_smooth = r_smooth * np.cos(theta_continuous)
    z_smooth = r_smooth * np.sin(theta_continuous)
    return x_smooth, z_smooth

# =============================================================================
# 3. GLAVNA OBRADA CSV FAJLOVA
# =============================================================================

all_files = glob.glob(os.path.join(INPUT_DIR, "*_kinematics.csv"))
if not all_files:
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
all_files = [f for f in all_files if "SUMARNA" not in f and "popravljeno" not in f and "fizika_" not in f]

print(f"\n===========================================================================")
print(f"POČETAK ANALIZE: Pronađeno {len(all_files)} CSV fajlova u '{INPUT_DIR}'")
print(f"===========================================================================\n")

summary_rows = []

for file in sorted(all_files):
    df_raw = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower()
    
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
    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file).split('_')[0].upper()

    n_frames = len(df_raw)
    if n_frames < 10 or ("x_0" not in df_raw.columns and "X_0" not in df_raw.columns):
        continue

    print(f"[OBRADA] {clean_name:<14} | Tip: {atype:<18} | Visina: {height_m}m | Masa: {weight_kg}kg")

    # Određivanje dt (vremenskog koraka)
    if "timestamp_sec" in df_raw.columns and len(df_raw["timestamp_sec"]) > 1:
        diffs = np.diff(df_raw["timestamp_sec"].values)
        valid_dt = diffs[diffs > 0.001]
        dt = np.median(valid_dt) if len(valid_dt) > 0 else 1.0 / 30.0
    else:
        dt = 1.0 / 30.0

    # Popunjavanje i interpolacija 33 tačke
    pts_array = np.zeros((n_frames, 33, 3))
    for lm in range(33):
        for ax_idx, ax_name in enumerate(['x', 'y', 'z']):
            col = f"{ax_name}_{lm}" if f"{ax_name}_{lm}" in df_raw.columns else f"{ax_name.upper()}_{lm}"
            if col in df_raw.columns:
                s = pd.Series(df_raw[col].values, dtype=float).replace(0.0, np.nan)
                pts_array[:, lm, ax_idx] = s.interpolate(method='linear', limit_direction='both').bfill().ffill().values

    # Detekcija stajne noge (31=Leva, 32=Desna)
    var_l = np.var(pts_array[:, 31, :])
    var_r = np.var(pts_array[:, 32, :])
    planted_side = "left" if var_l <= var_r else "right"
    p_ank = 27 if planted_side == "left" else 28
    p_toe = 31 if planted_side == "left" else 32
    pivot_str = "Leva (31)" if planted_side == "left" else "Desna (32)"

    raw_angles, com_points, bos_points, anat_heights = [], [], [], []

    for f_idx in range(n_frames):
        pts = {lm: pts_array[f_idx, lm, :] for lm in range(33)}
        
        ang = compute_torso_orientation_3d(pts)
        raw_angles.append(ang if ang is not None else np.nan)
        
        mid_shoulder = (pts[11] + pts[12]) / 2.0
        mid_hip = (pts[23] + pts[24]) / 2.0
        head_vertex = pts[0] + 0.5 * (pts[0] - mid_shoulder)
        anchor_pt = (pts[p_ank] + pts[p_toe]) / 2.0
        
        knee_pt = pts[25] if planted_side == "left" else pts[26]
        hip_pt = pts[23] if planted_side == "left" else pts[24]
        
        h_chain = (np.linalg.norm(head_vertex - mid_hip) + 
                   np.linalg.norm(hip_pt - knee_pt) + 
                   np.linalg.norm(knee_pt - pts[p_ank]))
        if h_chain > 0.2:
            anat_heights.append(h_chain)

        # De Leva CoM
        mid_knee = (pts[25] + pts[26]) / 2.0
        mid_ankle_pt = (pts[27] + pts[28]) / 2.0
        mid_wrist = (pts[15] + pts[16]) / 2.0
        
        c_trunk = mid_shoulder + 0.436 * (mid_hip - mid_shoulder)
        c_head = mid_shoulder + 0.500 * (pts[0] - mid_shoulder)
        c_thigh = mid_hip + 0.361 * (mid_knee - mid_hip)
        c_shank = mid_knee + 0.442 * (mid_ankle_pt - mid_knee)
        
        com = (0.0668 * c_head + 0.4257 * c_trunk + 0.0898 * mid_wrist + 
               0.2956 * c_thigh + 0.0962 * c_shank + 0.0258 * mid_ankle_pt)
               
        com_points.append(com)
        bos_points.append(anchor_pt)

    raw_angles = np.array(raw_angles, dtype=float)
    nans = np.isnan(raw_angles)
    if nans.any():
        valid_i = np.where(~nans)[0]
        raw_angles[nans] = np.interp(np.where(nans)[0], valid_i, raw_angles[valid_i]) if len(valid_i) > 2 else 0.0

    # Skaliranje na stvarne metre
    scale = height_m / np.median(anat_heights) if len(anat_heights) > 0 else 1.0
    com_m = np.array(com_points) * scale
    bos_m = np.array(bos_points) * scale
    
    pivot_x = np.median(bos_m[:, 0])
    pivot_z = np.median(bos_m[:, 2])
    
    com_x_raw_cm = (com_m[:, 0] - pivot_x) * 100.0
    com_z_raw_cm = (com_m[:, 2] - pivot_z) * 100.0

    # 1. KONTINUALNA AKUMULACIJA UGLA BEZ GUBLJENJA KRUGOVA
    theta_continuous = unwrap_high_speed_spin(raw_angles)
    com_x_clean, com_z_clean = smooth_orbit_polar(com_x_raw_cm, com_z_raw_cm, theta_continuous)

    # 2. KINEMATIČKE VELIČINE
    total_displacement_deg = np.degrees(theta_continuous[-1])
    total_rotations = total_displacement_deg / 360.0

    # Ugaona brzina ω
    omega_rad_s = np.gradient(theta_continuous, dt)
    if len(omega_rad_s) >= 7:
        omega_rad_s = savgol_filter(omega_rad_s, window_length=7, polyorder=2)
    omega_rad_s = np.maximum(omega_rad_s, 0.05)
    omega_deg_s = np.degrees(omega_rad_s)

    # Ugaono ubrzanje α
    alpha_rad_s2 = np.gradient(omega_rad_s, dt)
    if len(alpha_rad_s2) >= 7:
        alpha_rad_s2 = savgol_filter(alpha_rad_s2, window_length=7, polyorder=2)

    radii_cm = np.sqrt(com_x_clean**2 + com_z_clean**2)
    foot_rad_cm = float(FOOT_SIZES_CM.get(shoe_size, 24.3) / 2.0)
    
    # Moment inercije i Kinetička energija
    I_mean = 0.5 * weight_kg * (np.mean(radii_cm) / 100.0 + 0.12)**2
    ke_series = 0.5 * I_mean * (omega_rad_s ** 2)
    time_axis = np.arange(len(theta_continuous)) * dt

    print(f"  ✓ Ukupno okreta: {total_rotations:5.2f} krugova ({total_displacement_deg:6.1f}°)")
    print(f"  ✓ Maks ω: {np.max(omega_deg_s):6.1f} °/s | Srednja ω: {np.mean(omega_deg_s):5.1f} °/s")
    print(f"  ✓ Maks KE: {np.max(ke_series):5.1f} J | Srednja KE: {np.mean(ke_series):5.1f} J\n")

    # Čuvanje korigovanih podataka
    df_fixed = pd.DataFrame({
        "Frame": np.arange(len(theta_continuous)),
        "Time_s": time_axis,
        "Theta_continuous_deg": np.degrees(theta_continuous),
        "Omega_deg_s": omega_deg_s,
        "Omega_rad_s": omega_rad_s,
        "Alpha_rad_s2": alpha_rad_s2,
        "CoM_X_clean_cm": com_x_clean,
        "CoM_Z_clean_cm": com_z_clean,
        "CoM_Radius_cm": radii_cm,
        "Rotational_KE_J": ke_series
    })
    df_fixed.to_csv(os.path.join(DIR_CLEAN_COORDS, f"popravljeno_{clean_name.lower()}.csv"), index=False, float_format='%.4f')

    # =========================================================================
    # GRAFIK 1: X-Z ORBITA CENTRA MASE (DARK THEME)
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

    sc = ax.scatter(com_x_clean, com_z_clean, c=time_axis, cmap='plasma', s=45, zorder=5, edgecolors='none', alpha=0.95)
    ax.plot(com_x_clean, com_z_clean, color='#ffffff', alpha=0.35, linewidth=1.5, zorder=4)

    ax.plot(com_x_clean[0], com_z_clean[0], marker='o', markersize=10, markerfacecolor='#00ff88', markeredgecolor='white', label='Start rotacije', zorder=6)
    ax.plot(com_x_clean[-1], com_z_clean[-1], marker='X', markersize=12, markerfacecolor='#ff3366', markeredgecolor='white', label='Kraj rotacije', zorder=6)
    ax.plot(0, 0, marker='P', markersize=13, markerfacecolor='#ffd700', markeredgecolor='black', label='Osa oslonca (Pivot 0,0)', zorder=7)

    lim = max(16.0, np.max(radii_cm) + 3.0, foot_rad_cm + 3.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', 'box')
    ax.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    
    ax.set_title(f"X-Z PUTANJA CENTRA MASE U RAVNI ROTACIJE\n{clean_name} ({atype})", fontsize=13, fontweight='bold', color='#ffffff', pad=15)
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
    # GRAFIK 2: KINEMATIČKI PROFIL (BRZINA I KRUGOVI)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, facecolor='#0b0f19')
    ax1.set_facecolor('#0b0f19')
    ax2.set_facecolor('#0b0f19')
    
    # 1. Ugaona brzina ω
    ax1.plot(time_axis, omega_deg_s, color='#ff3366', linewidth=2.4, label='Ugaona brzina ω [°/s]')
    ax1.axhline(np.mean(omega_deg_s), color='#00e5ff', linestyle='--', linewidth=1.5, label=f'Srednja ω ({np.mean(omega_deg_s):.1f} °/s)')
    ax1.fill_between(time_axis, 0, omega_deg_s, color='#ff3366', alpha=0.15)
    ax1.set_ylabel("Ugaona brzina [°/s]", fontsize=10, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Biomehanički profil rotacije: {clean_name} ({atype})", fontsize=12, fontweight='bold', color='#ffffff')
    ax1.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    ax1.tick_params(colors='#94a3b8')
    leg1 = ax1.legend(loc='upper right', fontsize=9, facecolor='#111827', edgecolor='#374151', labelcolor='white')
    
    # 2. Kumulativni okreti
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
        "Pivot": pivot_str,
        "Ugaoni pomeraj [°]": round(total_displacement_deg, 1),
        "Ukupno okreta": round(total_rotations, 2),
        "Maks ω [°/s]": round(np.max(omega_deg_s), 1),
        "Srednja ω [°/s]": round(np.mean(omega_deg_s), 1),
        "Maks α [rad/s²]": round(np.max(np.abs(alpha_rad_s2)), 1),
        "Srednje α [rad/s²]": round(np.mean(np.abs(alpha_rad_s2)), 1),
        "Maks KE [J]": round(np.max(ke_series), 1),
        "Srednja KE [J]": round(np.mean(ke_series), 1)
    })

# =============================================================================
# 4. ZAVRŠNA TABELA (DARK THEME PNG I CSV)
# =============================================================================

summary_df = pd.DataFrame(summary_rows)

print("\n" + "="*135)
print("                                 FINALNI REZULTATI ROTACIONE DINAMIKE")
print("="*135)
print(summary_df.to_string(index=False))
print("="*135)

summary_csv = os.path.join(DIR_TABLES, "SUMARNA_TABELA_ROTACIJA.csv")
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

plt.title("FINALNA EVALUACIJA - BIOMEHANIKA ROTACIJE (KLIZANJE I BALET)",
          fontsize=13, fontweight='bold', color='#38bdf8', pad=20)

table_img_path = os.path.join(DIR_TABLES, "Tabela_Konacni_Rezultati.png")
plt.savefig(table_img_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print(f"\n✓ Sve popravljene koordinate sačuvane u: {DIR_CLEAN_COORDS}/")
print(f"✓ Svi generisani grafici sačuvani u:     {DIR_PLOTS}/")
print(f"✓ Zbirna tabela i statistika sačuvana u: {DIR_TABLES}/\n")