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
# 1. PARAMETRI I BAZA PODATAKA
# =============================================================================
ATHLETE_DB = {
    "marianela": {"height": 1.74, "weight": 52.0, "type": "Balet", "shoe_size": 39},
    "kapitonova": {"height": 1.68, "weight": 48.0, "type": "Balet", "shoe_size": 38},
    "khoreva": {"height": 1.73, "weight": 47.0, "type": "Balet", "shoe_size": 39},
    "trusova": {"height": 1.66, "weight": 50.0, "type": "Umetničko klizanje", "shoe_size": 37},
    "shcherbakova": {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje", "shoe_size": 36},
    "scerebakova": {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje", "shoe_size": 36},
    "liu": {"height": 1.58, "weight": 45.0, "type": "Umetničko klizanje", "shoe_size": 36}
}

FOOT_SIZES_CM = {36: 22.9, 37: 23.8, 38: 24.3, 39: 25.1, 40: 25.4}

INPUT_DIR = "konacne_koordinate"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "obradjene_koordinate" if os.path.exists("obradjene_koordinate") else "."

DIR_CLEAN_COORDS = "popravljene_koordinate"
DIR_PLOTS = "grafici_fizika"
DIR_TABLES = "rezultati_tabele"

os.makedirs(DIR_CLEAN_COORDS, exist_ok=True)
os.makedirs(DIR_PLOTS, exist_ok=True)
os.makedirs(DIR_TABLES, exist_ok=True)

FPS = 30.0
dt = 1.0 / FPS

# =============================================================================
# 2. POMOĆNE MATEMATIČKE I BIOMEHANIČKE FUNKCIJE
# =============================================================================
def interp_seg(p_prox, p_dist, ratio):
    return p_prox + ratio * (p_dist - p_prox)

def compute_body_angle(frame_pts):
    angles, weights = [], []
    if 11 in frame_pts and 12 in frame_pts:
        dx = frame_pts[12][0] - frame_pts[11][0]
        dz = frame_pts[12][2] - frame_pts[11][2]
        length = np.sqrt(dx**2 + dz**2)
        if length > 1e-5:
            angles.append(np.arctan2(dz, dx))
            weights.append(length * 2.0)
            
    if 23 in frame_pts and 24 in frame_pts:
        dx = frame_pts[24][0] - frame_pts[23][0]
        dz = frame_pts[24][2] - frame_pts[23][2]
        length = np.sqrt(dx**2 + dz**2)
        if length > 1e-5:
            angles.append(np.arctan2(dz, dx))
            weights.append(length)
            
    if not angles: return None
    weights = np.array(weights) / np.sum(weights)
    return np.arctan2(np.sum(weights * np.sin(angles)), np.sum(weights * np.cos(angles)))

def robust_directional_unwrap(angles_raw, is_skater=True, athlete_name=""):
    n = len(angles_raw)
    if n < 2: return np.zeros(n)
    unwrapped = np.unwrap(angles_raw)
    diffs = np.diff(unwrapped)
    valid_diffs = diffs[np.abs(diffs) < (np.pi * 0.9)]
    
    if len(valid_diffs) > 3:
        direction = np.sign(np.median(valid_diffs))
        nom_speed = np.abs(np.median(valid_diffs))
    else:
        direction = -1.0
        nom_speed = 0.5 if "trusova" in athlete_name.lower() else 0.4
        
    if direction == 0: direction = -1.0
    phi = direction * unwrapped
    phi_fixed = np.zeros(n)
    phi_fixed[0] = phi[0]
    step_nom = nom_speed if nom_speed > 0.05 else (0.45 if "trusova" in athlete_name.lower() else 0.35)
    is_trusova = "trusova" in athlete_name.lower()
    
    for i in range(1, n):
        raw_step = phi[i] - phi_fixed[i-1]
        candidates = [raw_step, raw_step - np.pi, raw_step + np.pi, raw_step - 2 * np.pi, raw_step + 2 * np.pi]
        best_c = min(candidates, key=lambda c: 1000.0 + abs(c) if c <= 0.01 else abs(c - step_nom))
        
        if is_trusova:
            if best_c <= 0.01: best_c = step_nom * 0.85
            phi_fixed[i] = phi_fixed[i-1] + best_c
            if 0.2 * step_nom < best_c < 3.0 * step_nom: step_nom = 0.85 * step_nom + 0.15 * best_c
        else:
            max_allowed_step = min(np.deg2rad(65), max(1.8 * step_nom, np.deg2rad(25)))
            if best_c > max_allowed_step: best_c = step_nom
            elif best_c <= 0.01: best_c = step_nom * 0.9
            phi_fixed[i] = phi_fixed[i-1] + best_c
            if 0.3 * step_nom < best_c < 2.2 * step_nom: step_nom = 0.85 * step_nom + 0.15 * best_c
            
    win = 9 if is_skater else 11
    if len(phi_fixed) > win:
        if win % 2 == 0: win += 1
        phi_smooth = savgol_filter(phi_fixed, window_length=win, polyorder=2)
    else:
        phi_smooth = phi_fixed
        
    return phi_smooth - phi_smooth[0]

def repair_local_angle_artifacts(theta, athlete_name):
    if "trusova" not in athlete_name.lower(): return theta
    theta = np.asarray(theta, dtype=float).copy()
    if len(theta) < 15: return theta
    dtheta = np.diff(theta)
    med = np.median(dtheta)
    mad = np.median(np.abs(dtheta - med))
    if mad < 1e-6: return theta
    robust_z = np.abs(dtheta - med) / (1.4826 * mad)
    bad_steps = np.where(robust_z > 5.0)[0]
    for i in bad_steps:
        if 2 <= i < len(theta) - 2:
            left, current, right = dtheta[i-1], dtheta[i], dtheta[i+1]
            if np.sign(left) == np.sign(right) and np.sign(current) != np.sign(left):
                theta[i+1] = 0.5 * (theta[i] + theta[i+2])
    return theta

def smooth_orbit_polar(com_x_cm, com_z_cm, theta_continuous):
    n = len(com_x_cm)
    if n < 5: return com_x_cm, com_z_cm
    r_raw = np.sqrt(com_x_cm**2 + com_z_cm**2)
    r_med = median_filter(r_raw, size=5)
    win_r = min(15, max(5, (n // 6) * 2 + 1))
    if win_r % 2 == 0: win_r += 1
    r_smooth = savgol_filter(r_med, window_length=win_r, polyorder=2)
    r_smooth = np.maximum(r_smooth, 0.5)
    return r_smooth * np.cos(theta_continuous), r_smooth * np.sin(theta_continuous)

# =============================================================================
# 3. GLAVNI PROGRAM ZA OBRADU
# =============================================================================
all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
all_files = [f for f in all_files if "SUMARNA" not in f and "popravljeno" not in f and "fizika_" not in f]

summary_rows = []

for file in sorted(all_files):
    df_raw = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower()
    
    athlete_key = None
    athlete_data = {"height": 1.65, "weight": 50.0, "type": "Nepoznato", "shoe_size": 38}
    for key, val in ATHLETE_DB.items():
        if key in filename_lower:
            athlete_key = key
            athlete_data = val
            break
            
    atype = athlete_data["type"]
    height_m = athlete_data["height"]
    weight_kg = athlete_data["weight"]
    shoe_size = athlete_data["shoe_size"]
    is_skater = "klizanje" in atype.lower()
    
    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file).split('.')[0].upper()
    print(f"\n[OBRADA] {clean_name} | {atype} | {height_m}m | {weight_kg}kg")

    coord_cols = [c for c in ['X_clean', 'Y_clean', 'Z_clean', 'X', 'Y', 'Z'] if c in df_raw.columns]
    x_c, y_c, z_c = coord_cols[0], coord_cols[1], coord_cols[2]
    
    df_raw[[x_c, y_c, z_c]] = df_raw.groupby('Landmark_ID')[[x_c, y_c, z_c]].transform(
        lambda s: s.interpolate(method='linear', limit=3, limit_direction='both').ffill().bfill()
    )

    foot_left = df_raw[df_raw['Landmark_ID'] == 31][[x_c, y_c, z_c]].values
    foot_right = df_raw[df_raw['Landmark_ID'] == 32][[x_c, y_c, z_c]].values
    var_l = np.sum(np.var(foot_left, axis=0)) if len(foot_left) > 0 else 999.0
    var_r = np.sum(np.var(foot_right, axis=0)) if len(foot_right) > 0 else 999.0
    
    planted_side = "left" if var_l <= var_r else "right"
    planted_ankle = 27 if planted_side == "left" else 28
    planted_toe = 31 if planted_side == "left" else 32
    pivot_str = "Leva (31)" if planted_side == "left" else "Desna (32)"

    frames = sorted(df_raw['Frame'].unique())
    raw_angles, com_points, bos_points, anat_heights = [], [], [], []

    for f_idx in frames:
        f_df = df_raw[df_raw['Frame'] == f_idx].set_index('Landmark_ID')
        pts = {lm: f_df.loc[lm, [x_c, y_c, z_c]].values.astype(float) for lm in f_df.index}
        
        if not all(k in pts for k in [0, 11, 12, 23, 24, planted_ankle, planted_toe]): continue
            
        ang = compute_body_angle(pts)
        raw_angles.append(ang if ang is not None else np.nan)
        
        mid_shoulder = (pts[11] + pts[12]) / 2.0
        mid_hip = (pts[23] + pts[24]) / 2.0
        head_vertex = pts[0] + 0.5 * (pts[0] - mid_shoulder)
        anchor_pt = (pts[planted_ankle] + pts[planted_toe]) / 2.0
        
        knee_pt = pts[25] if planted_side == "left" else pts[26]
        hip_pt = pts[23] if planted_side == "left" else pts[24]
        h_chain = (np.linalg.norm(head_vertex - mid_hip) + np.linalg.norm(hip_pt - knee_pt) + np.linalg.norm(knee_pt - pts[planted_ankle]))
        if h_chain > 0.2: anat_heights.append(h_chain)

        if atype == "Umetničko klizanje":
            mid_knee = (pts.get(25, pts[23]) + pts.get(26, pts[24])) / 2.0
            mid_ankle_pt = (pts.get(27, pts[23]) + pts.get(28, pts[24])) / 2.0
            mid_wrist = (pts.get(15, pts[11]) + pts.get(16, pts[12])) / 2.0
            c_trunk = mid_shoulder + 0.436 * (mid_hip - mid_shoulder)
            c_head = mid_shoulder + 0.5 * (pts[0] - mid_shoulder)
            c_thigh = mid_hip + 0.361 * (mid_knee - mid_hip)
            c_shank = mid_knee + 0.442 * (mid_ankle_pt - mid_knee)
            com = (0.0668 * c_head + 0.4257 * c_trunk + 0.0898 * mid_wrist + 0.2956 * c_thigh + 0.0962 * c_shank + 0.0258 * mid_ankle_pt)
        else:
            com = (0.0668 * pts[0] + 0.4257 * mid_hip + 0.10 * (pts[11] + pts[12])/2 + 0.40 * anchor_pt)
                   
        com_points.append(com)
        bos_points.append(anchor_pt)

    if len(raw_angles) < 10: continue

    raw_angles = np.array(raw_angles, dtype=float)
    nans = np.isnan(raw_angles)
    if nans.any():
        valid_i = np.where(~nans)[0]
        raw_angles[nans] = np.interp(np.where(nans)[0], valid_i, raw_angles[valid_i]) if len(valid_i) > 2 else 0.0

    scale = height_m / np.median(anat_heights) if len(anat_heights) > 0 else 1.0
    com_m = np.array(com_points) * scale
    bos_m = np.array(bos_points) * scale
    pivot_x, pivot_z = np.median(bos_m[:, 0]), np.median(bos_m[:, 2])
    
    com_x_raw_cm = (com_m[:, 0] - pivot_x) * 100.0
    com_z_raw_cm = (com_m[:, 2] - pivot_z) * 100.0

    theta_continuous = robust_directional_unwrap(raw_angles, is_skater=is_skater, athlete_name=clean_name)
    theta_continuous = repair_local_angle_artifacts(theta_continuous, clean_name)
    com_x_clean, com_z_clean = smooth_orbit_polar(com_x_raw_cm, com_z_raw_cm, theta_continuous)

    # Optimizovano izračunavanje ugaonih veličina sa stabilnim, glatkim izvodima bez ekstremnih skokova na krajevima
    theta_for_velocity = savgol_filter(theta_continuous, window_length=15, polyorder=2) if len(theta_continuous) >= 15 else theta_continuous
    
    total_displacement_deg = np.degrees(theta_continuous[-1])
    total_rotations = total_displacement_deg / 360.0
    
    omega_rad_s = np.gradient(theta_for_velocity, dt)
    omega_rad_s = savgol_filter(omega_rad_s, window_length=11, polyorder=2) if len(omega_rad_s) >= 11 else omega_rad_s
    omega_rad_s = np.maximum(omega_rad_s, 0.0) # Nema negativnih vrednosti brzine
    omega_deg_s = np.degrees(omega_rad_s)
    
    # ── UKLANJANJE LAŽNIH ŠILJAKA NA KRAJEVIMA (TRIMOVANJE POČETKA I KRAJA) ──
    # Ako su prva 3 ili poslednja 3 frejma dala šum, peglamo ih na susedne vrednosti
    if len(omega_deg_s) > 10:
        omega_deg_s[:3] = omega_deg_s[3]
        omega_deg_s[-3:] = omega_deg_s[-4]

    max_omega_deg = np.max(omega_deg_s)
    mean_omega_deg = np.mean(omega_deg_s)
    
    alpha_rad_s2 = np.gradient(omega_rad_s, dt)
    if len(alpha_rad_s2) >= 7: alpha_rad_s2 = savgol_filter(alpha_rad_s2, window_length=7, polyorder=2)
    max_alpha = np.max(np.abs(alpha_rad_s2))
    mean_alpha = np.mean(np.abs(alpha_rad_s2))

    radii_cm = np.sqrt(com_x_clean**2 + com_z_clean**2)
    foot_rad_cm = float(FOOT_SIZES_CM.get(shoe_size, 24.3) / 2.0)
    mean_r_cm = np.mean(radii_cm)
    max_r_cm = np.max(radii_cm)

    I_mean = 0.5 * weight_kg * (np.mean(radii_cm) / 100.0 + 0.12)**2
    ke_series = 0.5 * I_mean * (omega_rad_s ** 2)
    max_ke = np.max(ke_series)
    mean_ke = np.mean(ke_series)

    time_axis = np.arange(len(theta_continuous)) * dt

    print(f"  ✓ Ugaoni pomeraj: {total_displacement_deg:.1f}° | Ukupno okreta: {total_rotations:.2f} krugova")
    print(f"  ✓ Maksimalna ω: {max_omega_deg:.1f} °/s | Srednja ω: {mean_omega_deg:.1f} °/s")

    # =========================================================================
    # KREIRANJE GRAFIKA 1: X-Z PUTANJA (KRUGOVI)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 9), facecolor='#0b0f19')
    ax.set_facecolor('#0b0f19')
    
    for r_ring in [3.0, 6.0, 9.0, 12.0, 15.0]:
        ax.add_patch(plt.Circle((0, 0), r_ring, color='#1e293b', fill=False, linestyle=':', linewidth=1.0, alpha=0.8))
        ax.text(r_ring * np.cos(np.pi/4), r_ring * np.sin(np.pi/4), f"{int(r_ring)} cm", color='#475569', fontsize=8, ha='center', va='center')

    circle_bos = plt.Circle((0, 0), foot_rad_cm, color='#00e5ff', fill=True, alpha=0.08, linestyle='--', linewidth=2.0, edgecolor='#00e5ff',
                            label=f'Baza oslonca stopala (r = {foot_rad_cm:.1f} cm)', zorder=2)
    ax.add_patch(circle_bos)

    sc = ax.scatter(com_x_clean, com_z_clean, c=time_axis, cmap='plasma', s=45, zorder=5, edgecolors='none', alpha=0.95)
    ax.plot(com_x_clean, com_z_clean, color='#ffffff', alpha=0.35, linewidth=1.5, zorder=4)

    ax.plot(com_x_clean[0], com_z_clean[0], marker='o', markersize=10, markerfacecolor='#00ff88', markeredgecolor='white', label='Start rotacije', zorder=6)
    ax.plot(com_x_clean[-1], com_z_clean[-1], marker='X', markersize=12, markerfacecolor='#ff3366', markeredgecolor='white', label='Kraj rotacije', zorder=6)
    ax.plot(0, 0, marker='P', markersize=13, markerfacecolor='#ffd700', markeredgecolor='black', label='Osa oslonca (Pivot 0,0)', zorder=7)

    lim = max(16.0, max_r_cm + 3.0, foot_rad_cm + 3.0)
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
    # KREIRANJE GRAFIKA 2: KINEMATIČKI PROFIL (UGAONA BRZINA I OKRETI)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, facecolor='#0b0f19')
    ax1.set_facecolor('#0b0f19')
    ax2.set_facecolor('#0b0f19')
    
    ax1.plot(time_axis, omega_deg_s, color='#ff3366', linewidth=2.4, label='Ugaona brzina ω [°/s]')
    ax1.axhline(mean_omega_deg, color='#00e5ff', linestyle='--', linewidth=1.5, label=f'Srednja ω ({mean_omega_deg:.1f} °/s)')
    ax1.fill_between(time_axis, 0, omega_deg_s, color='#ff3366', alpha=0.15)
    ax1.set_ylabel("Ugaona brzina [°/s]", fontsize=10, fontweight='bold', color='#94a3b8')
    ax1.set_title(f"Biomehanički profil rotacije: {clean_name} ({atype})", fontsize=12, fontweight='bold', color='#ffffff')
    ax1.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    ax1.tick_params(colors='#94a3b8')
    leg1 = ax1.legend(loc='upper right', fontsize=9, facecolor='#111827', edgecolor='#374151')
    for text in leg1.get_texts(): text.set_color('#ffffff')
    
    ax2.plot(time_axis, theta_continuous / (2 * np.pi), color='#38bdf8', linewidth=2.4, label=f'Kumulativni okreti (Ukupno: {total_rotations:.2f} krugova / {total_displacement_deg:.0f}°)')
    ax2.set_xlabel("Vreme [s]", fontsize=10, fontweight='bold', color='#94a3b8')
    ax2.set_ylabel("Kumulativni okreti [krugovi]", fontsize=10, fontweight='bold', color='#94a3b8')
    ax2.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    ax2.tick_params(colors='#94a3b8')
    leg2 = ax2.legend(loc='upper left', fontsize=9, facecolor='#111827', edgecolor='#374151')
    for text in leg2.get_texts(): text.set_color('#ffffff')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, f"ugaona_brzina_{clean_name.lower()}.png"), dpi=300, facecolor=fig.get_facecolor())
    plt.close()

    summary_rows.append({
        "Atleta": clean_name,
        "Tip": atype,
        "Ugaoni pomeraj [°]": round(total_displacement_deg, 1),
        "Ukupno okreta": round(total_rotations, 2),
        "Maks ω [°/s]": round(max_omega_deg, 1),
        "Srednja ω [°/s]": round(mean_omega_deg, 1),
        "Maks α [rad/s²]": round(max_alpha, 1),
        "Srednje α [rad/s²]": round(mean_alpha, 1),
        "Maks KE [J]": round(max_ke, 1),
        "Srednja KE [J]": round(mean_ke, 1)
    })

# =============================================================================
# 4. SUMARNA TABELA
# =============================================================================
summary_df = pd.DataFrame(summary_rows)
print("\n" + "="*115)
print("                                 FINALNI REZULTATI ROTACIONE DINAMIKE")
print("="*115)
print(summary_df.to_string(index=False))
print("="*115)

summary_csv = os.path.join(DIR_TABLES, "SUMARNA_TABELA_ROTACIJA.csv")
summary_df.to_csv(summary_csv, index=False, float_format='%.2f')
print(f"\n✓ Svi grafici i tabele uspešno generisani u folderima: {DIR_PLOTS}/ i {DIR_TABLES}/\n")