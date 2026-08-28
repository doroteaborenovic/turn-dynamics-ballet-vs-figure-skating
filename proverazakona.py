import pandas as pd
import glob
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. ANTROPOMETRIJSKI I FIZIČKI PARAMETRI
# ---------------------------------------------------------
ATHLETE_DB = {
    "marianela": {"height": 1.74, "weight": 52.0, "type": "Balet"},
    "kapitonova": {"height": 1.68, "weight": 48.0, "type": "Balet"},
    "khoreva": {"height": 1.73, "weight": 47.0, "type": "Balet"},
    "trusova": {"height": 1.66, "weight": 50.0, "type": "Umetničko klizanje"},
    "scerebakova": {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje"},
    "shcherbakova": {"height": 1.61, "weight": 42.0, "type": "Umetničko klizanje"},
    "liu": {"height": 1.58, "weight": 45.0, "type": "Umetničko klizanje"}
}

DE_LEVA_MASS_FEMALE = {
    "Head": 0.0668, "Trunk": 0.4257, "UpperArm": 0.0255,
    "Forearm": 0.0138, "Hand": 0.0056, "Thigh": 0.1478,
    "Shank": 0.0481, "Foot": 0.0129
}

folder_path = r"C:\Users\PC\gitara\projekat.fizika\konacne_koordinate"
all_files = glob.glob(os.path.join(folder_path, "*_konacne.csv"))

output_table_folder = r"C:\Users\PC\gitara\projekat.fizika\tabele"
os.makedirs(output_table_folder, exist_ok=True)

FPS = 30
dt = 1.0 / FPS

# =================================================================
# POMOĆNE FUNKCIJE (METODOLOŠKI ČISTE)
# =================================================================

def robust_unwrap(angles, max_jump_deg=120.0):
    angles = np.array(angles, dtype=float)
    n = len(angles)
    if n < 2: return angles
    unwrapped = np.zeros(n)
    unwrapped[0] = angles[0]
    for i in range(1, n):
        diff = angles[i] - angles[i-1]
        while diff > np.pi: diff -= 2 * np.pi
        while diff < -np.pi: diff += 2 * np.pi
        unwrapped[i] = unwrapped[i-1] + diff
    return unwrapped

def compute_angular_velocity_robust(theta_unwrapped, fps=30.0):
    dt = 1.0 / fps
    n = len(theta_unwrapped)
    omega = np.zeros(n)
    for i in range(1, n-1):
        omega[i] = (theta_unwrapped[i+1] - theta_unwrapped[i-1]) / (2 * dt)
    omega[0] = (theta_unwrapped[1] - theta_unwrapped[0]) / dt
    omega[-1] = (theta_unwrapped[-1] - theta_unwrapped[-2]) / dt
    return omega

def count_rotations_by_cycles(x_data, z_data):
    x_centered = x_data - np.mean(x_data)
    z_centered = z_data - np.mean(z_data)
    x_crossings = np.where(np.diff(np.sign(x_centered)))[0]
    z_crossings = np.where(np.diff(np.sign(z_centered)))[0]
    x_rotations = len(x_crossings) / 2.0
    z_rotations = len(z_crossings) / 2.0
    cycle_estimate = (x_rotations + z_rotations) / 2.0
    return cycle_estimate

def compute_body_angle_multi_vector(frame_data_dict):
    angles, weights = [], []
    def get_xz(lm_id):
        return (frame_data_dict[lm_id][0], frame_data_dict[lm_id][2]) if lm_id in frame_data_dict else (None, None)
    
    # Ramena (11 -> 12)
    x11, z11 = get_xz(11); x12, z12 = get_xz(12)
    if x11 is not None and x12 is not None:
        dx, dz = x12 - x11, z12 - z11
        l = np.sqrt(dx**2 + dz**2)
        if l > 1e-6: angles.append(np.arctan2(dz, dx)); weights.append(l)
        
    # Kukovi (23 -> 24)
    x23, z23 = get_xz(23); x24, z24 = get_xz(24)
    if x23 is not None and x24 is not None:
        dx, dz = x24 - x23, z24 - z23
        l = np.sqrt(dx**2 + dz**2)
        if l > 1e-6: angles.append(np.arctan2(dz, dx)); weights.append(l)
        
    if not angles: return 0.0
    weights = np.array(weights); weights /= weights.sum()
    sin_avg = np.sum(weights * np.sin(angles))
    cos_avg = np.sum(weights * np.cos(angles))
    return np.arctan2(sin_avg, cos_avg)

def detect_and_fix_angle_outliers(theta, max_vel_deg=70.0):
    theta_fixed = theta.copy()
    max_jump = np.radians(max_vel_deg)
    outliers = 0
    for i in range(1, len(theta_fixed)-1):
        if abs(theta_fixed[i] - theta_fixed[i-1]) > max_jump and abs(theta_fixed[i+1] - theta_fixed[i]) > max_jump:
            theta_fixed[i] = (theta_fixed[i-1] + theta_fixed[i+1]) / 2.0
            outliers += 1
    return theta_fixed, outliers

def compute_moment_of_inertia(frame_pts, athlete_weight, pivot_id):
    if pivot_id not in frame_pts: return 0.0
    pivot_xz = np.array([frame_pts[pivot_id][0], frame_pts[pivot_id][2]])
    
    seg_centers = {}
    if 0 in frame_pts: seg_centers["Head"] = np.array([frame_pts[0][0], frame_pts[0][2]])
    
    trunk_pts = [frame_pts[i] for i in [11, 12, 23, 24] if i in frame_pts]
    if trunk_pts:
        t_avg = np.mean(trunk_pts, axis=0)
        seg_centers["Trunk"] = np.array([t_avg[0], t_avg[2]])
        
    if 11 in frame_pts and 13 in frame_pts:
        seg_centers["UpperArm_L"] = np.mean([frame_pts[11], frame_pts[13]], axis=0)[[0, 2]]
    if 12 in frame_pts and 14 in frame_pts:
        seg_centers["UpperArm_R"] = np.mean([frame_pts[12], frame_pts[14]], axis=0)[[0, 2]]
    if 13 in frame_pts and 15 in frame_pts:
        seg_centers["Forearm_L"] = np.mean([frame_pts[13], frame_pts[15]], axis=0)[[0, 2]]
    if 14 in frame_pts and 16 in frame_pts:
        seg_centers["Forearm_R"] = np.mean([frame_pts[14], frame_pts[16]], axis=0)[[0, 2]]
    if 15 in frame_pts: seg_centers["Hand_L"] = np.array([frame_pts[15][0], frame_pts[15][2]])
    if 16 in frame_pts: seg_centers["Hand_R"] = np.array([frame_pts[16][0], frame_pts[16][2]])
    
    if 23 in frame_pts and 25 in frame_pts:
        seg_centers["Thigh_L"] = np.mean([frame_pts[23], frame_pts[25]], axis=0)[[0, 2]]
    if 24 in frame_pts and 26 in frame_pts:
        seg_centers["Thigh_R"] = np.mean([frame_pts[24], frame_pts[26]], axis=0)[[0, 2]]
    if 25 in frame_pts and 27 in frame_pts:
        seg_centers["Shank_L"] = np.mean([frame_pts[25], frame_pts[27]], axis=0)[[0, 2]]
    if 26 in frame_pts and 28 in frame_pts:
        seg_centers["Shank_R"] = np.mean([frame_pts[26], frame_pts[28]], axis=0)[[0, 2]]
    if 31 in frame_pts: seg_centers["Foot_L"] = np.array([frame_pts[31][0], frame_pts[31][2]])
    if 32 in frame_pts: seg_centers["Foot_R"] = np.array([frame_pts[32][0], frame_pts[32][2]])

    I_total = 0.0
    for seg_name, pos in seg_centers.items():
        if seg_name in DE_LEVA_MASS_FEMALE:
            r = np.linalg.norm(pos - pivot_xz)
            I_total += DE_LEVA_MASS_FEMALE[seg_name] * athlete_weight * (r ** 2)
    return I_total

# =================================================================
# GLAVNI PROGRAM
# =================================================================
print(f"\nUčitavam fajlove iz: {folder_path}")
print(f"Pronađeno fajlova: {len(all_files)}\n")

table_rows = []

for file in sorted(all_files):
    df_raw = pd.read_csv(file)
    filename_lower = os.path.basename(file).lower()
    
    athlete_key = None
    athlete_data = {"height": 1.65, "weight": 50.0, "type": "Nepoznato"}
    for key, val in ATHLETE_DB.items():
        if key in filename_lower:
            athlete_key = key
            athlete_data = val
            break
            
    atype = athlete_data["type"]
    weight = athlete_data["weight"]
    num_frames = len(df_raw['Frame'].unique())

    # Interpolacija NaN
    df_raw[['X_clean', 'Y_clean', 'Z_clean']] = df_raw.groupby('Landmark_ID')[['X_clean', 'Y_clean', 'Z_clean']].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both').ffill().bfill()
    )

    # Pivot noga
    foot_left = df_raw[df_raw['Landmark_ID'] == 31][['X_clean', 'Y_clean', 'Z_clean']].values
    foot_right = df_raw[df_raw['Landmark_ID'] == 32][['X_clean', 'Y_clean', 'Z_clean']].values
    move_left = np.sum(np.std(foot_left, axis=0)) if len(foot_left) > 0 else 999.0
    move_right = np.sum(np.std(foot_right, axis=0)) if len(foot_right) > 0 else 999.0
    
    planted_foot_id = 31 if move_left <= move_right else 32
    pivot_name = "Leva (31)" if planted_foot_id == 31 else "Desna (32)"

    frames_sorted = sorted(df_raw['Frame'].unique())
    angle_list, com_list, inertia_list = [], [], []

    for frame_idx in frames_sorted:
        f_df = df_raw[df_raw['Frame'] == frame_idx].set_index('Landmark_ID')
        frame_pts = {lm_id: f_df.loc[lm_id, ['X_clean', 'Y_clean', 'Z_clean']].values.astype(float) for lm_id in f_df.index}
        
        # Ugao
        angle_list.append(compute_body_angle_multi_vector(frame_pts))
        
        # CoM i Moment inercije
        if planted_foot_id in frame_pts:
            anchor = frame_pts[planted_foot_id]
            seg_pos = {
                "Trunk": np.mean([frame_pts.get(i, anchor) for i in [11, 12, 23, 24]], axis=0),
                "Head": frame_pts.get(0, anchor),
                "Thigh": np.mean([frame_pts.get(i, anchor) for i in [23, 24, 25, 26]], axis=0),
                "UpperArm": np.mean([frame_pts.get(i, anchor) for i in [11, 12, 13, 14]], axis=0),
            }
            com = np.zeros(3); tw = 0.0
            for sn, sp in seg_pos.items():
                if sn in DE_LEVA_MASS_FEMALE:
                    com += DE_LEVA_MASS_FEMALE[sn] * (sp - anchor)
                    tw += DE_LEVA_MASS_FEMALE[sn]
            if tw > 0: com /= tw
            com_list.append(com)
            
            inertia_list.append(compute_moment_of_inertia(frame_pts, weight, planted_foot_id))
        else:
            com_list.append(np.zeros(3))
            inertia_list.append(0.0)

    angles_raw = np.array(angle_list)
    theta_unwrapped = robust_unwrap(angles_raw)  #nacrtati od pocetka svake rotacije koji je otklon psebno kuk i delovi tela; da se dobije iblik sve momente inercije na isti grafik oidradi stabilnost 
    theta_fixed, _ = detect_and_fix_angle_outliers(theta_unwrapped, max_vel_deg=70.0)
    
    # Blago glađenje tek NAKON unwrap-a
    theta_smooth = savgol_filter(theta_fixed, window_length=9, polyorder=2)

    total_displacement_deg = np.degrees(np.abs(theta_smooth[-1] - theta_smooth[0]))
    total_rotations = total_displacement_deg / 360.0

    # Brzina i Ubrzanje
    omega = compute_angular_velocity_robust(theta_smooth, FPS)
    omega_smooth = savgol_filter(omega, window_length=7, polyorder=2)
    max_omega_deg = np.max(np.abs(np.degrees(omega_smooth)))
    mean_omega_deg = np.mean(np.abs(np.degrees(omega_smooth)))

    alpha = np.gradient(omega_smooth, dt)
    max_alpha = np.max(np.abs(savgol_filter(alpha, window_length=9, polyorder=2)))

    # Verifikacija cycle counting
    sh_x = df_raw[df_raw['Landmark_ID'] == 11]['X_clean'].values
    sh_z = df_raw[df_raw['Landmark_ID'] == 11]['Z_clean'].values    
    cycle_est = count_rotations_by_cycles(sh_x, sh_z) if len(sh_x) == num_frames else 0.0

    com_arr = np.array(com_list)
    com_xz = np.sqrt(com_arr[:, 0]**2 + com_arr[:, 2]**2)
    com_fluctuation_cm = np.std(savgol_filter(com_xz, window_length=15, polyorder=3)) * 100.0

    inertia_arr = savgol_filter(np.array(inertia_list), window_length=11, polyorder=2)
    I_mean = np.mean(inertia_arr)
    L_mean = np.mean(inertia_arr * np.abs(omega_smooth))
    KE_max = np.max(0.5 * inertia_arr * (omega_smooth ** 2))

    clean_name = athlete_key.upper() if athlete_key else os.path.basename(file).replace("_konacne.csv", "").upper()
    
    table_rows.append({
        "Atleta": clean_name,
        "Tip": atype,
        "Pivot": pivot_name,
        "Ugaoni pomeraj [°]": round(total_displacement_deg, 1),
        "Broj okreta": round(total_rotations, 2),
        "Maks ω [°/s]": round(max_omega_deg, 1),
        "Srednja ω [°/s]": round(mean_omega_deg, 1),
        "Maks α [rad/s²]": round(max_alpha, 2),
        "I_mean [kg·m²]": round(I_mean, 3),
        "Max KE [J]": round(KE_max, 1),
        "CoM flukt. [cm]": round(com_fluctuation_cm, 2),
        "Verif. (cycles)": round(cycle_est, 1)
    })

summary_df = pd.DataFrame(table_rows)

print("\n\n" + "=" * 130)
print("rezultatt")
print("=" * 130)
print(summary_df.to_string(index=False))
print("=" * 130)

# Hot Pink slika
fig, ax = plt.subplots(figsize=(18, len(summary_df) * 0.9 + 3), facecolor='#fff0f5')
ax.set_facecolor('#fff0f5')
ax.axis('off')
ax.axis('tight')

display_cols = ["Atleta", "Tip", "Ugaoni pomeraj [°]", "Broj okreta", 
                "Maks ω [°/s]", "Maks α [rad/s²]", "I_mean [kg·m²]", 
                "Max KE [J]", "CoM flukt. [cm]"]
display_df = summary_df[display_cols]

table_plot = ax.table(cellText=display_df.values,
                      colLabels=display_df.columns,
                      cellLoc='center',
                      loc='center')

table_plot.auto_set_font_size(False)
table_plot.set_fontsize(9)
table_plot.scale(1.3, 2.2)

for key, cell in table_plot.get_celld().items():
    cell.set_edgecolor('#ffb6c1')
    if key[0] == 0:
        cell.set_facecolor('#ff1493')
        cell.set_text_props(weight='bold', color='white', family='sans-serif', size=10)
    else:
        cell.set_facecolor('#ffe4e1' if key[0] % 2 == 0 else '#ffffff')
        cell.set_text_props(color='#333333', family='sans-serif', size=9)

plt.title("analiza fizickih zakonaaaa", 
          fontsize=13, fontweight='bold', color='#c71585', pad=20, family='sans-serif')

image_path = os.path.join(output_table_folder, "tabelica.png")
plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print(f"\n tabelica je sutnuta na :\n{image_path}\n")